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

Three panels model a binary where there are three cases, and each one has
asserted the wrong half at some point, so each third state is now driven on
its own: a gate stage that has written every planned shard is not one stopped
at shard 82; an /api/detect that did not answer is not an idle sweep; and a
run still training has not won on cost against one that finished. The rule
that lets an element be hidden at all is derived here rather than listed for
the same reason -- an author ``display:`` outranks the UA's ``[hidden]``, and
``.swctl`` and ``.wrkeyi`` were each found by a person noticing.

The header that folds once the page has scrolled is graded three ways,
because it fails three ways. Its rules are read (does the fold shed height,
can prefers-reduced-motion still win, does the sentinel sit under the
header); its observer is driven under node over the crossings that broke it;
and the page is opened in chromium and measured, because the failure that
took two review passes to find is a layout feedback loop -- folding a sticky
header scrolls the page, and the scroll is an input to the decision that
caused it. Nothing static can see that one. No chromium is a loud SKIP.

What a caller sees after nvidia-smi exits is driven too. It was not: this
file called its own reset(), which sets the process to None, and then
asserted the process was None, without ever calling _gpu() again. A check
that passes against the broken code as readily as the fixed one certifies
safety it never looked at, and that one sat over a live defect for months.

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


def check_no_shadowing(html):
    """One name, one function: the inline script is a single scope.

    Two `function copyText(t)` declarations 300 lines apart is legal
    JavaScript. The later one silently wins everywhere, including inside the
    earlier one's callers, and there is no error and no warning -- the page
    carried exactly that pair, and the three icon-only copy buttons said
    nothing on success and nothing on failure for as long as it lasted. The
    substitution that builds this page pastes shared helpers (COPY_JS, LB_JS)
    into a scope that already has its own, which is how it happens.
    """
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    seen = {}
    for m in re.finditer(r'^function ([A-Za-z_$][\w$]*)\s*\(', script, re.M):
        seen.setdefault(m.group(1), []).append(script.count('\n', 0, m.start())
                                               + 1)
    dup = [f'{n}() declared {len(at)} times, at script lines {at} — the last '
           f'one wins for every caller, including the earlier one\'s'
           for n, at in sorted(seen.items()) if len(at) > 1]
    if dup:
        raise SystemExit('SHADOWED HELPERS: ' + ' | '.join(dup))
    print(f'ok   {len(seen)} top-level helpers, each declared once')


def check_copy_say(html):
    """The icon-only copy buttons say whether the copy landed, either way.

    All three are a glyph, so copyOnto() -- which writes "Copied" over the
    button's own label -- is not available to them and the toast is the only
    place they can speak. Driven with isSecureContext false, because that is
    how this dashboard is served: navigator.clipboard does not exist on a
    plain http origin, so the execCommand fallback is the only path anyone
    using this page ever takes, and it is the one that can quietly fail.
    """
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    fns = []
    for name in ('copyText', 'copySay'):
        m = re.search(r'^function %s\(' % name, script, re.M)
        if not m:
            raise SystemExit(f'{name}() is not a top-level helper of the '
                             f'built script — the copy buttons have nothing '
                             f'to call')
        fns.append(_take_fn(script, m.start()))
    drive = r'''
'use strict';
let said = [], left = 0, works = true;
global.window = {isSecureContext: false};
// node ships a real navigator, and it is a getter — defineProperty is the
// only way to put the clipboard-less one of a plain http origin in its place
Object.defineProperty(global, 'navigator', {value: {}, configurable: true});
global.document = {
  createElement: () => ({style: {}, setAttribute(){}, select(){},
                         setSelectionRange(){}}),
  body: {appendChild(){ left++ }, removeChild(){ left-- }},
  execCommand: () => works,
};
function toast(t){ said.push(String(t)) }
__FNS__
const settle = () => new Promise(r => setImmediate(r));
(async () => {
  const bad = [];
  copySay('Africa_west', 'Africa_west');
  await settle(); await settle();
  if (said.length !== 1)
    bad.push('a copy that worked produced ' + said.length + ' messages: '
      + JSON.stringify(said));
  else if (!said[0].includes('Africa_west'))
    bad.push('the button did not say what it copied: ' + JSON.stringify(said[0]));
  const good = said[0];
  said = []; works = false;
  copySay('Africa_west', 'Africa_west');
  await settle(); await settle();
  if (said.length !== 1)
    bad.push('a copy that FAILED produced ' + said.length + ' messages: '
      + JSON.stringify(said) + ' — silence is indistinguishable from a '
      + 'button that did nothing');
  else if (said[0] === good)
    bad.push('a failed copy says the same thing as one that worked: '
      + JSON.stringify(said[0]));
  if (left !== 0)
    bad.push(left + ' off-screen textarea(s) left in the body');
  if (bad.length) { bad.forEach(b => console.log('FAIL ' + b)); process.exit(1) }
  console.log('ok   copy buttons speak on success and on failure');
})();
'''.replace('__FNS__', '\n'.join(fns))
    with tempfile.TemporaryDirectory() as tmp:
        js = os.path.join(tmp, 'copy.js')
        with open(js, 'w', encoding='utf-8') as f:
            f.write(drive)
        r = subprocess.run(['node', js], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def check_no_host_paths(html):
    """The page tells a stranger nothing about the operator's machine.

    This dashboard is published now and its readers are volunteers. The front
    page used to print the sweep database's absolute path -- the same string
    the repository's own leak check refuses to let into a commit, naming the
    account and the drive it all sits on.
    """
    import re as _re
    hits = sorted(set(_re.findall(r'/(?:home|Users|media|mnt|srv)/'
                      r'[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*', html)))
    # a bare mount point with nothing after it is a word, not a path
    hits = [h for h in hits if h.count('/') > 2]
    if hits:
        print('FAIL the page shows the operator\'s filesystem to every reader: '
              + ', '.join(hits[:4]))
        return 1
    print('ok   no host paths on the page')
    return 0


def check_freshness(html):
    """The mark that says how current the page is, at three ages.

    It was rendered once, at build time, as a green pulsing dot -- so a tab
    left open overnight showed the same "live" mark as one opened a minute
    after the rebuild, and the element whose entire job is to say how old this
    is was the one element that could not say it. Driven here at four minutes,
    at three hours and at a day and a half old.
    """
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    mark = '/* \u2500\u2500 how old is what you are looking at'
    if mark not in script:
        print('FAIL the freshness mark is not decided in the browser at all')
        return 1
    s = script.index(mark)
    iife = script[s:script.index('})();', s) + 5]
    m = re.search(r'data-at="(\d+)"', html)
    if not m:
        print('FAIL the page does not carry the time it was built, so nothing '
              'can work out how old it is')
        return 1
    built = int(m.group(1))
    if abs(time.time() - built) > 6 * 3600:
        print('FAIL the page says it was built %s, which is not when this '
              'build ran' % (built,))
        return 1

    drive = r"""
'use strict';
let NOW = 0;
const dot = {className: 'dot'};
const txt = {textContent: 'updated 14:03', title: ''};
const box = {getAttribute: () => String(NOW / 1000 - AGE * 60)};
global.document = {getElementById: (id) =>
  id === 'upd' ? box : id === 'updDot' ? dot : id === 'updT' ? txt : null};
global.setInterval = () => 0;
let AGE = 0;
const realNow = Date.now;
Date.now = () => NOW;
const bad = [];
function at(mins, wantClass, why) {
  AGE = mins; NOW = 1787000000000;
  dot.className = 'dot'; txt.textContent = 'updated 14:03'; txt.title = '';
  __IIFE__
  if (dot.className !== wantClass)
    bad.push(mins + ' minutes old drew ' + JSON.stringify(dot.className) +
             ', want ' + JSON.stringify(wantClass) + ' -- ' + why);
  return txt.textContent;
}
const fresh = at(4, 'dot', 'the rebuild is hourly, so this is current');
if (!/14:03/.test(fresh))
  bad.push('a current page stopped naming the time it was built: ' + fresh);
const aging = at(180, 'dot aging', 'three hours is three missed rebuilds');
if (!/3h ago/.test(aging))
  bad.push('a three-hour-old page reads ' + JSON.stringify(aging) +
           ' -- the reader has to do the arithmetic');
const stale = at(2160, 'dot stale', 'a day and a half is not live');
if (!/1d ago|2d ago/.test(stale))
  bad.push('a day-and-a-half-old page reads ' + JSON.stringify(stale));
if (bad.length) { bad.forEach(b => console.log('FAIL ' + b)); process.exit(1) }
console.log('ok   the freshness mark is decided when you look, not when it '
            + 'was built');
""".replace('__IIFE__', iife)
    with tempfile.TemporaryDirectory() as tmp:
        js = os.path.join(tmp, 'fresh.js')
        with open(js, 'w', encoding='utf-8') as f:
            f.write(drive)
        r = subprocess.run(['node', js], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


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


def _css_rules(css):
    """[(offset, selector, declarations, enclosing at-rule)], comments gone.

    Flattens @media so a rule inside one is reported at its own offset, which
    is what the cascade actually goes on: a media query adds no specificity,
    so where its declaration SITS is the whole of its power over an equally
    specific rule. @keyframes and @font-face are skipped whole -- their inner
    blocks are percentages and descriptors, not selectors.
    """
    css = re.sub(r'/\*.*?\*/', ' ', css, flags=re.S)
    out, i, n, stack = [], 0, len(css), []
    while i < n:
        j = css.find('{', i)
        if j < 0:
            break
        head = css[i:j]
        for _ in range(min(head.count('}'), len(stack))):
            stack.pop()            # every } here closed an at-rule we entered
        # the closing brace of whatever block just ended leads the next
        # selector; leaving it on turned `h1` into `} h1` and every lookup
        # for a rule that happens to follow an @media silently missed
        sel = ' '.join(re.sub(r'^[\s}]+', '', head).split())
        if sel.startswith('@'):
            if sel.split('(')[0].split()[0] in ('@media', '@supports'):
                stack.append(sel)  # step inside: its children are real rules
                i = j + 1
                continue
            depth, k = 0, j
            while k < n:           # skip the block whole
                if css[k] == '{':
                    depth += 1
                elif css[k] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            i = k + 1
            continue
        k = css.find('}', j)
        if k < 0:
            break
        out.append((i, sel, css[j + 1:k], ' '.join(stack)))
        i = k + 1
    return out


def _px(decls, prop):
    """The px number a declaration block gives ``prop``, or None.

    Bare ``0`` counts -- ``font-size:0`` is how the header collapses its
    ambient text, and demanding the unit read that as "no rule at all".
    """
    m = re.search(r'(?:^|;)\s*%s\s*:\s*(-?[\d.]+)(px)?(?=[;\s}]|$)'
                  % re.escape(prop), decls)
    if not m or (m.group(2) is None and float(m.group(1)) != 0):
        return None
    return float(m.group(1))


def check_header_compact(html):
    """The header folds what is ambient once the page has scrolled.

    Three things here are only visible statically, and each one shipped.

    WHERE THE SENTINEL SITS. Folding takes 54px -- 140px at a narrow width --
    out of a POSITION:STICKY element, and the browser answers a shrink above
    the reader's position by scrolling back the same distance to keep the
    content under their eye still. A sentinel above the header does not move
    when the header does, so that scroll-back carried the page straight over
    a sentinel it had only just crossed; the unfold that followed was refused
    as a flutter and the header stayed folded at the top of the page, no
    title and no tagline, with no gesture that brought it back. Below the
    header the sentinel moves with the fold and the two cancel -- measured,
    the crossing point holds to within a pixel at every width.

    THE CASCADE. ``@media(prefers-reduced-motion:reduce){h1{transition:none}}``
    placed ABOVE ``h1{...;transition:font-size .18s}`` does nothing at all: a
    media query adds no specificity and both selectors are (0,0,1), so the
    later one wins in both media states. That left the largest text on the
    page as the one thing still easing for the reader who asked for no
    motion, which is louder than the uniform ease the setting exists to stop.

    WHAT IS SHED. The point of the whole thing is viewport, so the folded
    header has to be shorter. check_header_shrinks() measures that in a
    browser; here it is read out of the rules, which is the half that still
    works on a machine with no chromium.
    """
    body = html[html.index('</style>'):]
    css = '\n'.join(re.findall(r'<style[^>]*>(.*?)</style>', html, re.S))
    rules = _css_rules(css)
    bad = []

    def one(sel):
        got = [(o, d) for o, s, d, _a in rules if s == sel]
        if not got:
            bad.append(f'no {sel}{{...}} rule on the page')
            return -1, ''
        return got[-1]

    cue = re.search(r'<i[^>]*\bclass="scrollcue"[^>]*>', body)
    if not cue:
        bad.append('no <i class="scrollcue"> sentinel in the markup — nothing '
                   'for the observer to watch, so the header never folds')
    else:
        if 'id="scrollcue"' not in cue.group(0):
            bad.append('the sentinel carries no id="scrollcue" — the observer '
                       'looks it up by id and gives up quietly without one')
        if 'aria-hidden="true"' not in cue.group(0):
            bad.append('the sentinel is not aria-hidden — it is a scroll '
                       'landmark with no content, and a screen reader should '
                       'never be told about it')
        if not (body.index('</header>') < cue.start()):
            bad.append('the scrollcue sentinel sits ABOVE the header. Folding '
                       'the header scrolls the viewport back by what it shed, '
                       'and a sentinel that did not move with it is re-crossed '
                       'by that scroll-back — the unfold is then refused as a '
                       'flutter and the header stays folded at the top of the '
                       'page. It has to sit after </header>')

    # the sentinel must cost no layout in EITHER state, or folding the header
    # would move the page by the sentinel as well as by the header
    box = {}
    for sel, why in (
            ('.scrollcue', 'there is no sentinel to observe, so the header '
                           'never folds'),
            ('body.compact .scrollcue',
             'the folded sentinel then measures the same as the unfolded one, '
             'and unfolding happens on the exact pixel folding did — the '
             'sentinel rides under the header, so the fold moves it and the '
             'browser\'s compensating scroll moves the viewport by the same '
             'amount, landing the crossing back on itself. An exact boundary '
             'ping-pongs on a rounding error')):
        got = [d for o, s, d, _a in rules if s == sel]
        if not got:
            bad.append(f'no {sel}{{...}} rule — {why}')
            continue
        h, mb = _px(got[-1], 'height'), _px(got[-1], 'margin-bottom')
        if h is None or mb is None:
            bad.append(f'{sel} does not set both a height and a margin-bottom')
        elif h + mb != 0:
            bad.append(f'{sel} is {h}px tall with {mb}px of margin-bottom — '
                       f'the negative margin has to cancel the height exactly '
                       f'or the sentinel costs {h + mb}px of layout')
        else:
            box[sel] = h
    if len(box) == 2 and not box['body.compact .scrollcue'] < box['.scrollcue']:
        bad.append(f'the folded sentinel is {box["body.compact .scrollcue"]}px '
                   f'against {box[".scrollcue"]}px unfolded — it has to be '
                   f'SHORTER, so the point the header unfolds at sits below '
                   f'the point it folded at rather than exactly on it')

    # ── what the fold actually takes off the height ──
    # A floor, not a measurement: three declared sheds that each stand for
    # real height, added up at the page's own line-height of 1.5. The browser
    # is what measures; this is what still runs without one, and what says
    # WHICH rule went missing when the number collapses.
    shed = []
    _, hdr = one('header')
    _, hdrc = one('body.compact header')
    pad = re.search(r'(?:^|;)\s*padding\s*:\s*(-?[\d.]+)px\s+[\d.]+px\s+'
                    r'(-?[\d.]+)px', hdr)
    if not pad:
        bad.append('the header rule no longer sets a three-value padding, so '
                   'there is nothing to compare the folded one against')
    else:
        was = float(pad.group(1)) + float(pad.group(2))
        now = (_px(hdrc, 'padding-top') or 0) + (_px(hdrc, 'padding-bottom') or 0)
        shed.append(('header padding', was - now))
    _, h1 = one('h1')
    _, h1c = one('body.compact h1')
    big = re.search(r'font-size\s*:\s*clamp\([^)]*?(-?[\d.]+)px\s*\)', h1)
    if not big:
        bad.append('the h1 rule no longer sets a clamp() font-size')
    else:
        shed.append(('title', float(big.group(1)) - (_px(h1c, 'font-size') or 0)))
    _, sub = one('.sub')
    subc = [d for o, s, d, _a in rules
            if 'body.compact .sub' in s.split(',')]
    grown = (_px(sub, 'font-size') or 0) * 1.5 + (_px(sub, 'margin-top') or 0)
    if not subc or _px(subc[-1], 'font-size') != 0:
        bad.append('body.compact does not collapse .sub — the tagline is the '
                   'tallest ambient thing in the header and shedding it is '
                   'most of the point')
    else:
        shed.append(('tagline', grown))
    thin = sum(v for _, v in shed)
    if any(v <= 0 for _, v in shed) or thin < 30:
        bad.append('the folded header is not measurably shorter: '
                   + ', '.join(f'{n} {v:+.1f}px' for n, v in shed)
                   + f' = {thin:+.1f}px. The whole point is viewport')

    # ── prefers-reduced-motion has to be able to win ──
    for off, sel, decls, at in rules:
        # only rules inside a reduced-motion query -- an ordinary rule turning
        # a transition off is just a rule
        if 'reduced-motion' not in at:
            continue
        if 'transition:none' not in decls.replace(' ', ''):
            continue
        for name in [s.strip() for s in sel.split(',')]:
            for o2, s2, d2, at2 in rules:
                if o2 <= off or 'reduced-motion' in at2:
                    continue
                if not re.search(r'(?:^|;)\s*transition\s*:', d2):
                    continue
                if name in [x.strip() for x in s2.split(',')]:
                    bad.append(
                        f'{name}{{transition:...}} is declared AFTER the '
                        f'reduced-motion block that tries to silence it '
                        f'({sel}). A media query adds no specificity, so the '
                        f'later rule wins in both media states and {name} '
                        f'keeps easing for the one reader who asked for no '
                        f'motion — move the base rule above the query')

    # the file has been bitten twice by an author display: outranking the UA's
    # [hidden]{display:none}; the fold is a place it would be easy to reach for
    for off, sel, decls, at in rules:
        if sel.startswith('body.compact') and re.search(r'(?:^|;)\s*display\s*:',
                                                        decls):
            bad.append(f'{sel} names a display. Fold with max-height/opacity/'
                       f'font-size instead: an author display: outranks the '
                       f'UA\'s [hidden]{{display:none}}, which is how .swctl '
                       f'and .wrkeyi each stayed on screen while the page '
                       f'thought they were hidden')
    if bad:
        raise SystemExit('HEADER FOLD FAILURES: ' + ' | '.join(bad))
    print('ok   header fold: sentinel under the header and aria-hidden, '
          'reduced-motion can win, folded header sheds '
          + ' + '.join(f'{n} {v:.0f}px' for n, v in shed)
          + f' = {thin:.0f}px')


HEADER_FOLD_STUB = r'''
'use strict';
var BAD = [];
function ck(c, m) { if (!c) BAD.push(m) }
var CLASS = {}, TIMERS = [], CB = null, THREW = null;
global.AFTER = false;
global.setTimeout = function (f) { TIMERS.push(f); return TIMERS.length };
function runTimers() { var t = TIMERS.slice(); TIMERS.length = 0;
                       t.forEach(function (f) { if (f) f() }) }
function compact() { return !!CLASS.compact }
// Cross the sentinel the way a scroll does, with the timestamp the browser
// would have stamped the crossing with -- `time` is what the page reads, so
// the hold window can be driven from both sides without the test sleeping.
function cross(out, t) { CB([{isIntersecting: !out, time: t}]) }
function start(withIO, withCue) {
  CLASS = {}; TIMERS = []; CB = null; THREW = null; global.AFTER = false;
  global.document = {
    getElementById: function (id) {
      return (withCue && id === 'scrollcue') ? {id: id} : null;
    },
    body: {classList: {
      toggle: function (n, on) { CLASS[n] = !!on },
      contains: function (n) { return !!CLASS[n] },
    }},
  };
  if (withIO) {
    global.IntersectionObserver = function (cb) {
      CB = cb;
      return {observe: function () {}, disconnect: function () {}};
    };
  } else {
    delete global.IntersectionObserver;
  }
  // AFTER stands for every handler the page binds below this block: they are
  // in the same <script>, so a throw here takes all of them with it.
  try { (0, eval)(SRC + '\nglobal.AFTER = true;') } catch (e) { THREW = e }
}
var SRC = require('fs').readFileSync(process.argv[2], 'utf8');

start(false, true);
ck(THREW === null, 'a browser with no IntersectionObserver threw: ' + THREW);
ck(global.AFTER, 'no IntersectionObserver killed the rest of the script — '
   + 'every handler bound after this block is gone');
start(true, false);
ck(THREW === null, 'a page with no #scrollcue threw: ' + THREW);
ck(global.AFTER, 'a missing #scrollcue killed the rest of the script');

start(true, true);
ck(THREW === null, 'the fold block threw on a normal page: ' + THREW);
ck(CB !== null, 'no IntersectionObserver was constructed — nothing watches '
   + 'the sentinel, so the header never folds');
if (CB) {
  // the callback observe() delivers straight away, before anything scrolled
  cross(false, 120);
  ck(!compact(), 'the header starts folded, before anything has scrolled');
  // 19 ms later: a wheel flicked while the page is still painting, which is
  // how an ops page actually gets used
  cross(true, 139);
  ck(compact(), 'a scroll 19ms after load did not fold the header — the '
     + 'load-time callback armed the hold window, so the first real scroll '
     + 'lands inside it and is refused');
  cross(false, 170);
  ck(compact(), 'a reversal 31ms later unfolded it — that is the flutter the '
     + 'hold window exists to stop');
  runTimers();
  ck(!compact(), 'the refused reversal was DROPPED, not held. An '
     + 'IntersectionObserver only reports changes, so it is never offered '
     + 'again: the header stays folded at the top of the page with nothing '
     + 'left to unfold it');
  cross(true, 1500);
  ck(compact(), 'a crossing well outside the hold window was refused too');
  cross(false, 1550);
  ck(compact(), 'a reversal 50ms later was applied at once — the hold window '
     + 'is not being applied');
  cross(true, 1600);            // the reader put it back themselves
  runTimers();
  ck(compact(), 'a held reversal the reader had already undone was applied '
     + 'anyway when the window closed');
}
if (BAD.length) { BAD.forEach(function (b) { console.log('FAIL ' + b) });
                  process.exit(1) }
console.log('ok   header fold observer (guarded, folds, holds a reversal '
            + 'without losing it, and is not armed by the load)');
'''


def check_header_fold(html):
    """Drive the fold observer itself, over the sequences that broke it.

    Node has no IntersectionObserver, which is the point twice over: it is
    the case the page's own guard exists for, and it means the block has to
    be handed a stub before any of its behaviour can be looked at. Without
    one the whole thing returns on its first line and every assertion below
    would pass against a page that never folds.
    """
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    mark = '/* Fold the header once a screenful'
    if mark not in script:
        raise SystemExit('the header-fold block is not in the built script — '
                         'the header never folds and nothing below can see it')
    s = script.index(mark)
    src = script[s:script.index('})();', s) + 5]
    with tempfile.TemporaryDirectory() as tmp:
        js = os.path.join(tmp, 'fold.js')
        with open(js, 'w', encoding='utf-8') as f:
            f.write(src)
        run = os.path.join(tmp, 'run.js')
        with open(run, 'w', encoding='utf-8') as f:
            f.write(HEADER_FOLD_STUB)
        r = subprocess.run(['node', run, js], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def check_header_shrinks():
    """Measure the folded header, and prove it unfolds again, in a browser.

    Everything this catches is a layout feedback loop, and no amount of
    reading the rules can see one. Folding removes height from a sticky
    element, the browser compensates by scrolling, and the scroll is an input
    to the same decision that caused it -- so the header could fold at
    scrollY 110, be carried back to 56 by that compensation, refuse the
    unfold as a flutter, and sit folded at the very top of the page for the
    rest of the session. Every static check in this file passed on that page.

    Loud SKIP, not a quiet pass, when there is no browser: what goes
    unchecked is named, because this is the half that found the bug.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f'SKIP: no playwright ({e}) — the folded header was not '
              f'measured and the scroll-anchoring latch went unchecked')
        return 0
    bad, url = [], 'file://' + INDEX
    try:
        pw = sync_playwright().start()
    except Exception as e:
        print(f'SKIP: playwright would not start ({e}) — the folded header '
              f'was not measured')
        return 0
    try:
        try:
            br = pw.chromium.launch()
        except Exception as e:
            print(f'SKIP: no chromium ({str(e).splitlines()[0]}) — the folded '
                  f'header was not measured')
            return 0
        tall = 'return document.querySelector("header").getBoundingClientRect().height'
        on = 'return document.body.classList.contains("compact")'
        sizes = []
        for w in (1440, 760):
            pg = br.new_page(viewport={'width': w, 'height': 900})
            pg.goto(url, wait_until='load')
            pg.wait_for_timeout(400)
            # measured with the easing off: mid-transition is not a height
            pg.add_style_tag(content='*{transition:none!important;'
                                     'animation:none!important}')
            pg.evaluate('()=>document.body.classList.remove("compact")')
            e = pg.evaluate('()=>{%s}' % tall)
            pg.evaluate('()=>document.body.classList.add("compact")')
            c = pg.evaluate('()=>{%s}' % tall)
            sizes.append((w, e, c))
            if e - c < 24:
                bad.append(f'at {w}px the folded header is {c:.0f}px against '
                           f'{e:.0f}px unfolded — it sheds {e - c:.0f}px, '
                           f'which is not worth moving the page for. The '
                           f'point of folding it is viewport')
            pg.close()

        pg = br.new_page(viewport={'width': 1440, 'height': 900})
        pg.goto(url, wait_until='load')
        pg.wait_for_timeout(600)
        # count every change of the class, not just where it ends up: the
        # loop this guards can settle into a flip every 260ms rather than
        # latching, and a page read only at rest calls that fine
        pg.evaluate('()=>{window.__n=0;new MutationObserver(function(){'
                    'window.__n++}).observe(document.body,{attributes:true,'
                    'attributeFilter:["class"]})}')
        folded = False
        # 110 and 140 are inside the band a sentinel ABOVE the header cannot
        # survive: folding there scrolls the page back over a sentinel it has
        # only just crossed, and it is asked to unfold again straight away.
        for y in (110, 140, 420):
            pg.evaluate('()=>{window.__n=0;window.scrollTo(0,%d)}' % y)
            pg.wait_for_timeout(1100)
            folded = folded or pg.evaluate('()=>{%s}' % on)
            n = pg.evaluate('()=>window.__n')
            if n > 1:
                bad.append(f'parked at y={y} the header folded and unfolded '
                           f'{n} times over a second with nobody touching it '
                           f'— folding scrolls the page back over the sentinel '
                           f'and the crossing feeds its own cause')
            pg.evaluate('()=>window.scrollTo(0,0)')
            pg.wait_for_timeout(650)
            if pg.evaluate('()=>{%s}' % on):
                bad.append(f'scrolled to y={y} and back to the top, and the '
                           f'header is STILL folded — the reader is looking at '
                           f'the top of the page with no title and no tagline, '
                           f'and no scrolling gets them back')
        if not folded:
            bad.append('the header never folded at any scroll position — the '
                       'sentinel is never crossed, or nothing is watching it')
        # the wheel flicked down and straight back up: two crossings inside
        # the hold window, and the second one used to be thrown away
        pg.mouse.move(700, 500)
        pg.mouse.wheel(0, 700)
        pg.mouse.wheel(0, -700)
        pg.wait_for_timeout(1400)
        if pg.evaluate('()=>{%s}' % on) and not pg.evaluate('()=>window.scrollY'):
            bad.append('a wheel flicked down and straight back up left the '
                       'header folded at scrollY 0 — the reversal was refused '
                       'as a flutter and then dropped')
        pg.close()
        br.close()
    finally:
        pw.stop()
    if bad:
        for b in bad:
            print('FAIL ' + b)
        return 1
    print('ok   folded header measures '
          + ', '.join(f'{c:.0f}px against {e:.0f} at {w}px' for w, e, c in sizes)
          + ', and unfolds again from every scroll position')
    return 0


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

            # Every call names its annotator: flag_crop refuses a write it
            # cannot attribute, so a guard that left it out would be driving
            # the refusal path and asserting nothing about flagging. The
            # attribution itself is graded in adv_attribution.py.
            who = 'flagtester'
            b, c = dash.flag_crop('../../etc/passwd', by=who)
            want(c == 400 and not b['ok'], 'malformed name did not 400')
            b, c = dash.flag_crop('', by=who)
            want(c == 400, 'empty name did not 400')

            b, c = dash.flag_crop(name, by=who)
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

            b, _ = dash.flag_crop(name, by=who)  # idempotent
            want(b['ok'] and b.get('duplicate') is True, f're-flag: {b}')
            want(len(lines()) == 1, 'a duplicate flag appended a second line')

            # lost the race with the pruner entirely: record it anyway
            gone = '1785663399999_777777777777777_042.jpg'
            b, c = dash.flag_crop(gone, by=who)
            want(c == 200 and b['ok'] and b['copied'] is False,
                 f'pruned crop should be ok/copied:false, got {b}')
            want(len(lines()) == 2, 'pruned crop did not get a ledger line')
            want([r for r in lines() if r['image_id'] == '777777777777777'],
                 'pruned crop lost its image_id')

            # 12 threads racing on one name -> exactly one line
            race = '1785663311111_888888888888888_055.jpg'
            with open(os.path.join(crops, race), 'wb') as f:
                f.write(b'\xff\xd8x')
            ts = [threading.Thread(target=dash.flag_crop, args=(race,),
                                   kwargs={'by': who})
                  for _ in range(12)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            want(len([r for r in lines() if r['crop'] == race]) == 1,
                 'concurrent flags of one crop duplicated the ledger line')

            b, _ = dash.flag_crop(name, undo=True, by=who)
            want(b['ok'] and b['undone'], f'undo: {b}')
            want(not os.path.exists(os.path.join(dash.HN_CROPS, name)),
                 'undo left the copied crop behind')
            want(not os.path.exists(os.path.join(dash.HN_FULL, name)),
                 'undo left the copied full frame behind')
            want([r['crop'] for r in lines()] == [gone, race],
                 f'undo mangled the ledger: {[r["crop"] for r in lines()]}')
            b, _ = dash.flag_crop(name, undo=True, by=who)  # no-op
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
// A fresh copy, because the panel's memory (lastJ, the remembered roster it
// reads from localStorage at construction) is what several of these cases are
// about, and a panel that has already seen a frame cannot show what the first
// poll of the day looks like.
function freshRender() {
  try {
    return new Function(src + '\nreturn render;')();
  } catch (e) {
    console.log('FAIL: could not evaluate detect IIFE body: ' + e);
    process.exit(1);
  }
}
const render = freshRender();

for (const [name, p] of Object.entries(payloads)) {
  // "the server did not answer" is neither running nor idle; it is driven
  // below, where the frame BEFORE it is what the assertion is about
  if (p === null || p.ever === false) continue;
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
      // A SECTION WITH NOTHING TO REPORT IS NOT SHOWN. It used to spend two
      // lines saying so -- a heading, and a sentence under it explaining the
      // absence -- and both appear with the first measurement instead.
      if ((p.crops_classified || 0) > 0) {
        if (!els['detHealth']._innerHTML.includes('dband'))
          failures.push(name + ': health gauge missing');
        if (els['detHealthHead'].hidden || els['detHealth'].hidden)
          failures.push(name + ': the classifier section is hidden though '
            + p.crops_classified + ' crops were classified');
      } else {
        if (els['detHealth']._innerHTML)
          failures.push(name + ': the classifier section says '
            + JSON.stringify(els['detHealth']._innerHTML)
            + ' with nothing classified');
        if (!els['detHealthHead'].hidden)
          failures.push(name + ': a Classifier heading is shown over nothing');
      }
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
      // Cumulative, both of these: an error tally and a classifier share are
      // facts about work already done, and the moment you want them is the
      // postmortem — a run killed mid-sweep left 64 decode errors with no
      // state of the page in which they could be read.
      const er = els['detErrs']._innerHTML,
            errN = Object.values(p.errors || {}).reduce((a, b) => a + b, 0);
      if (errN && !er.includes(errN + ' error'))
        failures.push(name + ': ' + errN + ' errors on a stopped run render as '
          + JSON.stringify(er) + ' — they count frames that already failed');
      if (!errN && !/0 errors/.test(er))
        failures.push(name + ': a stopped run with no errors does not say so: '
          + JSON.stringify(er));
      // ...and a measurement OUTLIVES the run that made it: a stopped run
      // that classified crops still reports the share it measured.
      if ((p.crops_classified || 0)
          && !els['detHealth']._innerHTML.includes('dband'))
        failures.push(name + ': the classifier line reads '
          + JSON.stringify(els['detHealth']._innerHTML) + ' on a stopped run '
          + '— what share of crops were classified outlives the run that '
          + 'classified them');
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
      for (const id of ['detDrives', 'detRegions', 'detErrs'])
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
// ── unknown is not idle ───────────────────────────────────────────────────
// A fetch that failed and a status document that is not there both say
// NOTHING about the sweep, and painting either as "idle" redrew a finished
// 32.5M-image harvest as "sweep idle · 0 of 17 complete": the remembered
// region roster with every bar at zero and a header counting them as not
// started. Via a missing status.json it does not self-correct on the next
// poll — it is what the panel says from then on.
(function () {
  const R17 = ['Africa_west', 'Africa_east', 'Africa_south', 'Africa_north',
    'Europe', 'Europe_east', 'South_Asia', 'Southeast_Asia', 'East_Asia',
    'Central_Asia', 'Middle_East', 'North_America', 'Central_America',
    'South_America', 'Oceania', 'Caribbean', 'Arctic'];
  const snap = () => JSON.stringify(Object.keys(els).sort().map(k =>
    [k, String(els[k].textContent), String(els[k]._innerHTML),
     JSON.stringify(els[k].style)]));
  for (const [tag, ans] of [['fetch failed', null],
                            ['no status.json', {ever: false}],
                            ['the route threw', {ever: false, error: 'boom'}]]) {
    const R = freshRender();
    R(payloads.finished_idle);
    const held = snap(), reg = String(els['detRegHead'].textContent);
    R(ans);
    if (snap() !== held)
      failures.push('unknown/' + tag + ': the panel repainted from an answer '
        + 'that says nothing about the sweep. It has to hold the last good '
        + 'frame, the way the machine and gate panels beside it do — region '
        + 'header was ' + JSON.stringify(reg) + ', now '
        + JSON.stringify(String(els['detRegHead'].textContent)));
  }
  // The first poll of the day: nothing to hold, and a roster remembered from
  // the last visit. It may say it does not know; it may not read that roster
  // as seventeen regions nobody has started.
  const realGet = localStorage.getItem;
  localStorage.getItem = () =>
    JSON.stringify({ drives: ['lynx', 'bobcat'], regions: R17 });
  const blank = () => Object.keys(els).forEach(k => {
    els[k].textContent = ''; els[k]._innerHTML = '';
  });
  try {
    blank();
    const R = freshRender();
    R(null);
    const said = String(els['detOff'].textContent);
    if (!said)
      failures.push('unknown/first poll: the status line says nothing at all');
    if (/idle/.test(said))
      failures.push('unknown/first poll: "' + said + '" — no answer is not a '
        + 'report that nothing is running');
    if (/\d+ of \d+/.test(String(els['detRegHead'].textContent)))
      failures.push('unknown/first poll: region header reads '
        + JSON.stringify(els['detRegHead'].textContent)
        + ' from a remembered roster and no answer');
    // ...and the same roster under an idle document that carries no regions:
    // the rows are drawn (dashed) so the panel does not jump, but a roster is
    // not an answer about how much of it is done
    blank();
    const R2 = freshRender();
    R2({ running: false, finished: true, state: 'stopped', age_s: 4269,
         imgs_done: 32542334, imgs_total: 32542334, regions: {}, drives: {},
         errors: {} });
    if (/\d+ of \d+ complete/.test(String(els['detRegHead'].textContent)))
      failures.push('idle/no regions: region header reads '
        + JSON.stringify(els['detRegHead'].textContent) + ' — the count is a '
        + 'count of what the payload said, and this one said nothing');
  } finally {
    localStorage.getItem = realGet;
  }
  console.log('ok   unknown holds the last good frame, and a remembered '
    + 'roster is not a report');
})();

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
        # Every planned shard is on disk. That is a third state beside running
        # and stopped-part-way, and the panel did not have it: a gate that had
        # judged all 4,688,510 of its detections said "paused", offered
        # "Resume", and put "3,292,062 frames to open" under a full bar
        # reading 100.0%.
        'gate_done': {
            'ever': True, 'stage': 'gate', 'planned': True, 'running': False,
            'can_run': True, 'done': True, 'rows': 4688510, 'total': 4688510,
            'shards': 165, 'shards_total': 165, 'pct': 1.0,
            'dog_share': 0.184, 'dogs': 862687, 'dogs_of': 4688510,
            'rate': 0, 'sustained': 66.2, 'eta_s': None,
            'model': 'dog-bin gate', 'images': 3292062, 'images_done': None,
            'img_s': None, 'created': '2026-08-07 22:28:52', 'upstream': None},
        # stopped at shard 82, which is what paused actually means
        'gate_paused': {
            'ever': True, 'stage': 'gate', 'planned': True, 'running': False,
            'can_run': True, 'done': False, 'rows': 2330000, 'total': 4688510,
            'shards': 82, 'shards_total': 165, 'pct': 0.4969,
            'dog_share': 0.184, 'dogs': 428720, 'dogs_of': 2330000,
            'rate': 0, 'sustained': 66.2, 'eta_s': None,
            'model': 'dog-bin gate', 'images': 3292062, 'images_done': None,
            'img_s': None, 'created': '2026-08-07 22:28:52', 'upstream': None},
        # rows reaching the total is NOT the finish line, and deriving `done`
        # from it here would put the finished state on a run with five shards
        # still to write: `total` is the plan's estimate, and the runner's own
        # test is the shards, which is why the server ships `done` at all.
        'gate_rows_full_not_done': {
            'ever': True, 'stage': 'gate', 'planned': True, 'running': False,
            'can_run': True, 'done': False, 'rows': 4688510, 'total': 4688510,
            'shards': 160, 'shards_total': 165, 'pct': 1.0,
            'dog_share': 0.184, 'dogs': 862687, 'dogs_of': 4688510,
            'rate': 0, 'sustained': 66.2, 'eta_s': None,
            'model': 'dog-bin gate', 'images': 3292062, 'images_done': None,
            'img_s': None, 'created': '2026-08-07 22:28:52', 'upstream': None},
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
// ── the finished stage, and the two states it must stay distinct from ──
function fresh(p){
  for (var k in els) { els[k].textContent=''; els[k].title='';
                       els[k].disabled=false }
  window.__stage = p.stage; paint(p);
}
fresh(P.gate_done);
if (String(els.gateState.textContent) !== 'complete')
  bad.push('a stage with every planned shard on disk reads '
    + JSON.stringify(els.gateState.textContent) + ', not "complete"');
if (!els.gateBtn.disabled || /Resume/.test(String(els.gateBtn.textContent)))
  bad.push('a finished stage offers '
    + JSON.stringify(els.gateBtn.textContent) + ' (disabled='
    + els.gateBtn.disabled + ') — the runner skips every shard already '
    + 'written, so the click starts a job that exits having done nothing, '
    + 'and "Resume" names work that does not exist');
if (!/nothing left to resume/.test(String(els.gateBtn.title)))
  bad.push('the unavailable button does not say why: '
    + JSON.stringify(els.gateBtn.title));
if (String(els.gEta.textContent) !== 'complete')
  bad.push('a finished stage reports an ETA of '
    + JSON.stringify(els.gEta.textContent) + ', not "complete"');
if (/frames to open/.test(String(els.gRun.textContent)))
  bad.push('a finished stage still has frames to open: '
    + JSON.stringify(els.gRun.textContent));
if (!/frames opened/.test(String(els.gRun.textContent)))
  bad.push('a finished stage does not say what it opened: '
    + JSON.stringify(els.gRun.textContent));
// stopped at shard 82 — every one of those readings has to flip back
['gate_paused', 'gate_rows_full_not_done'].forEach(function (name) {
  fresh(P[name]);
  if (String(els.gateState.textContent) !== 'paused')
    bad.push(name + ': an unfinished stage reads '
      + JSON.stringify(els.gateState.textContent) + ', not "paused" — '
      + 'shards ' + P[name].shards + ' of ' + P[name].shards_total
      + ' are written');
  if (els.gateBtn.disabled || !/Resume/.test(String(els.gateBtn.textContent)))
    bad.push(name + ': there is work left and the button says '
      + JSON.stringify(els.gateBtn.textContent) + ' (disabled='
      + els.gateBtn.disabled + ')');
  if (String(els.gEta.textContent) === 'complete')
    bad.push(name + ': an unfinished stage reports "complete"');
  if (!/frames to open/.test(String(els.gRun.textContent)))
    bad.push(name + ': an unfinished stage says '
      + JSON.stringify(els.gRun.textContent));
});
if (bad.length) { bad.forEach(function(b){ console.log('FAIL ' + b) });
  process.exit(1) }
console.log('ok   gate panel: two stages, an unplanned one says so, and a '
  + 'finished one is not a paused one');
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
        # Every metric also reports the ceiling it reached. Only recall carried
        # one, so a run whose precision touched 0.8431 was compared on the
        # 0.7855 its promoted checkpoint happened to hold and read as further
        # behind than it was.
        for want in ('best recall at any epoch', 'best precision at any epoch',
                     'best mAP50 at any epoch'):
            if want not in html:
                bad.append(f'the comparison omits "{want}" — the run is '
                           f'reported below the best it ever scored')
        # a ceiling without its epoch reads as one model's scorecard, and these
        # peaks land on different epochs: no shipped checkpoint holds them all
        cells = _diff_peak_cells(html)
        for label, txt in cells.items():
            if 'epoch' not in txt:
                bad.append(f'"{label}" states a peak with no epoch beside it, '
                           f'so it reads as a score some checkpoint held')
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('ok   run comparison: subtracts A from B, and shows what the model '
          'is shipped on')
    return 0


def _diff_peak_cells(html):
    """{row label: both value cells} for every "best ... at any epoch" row."""
    out = {}
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S):
        m = re.match(r'\s*<th>(.*?)</th>(.*)', tr, re.S)
        if not m:
            continue
        label = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        if not (label.startswith('best ')
                and (label.endswith('at any epoch') or label == 'best mAP50-95')):
            continue
        cells = re.findall(r'<td[^>]*>(.*?)</td>', m.group(2), re.S)
        if len(cells) == 3:
            out[label] = cells[0] + ' ' + cells[1]
    return out


def _diff_change_cells(html):
    """{row label: the change cell} for a rendered comparison table."""
    out = {}
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S):
        m = re.match(r'\s*<th>(.*?)</th>(.*)', tr, re.S)
        if not m:
            continue
        cells = re.findall(r'<td[^>]*>.*?</td>', m.group(2), re.S)
        if len(cells) == 3:
            out[re.sub(r'<[^>]+>', '', m.group(1)).strip()] = cells[2]
    return out


def check_run_diff_live():
    """A run that has not finished has not won on cost.

    Its epoch count, the position of its peak and its wall clock are partial
    totals, and against a finished run every one of them comes out smaller and
    therefore green -- three of ten rows in the improvement colour on the one
    screen a promotion is decided from. Every value is correct; only the
    colour lies, which is why nothing downstream catches it.

    Driven on runs made up here rather than whatever is on disk: the defect
    needs one live run and one finished one, and which of those exist is a
    fact about what the GPU happens to be doing.
    """
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    try:
        import dashboard as d
    except ImportError as e:
        print(f'SKIP: cannot import the dashboard ({e})')
        return 0
    bad = []
    # the retrain, part way: behind on every metric, ahead on every cost
    # simply because it has not paid all of it yet
    live = {'project': 'dogdetection', 'name': 'dogdet_v3_002',
            'dir': os.path.join(REPO, 'no', 'such', 'run'), 'task': 'detect',
            'status': 'running', 'headline_key': 'mAP50-95',
            'headline_label': 'mAP50-95', 'best_headline': 0.3699,
            'epochs_done': 48, 'best_epoch': 41, 'secs_per_epoch': 1900.0,
            'wall_clock_s': 91200.0, 'latest_val_loss': 1.20, 'curve': [],
            'latest': [{'key': 'recall', 'at_best': 0.5500, 'peak': 0.5900},
                       {'key': 'precision', 'at_best': 0.7100, 'peak': 0.74},
                       {'key': 'mAP50', 'at_best': 0.6000, 'peak': 0.62}]}
    done = dict(live, name='dogdet_v2_001', status='finished',
                best_headline=0.4968, epochs_done=300, best_epoch=262,
                secs_per_epoch=2100.0, wall_clock_s=630000.0,
                latest_val_loss=1.05,
                latest=[{'key': 'recall', 'at_best': 0.6800, 'peak': 0.7000},
                        {'key': 'precision', 'at_best': 0.7600, 'peak': 0.78},
                        {'key': 'mAP50', 'at_best': 0.7300, 'peak': 0.75}])
    COST = ('epochs run', 'best epoch', 'wall clock')
    keep = d.training_runs
    try:
        d.training_runs = lambda: [live, done]
        rows = _diff_change_cells(
            d.render_run_diff('dogdetection/dogdet_v3_002',
                              'dogdetection/dogdet_v2_001'))
        for label in COST:
            cell = rows.get(label)
            if cell is None:
                bad.append(f'the comparison has no "{label}" row at all')
            elif 'dup' in cell or 'ddn' in cell:
                bad.append(f'"{label}" is painted {cell} against a run that '
                           f'is still training — 48 epochs is not fewer than '
                           f'300, it is 252 not yet run')
        # ...but only the cost rows. A rate is comparable at any point, and a
        # metric that went the wrong way still went the wrong way.
        if 'dup' not in (rows.get('seconds per epoch') or ''):
            bad.append(f'"seconds per epoch" lost its colour: '
                       f'{rows.get("seconds per epoch")!r} — it is a rate, '
                       f'and a rate is true of a run at any point in it')
        if 'ddn' not in (rows.get('recall at best epoch') or ''):
            bad.append(f'"recall at best epoch" lost its colour: '
                       f'{rows.get("recall at best epoch")!r} — the retrain '
                       f'really is behind on recall and the screen has to '
                       f'say so')
        html = d.render_run_diff('dogdetection/dogdet_v3_002',
                                 'dogdetection/dogdet_v2_001')
        if 'dwarn' not in html or 'dogdet_v3_002' not in html.split(
                '<table')[0]:
            bad.append('no note above the table naming the run that is still '
                       'training, so the blank change cells look like missing '
                       'data')
        # and two finished runs are untouched: a guard that blanked every
        # comparison would pass everything above and cost the table its point
        d.training_runs = lambda: [dict(live, status='finished'), done]
        rows = _diff_change_cells(
            d.render_run_diff('dogdetection/dogdet_v3_002',
                              'dogdetection/dogdet_v2_001'))
        for label in COST:
            cell = rows.get(label) or ''
            if 'dup' not in cell and 'ddn' not in cell:
                bad.append(f'two finished runs: "{label}" renders {cell!r} — '
                           f'both totals are final and the comparison is the '
                           f'whole point of the screen')
        if 'dwarn' in d.render_run_diff('dogdetection/dogdet_v3_002',
                                        'dogdetection/dogdet_v2_001'):
            bad.append('two finished runs still carry the still-training note')
    finally:
        d.training_runs = keep
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('ok   run comparison: a live run does not win on cost, and two '
          'finished ones still compare')
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

    holds = []

    class FakeSmi:
        """One nvidia-smi -l 1, as a stream of rows.

        Given a `hold` the pipe stays open after the last row, which is what
        the real one does -- it prints a line a second until something stops
        it. Without one the stream ends, which is what a driver reset, an
        unplugged eGPU or a killed nvidia-smi looks like from in here.
        """

        def __init__(self, rows, hold):
            self.drained = threading.Event()
            self.stdout = self._lines(rows, hold)

        def _lines(self, rows, hold):
            for r in rows:
                yield r
            # set after the reader has appended the last row, so a caller
            # waiting on it is not racing the thread for the window
            self.drained.set()
            if hold is not None:
                hold.wait(10)

    def feed(rows, hold=None):
        reset()
        made = []
        d.subprocess.Popen = lambda *a, **k: made.append(
            FakeSmi(rows, hold)) or made[-1]
        try:
            d._gpu()                            # spawns and drains the reader
            if made:
                made[-1].drained.wait(5)
                for _ in range(500):            # ...and, with no hold, exits
                    if hold is not None or d._GPU['proc'] is None:
                        break
                    time.sleep(0.01)
            return d._gpu()
        finally:
            d.subprocess.Popen = real

    def held():
        """A pipe that stays open until this function is done with it."""
        holds.append(threading.Event())
        return holds[-1]

    # A single reading is not a measurement of a bursty workload. Measured on
    # a real run: forty zeroes and four bursts in twenty-two seconds, so a
    # glance taken when a browser asks lands on zero nine times in ten and the
    # card reads 0% forever. The headline has to be the window.
    burst = ['NVIDIA X, 0, 4103, 16303, 33, 40, 360'] * 9 + \
            ['NVIDIA X, 63, 4103, 16303, 35, 190, 360']
    g = feed(burst, held()) or {}
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
    g = feed(['n/a, [N/A], , not supported', 'garbage', '', 'a,b'], held())
    if g and g.get('util') is not None:
        bad.append(f'"[N/A]" read as a measurement: {g}')
    # One field the driver will not answer for (power on a laptop card, temp
    # in a container) must cost that field, not the whole readout.
    g = feed(['Card X, 74, 4103, 16303, 51, [N/A], [N/A]'], held()) or {}
    if g.get('util') != 74 or g.get('mem_total') != 16303:
        bad.append(f'an unsupported power reading took the whole card down '
                   f'with it: {g}')
    if g.get('power') is not None:
        bad.append(f'"[N/A]" power read as a number: {g.get("power")}')

    # A card that goes away mid-run (driver reset, eGPU unplugged, nvidia-smi
    # killed) closes the pipe. What a CALLER SEES after that is the whole
    # point, and for a long time this asked the wrong question: it called its
    # own reset(), which sets proc to None, and then asserted proc is None,
    # without ever calling _gpu() again. A tautology passes against the broken
    # code and the fixed code alike, and while it did, a dead reader's last
    # frame was served as the live card for the whole 60 s back-off -- 90% and
    # "gpu bound" in the header from a process that had already exited.
    gone = feed(['NVIDIA X, 90, 4103, 16303, 61, 300, 360'] * 5)
    if gone is not None:
        bad.append(f'nvidia-smi exited and the panel still reports '
                   f'{gone.get("util")}% from the window it left behind — a '
                   f'reading has to belong to a reader that is still there')
    if d._GPU['proc'] is not None:
        bad.append('a closed nvidia-smi stream left a live process behind')
    if d._GPU['retry_at'] <= time.time():
        bad.append('a closed stream left no back-off — every request would '
                   'fork a fresh nvidia-smi')
    # The same claim by the route the back-off cannot reach: an nvidia-smi
    # that wedges with its pipe open leaves `proc` set, so nothing respawns
    # and nothing else would ever age its last window out.
    stuck = feed(['NVIDIA X, 88, 4103, 16303, 60, 300, 360'] * 4, held()) or {}
    if stuck.get('util') != 88:
        bad.append(f'a live reader\'s own window was not reported: {stuck}')
    d._GPU['samples'] = [(t - d.GPU_WINDOW - 5, u)
                         for t, u in d._GPU['samples']]
    if d._gpu() is not None:
        bad.append(f'readings older than the {d.GPU_WINDOW} s window are still '
                   f'served as "the card now" — they are the last half minute '
                   f'the reader managed before it stopped answering')
    for h in holds:
        h.set()
    reset()
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


# Every start tag, and the two ways an element is named to CSS.
_TAG = re.compile(r'<[a-z][a-z0-9]*\b[^<>]*>')
# The attribute, not `aria-hidden` and not `overflow: hidden` in a style="".
_HIDDEN_ATTR = re.compile(r'(?<![-\w:])hidden(?:=|[\s/>])')
# document.getElementById('x'), el.querySelector('.y'), $('x') — every way
# this page reaches an element it is going to toggle.
_LOOKUP = (r"(?:[\w$.\[\]]*\.)?(?P<how>getElementById|querySelectorAll"
           r"|querySelector)\(\s*'(?P<arg>[^']*)'\s*\)"
           r"|(?P<dollar>\$)\(\s*'(?P<darg>[^']*)'\s*\)")


def _el_tokens(tag):
    """The #ids and .classes one start tag answers to."""
    out = ['#' + m for m in re.findall(r'\bid="([^"]+)"', tag)]
    return out + ['.' + c for a in re.findall(r'\bclass="([^"]+)"', tag)
                  for c in a.split()]


def _sel_tokens(sel):
    """The tokens the ELEMENT itself must match: ".a .b" selects .b, not .a."""
    return re.findall(r'[.#][A-Za-z0-9_-]+',
                      re.split(r'[\s>+~,]+', sel.strip())[-1])


def _lookup_tokens(m):
    if m.group('dollar'):
        return ['#' + m.group('darg')]
    if m.group('how') == 'getElementById':
        return ['#' + m.group('arg')]
    return _sel_tokens(m.group('arg'))


def hidden_that_still_shows(html, src=''):
    """[(selector, why)] the page can hide but styles into view anyway.

    The UA's ``[hidden]{display:none}`` is the weakest rule there is: any
    author rule that names a display wins, so ``.swctl{display:flex}`` made
    ``<span class="swctl" hidden>`` fully visible. That put four buttons in
    the section header -- Resume sweep AND Run gate -- when only one stage was
    on screen. The page carries eighteen hand-written ``.x[hidden]{display:
    none}`` rules for exactly that, which is the tell: it is a rule everyone
    must remember, so nobody does.

    They were found one at a time all the same -- ``.swctl`` by somebody
    looking at the header, ``.wrkeyi`` by a review a year later -- because
    this only ever asked about elements carrying a literal ``hidden`` in the
    BUILT page. Two whole populations sat outside it. The first is markup the
    server renders into the page rather than baking into it: the mistakes
    panel's key line lives in a dashboard.py f-string and never appears in
    index.html, which is why ``.wrkeyi`` was invisible here. The second is
    every element the script toggles with ``el.hidden = ...`` and no attribute
    in the markup at all -- ``.mapgate`` and ``.wrtile`` among them. So
    "can be hidden" is derived now, from three sources, rather than listed:

      * a ``hidden`` attribute on a tag, in the built body or in the source's
        own fragments,
      * ``NAME.hidden =`` / ``NAME.setAttribute('hidden'`` in the inline
        script, with NAME resolved to the nearest lookup that named it (a
        page-wide map would be wrong -- ``b`` is four different elements in
        four different IIFEs),
      * a callback parameter over a collection that was looked up the same
        way, which is how the wrong-tile pager hides a page of them.

    A hit is reported per ELEMENT, not per class: ``<button class="rbtn
    maplock" hidden>`` is safe because ``.maplock[hidden]`` outranks
    ``.rbtn{display:inline-flex}``, and demanding a guard on ``.rbtn`` itself
    would be demanding it on every button on the page.

    ``src`` is dashboard.py. The /review page's template is cut out of it --
    that document has its own stylesheet and its own guard, and ``.chips`` is
    a class on both pages with a display rule on only one.
    """
    css = '\n'.join(re.findall(r'<style[^>]*>(.*?)</style>', html, re.S))
    css = re.sub(r'/\*.*?\*/', ' ', css, flags=re.S)
    if re.search(r'(?:^|[,}])\s*\[hidden\]\s*\{[^{}]*display\s*:\s*none', css):
        return []                       # a global rule covers everything
    shows, hides = {}, set()
    for sel, decls in re.findall(r'([^{}]+)\{([^{}]*)\}', css):
        disp = re.search(r'(?:^|;)\s*display\s*:\s*([a-z-]+)', decls)
        if not disp:
            continue
        for one in sel.split(','):
            if not one.strip():
                continue
            leaf = re.split(r'[\s>+~]+', one.strip())[-1]
            for tok in re.findall(r'[.#][A-Za-z0-9_-]+', leaf):
                if '[hidden]' in leaf and disp.group(1) == 'none':
                    hides.add(tok)
                elif disp.group(1) != 'none':
                    shows.setdefault(tok, one.strip())

    if 'REVIEW_HTML = r"""' in src:
        i = src.index('REVIEW_HTML = r"""')
        src = src[:i] + src[src.index('"""', i + 18) + 3:]
    body = html[html.index('</style>'):]
    els = [(_el_tokens(t), t) for t in _TAG.findall(body + '\n' + src)]
    els = [e for e in els if e[0]]
    hideable = [(tk, 'hidden in the markup: ' + tag.strip()[:70])
                for tk, tag in els if _HIDDEN_ATTR.search(tag)]

    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    named = {}
    for m in re.finditer(r'(?<![\w$.])([A-Za-z_$][\w$]*)\s*=\s*[^;,\n]*?(?:'
                         + _LOOKUP + ')', script):
        named.setdefault(m.group(1), []).append((m.start(), _lookup_tokens(m)))

    def nearest(name, at):
        """What NAME held where it was used. Nearest declaration above it, or
        the first below when the use is inside a hoisted function."""
        seen = named.get(name) or []
        above = [tk for pos, tk in seen if pos < at]
        return above[-1] if above else (seen[0][1] if seen else [])

    for pat in (r'([A-Za-z_$][\w$]*)\.(?:forEach|map|filter)'
                r'\(\s*function\s*\(\s*([A-Za-z_$][\w$]*)',
                r'\[\]\.(?:forEach|map|filter)\.call\(\s*([A-Za-z_$][\w$]*)'
                r'\s*,\s*function\s*\(\s*([A-Za-z_$][\w$]*)'):
        for m in re.finditer(pat, script):
            tk = nearest(m.group(1), m.start())
            if tk:
                named.setdefault(m.group(2), []).append((m.start(), tk))
    for v in named.values():
        v.sort()

    unresolved = []
    for m in re.finditer(r'(?<![\w$.])((?:[\w$.\[\]]*\.)?(?:getElementById|'
                         r"querySelector)\(\s*'[^']*'\s*\)|\$\(\s*'[^']*'\s*\)"
                         r"|[A-Za-z_$][\w$]*)\s*\."
                         r"(?:hidden\s*=[^=]|setAttribute\(\s*'hidden')",
                         script):
        one = re.fullmatch(_LOOKUP, m.group(1))
        got = _lookup_tokens(one) if one else nearest(m.group(1), m.start())
        if not got:
            unresolved.append(m.group(1))
            continue
        why = ('hidden by the script: '
               + ' '.join(script[m.start():m.start() + 48].split()))
        for tok in got:
            for tk in [t for t, _ in els if tok in t] or [[tok]]:
                hideable.append((tk, why))

    out, seen = [], set()
    for tk, why in hideable:
        if any(t in hides for t in tk):
            continue                    # one guarded class covers the element
        for t in tk:
            if t in shows and t not in seen:
                seen.add(t)
                out.append((t, f'{shows[t]} names a display; {why}'))
                break
    return out, sorted(set(unresolved))


def check_map_layers(html):
    """The atlas's model layers: tabs present, payloads sane, captions that
    say MODEL.

    Three separate failure modes, each seen or nearly shipped:

      - the chip exists but nothing serves its file (STATIC_FILES is an
        allowlist, so forgetting the entry is a silent 404 on click);
      - the layer file exists but is stale forever, because nothing in the
        build path calls map_layers.refresh();
      - the caption drops the one word that marks the numbers as a model's,
        and 736K unreviewed classifier calls read as counted dogs.

    Plus the payload itself: the page trusts the file's totals for its
    caption, so a file whose grids do not sum to its own totals would put a
    number on screen that the map contradicts on hover.
    """
    bad = []
    src_path = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')
    src = open(src_path, encoding='utf-8').read()

    # ── tab markup ──────────────────────────────────────────────────────
    chips = re.findall(
        r'<button[^>]*class="(mchip[^"]*)"[^>]*data-l="([a-z]+)"', html)
    have = {l for _, l in chips}
    for want in ('harvest', 'dogs', 'rate', 'gate', 'leash'):
        if want not in have:
            bad.append(f'the map card has no "{want}" layer chip — the tab '
                       f'strip lost a layer')
    on = [l for cls, l in chips if 'on' in cls.split()]
    if on != ['harvest']:
        bad.append(f'the default map layer should be harvest alone, but the '
                   f'"on" chip(s) at build time are {on or "none"}')

    # ── the page can actually fetch what the chips ask for ──────────────
    for fn in ('map_layer_dogs.json', 'map_layer_leash.json'):
        if fn not in html:
            bad.append(f'the page script never fetches {fn} — the chip is '
                       f'a control that does nothing')
        if f"'/{fn}'" not in src.split('STATIC_FILES', 1)[-1][:600]:
            bad.append(f'/{fn} is not in STATIC_FILES — the allowlist 404s '
                       f'the fetch the moment the chip is clicked')

    # ── the build path keeps the layers fresh ───────────────────────────
    if 'def _map_layers():' not in src or 'import map_layers' not in src:
        bad.append('dashboard.py has no lazy map_layers import — either the '
                   'layers are never refreshed or a broken map_layers.py '
                   'breaks the server at boot')
    m = re.search(r'\ndef build\(args\):.*?\ndef ', src, re.S)
    body = m.group(0) if m else ''
    if '_map_layers()' not in body or '.refresh()' not in body:
        bad.append('build() never calls map_layers refresh — the model '
                   'layers fossilize while the sweep keeps scoring crops')

    # ── the caption owns the model label ────────────────────────────────
    if 'model-called dog crops on' not in html:
        bad.append('the gate layer caption lost its "model-called" label — '
                   'unreviewed classifier counts would read as counted dogs')
    if 'model-called leashed' not in html:
        bad.append('the leash layer caption lost its "model-called" label — '
                   'the split would read as human verdicts')
    if html.count(".source||'model'") < 2:
        bad.append('the legend no longer surfaces the layer files\' source '
                   'field — nobody can tell WHICH model produced the layer')

    # ── payload self-consistency ────────────────────────────────────────
    out_dir = os.path.join(REPO, 'data', 'dashboard')

    def grid_ok(levels, name, want_total):
        for res_key in ('0.5', '0.15', '0.05'):
            lv = levels.get(res_key)
            if not lv:
                bad.append(f'{name} has no "{res_key}" grid — the page '
                           f'paints an empty layer at that zoom')
                continue
            res = lv['res']
            pts = lv['points']
            total = sum(p[2] for p in pts)
            if total != want_total:
                bad.append(f'{name} @ {res_key}° sums to {total:,} but the '
                           f'file claims {want_total:,} — the caption and '
                           f'the map would disagree')
            if pts and lv['max'] != max(p[2] for p in pts):
                bad.append(f'{name} @ {res_key}° max field is wrong — the '
                           f'colour ramp would be scaled to a lie')
            for p in pts[:200]:
                f_ = (p[0] - res / 2) / res
                if abs(f_ - round(f_)) > 0.01:
                    bad.append(f'{name} @ {res_key}° cell centre {p[0]} is '
                               f'off the harvest grid — the layers would '
                               f'not land cell-for-cell on the map')
                    break

    for fn, kind in (('map_layer_dogs.json', 'dogs'),
                     ('map_layer_leash.json', 'leash')):
        path = os.path.join(out_dir, fn)
        if not os.path.exists(path):
            bad.append(f'{fn} missing — a build should have written it '
                       f'(python tools/dashboard/map_layers.py)')
            continue
        try:
            doc = json.load(open(path, encoding='utf-8'))
        except ValueError as e:
            bad.append(f'{fn} is not valid JSON ({e})')
            continue
        if doc.get('schema') != 1:
            bad.append(f'{fn} schema is {doc.get("schema")!r}, expected 1')
        if not str(doc.get('source', '')).startswith('model:'):
            bad.append(f'{fn} source is {doc.get("source")!r} — the layer '
                       f'no longer says it is model output (rule 5)')
        if kind == 'dogs':
            grid_ok(doc.get('levels', {}), fn, doc.get('total', -1))
        else:
            grid_ok(doc.get('leashed_levels', {}), fn + ' (leashed)',
                    doc.get('leashed_total', -1))
            grid_ok(doc.get('loose_levels', {}), fn + ' (loose)',
                    doc.get('loose_total', -1))

    if bad:
        for b in bad:
            print('FAIL ' + b)
        return 1
    print('ok   map model layers: five chips, files served and refreshed, '
          'captions say model-called, payloads sum to their own totals')
    return 0


def check_map_tabs_live():
    """Click the model-layer chips in a real chromium against a served build.

    Static checks prove the markup and the strings exist; this proves the
    chips DO something -- the legend and caption follow the tab, the tab
    state follows the click, and the switch does not throw. The failure this
    exists for is invisible to grep: a lazy-loaded layer whose success path
    never applies the layer leaves a chip that highlights, fetches, and then
    shows the harvest forever.

    The page is served over a loopback socket from data/dashboard (never the
    live server): file:// would kill every fetch the map is built on. Loud
    SKIP without playwright or chromium, naming what went unchecked.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f'SKIP: no playwright ({e}) — nobody clicked the map layer '
              f'chips')
        return 0
    import functools
    import http.server
    import socketserver

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    out_dir = os.path.join(REPO, 'data', 'dashboard')
    httpd = socketserver.TCPServer(
        ('127.0.0.1', 0), functools.partial(Quiet, directory=out_dir))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{httpd.server_address[1]}/index.html'
    bad, errs = [], []
    try:
        pw = sync_playwright().start()
    except Exception as e:
        print(f'SKIP: playwright would not start ({e}) — nobody clicked the '
              f'map layer chips')
        httpd.shutdown()
        return 0
    try:
        try:
            br = pw.chromium.launch()
        except Exception as e:
            print(f'SKIP: no chromium ({str(e).splitlines()[0]}) — nobody '
                  f'clicked the map layer chips')
            return 0
        pg = br.new_page(viewport={'width': 1440, 'height': 950})
        # resource 404s are expected (the static server has no /api); a page
        # error or any other console error is not
        pg.on('console', lambda m: errs.append('console: ' + m.text)
              if m.type == 'error'
              and not m.text.startswith('Failed to load resource') else None)
        pg.on('pageerror', lambda e: errs.append('pageerror: ' + str(e)))
        pg.goto(url, wait_until='load')
        stats = 'document.getElementById("mapStats").textContent'

        def wait(cond, why):
            try:
                pg.wait_for_function('()=>' + cond, timeout=20000)
                return True
            except Exception:
                bad.append(why + f' — after 20s the caption still reads: '
                           + pg.evaluate('()=>' + stats)[:120])
                return False

        # The outlier toggle acts on the harvest grids and only those: the
        # model layers count from their own files, which place every crop at
        # its raw coordinate with no outlier split, so a ticked "exclude GPS
        # outliers" over one of them claimed an exclusion that was not
        # happening -- 154 gate crops sit in 89 cells the harvest layer
        # hides. It has to leave the screen with the layer and come back
        # with it, so the reader is never offered an inert control.
        vis = ('(function(){var e=document.getElementById("mapCleanWrap");'
               'return !!e&&getComputedStyle(e).display!=="none"})()')

        def toggle_shown():
            return bool(pg.evaluate('()=>' + vis))

        if wait(stats + '.length>0', 'the map never initialised'):
            # without outlier grids in map_points.json the control is absent
            # on every layer and there is nothing here to follow
            outlier_grids = toggle_shown()
            if not outlier_grids:
                print('SKIP: this build of map_points.json carries no '
                      'outlier grids — the GPS-outlier toggle was not '
                      'followed across the layers')
            s0 = pg.evaluate('()=>' + stats)
            if 'model-called' in s0:
                bad.append(f'the DEFAULT layer caption already says '
                           f'model-called: "{s0[:90]}" — harvest counts are '
                           f'not model output')
            pg.click('.mchip[data-l="gate"]')
            if wait(stats + '.indexOf("model-called dog crops")>=0',
                    'clicking the Dogs found chip never switched the layer'):
                on = pg.eval_on_selector_all(
                    '.mchip.on', 'es=>es.map(e=>e.dataset.l)')
                if on != ['gate']:
                    bad.append(f'after clicking the gate chip the "on" '
                               f'chip(s) are {on} — the tab state does not '
                               f'follow the click')
                lab = pg.eval_on_selector('#mapRampLab', 'e=>e.textContent')
                if 'model:' not in lab:
                    bad.append(f'the gate legend names no model ("{lab}") — '
                               f'the source field is not surfaced')
                if outlier_grids and toggle_shown():
                    bad.append('the "exclude GPS outliers" toggle is still '
                               'on screen over the Dogs found layer — that '
                               'layer has no outlier split, so a ticked box '
                               'promises an exclusion nothing performs')
            pg.click('.mchip[data-l="leash"]')
            if wait(stats + '.indexOf("model-called leashed")>=0',
                    'clicking the Leashed vs loose chip never switched'):
                mn = pg.eval_on_selector('#mapMin', 'e=>e.textContent')
                mx = pg.eval_on_selector('#mapMax', 'e=>e.textContent')
                if (mn, mx) != ('all unleashed', 'all leashed'):
                    bad.append(f'the leash ramp ends read "{mn}"/"{mx}", '
                               f'not all unleashed/all leashed — the two-colour '
                               f'share axis lost its labels')
                if outlier_grids and toggle_shown():
                    bad.append('the "exclude GPS outliers" toggle is still '
                               'on screen over the Leashed vs loose layer, '
                               'which is drawn from raw coordinates and '
                               'excludes nothing')
            pg.click('.mchip[data-l="harvest"]')
            if wait(stats + '.indexOf("model-called")<0',
                    'switching back to Harvest left a model caption up'):
                # hidden and never restored is the same broken control, one
                # layer along
                if outlier_grids and not toggle_shown():
                    bad.append('the "exclude GPS outliers" toggle did not '
                               'come back on the Harvest layer — the layer '
                               'it does apply to lost its control')
        br.close()
    finally:
        pw.stop()
        httpd.shutdown()
    for e in errs:
        bad.append('the console is not clean: ' + e[:200])
    if bad:
        for b in bad:
            print('FAIL ' + b)
        return 1
    print('ok   map layer chips switch layer, caption and tab state in '
          'chromium with a clean console')
    return 0


# ── the login gate ──────────────────────────────────────────────────────────
# Everything this dashboard serves is behind a session: the pages, the whole
# /api surface, every image route and the static fallback. Three things can
# take that away and none of them look broken from the outside -- the gate
# called after the routing instead of before it, a route added to the
# allow-list that should not be on it, and a gate that fails open when its
# module will not load. So the source is read for the first, and the handler
# is driven over a real socket for the other two.

# What a signed-out caller must get. Not a list of every route -- one of each
# KIND, because the point is that the answer does not depend on the route.
GATE_DENIED = (
    ('/', 'redirect', 'the front page'),
    ('/index.html', 'redirect', 'the front page by its filename'),
    ('/audit/review', 'redirect', 'the review queue'),
    ('/audit/gate', 'redirect', 'an audit page'),
    ('/datasets', 'redirect', 'the datasets page'),
    ('/nope', 'redirect', 'an address that does not exist'),
    ('/admin', 'redirect', 'the admin page'),
    ('/api/board', 'json401', 'the board API'),
    ('/api/sys', 'json401', 'the machine API'),
    ('/api/review/count', 'json401', 'the review count'),
    ('/hq?name=x.jpg', 'json401', 'a full-size frame'),
    ('/orig?name=x.jpg', 'json401', 'an original frame'),
    ('/flagged', 'json401', 'the flag ledger'),
    ('/audit/crop/gate/x.jpg', 'json401', 'an audit crop'),
    ('/audit/frame/gate/x.jpg', 'json401', 'an audit frame'),
    ('/datasets/thumb?key=x&rel=y', 'json401', 'a dataset thumbnail'),
    ('/datasets/image?key=x&rel=y', 'json401', 'a dataset image'),
    ('/recent_crops/x.jpg', 'json401', 'a crop from the rolling pool'),
    ('/review_set/x.jpg', 'json401', 'a crop from the review set'),
    ('/echarts.min.js', 'redirect', 'a file from the static allow-list'),
    ('/map_points.json', 'redirect', 'the map data'),
    ('/history.duckdb', 'redirect', 'a working file in the document root'),
    ('/accounts.db', 'redirect', 'the accounts database'),
    ('/session.key', 'redirect', 'the cookie signing key'),
)


def gate_traversal_checks():
    """The static allow-list must hold for a signed-in MEMBER, not just a
    stranger.

    THE ALLOW-LIST IS THE BOUNDARY, and it is only a boundary if it is
    applied to the name that gets opened. SimpleHTTPRequestHandler unquotes
    the path and runs posixpath.normpath over it before opening anything, so
    /recent_crops/../accounts.db starts with an allow-listed prefix -- a
    startswith() says yes -- and then resolves to OUT/accounts.db. Everything
    the server owns is a sibling of the page in that directory: the accounts
    database with every scrypt hash in it, the -wal carrying recent rows in
    the clear, session.key, serve.log. A member is the lowest-trust account
    this dashboard issues, created by an invite, and with session.key in hand
    they mint a cookie for the admin row and walk into the invite page.

    A DECOY DOCUMENT ROOT, never data/dashboard: a guard that has to read the
    machine's real password hashes to decide whether it can is a guard that
    reads them every time it fails. The names are the real ones -- taken from
    accounts.PRIVATE_FILES and auth.PRIVATE_FILES, so a private file added
    there later is covered here without anybody remembering to add it.
    """
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import dashboard as dash  # noqa: E402
    try:
        import accounts
        import auth
    except Exception as e:
        print(f'SKIP: the gate modules would not import ({e}) — nobody '
              f'checked the static allow-list against a member')
        return []

    import functools
    import http.client
    from http.server import ThreadingHTTPServer

    bad = []
    tmp = tempfile.mkdtemp(prefix='gate-trav-')
    root = os.path.join(tmp, 'out')
    db = os.path.join(tmp, 'accounts.db')
    key = os.path.join(tmp, 'session.key')
    private = sorted(accounts.PRIVATE_FILES | auth.PRIVATE_FILES
                     | {'serve.log', 'history.duckdb'})
    for d in dash.STATIC_DIRS:
        os.makedirs(os.path.join(root, d.strip('/')), exist_ok=True)
    for name in private:
        with open(os.path.join(root, name), 'wb') as f:
            f.write(b'DECOY ' + name.encode() + b' ' + b'x' * 64)
    crop = os.path.join(root, dash.STATIC_DIRS[0].strip('/'), 'a.jpg')
    with open(crop, 'wb') as f:
        f.write(b'\xff\xd8decoy-jpeg')
    boot = auth.bootstrap(db_path=db, key_path=key, env={
        'DASHBOARD_USER': 'admin', 'DASHBOARD_PASSWORD': 'a-password-long'})
    if not boot.get('ok'):
        shutil.rmtree(tmp, ignore_errors=True)
        print(f'SKIP: the throwaway store would not bootstrap '
              f'({boot.get("detail")}) — the allow-list was not driven')
        return []
    member = accounts.create_user('trav-member', 'a-password-long-enough',
                                  role='member', path=db)
    cookie = auth.COOKIE + '=' + auth.mint(member, key_path=key)[0]
    srv = ThreadingHTTPServer(
        ('127.0.0.1', 0),
        functools.partial(dash.BoardHandler, directory=root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def hit(path, method='GET'):
        # A connection per request, and the path sent verbatim: http.client
        # is happy to forward '..' unchanged, which is what a raw socket or
        # `curl --path-as-is` does and what a browser would not.
        c = http.client.HTTPConnection('127.0.0.1', srv.server_port,
                                       timeout=60)
        try:
            c.putrequest(method, path, skip_host=False,
                         skip_accept_encoding=True)
            c.putheader('Cookie', cookie)
            c.endheaders()
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()

    try:
        # The escape, in every spelling that reaches the same file. The last
        # two are the ones a check written against '..' alone lets through:
        # the server cuts the request at the '?' and the '#' before it
        # normalises, so a tail after either can steer the allow-list's
        # reading of the path somewhere the server will never look, while the
        # part in front of it resolves to the private file.
        tails = ('', '#/x/../../' + dash.STATIC_DIRS[0].strip('/') + '/y',
                 '?/../../' + dash.STATIC_DIRS[0].strip('/') + '/y')
        shapes = [d + hop + '/' + name + tail
                  for d in dash.STATIC_DIRS
                  for name in private
                  for hop in ('..', '../..', '%2e%2e', '.%2e')
                  for tail in tails]
        for p in shapes:
            for method in ('GET', 'HEAD'):
                code, head, body = hit(p, method=method)
                if code == 200:
                    bad.append(
                        f'{method} {p} answered 200 (Content-Length '
                        f'{head.get("Content-Length")}) to a signed-in '
                        f'member — the static allow-list matched a reading '
                        f'of the path that is not the one the server opens, '
                        f'so every file beside the page is readable by the '
                        f'lowest account this dashboard issues')
        # A bare allow-listed directory is a listing of every crop in the
        # pool, which nothing on any page asks for.
        for d in dash.STATIC_DIRS:
            code, head, body = hit(d)
            if code == 200:
                bad.append(f'GET {d} answered 200 with {len(body)} bytes — '
                           f'the allow-list let a directory through and the '
                           f'base class answered with an index of it')
        # ...and the allow-list still allows what the pages actually fetch,
        # or "deny everything" would pass every check above.
        code, head, body = hit(dash.STATIC_DIRS[0] + 'a.jpg')
        if code != 200 or not body:
            bad.append(f'{dash.STATIC_DIRS[0]}a.jpg answered {code} to a '
                       f'signed-in member — the allow-list closed the route '
                       f'the review grid loads every crop through')
        code, head, body = hit(dash.STATIC_DIRS[0] + './a.jpg')
        if code != 200:
            bad.append(f'{dash.STATIC_DIRS[0]}./a.jpg answered {code} — a '
                       f'./ that resolves inside the allow-listed directory '
                       f'is not an escape from it')
    finally:
        srv.shutdown()
        srv.server_close()
        shutil.rmtree(tmp, ignore_errors=True)
    return bad


def gate_source_checks(src):
    """The gate runs BEFORE the routing, in every verb, and stays wired.

    Read out of the source because it is an ordering, and an ordering is
    invisible from outside until the route that got past it exists. The rule
    is that nothing may look at self.path before self._gate() has had it:
    gating the routes we recognise, after we have recognised them, is exactly
    how the page added next month ships unprotected.
    """
    bad = []
    for verb in ('do_GET', 'do_POST', 'do_HEAD'):
        m = re.search(r'\n    def %s\(self\):\n(.*?)(?=\n    def )'
                      % verb, src, re.S)
        if not m:
            bad.append(f'BoardHandler.{verb} is gone — every verb has to be '
                       f'gated, and the one that is missing is the one the '
                       f'base class answers out of the document root '
                       f'unchecked')
            continue
        body = m.group(1)
        if 'self._gate()' not in body:
            bad.append(f'BoardHandler.{verb} never calls self._gate(), so '
                       f'everything it serves is served to anybody')
            continue
        before = body.split('self._gate()', 1)[0]
        # comments and the docstring say nothing to the router
        code = '\n'.join(ln for ln in before.split('\n')
                         if not ln.strip().startswith(('#', '"""', "'''")))
        for token in ('self.path', 'send_response', 'send_error', 'self._json',
                      'self._html'):
            if token in code:
                bad.append(f'BoardHandler.{verb} touches {token} BEFORE '
                           f'self._gate() — the gate is being asked about '
                           f'requests the router has already begun to '
                           f'answer, which is how a new route ships '
                           f'unprotected')
                break
    m = re.search(r'AUTH_FREE = frozenset\(\{([^}]*)\}\)', src)
    if not m:
        bad.append('BoardHandler.AUTH_FREE is gone — the one path an '
                   'unauthenticated caller may reach is no longer written '
                   'down anywhere a reader can check it')
    else:
        allowed = set(re.findall(r"'([^']+)'", m.group(1)))
        if allowed != {'/favicon.ico'}:
            bad.append(f'AUTH_FREE is {sorted(allowed)} — the public surface '
                       f'is exactly the tab icon plus what auth.py owns, and '
                       f'anything else on it is served before anybody has '
                       f'proved who they are')
    watched = re.search(r'for _m in \(([^)]*)\)', src)
    names = watched.group(1) if watched else ''
    for mod in ('auth.py', 'accounts.py'):
        if mod not in names:
            bad.append(f"{mod} is not in serve()'s watch list, so an edit to "
                       f'the lock sits invisible behind a healthy-looking '
                       f'server until somebody restarts it by hand')
    serve = re.search(r'\ndef serve\(args\):\n(.*?)(?=\ndef )', src, re.S)
    if serve:
        # comments stripped first: this file explains bootstrap() in prose
        # right above the call, and a check that greps for the word passed a
        # serve() from which the call itself had been deleted.
        code = '\n'.join(ln for ln in serve.group(1).split('\n')
                         if not ln.lstrip().startswith('#'))
        if not re.search(r'\.bootstrap\(', code):
            bad.append("serve() never calls the gate's bootstrap(), so .env "
                       'is never read and DASHBOARD_PASSWORD never becomes '
                       'an account — the first request pays for it or nobody '
                       'does')
    if not re.search(r'def log_message\(self, \*a\):\s*\n\s*pass', src):
        bad.append('BoardHandler.log_message is no longer a no-op, so every '
                   'request line is written down — including /signup?t=, '
                   'which carries an invite token in its query string')
    return bad


def gate_live_checks():
    """Drive the real handler over a socket, signed out and signed in.

    A THROWAWAY store in a temp directory, never data/dashboard/accounts.db:
    this must not create an account on the machine it runs on, and it must
    not care whether one already exists.

    ONE keep-alive connection for the whole run, the way a browser does it.
    A fresh connection per request lands each one on a new server thread, and
    a duckdb read that crosses threads takes this process down with a
    segfault -- a defect that predates the gate and has nothing to do with
    what is being measured here.
    """
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import dashboard as dash  # noqa: E402
    try:
        import accounts
        import auth
    except Exception as e:
        print(f'SKIP: the gate modules would not import ({e}) — nobody '
              f'checked that a signed-out caller is served nothing')
        return []

    import functools
    import http.client
    import urllib.parse
    from http.server import ThreadingHTTPServer

    bad = []
    tmp = tempfile.mkdtemp(prefix='gate-')
    db = os.path.join(tmp, 'accounts.db')
    key = os.path.join(tmp, 'session.key')
    pw = 'a-password-long-enough'
    boot = auth.bootstrap(db_path=db, key_path=key, env={
        'DASHBOARD_USER': 'admin', 'DASHBOARD_PASSWORD': pw})
    if not boot.get('ok'):
        print(f'SKIP: the throwaway store would not bootstrap '
              f'({boot.get("detail")}) — nobody checked the gate over HTTP')
        return []
    srv = ThreadingHTTPServer(
        ('127.0.0.1', 0),
        functools.partial(dash.BoardHandler, directory=dash.OUT))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    conn = http.client.HTTPConnection('127.0.0.1', srv.server_port,
                                      timeout=120)

    def hit(path, cookie=None, method='GET', body=None, ctype=None):
        head = {}
        if cookie:
            head['Cookie'] = cookie
        if ctype:
            head['Content-Type'] = ctype
        conn.request(method, path, body=body, headers=head)
        r = conn.getresponse()
        return r.status, dict(r.getheaders()), r.read()

    try:
        # ── signed out: nothing, in every shape ─────────────────────────
        for path, want, what in GATE_DENIED:
            code, head, body = hit(path)
            if code == 200:
                bad.append(f'{path} answered 200 with {len(body)} bytes to a '
                           f'caller with no session — {what} is being served '
                           f'to anybody who knows the address')
                continue
            if want == 'redirect':
                if code not in (301, 302, 303) or not head.get(
                        'Location', '').startswith('/login'):
                    bad.append(f'{path} answered {code} '
                               f'({head.get("Location", "no Location")}) to a '
                               f'signed-out caller, not a redirect to the '
                               f'login form')
                elif body:
                    bad.append(f'the gate redirect for {path} carries a '
                               f'{len(body)} byte body — a bounce to the '
                               f'login form has nothing to say about {what}')
            elif code != 401:
                bad.append(f'{path} answered {code} to a signed-out caller '
                           f'and should have answered 401: {what} is fetched '
                           f'by script or by <img>, and neither follows a '
                           f'redirect into a login form usefully')
        # HEAD is a verb too, and the base class answers it out of the
        # document root without passing through this class at all.
        for path in ('/', '/index.html', '/history.duckdb', '/echarts.min.js'):
            code, head, body = hit(path, method='HEAD')
            if code == 200:
                bad.append(f'HEAD {path} answered 200 (Content-Length '
                           f'{head.get("Content-Length")}) to a signed-out '
                           f'caller — HEAD names the file and gives its size')
        # A POST is not a way round it either.
        for path in ('/api/audit/verdict', '/api/board', '/api/refresh'):
            code, head, body = hit(path, method='POST', body='{}',
                                   ctype='application/json')
            if code != 401:
                bad.append(f'POST {path} answered {code} to a signed-out '
                           f'caller, not 401')
        # The answer for a crop that exists and a name somebody made up has
        # to be the same bytes. Anything else is a lookup service for what
        # this dashboard holds, open to whoever asks.
        a = hit('/audit/crop/gate/1785663300000_1606751523958968_073.jpg')
        b = hit('/audit/crop/gate/there-is-no-such-crop.jpg')
        if (a[0], a[2]) != (b[0], b[2]):
            bad.append(f'a real-looking crop name answers {a[0]}/{len(a[2])} '
                       f'bytes and an invented one {b[0]}/{len(b[2])} — the '
                       f'refusal tells a stranger which crops exist')
        # The tab icon is the one exception, and it is a constant in the
        # source rather than anything read off the disk.
        code, head, body = hit('/favicon.ico')
        if code != 200 or body != dash.FAVICON_SVG:
            bad.append(f'/favicon.ico answered {code} with {len(body)} bytes '
                       f'to a signed-out caller — it is on AUTH_FREE so that '
                       f'the login page has a tab icon, and it must serve '
                       f'exactly FAVICON_SVG')

        # ── signed in: the same routes answer ───────────────────────────
        code, head, body = hit(
            '/login?next=/audit/review', method='POST',
            body='username=admin&password=' + urllib.parse.quote(pw),
            ctype='application/x-www-form-urlencoded')
        cookie = head.get('Set-Cookie', '').split(';', 1)[0]
        if code not in (302, 303) or auth.COOKIE not in cookie:
            bad.append(f'signing in with the .env credential answered {code} '
                       f'and set no session cookie — the gate cannot be let '
                       f'through by the account it just created')
            return bad
        for path in ('/', '/audit/review', '/api/sys', '/echarts.min.js',
                     '/favicon.ico'):
            code, head, body = hit(path, cookie=cookie)
            if code != 200 or not body:
                bad.append(f'{path} answered {code} with {len(body)} bytes to '
                           f'a signed-in admin — the cookie does not open '
                           f'what the gate closed')
        # The header says who is reading, and offers an admin the way to the
        # accounts page. It is spliced per request: built into the page it
        # would name whoever rebuilt it last.
        code, head, page = hit('/', cookie=cookie)
        page = page.decode('utf-8', 'replace')
        # A POST form, never an <a href>. Signing out ends the session on
        # every device, and a state change a GET can make is one that any
        # page the reader happens to be on can make for them: an
        # <img src="/logout"> would sign an annotator out of their phone and
        # their laptop from across the internet, on a loop.
        if 'action="%s"' % (auth.LOGOUT_PATH,) not in page:
            bad.append('the front page carries no sign-out control, so a '
                       'dashboard left open on a phone stays signed in with '
                       'nothing on screen to end it')
        elif ('name="%s"' % (auth.CSRF_FIELD,)) not in page:
            bad.append('the front page\'s sign-out form carries no '
                       f'{auth.CSRF_FIELD} field, so submitting it is '
                       f'refused and the only way out of the dashboard is to '
                       f'wait seven days')
        if 'href="%s"' % (auth.LOGOUT_PATH,) in page:
            bad.append(f'the front page links to {auth.LOGOUT_PATH} with an '
                       f'<a href>, and signing out now ends the session on '
                       f'every device — as a GET, any page the reader visits '
                       f'can fire it for them with one <img src>')
        if 'href="/admin"' not in page:
            bad.append('the front page offers an admin no link to /admin — '
                       'the accounts page exists and nothing on the '
                       'dashboard says so')
        if 'admin</span>' not in page and '>admin<' not in page:
            bad.append('the front page does not name the account reading it, '
                       'so nobody can tell which of two accounts a browser '
                       'is holding')

        # ── a member is not an admin ────────────────────────────────────
        member = accounts.create_user('member-guard', 'a-password-long-enough',
                                      role='member', path=db)
        mval, _ = auth.mint(member, key_path=key)
        mcookie = auth.COOKIE + '=' + mval
        code, head, body = hit('/admin', cookie=mcookie)
        if code != 404 or len(body) > 600:
            bad.append(f'/admin answered {code} ({len(body)} bytes) to a '
                       f'member — it has to be the same empty 404 an address '
                       f'that does not exist gets, or a member learns the '
                       f'page is there and worth attacking')
        code, head, mpage = hit('/', cookie=mcookie)
        mpage = mpage.decode('utf-8', 'replace')
        if code != 200:
            bad.append(f'the front page answered {code} to a member — a '
                       f'member reviews crops like anybody else')
        elif 'href="/admin"' in mpage:
            bad.append('the front page offers a MEMBER a link to /admin, '
                       'which answers them with a 404 — the link undoes the '
                       'silence the 404 exists for')
        # One UPDATE ends every session for an account. It is the whole
        # revocation story, so it is checked over the wire and not in unit.
        accounts.bump_session_epoch('member-guard', path=db)
        code, head, body = hit('/', cookie=mcookie)
        if code == 200:
            bad.append('a session survived bump_session_epoch() — signing an '
                       'account out everywhere does nothing, and a stolen '
                       'cookie cannot be taken back')

        # ── the gate itself will not load ───────────────────────────────
        # Fail closed on data, fail open on uptime: no module, no data, but
        # also no traceback and no exit. The dashboard re-execs itself
        # unattended, so this state has to be survivable and legible.
        keep = dict(dash._AUTH)
        dash._AUTH.update({'mod': None, 'tried': True,
                           'why': 'ImportError: no auth for you'})
        # The handler says so once on stderr, which is right of it and wrong
        # here: an alarming line above a run of ok lines reads as a fault in
        # the run rather than as the state this block is asking for.
        import contextlib
        import io
        try:
            stack = contextlib.redirect_stderr(io.StringIO())
            stack.__enter__()
            code, head, body = hit('/')
            if code != 503 or b'<h1' in body or len(body) > 2000:
                bad.append(f'with the gate module missing, / answered {code} '
                           f'with {len(body)} bytes — a gate that cannot run '
                           f'must serve the explanation and nothing else')
            code, head, body = hit('/api/board')
            if code != 503:
                bad.append(f'with the gate module missing, /api/board '
                           f'answered {code} — a gate that cannot run must '
                           f'not answer with data')
            code, head, body = hit('/audit/crop/gate/x.jpg')
            if code == 200:
                bad.append('with the gate module missing, an audit crop was '
                           'served — the failure opened the door instead of '
                           'closing it')
        finally:
            stack.__exit__(None, None, None)
            dash._AUTH.clear()
            dash._AUTH.update(keep)
    finally:
        conn.close()
        srv.shutdown()
        srv.server_close()
        shutil.rmtree(tmp, ignore_errors=True)
    return bad


def check_login_gate():
    """Nothing is served to a caller with no session. Anything else is a leak.

    First of all the checks in this file, and deliberately: a CSS collision
    is a page that looks wrong, and this is the harvest, the review queue and
    every frame in the store being handed to whoever finds the address.
    """
    try:
        src = open(os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py'),
                   encoding='utf-8').read()
    except OSError as e:
        print(f'FAIL could not read dashboard.py: {e}')
        return 1
    bad = gate_source_checks(src)
    bad += gate_live_checks()
    bad += gate_traversal_checks()
    # The strip is spliced between two sentinels on the way out, so a build
    # that lost them renders no account, no sign-out and no way to the
    # accounts page -- and looks exactly like a build that has them.
    try:
        page = open(INDEX, encoding='utf-8').read()
        if '<!--ACCT-->' not in page or '<!--/ACCT-->' not in page:
            bad.append('the built page carries no <!--ACCT--> sentinels, so '
                       'the header cannot be told who is reading it and the '
                       'sign-out link never appears')
    except OSError as e:
        bad.append(f'could not read the built page: {e}')
    if bad:
        for b in bad:
            print('FAIL ' + b)
        return 1
    print('ok   the gate runs before every route in every verb: signed out '
          'is a redirect for a page, a 401 for an API or an image and the '
          'same bytes for a crop that exists as for one that does not; '
          'signed in opens all of it; a member gets the empty 404 at /admin '
          'and cannot walk out of the static allow-list to the accounts '
          'database or the signing key; a missing gate module serves an '
          'explanation, not the data')
    return 0


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
    #
    # FIRST, before any check reads the page. It used to sit ten lines below
    # css_collisions(), so a stale run graded the old build's CSS and could
    # report FAIL against a collision the source no longer has -- sending the
    # reader to hunt a defect that is not there -- or print "ok every section
    # keeps to its own class prefix" about a build nobody is testing and only
    # then abort. It needs nothing but two mtimes; nothing may precede it.
    src = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')
    if os.path.exists(src) and os.path.getmtime(src) > os.path.getmtime(INDEX):
        raise SystemExit(
            'STALE PAGE: tools/dashboard/dashboard.py is newer than\n'
            f'  {INDEX}\n'
            'so this run would grade the previous build. Rebuild first:\n'
            '  python tools/dashboard/dashboard.py build --no-refresh')

    # Before every other check in this file. A CSS collision is a page that
    # looks wrong; this one is the harvest, the review queue and every frame
    # in the store being handed to whoever finds the address, and a reader
    # who has to scroll past forty ok lines to find that out is a reader who
    # finds it out later than they should.
    if check_login_gate():
        return 1

    stray = css_collisions(INDEX)
    if stray:
        for c, sid in stray[:8]:
            print(f'FAIL {sid} wears class "{c}", which is neither its own '
                  f'prefix nor a shared component — one stylesheet, so it is '
                  f'styling or being styled by another section')
        return 1
    print('ok   every section keeps to its own class prefix')

    html = open(INDEX, encoding='utf-8').read()
    showing, opaque = hidden_that_still_shows(
        html, open(src, encoding='utf-8').read() if os.path.exists(src) else '')
    if showing:
        for sel, why in showing[:8]:
            print(f'FAIL {sel} stays on screen when the page hides it — '
                  f'{why}\n     add {sel}[hidden]{{display:none}} beside the '
                  f'rule that shows it')
        return 1
    # Said out loud rather than swallowed: these are receivers the resolver
    # could not follow back to a class, and every one of them is a hole in
    # the check above.
    print('ok   every element the page can hide is actually hidden'
          + (f' ({len(opaque)} unresolved toggle(s): '
             f'{", ".join(opaque)})' if opaque else ''))
    if check_machine_stats():
        return 1
    if check_progress_ramp():
        return 1
    if check_run_diff():
        return 1
    if check_run_diff_live():
        return 1
    if check_gate_panel(html, extract_snippets(html)):
        return 1
    check_whole_script(html)
    check_no_shadowing(html)
    if check_copy_say(html):
        return 1
    if check_no_host_paths(html):
        return 1
    if check_freshness(html):
        return 1
    check_markup(html)
    if check_map_layers(html):
        return 1
    check_header_compact(html)
    if check_header_fold(html):
        return 1
    if check_header_shrinks():
        return 1
    if check_map_tabs_live():
        return 1
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
                'class_split': {}, 'errors': {'decode': 64},
                'last_error': 'decode: 1049220049202801.jpg',
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
