#!/usr/bin/env python3
"""
Adversarial test for the /review bulk-flagging page's client JS.

``node --check`` only proves the script parses. This EXECUTES the script
extracted from dashboard.REVIEW_HTML under node against a stub DOM and drives
the real user path: load, flag, backfill, undo, paginate, keyboard, lightbox.

The stub DOM is deliberately small but honest about the three things this page
actually leans on:

  * ``innerHTML`` assignment creates queryable children AND registers their
    ids, because tile()/showUndo()/openLb() all build markup that way and then
    immediately look the pieces back up.
  * ``querySelector('.card[data-name="..."]')`` -- the page addresses tiles by
    crop name, not index, so a stale index can never flag the wrong image.
  * ``getComputedStyle(grid).gridTemplateColumns`` -- arrow-key up/down needs
    the live column count.

Cases cover the normal payload, a flag that the server REFUSES (must roll
back, not silently drop the tile), undo restoring position, a fetch that
fails, the empty queue, quote/tag injection in image_id, and every keyboard
binding. A ReferenceError, a TypeError or any other throw fails the test.

Both of the page's modes are driven, and the audit one is not an afterthought:
flag() has a whole second branch for it, and while that branch went undriven an
undefined toast() sat in it throwing into the chain's own .catch, and markSeen
banked the crop whose annotation had just been removed. A throw inside a .then
is invisible from outside -- the page swallows it and the chain resolves -- so
watchThrows() below makes one visible without changing what the page does with
it.


Requires node on PATH; skips (exit 0, loud message) if absent.
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
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DASH = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')
AUDIT = os.path.join(REPO, 'tools', 'dashboard', 'audit.py')


def load_dashboard():
    spec = importlib.util.spec_from_file_location('dash_under_test', DASH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def crop(i, conf=0.5, full=True, iid=None):
    return {'name': '%d_%s_%03d.jpg' % (1_700_000_000_000 + i,
                                        iid or ('img%d' % i),
                                        int(round(conf * 100))),
            'image_id': iid or ('img%d' % i),
            'ts': 1_700_000_000_000 + i,
            'conf': conf,
            'has_full': full}


def tab_strip_checks(html):
    """The shared tab strip, exactly as the contract spells it.

    Three judging surfaces carry one strip, rendered by two owners --
    audit.py on the two audit pages, dashboard.py here -- and
    adv_fn_audit.tab_checks pins the audit copies byte for byte. This is the
    same pin on the review copy, so the two owners cannot drift apart: same
    single-line markup, 'jtab on' and aria-current on THIS page's tab, and
    nothing between the strip and the header.
    """
    bad = []
    labels = {'review': 'Review queue', 'gate': 'Dog-bin audit',
              'leash': 'Leash audit'}
    order = ('review', 'gate', 'leash')
    nav = '<nav class="jtabs" aria-label="judging surfaces">'
    if html.count(nav) != 1:
        bad.append('the page carries %d shared tab strips, not one'
                   % html.count(nav))
        return bad
    if not re.search(r'</header>\s*<nav class="jtabs"', html):
        bad.append('the tab strip is not directly under the header — the '
                   'contract puts it in the same place on every judging page')
    at = []
    for k in order:
        a = (f'<a href="/audit/{k}" class="jtab on" '
             f'aria-current="page">{labels[k]}</a>' if k == 'review'
             else f'<a href="/audit/{k}" class="jtab">{labels[k]}</a>')
        if a not in html:
            bad.append(f'the strip is missing the exact tab {a!r} — three '
                       f'agents render this markup and they must render it '
                       f'identically')
            at.append(-1)
        else:
            at.append(html.index(a))
    if -1 not in at and at != sorted(at):
        bad.append('the tabs are out of order — review queue, then the two '
                   'audits')
    if html.count('aria-current') != 1:
        bad.append('aria-current appears %d times; exactly one tab IS the '
                   'current page' % html.count('aria-current'))
    # ...and the strip has to LOOK like the audit pages' one: the four rules
    # are audit.py's pill vocabulary, compared with whitespace folded so a
    # rewrap is not a difference but a colour or a padding is.
    try:
        with open(AUDIT) as fh:
            audit_src = fh.read()
    except OSError:
        return bad          # no audit checkout to compare against
    for sel in (r'\.jtabs\{', r'\.jtab\{', r'\.jtab:hover\{', r'\.jtab\.on\{'):
        want = re.search(sel + r'[^}]*\}', audit_src)
        got = re.search(sel + r'[^}]*\}', html)
        if not want:
            continue
        if not got:
            bad.append('the review page has no %s rule — the strip is '
                       'unstyled here' % sel.replace('\\', ''))
        elif re.sub(r'\s+', '', got.group(0)) != \
                re.sub(r'\s+', '', want.group(0)):
            bad.append('the strip is styled differently from the audit '
                       'pages: %s vs %s' % (got.group(0), want.group(0)))
    # ...INCLUDING THE KEYBOARD RING. The audit pages ring everything
    # focusable with one blanket :focus-visible; this page rings per class,
    # so without a rule of its own the shared strip falls through to the
    # browser's default outline here and one strip wears two rings across
    # three surfaces.
    blanket = re.search(r'(?:^|[;}])\s*:focus-visible\s*\{([^}]*)\}',
                        audit_src, re.M)
    ring = re.search(r'\.jtab:focus-visible\s*\{([^}]*)\}', html)
    if blanket and not ring:
        bad.append('the review page has no .jtab:focus-visible rule — the '
                   'audit pages ring the same tabs with '
                   '{%s} and this one falls through to the browser default'
                   % re.sub(r'\s+', '', blanket.group(1)))
    elif blanket and ring and re.sub(r'\s+', '', ring.group(1)) != \
            re.sub(r'\s+', '', blanket.group(1)):
        bad.append('the strip focuses differently from the audit pages: '
                   '{%s} vs {%s}' % (re.sub(r'\s+', '', ring.group(1)),
                                     re.sub(r'\s+', '', blanket.group(1))))
    return bad


# What the guess feature WAS, by the tokens that carried it. Markup ids,
# copy, wire parameters and the poll target: any of these coming back is the
# feature coming back, whatever it is called.
GONE_MARKUP = ('id="suggest"', 'id="gatef"', 'id="trg"', 'id="trgRun"',
               'id="trgModel"', 'id="ngrpWho"', 'id="ngrpLooks"',
               'Run guesses', 'Guesses by', 'Any guess', 'class="sg')
GONE_SCRIPT = ('/api/triage', "'&suggest=", "'&gate=", "'&backend=",
               'paintSuggest', 'paintGate', 'BACKEND', 'SG_WORD',
               'start the guesser')


def guess_absence_checks(html, script):
    """The guesses feature is REMOVED, not hidden.

    The user asked for it gone: the suggest filter, the gate filter, the
    backend picker, the run strip and the tile badges. Hidden-but-wired is
    how a phantom filter emptied this queue once before, so the pin is on
    the bytes: none of the ids, none of the copy, no request parameter and
    no /api/triage traffic from this page.
    """
    bad = []
    for tok in GONE_MARKUP:
        if tok in html:
            bad.append('the guess feature is back in the markup: %r' % tok)
    for tok in GONE_SCRIPT:
        if tok in script:
            bad.append('the guess feature is back in the script: %r' % tok)
    return bad


def sign_in(mod):
    """A session cookie header for the handler under test, or {}.

    Every route below is behind the login gate now, so a guard that knocks
    without one is answered with the login form and grades that instead --
    which is how this file started reporting that /audit/review had lost the
    tab strip. It signs in rather than reaching past the gate: the routes
    have to work for somebody who is holding a real cookie, and that is the
    only way anybody ever reaches them.

    The store is a THROWAWAY in a temp directory. It never opens
    data/dashboard/accounts.db, never reads the repo's .env, and does not
    care whether the machine it runs on has an account at all.
    """
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
        import auth
    except Exception:
        return {}, None            # no gate in this checkout; nothing to open
    tmp = tempfile.mkdtemp(prefix='review-gate-')
    got = auth.bootstrap(db_path=os.path.join(tmp, 'accounts.db'),
                         key_path=os.path.join(tmp, 'session.key'),
                         env={'DASHBOARD_USER': 'guard',
                              'DASHBOARD_PASSWORD': 'a-password-long-enough'})
    if not got.get('ok'):
        return {}, tmp
    import accounts
    user = accounts.get_user('guard', path=os.path.join(tmp, 'accounts.db'))
    value, _ = auth.mint(user, key_path=os.path.join(tmp, 'session.key'))
    return {'Cookie': auth.COOKIE + '=' + value}, tmp


def score_checks(html, script):
    """The detector's score, on every tile, without the cursor.

    It used to be the last item of the caption line -- and the caption line is
    the bottom of the card, which is exactly where the verdict row rides over
    it. So the number was legible only on a tile nobody was pointing at, and
    gone from every tile in audit mode, where that row never hides at all.
    The user reported it as the score having disappeared, which is what it had
    done.

    Three pins, because there are three ways to lose it again: stop drawing
    it, draw it inside the overlay that covers the caption, or hide it behind
    a hover. The last is the one that reads as a design choice.
    """
    bad = []
    if 'class="cfx"' not in script:
        bad.append('the tile draws no score chip at all (class="cfx" gone)')
    # It must be built BEFORE the action overlay, which is the element that
    # covers the foot of the card -- inside it, the score is hidden again by
    # the very thing that hid it the first time.
    i_chip, i_wrap = script.find('class="cfx"'), script.find('class="actwrap"')
    if i_chip >= 0 and i_wrap >= 0 and i_chip > i_wrap:
        bad.append('the score chip is built inside/after the action overlay, '
                   'which is what covered it before')
    rule = re.search(r'\.cfx\{[^}]*\}', html)
    if not rule:
        bad.append('no .cfx rule: the chip has no styling and no position')
    else:
        body = rule.group(0)
        if re.search(r'opacity:\s*0(?![.\d])', body):
            bad.append('the score chip starts invisible (opacity:0)')
        if 'position:absolute' not in body:
            bad.append('the score chip is not positioned over the frame, so '
                       'it is back in the flow the overlay covers')
        # Permanent furniture over a PHOTOGRAPH cannot be a wash: whatever the
        # crop shows comes through the number on exactly the bright frames
        # that need reading. Same defect the verdict buttons were fixed for.
        m = re.search(r'background:([^;}]+)', body)
        if m and re.search(r'rgba\([^)]*,\s*0?\.\d+\s*\)', m.group(1)):
            bad.append('the score chip is translucent (%s) -- the crop shows '
                       'through a number that is now always on screen'
                       % (m.group(1).strip(),))
    for m in re.finditer(r'([^{}]*)\{[^}]*\}', html):
        sel = m.group(1)
        if '.cfx' in sel and (':hover' in sel or ':focus-within' in sel):
            bad.append('a hover rule targets the score chip again: %s'
                       % (sel.strip()[:80],))
    return bad


def period_markup_checks(html):
    """The window control says what it is, on this page and on the audits.

    Two date fields with nothing beside them are a range over something
    unnamed. The audit pages carry the word already (adv_fn_audit pins it
    there); this is the same pin here, so the judging surfaces go on speaking
    one language.
    """
    bad = []
    for need, why in (
            ('type="date"', 'the window is not two calendars any more'),
            ('id="pfrom"', 'the near end of the window is gone'),
            ('id="pto"', 'the far end of the window is gone'),
            ('id="pclr"', 'there is no way back to any time'),
            ('aria-label="judged on or after',
             'the near calendar is unnamed for a screen reader'),
            ('aria-label="judged on or before',
             'the far calendar is unnamed for a screen reader'),
            ('>judged<', 'the control does not say what it filters — two '
             'bare date fields beside a verdict select'),
            ('color-scheme:dark', "the picker draws in the browser's light "
             'theme over a dark page')):
        if need not in html:
            bad.append(why + ' (%r)' % (need,))
    for gone in ('value="7d"', 'value="30d"', '>last 7 days<', '>any time<'):
        if gone in html:
            bad.append('a preset is still in the markup beside the calendars '
                       '(%r)' % (gone,))
    # The hidden input is the FILTER: one element, one value, one onchange,
    # which is what the chip row and every clear are written against.
    if 'type="hidden" id="period"' not in html:
        bad.append('the filter is no longer one element with one value — the '
                   'chip row and the clear-all are written against that')
    return bad


def route_checks(mod):
    """/audit/review serves the queue; /review answers with the new address.

    Driven over HTTP against the real handler, because the defect this
    guards against is a routing-table one: /audit/review has to be claimed
    before the generic /audit dispatch, and the redirect has to keep the
    query string a bookmark was made for.
    """
    bad = []
    session, tmp = sign_in(mod)

    def fetch(url, timeout=30):
        """One request, carrying the session. urlopen() with a header."""
        return urllib.request.urlopen(
            urllib.request.Request(url, headers=session), timeout=timeout)

    class Quiet(mod.BoardHandler):
        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(('127.0.0.1', 0), Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = 'http://127.0.0.1:%d' % srv.server_port

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    try:
        # The gate is a route too, and it comes first: if this is answered
        # with a bounce to /login, everything below is grading the login
        # form and every failure it prints names the wrong file.
        if session:
            try:
                probe = opener.open(
                    urllib.request.Request(base + '/audit/review',
                                           headers=session), timeout=10)
                if probe.status != 200:
                    bad.append('a signed-in request for /audit/review '
                               'answered %d — the guard could not get past '
                               'the login gate, so nothing below was '
                               'actually tested' % probe.status)
            except urllib.error.HTTPError as e:
                bad.append('a signed-in request for /audit/review answered '
                           '%d — the guard could not get past the login '
                           'gate, so nothing below was actually tested'
                           % e.code)
        try:
            r = opener.open(
                urllib.request.Request(base + '/review?country=JPN&x=1',
                                       headers=session), timeout=10)
            bad.append('/review answered %d itself instead of redirecting '
                       'to /audit/review' % r.status)
        except urllib.error.HTTPError as e:
            loc = e.headers.get('Location')
            if e.code not in (301, 302):
                bad.append('/review answered %d, not a redirect' % e.code)
            elif loc != '/audit/review?country=JPN&x=1':
                bad.append('the redirect dropped or rewrote the query '
                           'string: Location=%r' % loc)
        # ── a stale bookmark's guess params change nothing ───────────────
        # review_payload() still IMPLEMENTS suggest/backend/gate -- the
        # triage guard calls it that way -- and only the handler withholds
        # them. So the whole contract lives at one call site, where one
        # re-added positional argument silently narrows the queue for anyone
        # whose bookmark predates the removal, with no control on the page
        # to see or clear the narrowing. Driven over HTTP, because that is
        # the layer the promise is made at.
        qs = 'page=0&size=50&sort=low'
        stale = '&suggest=object&backend=rfdetr&gate=not_dog'
        try:
            def api(q):
                return json.loads(
                    fetch(base + '/api/review?' + q).read().decode('utf-8'))
            bare, wired = api(qs), api(qs + stale)
            if bare.get('error') or wired.get('error'):
                bad.append('/api/review answered with an error (%r / %r) — '
                           'the stale-parameter contract could not be '
                           'tested' % (bare.get('error'), wired.get('error')))
            elif ([c.get('name') for c in bare.get('items') or []]
                    != [c.get('name') for c in wired.get('items') or []]
                    or bare.get('pages') != wired.get('pages')
                    or bare.get('total_unflagged')
                    != wired.get('total_unflagged')):
                bad.append(
                    'suggest=/backend=/gate= still narrow the queue on the '
                    'server: %d items over %r pages bare, %d over %r with '
                    'the removed params — a stale bookmark shows a filtered '
                    'queue with nothing on the page to clear it'
                    % (len(bare.get('items') or []), bare.get('pages'),
                       len(wired.get('items') or []), wired.get('pages')))
        except (urllib.error.URLError, ValueError) as e:
            bad.append('/api/review could not be read with the removed '
                       'guess params set: %s' % e)
        try:
            page = fetch(base + '/audit/review',
                         timeout=10).read().decode('utf-8')
            if '<nav class="jtabs"' not in page:
                bad.append('/audit/review serves a page without the shared '
                           'tab strip')
            if 'class="jtab on" aria-current="page">Review queue' not in page:
                bad.append('/audit/review does not mark the Review queue '
                           'tab as current')
        except urllib.error.HTTPError as e:
            bad.append('/audit/review answered %d — the generic /audit '
                       'dispatch swallowed it' % e.code)
    finally:
        srv.shutdown()
        srv.server_close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    # every link dashboard.py emits has to point at the new address; the old
    # one only survives as the redirect
    with open(DASH) as fh:
        src = fh.read()
    if 'href="/review"' in src:
        bad.append('dashboard.py still emits a link to the old /review '
                   'address')
    # and the one tracked document that lists the served pages moves with
    # them: /review survives as a 301, so a reader following the README does
    # not hit an error, they just never learn the address every judging page
    # links from
    readme = os.path.join(REPO, 'README.md')
    if os.path.exists(readme):
        with open(readme, encoding='utf-8') as fh:
            doc = fh.read()
        if '`/review`' in doc:
            bad.append('README.md still lists `/review` among the pages the '
                       'dashboard serves — the queue moved to /audit/review '
                       'and the old address is only a redirect')
        elif '`/audit/review`' not in doc:
            bad.append('README.md names no review address at all — the '
                       'module table stopped describing a page the '
                       'dashboard serves')
    return bad


def period_payload_checks(mod):
    """The annotated-date window is server-side, truthful, and shared maths.

    The rows carry flagged_at, so annotated_payload narrows by it BEFORE
    counting and paginating -- a client-side hide would leave total/pages
    describing rows nobody can see. The maths is the audit pages' (audit.py
    keeps a character-identical copy): same wire format, same server-local
    days, same answer about which rows a day covers.
    """
    bad = []
    # ── the window itself ──
    lo, hi = mod.period_range('2026-08-12..2026-08-12')
    if lo != datetime(2026, 8, 12).timestamp():
        bad.append('a window does not open at the server\'s local midnight')
    if hi != datetime(2026, 8, 13).timestamp():
        bad.append('the far date does not cover its whole day — picking one '
                   'date returns the rows judged at 00:00 and calls it a day')
    if mod.period_range('2026-08-12..') != (
            datetime(2026, 8, 12).timestamp(), None):
        bad.append('an open far end is not open')
    if mod.period_range('..2026-08-12') != (
            None, datetime(2026, 8, 13).timestamp()):
        bad.append('an open near end is not open')
    if mod.period_range('') != (None, None):
        bad.append('no window is not no window')
    # A date that is not one must mean no filter, never a filter over the
    # wrong days: mktime turns the 31st of February into the 3rd of March,
    # and a window over the wrong days LOOKS like an answer.
    for junk in ('2026-02-31..', '2026-8-1..', 'yesterweek', '7d',
                 'today..tomorrow', '..', 'x' * 200):
        if mod.period_norm(junk) or mod.period_range(junk) != (None, None):
            bad.append('%r is read as a window (%r) — a typo must mean any '
                       'time' % (junk, mod.period_norm(junk)))
    # backwards is a reader's slip, not a request for nothing
    if mod.period_norm('2026-08-20..2026-08-12') != '2026-08-12..2026-08-20':
        bad.append('a backwards window is not turned around, so it selects '
                   'nothing and the page blames the ledger')
    # a row with no stamp cannot prove it is inside anything
    if mod.in_period(0, datetime(2026, 8, 12).timestamp(), None) or \
            mod.in_period(None, None, datetime(2026, 8, 13).timestamp()):
        bad.append('a row with no stamp sits inside a window')
    if not mod.in_period(0, None, None):
        bad.append('no window drops the rows it cannot place')
    # AND THE AUDIT PAGES DO THE SAME MATHS, character for character. Three
    # surfaces answer "what did I judge on the 12th" and two of them
    # disagreeing about which rows a day covers would be a bug report that
    # takes a week to believe. The files each keep a copy because neither
    # imports the other; this is what stops the copies drifting.
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
        import audit as _a
        import inspect as _i
        for fn in ('_day_start', 'period_norm', 'period_range', 'in_period'):
            here = _i.getsource(getattr(mod, fn))
            there = _i.getsource(getattr(_a, fn, None) or (lambda: None))
            if here != there:
                bad.append('dashboard.py and audit.py no longer agree on '
                           '%s() — two surfaces, two answers about which '
                           'rows a day holds' % (fn,))
        if getattr(_a, 'PERIOD_SEP', None) != mod.PERIOD_SEP:
            bad.append('the two files spell the wire format differently')
    except ImportError:
        pass                      # no audit module in this checkout

    keep = {k: getattr(mod, k) for k in
            ('HN_DIR', 'HN_CROPS', 'HN_FULL', 'HN_LABELS', 'HP_DIR')}
    kept_cache = mod._flagged
    try:
        with tempfile.TemporaryDirectory() as tmp:
            hn, hp = os.path.join(tmp, 'hn'), os.path.join(tmp, 'hp')
            os.makedirs(hn)
            os.makedirs(hp)
            now = int(time.time())

            def row(i, at):
                r = {'crop': '%d_p%03d_%03d.jpg'
                             % (1_700_000_000_000 + i, i, 50 + i % 40)}
                if at is not None:
                    r['flagged_at'] = at
                return r
            # two fresh positives, three older negatives, one row that
            # cannot prove when it was judged, and filler enough that the
            # unfiltered list takes two pages
            pos = [row(1, now), row(2, now - 3 * 86400)]
            neg = ([row(3, now - 10 * 86400), row(4, now - 40 * 86400),
                    row(5, None)]
                   + [row(100 + i, now - 40 * 86400) for i in range(55)])
            # the ledger FILENAMES come from the module, never spelt here:
            # this file is on adv_triage_isolation's no-ledger allowlist, and
            # naming a store would read as this guard touching one
            mod.HN_DIR, mod.HP_DIR = hn, hp
            mod.HN_CROPS = os.path.join(hn, 'crops')
            mod.HN_FULL = os.path.join(hn, 'full')
            mod.HN_LABELS = os.path.join(
                hn, os.path.basename(keep['HN_LABELS']))
            mod._flagged = None       # forget any live ledger already cached
            with open(mod._store_for(mod.POS_LABEL)['labels'], 'w') as fh:
                fh.write('\n'.join(json.dumps(r) for r in pos) + '\n')
            with open(mod._store_for(mod.FLAG_LABEL)['labels'], 'w') as fh:
                fh.write('\n'.join(json.dumps(r) for r in neg) + '\n')

            j_all = mod.annotated_payload(size=50)
            if j_all['total'] != 60 or j_all['pages'] != 2:
                bad.append('no period: total/pages %s/%s, want 60/2'
                           % (j_all['total'], j_all['pages']))
            # the windows the four presets used to be, spelled as dates
            def day(back):
                return time.strftime('%Y-%m-%d',
                                     time.localtime(now - back * 86400))
            j7 = mod.annotated_payload(size=50, period=day(7) + '..')
            if j7['total'] != 2 or j7['pages'] != 1:
                bad.append('since seven days back: total/pages %s/%s, want '
                           '2/1 — the filter must run before pagination'
                           % (j7['total'], j7['pages']))
            if j7.get('period') != day(7) + '..':
                bad.append('the payload does not say which window it '
                           'applied: %r' % j7.get('period'))
            if (j7['n_true_positive'], j7['n_false_positive']) != (2, 0):
                bad.append('per-verdict counts %s/%s describe more than the '
                           'window'
                           % (j7['n_true_positive'], j7['n_false_positive']))
            if j7['leash_counts']['all'] != 2:
                bad.append('the leash options were counted over the '
                           'unfiltered list (%s)' % j7['leash_counts']['all'])
            if j7['pool_unfiltered'] != 60:
                bad.append('pool_unfiltered moved with the filter (%s) — the '
                           '"narrowed from" baseline must not'
                           % j7['pool_unfiltered'])
            # ONE DAY, which is the window the presets could never express
            if mod.annotated_payload(
                    size=50, period=day(0) + '..' + day(0))['total'] != 1:
                bad.append("one day counts more than that day's row")
            if mod.annotated_payload(
                    size=50, period=day(3) + '..' + day(3))['total'] != 1:
                bad.append('a single day three days back does not hold the '
                           'one row judged in it — the far date must cover '
                           'its whole day')
            # an open near end, up to and including today
            jup = mod.annotated_payload(size=50, period='..' + day(0))
            if jup['total'] != 59:
                bad.append('up to today holds %s, want 59 (everything but '
                           'the row with no stamp)' % jup['total'])
            j30 = mod.annotated_payload(size=50, period=day(30) + '..')
            if j30['total'] != 3:
                bad.append('since thirty days back: total %s, want 3'
                           % j30['total'])
            if any(it['name'].startswith('1700000000005_')
                   for it in j30['items']):
                bad.append('a row with no flagged_at sits inside a window '
                           'it cannot prove it belongs to')
            # backwards, and it still selects the same three
            if mod.annotated_payload(
                    size=50, period=day(0) + '..' + day(30))['total'] != 3:
                bad.append('a backwards window selects nothing instead of '
                           'the days between its two dates')
            for junk in ('yesterweek', '7d', '2026-02-31..'):
                jx = mod.annotated_payload(size=50, period=junk)
                if jx['total'] != 60 or jx.get('period'):
                    bad.append('%r must fall back to any time AND say so, '
                               'never to an empty or surprise page (total '
                               '%s, echoed %r)'
                               % (junk, jx['total'], jx.get('period')))
    finally:
        for k, v in keep.items():
            setattr(mod, k, v)
        mod._flagged = kept_cache
    return bad


HARNESS = r"""
// ── stub DOM ────────────────────────────────────────────────────────────────
const failures = [];
let COLS = 5;

function parseKids(html) {
  // shallow: every tag with a class= and/or id= becomes one child node
  const out = [];
  const re = /<(\w+)([^>]*)>/g;
  let m;
  while ((m = re.exec(html))) {
    const attrs = m[2];
    const cm = /class="([^"]*)"/.exec(attrs);
    const im = /id="([^"]*)"/.exec(attrs);
    const sm = /src="([^"]*)"/.exec(attrs);
    if (!cm && !im) continue;
    const el = new El(m[1]);
    if (cm) el.className = cm[1];
    if (im) { el.id = im[1]; byId[im[1]] = el; }
    if (sm) el.src = sm[1];
    // every data-* attribute, not one hardcoded name: delegated handlers use
    // them to work out which control a click came from
    let dm; const dre = /data-([\w-]+)="([^"]*)"/g;
    while ((dm = dre.exec(attrs))) el.dataset[dm[1]] = dm[2];
    const am = /aria-expanded="([^"]*)"/.exec(attrs);
    if (am) el._attrs['aria-expanded'] = am[1];
    out.push(el);
  }
  return out;
}

const byId = {};
const allEls = [];

// what a browser's textContent -> innerHTML round trip actually escapes.
// NOT quotes -- which is exactly why the page needs att() for title="".
function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;');
}

class El {
  constructor(tag) {
    this.tagName = (tag || 'div').toUpperCase();
    this.className = ''; this._id = ''; this.dataset = {}; this.style = {};
    this.children = []; this.parentNode = null; this._text = '';
    this.hidden = false; this.disabled = false; this.src = '';
    this._html = ''; this.onclick = null; this.onchange = null;
    this._attrs = {};
    this.onmousedown = null; this.value = ''; this._listeners = {};
    this.onload = null; this.naturalWidth = 0; this.naturalHeight = 0;
    this.clientWidth = 0; this.clientHeight = 0;
    this.offsetLeft = 0; this.offsetTop = 0;
    this.scrollLeft = 0; this.scrollTop = 0;
    allEls.push(this);
  }
  // assigning .id must make the node findable, exactly as in a real document
  // (showUndo() builds the toast with `t.id='tbox'` and later looks it up)
  set id(v) { this._id = v; if (v) byId[v] = this; }
  get id() { return this._id; }
  set innerHTML(v) {
    this._html = v; this._text = '';
    for (const c of this.children) {
      const k = allEls.indexOf(c); if (k >= 0) allEls.splice(k, 1);
    }
    this.children = parseKids(v);
    for (const c of this.children) c.parentNode = this;
  }
  get innerHTML() { return this._html; }
  // A <select> is not just a box with a value: four painters on this page
  // BUILD its options and a fifth reads the chosen one's text back out to
  // label a chip. Without options/selectedIndex the read threw, load()'s
  // promise swallowed it, and every later assertion failed for the wrong
  // reason.
  get options() {
    const out = [];
    const re = /<option([^>]*)>([\s\S]*?)<\/option>/g;
    let m;
    while ((m = re.exec(this._html))) {
      const v = /value="([^"]*)"/.exec(m[1]);
      out.push({ value: v ? v[1].replace(/&quot;/g, '"') : m[2],
                 // a browser hands back option.text already decoded, so
                 // the stub must too or a label reads as raw entities
                 text: m[2].replace(/&middot;/g, '·')
                           .replace(/&mdash;/g, '—')
                           .replace(/&ldquo;/g, '\u201c')
                           .replace(/&rdquo;/g, '\u201d')
                           .replace(/&amp;/g, '&').replace(/&lt;/g, '<')
                           .replace(/&gt;/g, '>') });
    }
    return out;
  }
  get selectedIndex() {
    const o = this.options;
    for (let i = 0; i < o.length; i++) if (o[i].value === this.value) return i;
    return o.length ? 0 : -1;
  }
  // esc() in the page is `d.textContent = t; return d.innerHTML` -- model it
  set textContent(v) { this._text = String(v); this._html = escHtml(v);
                       this.children = []; }
  get textContent() {
    if (this._text) return this._text;
    // set via innerHTML: a browser still reports the text inside it
    return this._html.replace(/<[^>]*>/g, '')
      .replace(/&middot;/g, '·').replace(/&mdash;/g, '—')
      .replace(/&times;/g, '×').replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<').replace(/&gt;/g, '>');
  }
  get classList() {
    const self = this;
    return {
      add(c) { if (!self.className.split(' ').includes(c))
                 self.className = (self.className + ' ' + c).trim(); },
      remove(c) { self.className = self.className.split(' ')
                    .filter(x => x && x !== c).join(' '); },
      contains(c) { return self.className.split(' ').includes(c); },
      toggle(c, on) { on ? this.add(c) : this.remove(c); },
    };
  }
  appendChild(n) {
    if (n && n.__frag) { for (const c of n.children) this.appendChild(c); return n; }
    n.parentNode = this; this.children.push(n); return n;
  }
  insertBefore(n, ref) {
    n.parentNode = this;
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(n); else this.children.splice(i, 0, n);
    return n;
  }
  removeChild(n) {
    const i = this.children.indexOf(n);
    if (i >= 0) this.children.splice(i, 1);
    n.parentNode = null; return n;
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  addEventListener(t, f) { (this._listeners[t] = this._listeners[t] || []).push(f); }
  focus() {}
  select() {}
  setSelectionRange() {}
  scrollIntoView() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth,
                                     height: this.clientHeight }; }
  matches(sel) { return matchSel(this, sel); }
  // delegated handlers walk up from the event target; the chip row and
  // the grid both rely on it
  closest(sel) {
    for (let n = this; n; n = n.parentNode) if (matchSel(n, sel)) return n;
    return null;
  }
  querySelector(sel) { return descendants(this).find(e => matchSel(e, sel)) || null; }
  querySelectorAll(sel) { return descendants(this).filter(e => matchSel(e, sel)); }
  getAttribute(k) {
    if (k.startsWith('data-')) {
      const v = this.dataset[k.slice(5)];
      return v === undefined ? null : v;
    }
    return this._attrs[k] === undefined ? null : this._attrs[k];
  }
  setAttribute(k, v) { this._attrs[k] = String(v); }
}

function descendants(root) {
  const out = [];
  (function walk(n) { for (const c of n.children) { out.push(c); walk(c); } })(root);
  return out;
}
function matchSel(el, sel) {
  // supports ".cls", ".cls[attr="v"]", "#id"
  const am = /^\.([\w-]+)\[([\w-]+)="(.*)"\]$/.exec(sel);
  if (am) {
    if (!el.classList.contains(am[1])) return false;
    const key = am[2].replace(/^data-/, '');
    return String(el.dataset[key]) === am[3].replace(/\\(.)/g, '$1');
  }
  // compound classes: '.fbtn.no' must not be read as one class named 'fbtn.no'
  if (sel[0] === '.')
    return sel.slice(1).split('.').every(c => el.classList.contains(c));
  if (sel[0] === '#') return el.id === sel.slice(1);
  return el.tagName === sel.toUpperCase();
}

const root = new El('body');
// getElementById only sees attached nodes -- otherwise a removed toast stays
// "findable" and the page silently reuses a detached element forever
function attached(el) {
  for (let n = el; n; n = n.parentNode) if (n === root) return true;
  return false;
}
const document = {
  body: root,
  // the page copies through a detached textarea when navigator.clipboard is
  // absent, which is the case on every non-https origin -- including this one
  execCommand(cmd) {
    if (cmd !== 'copy') return false;
    const ta = descendants(root).find(e => e.tagName === 'TEXTAREA');
    if (ta) COPIED = ta.value;
    return EXEC_OK;
  },
  createElement: t => new El(t),
  createDocumentFragment: () => { const f = new El('frag'); f.__frag = true; return f; },
  getElementById: id => {
    const e = byId[id];
    return e && (attached(e) || e.__page) ? e : null;
  },
  querySelector: s => descendants(root).find(e => matchSel(e, s)) || null,
  querySelectorAll: s => descendants(root).filter(e => matchSel(e, s)),
  addEventListener: (t, f) => (docL[t] = docL[t] || []).push(f),
};
const docL = {};
const CSS = { escape: s => String(s).replace(/([^\w-])/g, '\\$1') };
const beacons = [];
let scrolls = [];
const window = {
  matchMedia: () => ({ matches: false }),
  addEventListener: (t, f) => (winL[t] = winL[t] || []).push(f),
  scrollTo: (a) => scrolls.push(a),
  // false, as it is over plain http on a LAN address: the page must not take
  // the navigator.clipboard branch, because that API is not there
  isSecureContext: false,
};
const winL = {};
const navigator = { sendBeacon: (u, b) => { beacons.push(u); return true; } };
// No navigator.clipboard: it does not exist on a plain http origin, which is
// how this dashboard is actually served. The page must reach the fallback.
const Blob = function (parts, opts) { this.parts = parts; this.type = opts && opts.type; };
function getComputedStyle() {
  return { gridTemplateColumns: new Array(COLS).fill('100px').join(' ') };
}
function requestAnimationFrame(f) { f(); }
function setTimeout(f, ms) { timers.push({ f, ms }); return timers.length; }
// IntersectionObserver: the header sheds its ambient rows once the page has
// scrolled, and without a stub the guard skips that whole branch -- so the
// compact behaviour would look tested while never running.
let COPIED = null;              // what reached the clipboard
let SECURE = false;             // http origin by default, as the box serves it
let EXEC_OK = true;
let IO_CB = null;
function IntersectionObserver(cb) {
  IO_CB = cb;
  return { observe() {}, disconnect() {} };
}
// setInterval stubbed to a no-op handle: a real one keeps node's event
// loop alive forever, hanging this test the moment the page starts a poll
function setInterval() { return 0; }
function clearInterval() {}
function clearTimeout(h) { if (h) timers[h - 1] = null; }
const timers = [];
function runTimers() { const t = timers.slice(); timers.length = 0;
                       for (const x of t) if (x) x.f(); }

// ── controllable fetch ──────────────────────────────────────────────────────
let RESP = {};           // url-substring -> () => value | 'reject'
// The status of the answer, by the same keys, 200 for anything unnamed. A
// stub that could only ever hand back 200 could not pose the question the
// login gate poses: it refuses with 401 and a body carrying no `ok` at all,
// which is a shape no endpoint on this page has ever produced by itself.
let STATUS = {};
const CALLS = [];
function fetch(url, opts) {
  CALLS.push({ url, body: opts && opts.body ? JSON.parse(opts.body) : null });
  // longest key first: '/api/review' is a prefix of '/api/review/seen', so
  // insertion order would silently answer the seen POST with a page payload
  for (const k of Object.keys(RESP).sort((a, b) => b.length - a.length)) {
    if (String(url).includes(k)) {
      const v = RESP[k](url, opts);
      if (v === 'reject') return Promise.reject(new Error('boom'));
      const st = STATUS[k] || 200;
      return Promise.resolve({ ok: st < 400, status: st,
                               json: () => Promise.resolve(v) });
    }
  }
  return Promise.reject(new Error('unstubbed ' + url));
}

// ── the page's own element graph (built from the real markup ids) ───────────
for (const id of ['left','done','seen','dups','unkeep','pg','pg2','next','next2','mode','verdict',
                  // The audit view's annotated-date window. 'period' is the
                  // FILTER -- one hidden input, one value, one onchange, which
                  // is what the chips and the clear-all are written against --
                  // and the two calendars beside it are how it gets set.
                  'period','periodwrap','pfrom','pto','pclr',
                  'foot','grid','state','sort','size','reload','country','leftlab',
                  'leashN',
                  // findmsg is what says the search cannot work; leaving it
                  // out of the stub makes paintFind's guard skip the whole
                  // branch, so every state would 'pass' untested
                  'find','findterms','findmsg',
                  // the redesigned block: caption, applied-filter chips
                  // and the disclosure holding the controls
                  'cap','chips','narrow','npanel',
                  // the sentinel the header watches to know it has scrolled
                  'scrollcue']) {
  const e = new El(id === 'grid' || id === 'state' || id === 'foot' ? 'div' : 'span');
  e.id = id; e.__page = true; root.appendChild(e);
}
byId['sort'].value = 'conf';
byId['size'].value = '50';
byId['country'].value = '';

// ── run the page script ─────────────────────────────────────────────────────
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
// Stale preferences from the build that HAD the guess feature. Seeded before
// the script runs, because that is when restorePrefs reads them: a stored
// suggest/backend/gatef must never put a removed filter back into a request.
global.localStorage = {
  _o: JSON.stringify({suggest: 'dog', backend: 'rfdetr', gatef: 'dog'}),
  getItem(k) { return k === 'sdReview' ? this._o : null; },
  setItem(k, v) { if (k === 'sdReview') this._o = v; },
};
// The address this page was opened at. /review answers 301 to /audit/review
// carrying its query string, so a bookmark made on ?country=ECU arrives here
// with one -- and it is only worth carrying if the page reads it. Seeded
// before the script runs, because that is when restorePrefs looks.
global.location = { search: '?country=ECU&page=2', pathname: '/audit/review',
                    href: 'http://stub/audit/review?country=ECU&page=2' };
global.URLSearchParams = URLSearchParams;
let API;
try {
  API = new Function('document','window','CSS','fetch','getComputedStyle',
    'requestAnimationFrame','setTimeout','clearTimeout','setInterval','clearInterval','docL','navigator','Blob','IntersectionObserver',
    src + '\nreturn {load,render,flag,undo,openLb,closeLb,stepLb,tile,score,'
        + 'idx,mark,cols,hideToast,showUndo,'
        + 'markSeen,imgScale,saveBox,paintBox,fitBox,fitImage,zoomBy,'
        + 'flushSave,dirty,'
        + 'st:()=>({page,size,sort,items,reserve,pages,sel,todoN,flaggedN,'
        + 'seenN,session,lastUndo,lb})};')(
    document, window, CSS, fetch, getComputedStyle, requestAnimationFrame,
    setTimeout, clearTimeout, setInterval, clearInterval, docL, navigator, Blob,
    IntersectionObserver);
} catch (e) {
  console.log('FAIL: could not evaluate the review script: ' + e);
  process.exit(1);
}

// What the page asked the server for on BOOT, before any case rewrites RESP.
const BOOT_CALLS = CALLS.slice();

const FIX = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const CROPS = FIX.crops;
// Which page elements the real markup ships hidden. Taken from the
// markup rather than assumed, because a stub that starts everything
// visible lets a panel 'pass' a test for being shut.
for (const id of (FIX.hidden || [])) if (byId[id]) byId[id].hidden = true;
// ...and the options a <select> ships in the markup. Four painters BUILD
// their options at runtime, but #mode, #verdict, #sort and #size carry
// theirs in the HTML -- so a stub that starts them empty makes every
// read of a chosen option's text return '' and any test of one vacuous.
for (const [id, html] of Object.entries(FIX.options || {}))
  if (byId[id]) { byId[id].innerHTML = html; byId[id].value =
    (byId[id].options[0] || {}).value || ''; }
// The panel's shape: which controls sit in which group, read off the
// markup. trimGroups() walks that tree to decide whether a group still
// offers anything, and a flat stub gave it nothing to walk -- so the
// heading-over-nothing it exists to prevent could not be tested at all.
for (const [gid, ids] of Object.entries(FIX.groups || {})) {
  const g = byId[gid]; if (!g) continue;
  g.className = ((g.className || '') + ' ngrp').trim();
  const row = new El('div'); row.className = 'nrow'; g.appendChild(row);
  for (const id of ids) if (byId[id]) row.appendChild(byId[id]);
}
function payload(items, reserve, extra) {
  return Object.assign({ items, reserve: reserve || [], page: 0, pages: 2,
                         size: 50, sort: 'conf',
                         total_unflagged: 120, flagged_total: 30 }, extra || {});
}
function key(k) {
  for (const f of (docL['keydown'] || []))
    f({ key: k, preventDefault(){}, target: { tagName: 'BODY' } });
}
function ck(cond, msg) { if (!cond) failures.push(msg); }
// keydown handlers fire flag()/undo() without awaiting; drain the microtask
// queue deep enough that their fetch chains have settled
async function flush(n) { for (let i = 0; i < (n || 12); i++) await Promise.resolve(); }
function toastUp() { return root.children.some(c => c.id === 'tbox'); }
// What the page's one notice surface is currently saying. Both halves are
// needed: showUndo builds the box with innerHTML, where the text is in the
// markup, while leashNote appends a span it set textContent on, which the
// stub's own textContent getter cannot see from the parent.
function noticed() {
  const t = root.children.filter(c => c.id === 'tbox')[0];
  if (!t) return '';
  return (String(t.textContent || '') + ' ' +
          descendants(t).map(e => String(e._text || '')).join(' ')).trim();
}
// A throw inside one of the page's .then handlers never reaches this harness:
// every fetch chain here ends in its own .catch, which absorbs it and resolves
// as though the work had finished. That is how a free identifier in the audit
// branch stayed hidden through a green suite. Wrapping .then for the length of
// one drive records the throw on its way past without altering it.
async function watchThrows(fn) {
  const seen = [];
  const realThen = Promise.prototype.then;
  Promise.prototype.then = function (ok, bad) {
    return realThen.call(this, typeof ok === 'function' ? function (v) {
      try { return ok(v) } catch (e) { seen.push(e); throw e }
    } : ok, bad);
  };
  try { await fn(); } finally { Promise.prototype.then = realThen; }
  return seen;
}
function why(e) { return String((e && (e.message || e.name)) || e); }

// Cross the sentinel the way a scroll does. IO_CB is the page's own callback,
// captured when it constructed the observer.
let IO_T = 0;
// `ms` advances the crossing timestamp the page reads, so the hysteresis that
// stops a sticky header fluttering can be exercised from both sides without
// the test sleeping.
function scrolled(down, ms) {
  IO_T += (ms === undefined ? 1000 : ms);
  if (IO_CB) IO_CB([{isIntersecting: !down, time: IO_T}]);
}

// ── 1. load + render ────────────────────────────────────────────────────────
async function t1() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6),
                                        CROPS.normal.slice(6, 9)) };
  await API.load(); await flush();
  const cards = document.querySelectorAll('.card');
  ck(cards.length === 6, 't1: rendered ' + cards.length + ' tiles, want 6');
  ck(byId['left'].textContent === '120', 't1: left=' + byId['left'].textContent);
  ck(byId['done'].textContent === '30', 't1: done=' + byId['done'].textContent);
  /* The queue is consumed from the head, not paged through: nav() banks the
     screen before loading, so an offset on top of that skipped a screenful
     every turn. The label counts what is LEFT rather than naming an offset
     that no longer moves, and Prev is gone -- Restore kept is the way back. */
  ck(/^6 shown \u00b7 \d+ left$/.test(byId['pg'].textContent),
     't1: pg=' + byId['pg'].textContent);
  ck(byId['next'].disabled === false, 't1: next disabled with crops left');
  ck(byId['prev'] === undefined || byId['prev'] === null ||
     !byId['prev'].onclick, 't1: Prev still wired');
  ck(byId['next'].disabled === false, 't1: next disabled with 2 pages');
  // nothing may be pre-selected: a highlighted first tile reads as a choice
  // the user did not make
  ck(API.st().sel === -1, 't1: something was pre-selected, sel=' + API.st().sel);
  ck(!document.querySelectorAll('.card').some(c => c.classList.contains('sel')),
     't1: a tile is marked selected on load');
  // the confidence rail must reflect conf, not be a constant
  const rails = document.querySelectorAll('.rail');
  ck(rails.length === 6, 't1: ' + rails.length + ' rails for 6 tiles');
}

// ── 2. flag: surgical removal + backfill from reserve ───────────────────────
async function t2() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6),
                                        CROPS.normal.slice(6, 9)),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  const before = API.st().items.map(c => c.name);
  await API.flag(2); await flush();
  const after = API.st().items.map(c => c.name);
  ck(!after.includes(before[2]), 't2: flagged crop still in items');
  ck(after.length === 6, 't2: grid shrank to ' + after.length + ', want backfill to 6');
  ck(after[5] === CROPS.normal[6].name, 't2: backfilled with the wrong crop');
  ck(API.st().reserve.length === 2, 't2: reserve not consumed');
  ck(document.querySelectorAll('.card').length === 6,
     't2: DOM has ' + document.querySelectorAll('.card').length + ' tiles');
  ck(byId['left'].textContent === '119', 't2: left=' + byId['left'].textContent);
  ck(byId['done'].textContent === '31', 't2: done=' + byId['done'].textContent);
  const post = CALLS[CALLS.length - 1];
  ck(post.body && post.body.name === before[2] && post.body.label === 'false_positive',
     't2: wrong flag POST body: ' + JSON.stringify(post.body));
  ck(!!byId['undoB'], 't2: no undo control in the toast');
  // DOM order must still track items order, or arrow keys select the wrong tile
  const dom = document.querySelectorAll('.card').map(e => e.dataset.name);
  ck(JSON.stringify(dom) === JSON.stringify(after),
     't2: DOM order diverged from items order');
}

// ── 3. undo restores the crop AT ITS OLD INDEX ─────────────────────────────
async function t3() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6),
                                        CROPS.normal.slice(6, 9)),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  const before = API.st().items.map(c => c.name);
  const s0 = API.st().session;
  await API.flag(2); await flush();
  await API.undo(); await flush();
  const after = API.st().items.map(c => c.name);
  ck(after[2] === before[2], 't3: undo put the crop at index ' +
     after.indexOf(before[2]) + ', want 2');
  ck(byId['left'].textContent === '120', 't3: left=' + byId['left'].textContent);
  ck(byId['done'].textContent === '30', 't3: done=' + byId['done'].textContent);
  ck(API.st().session === s0, 't3: session counter not decremented');
  const dom = document.querySelectorAll('.card').map(e => e.dataset.name);
  ck(JSON.stringify(dom) === JSON.stringify(after), 't3: DOM order wrong after undo');
  ck(!toastUp(), 't3: toast still present after undo');
  // a flag pulls one crop out of `reserve`; undo must hand it back, or
  // repeated flag/undo cycles grow the page without bound
  ck(after.length === before.length,
     't3: page length drifted ' + before.length + ' -> ' + after.length);
  ck(API.st().reserve.length === 3, 't3: reserve not restored, has ' +
     API.st().reserve.length + ' want 3');
}

// ── 4. a REFUSED flag must roll back, never drop the tile ──────────────────
async function t4() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6), []),
           '/api/detect/flag': () => ({ ok: false, error: 'nope' }) };
  await API.load(); await flush();
  const n = API.st().items.length;
  const name = API.st().items[1].name;
  await API.flag(1); await flush();
  ck(API.st().items.length === n, 't4: refused flag still removed the crop');
  ck(API.st().items[1].name === name, 't4: refused flag reordered items');
  const card = document.querySelector('.card[data-name="' + name + '"]');
  ck(card && !card.classList.contains('go'),
     't4: tile left in the exiting state after a refusal');
  ck(byId['left'].textContent === '120', 't4: counters moved on a refusal');
}

// ── 5. fetch failure -> error state with a retry, not a blank page ─────────
async function t5() {
  RESP = { '/api/review': () => 'reject' };
  await API.load(); await flush(); await Promise.resolve();
  ck(/Could not reach/.test(byId['state'].innerHTML), 't5: no error state shown');
  ck(!!byId['retry'], 't5: no retry control');
  ck(byId['foot'].hidden === true, 't5: pager left visible over an error');
  ck(document.querySelectorAll('.card').length === 0, 't5: stale tiles kept');
}

// ── 6. empty queue -> an invitation, not a void ────────────────────────────
async function t6() {
  RESP = { '/api/review': () => payload([], [], { total_unflagged: 0,
                                                  flagged_total: 500, pages: 1 }) };
  await API.load(); await flush();
  ck(/Queue is clear/.test(byId['state'].innerHTML), 't6: no empty state');
  ck(!!byId['rl2'], 't6: no way to re-check from the empty state');
  ck(byId['left'].textContent === '0', 't6: left=' + byId['left'].textContent);
  ck(byId['done'].textContent === '500', 't6: done=' + byId['done'].textContent);
  ck(byId['foot'].hidden === true, 't6: pager shown for a single page');
}

// ── 7. injection in image_id must not reach markup unescaped ───────────────
async function t7() {
  RESP = { '/api/review': () => payload(CROPS.hostile, []) };
  await API.load(); await flush();
  const h = byId['grid'].children.map(c => c.innerHTML).join('');
  // no raw tag may appear that we did not write ourselves
  ck(!/<script/i.test(h), 't7: <script survived into tile markup');
  ck(!/<img\s+src=x/i.test(h), 't7: <img src=x survived into tile markup');
  ck(h.includes('&lt;'), 't7: nothing was escaped at all');
  // The id also lands in a title="". esc() does NOT touch quotes, so a bare
  // esc() there lets `"><script>` close the attribute AND the tag. Assert the
  // exact fully-escaped attribute value rather than hunting for fragments.
  ck(h.includes('title="&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;"'),
     't7: title="" not fully escaped -- got ' +
     String((/title="([^"]*)"/.exec(h) || [])[1]));
  const src = byId['grid'].children[0].querySelector('.thumb').src;
  ck(!src.includes('"') && !src.includes('<'), 't7: thumb src not URL-encoded: ' + src);
}

// ── 8. keyboard: arrows honour the column count, F/U/Enter/Esc all bound ───
async function t8() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 12), []),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  COLS = 5;
  const last = API.st().items.length - 1;
  ck(API.cols() === 5, 't8: cols()=' + API.cols());
  // first arrow from "nothing selected" lands on tile 0, not tile 1
  ck(API.st().sel === -1, 't8: page did not open unselected');
  key('ArrowRight'); ck(API.st().sel === 0, 't8: first arrow -> ' + API.st().sel);
  key('ArrowRight'); ck(API.st().sel === 1, 't8: right -> ' + API.st().sel);
  key('ArrowDown');  ck(API.st().sel === 6, 't8: down -> ' + API.st().sel + ', want 6');
  key('ArrowUp');    ck(API.st().sel === 1, 't8: up -> ' + API.st().sel);
  key('ArrowLeft');  ck(API.st().sel === 0, 't8: left -> ' + API.st().sel);
  key('ArrowLeft');  ck(API.st().sel === 0, 't8: left ran past 0');
  for (let i = 0; i < 40; i++) key('ArrowDown');
  ck(API.st().sel === last, 't8: down ran past the end -> ' + API.st().sel +
     ', last is ' + last);
  for (let i = 0; i < 40; i++) key('ArrowUp');
  ck(API.st().sel === 0, 't8: up ran past 0 -> ' + API.st().sel);
  // Enter opens the lightbox on a crop that has a full frame
  key('Enter');
  ck(!!API.st().lb, 't8: Enter did not open the lightbox');
  key('Escape'); ck(!API.st().lb, 't8: Escape did not close the lightbox');
  // F flags the selection
  const n0 = API.st().items.length;
  key('f'); await flush();
  ck(API.st().items.length === n0 - 1, 't8: F did not flag the selection');
  // U undoes it
  key('u'); await flush();
  ck(API.st().items.length === n0, 't8: U did not undo');
  // typing in a control must not steal the key
  const sel0 = API.st().sel, n1 = API.st().items.length;
  for (const f of (docL['keydown'] || []))
    f({ key: 'f', preventDefault(){}, target: { tagName: 'SELECT' } });
  await flush();
  ck(API.st().items.length === n1 && API.st().sel === sel0,
     't8: F fired while focus was in a <select>');
}

// ── 9. lightbox: only steps to crops that HAVE a full frame ───────────────
async function t9() {
  RESP = { '/api/review': () => payload(CROPS.mixed, []) };
  await API.load(); await flush();
  // has_full only says whether a burned-in PREVIEW was saved; the editor
  // reads the original jpg, so every crop opens
  API.openLb(0);
  ck(!!API.st().lb, 't9: refused a crop with no preview frame');
  ck(String(byId['lbi'].src).startsWith('/orig?name='),
     't9: lightbox did not load the ORIGINAL (needed to edit): ' + byId['lbi'].src);
  ck(!String(byId['lbi'].src).includes(' '),
     't9: lightbox src not URL-encoded: ' + byId['lbi'].src);
  API.openLb(1);
  ck(API.st().sel === 1, 't9: opening did not move the selection');
  const first = byId['lbi'].src;
  await API.stepLb(1); await flush();
  ck(byId['lbi'].src !== first, 't9: step(1) did not advance');
  await API.stepLb(1); await API.stepLb(1); await API.stepLb(1); await flush();
  ck(!!API.st().lb, 't9: stepping past the end closed the lightbox');
  ck(API.st().sel >= 0 && API.st().sel < CROPS.mixed.length,
     't9: stepped outside the page, sel=' + API.st().sel);
  API.closeLb(); ck(!API.st().lb, 't9: closeLb left the overlay up');
  ck(document.body.style.overflow === '', 't9: page scroll not restored');
}

// ── 10. flagging the LAST crop falls into the empty state ─────────────────
async function t10() {
  RESP = { '/api/review': () => payload([CROPS.normal[0]], []),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  await API.flag(0); await flush();
  ck(/Queue is clear/.test(byId['state'].innerHTML),
     't10: no empty state after the last crop was flagged');
  ck(API.st().sel === -1, 't10: sel should be -1 on an empty page, got ' + API.st().sel);
  // and undo must climb back out of the empty state
  await API.undo(); await flush();
  ck(API.st().items.length === 1, 't10: undo did not restore the last crop');
  ck(document.querySelectorAll('.card').length === 1,
     't10: undo restored state but not the tile');
}

// ── 11. double-flag of the same crop must not double-post ─────────────────
async function t11() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 4), []),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  CALLS.length = 0;
  const p1 = API.flag(1), p2 = API.flag(1);
  await p1; await p2; await Promise.resolve(); await Promise.resolve();
  const posts = CALLS.filter(c => String(c.url).includes('/api/detect/flag'));
  ck(posts.length === 1, 't11: ' + posts.length + ' POSTs for one crop');
  ck(API.st().items.length === 3, 't11: items=' + API.st().items.length);
}

// ── 12. the 5 s undo window really expires ────────────────────────────────
async function t12() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 4), []),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  await API.flag(0); await flush();
  ck(!!API.st().lastUndo, 't12: nothing staged for undo right after a flag');
  runTimers();
  ck(!API.st().lastUndo, 't12: undo still live after the timer fired');
  const n = API.st().items.length;
  await API.undo(); await Promise.resolve();
  ck(API.st().items.length === n, 't12: undo worked after the window closed');
}

// ── 13. flagging must not advance the selection on its own ───────────────
async function t13() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6), CROPS.normal.slice(6, 9)),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  // MOUSE path: pressing the flag button must not select the tile, and after
  // removal nothing may be selected -- otherwise the highlight lands on
  // whatever slid into that index, which reads as an auto-advance
  const card = document.querySelectorAll('.card')[1];
  card.onmousedown({ target: { closest: sel => sel === '.acts' ? {} : null } });
  ck(API.st().sel === -1, 't13: pressing a verdict button selected the tile');
  await API.flag(1); await flush();
  ck(API.st().sel === -1,
     't13: mouse flag left a selection (auto-advance), sel=' + API.st().sel);
  ck(!document.querySelectorAll('.card').some(c => c.classList.contains('sel')),
     't13: a tile is highlighted after a mouse flag');
  // pressing elsewhere on the tile still selects it (needed for the lightbox)
  const c2 = document.querySelectorAll('.card')[2];
  c2.onmousedown({ target: { closest: () => null } });
  ck(API.st().sel === 2, 't13: clicking the tile body no longer selects');
  // KEYBOARD path: F keeps the position so the next crop can be judged.
  // items.length does NOT drop -- the reserve backfills -- so assert on
  // identity: the crop that was selected must be gone.
  const gone = API.st().items[2].name;
  key('f'); await flush();
  ck(!API.st().items.some(c => c.name === gone),
     't13: F did not flag the selected crop');
  ck(API.st().sel === 2,
     't13: F flow lost its position, sel=' + API.st().sel + ' want 2');
  // D marks a low-confidence detection as a REAL dog -> the other ledger
  const lbl = [];
  RESP['/api/detect/flag'] = (u, o) => { lbl.push(JSON.parse(o.body).label);
                                         return { ok: true }; };
  const dogCrop = API.st().items[2].name;
  key('d'); await flush();
  ck(lbl[lbl.length - 1] === 'true_positive',
     't13: D sent label ' + lbl[lbl.length - 1]);
  ck(!API.st().items.some(c => c.name === dogCrop),
     't13: D did not remove the crop from the queue');
  // undo must return it under the SAME label, or it is undone in the wrong
  // ledger and stays flagged forever in the other one
  await API.undo(); await flush();
  const last = CALLS[CALLS.length - 1];
  ck(last.body && last.body.undo === true && last.body.label === 'true_positive',
     't13: undo sent ' + JSON.stringify(last.body));
}

// ── 14. paging banks the screen as reviewed, so it never comes back ──────
async function t14() {
  let posted = null;
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6), []),
           '/api/review/seen': (u, o) => {
             posted = JSON.parse(o.body).names; return { ok: true, seen_total: 99 };
           } };
  await API.load(); await flush();
  const onScreen = API.st().items.map(c => c.name);
  CALLS.length = 0;
  byId['next'].onclick(); await flush();
  ck(posted !== null, 't14: paging did not record the screen as reviewed');
  ck(JSON.stringify(posted) === JSON.stringify(onScreen),
     't14: banked the wrong crops');
  ck(API.st().seenN === 99, 't14: reviewed total not tracked, ' + API.st().seenN);
  // the seen POST must land BEFORE the next page is fetched, or the server
  // computes the next page from a pool that still contains what we just kept
  const order = CALLS.map(c => String(c.url).includes('/seen') ? 'seen' : 'page');
  ck(order.indexOf('seen') >= 0 && order.indexOf('seen') < order.indexOf('page'),
     't14: fetched the next page before banking this one: ' + order.join(','));
  // an empty grid must not POST an empty list
  RESP['/api/review'] = () => payload([], []);
  posted = null;
  await API.load(); await flush();
  await API.markSeen(); await flush();
  ck(posted === null, 't14: posted an empty screen as reviewed');
}

// ── 15. Restore kept: confirmed, scoped, and never touches the flags ─────
async function t15() {
  let body = null, alerted = null, kept = 7;   // server-side kept count
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 4), [], {seen_total: kept}),
           '/api/review/seen': (u, o) => { body = JSON.parse(o.body);
                                           if (body.reset) kept = 0;
                                           return { ok: true, restored: 7, seen_total: kept }; },
           '/api/dataset': () => ({ dog: 10, not_dog: 2, new_flags: 0,
                                    yield_per_flag: 0.822, dataset: 'x' }) };
  await API.load(); await flush();
  ck(API.st().seenN === 7, 't15: kept total not read from the payload');

  // declining the confirm must do nothing at all
  window.confirm = () => false;
  body = null;
  byId['unkeep'].onclick(); await flush();
  ck(body === null, 't15: acted despite the confirm being declined');
  ck(API.st().seenN === 7, 't15: cleared the counter on a declined confirm');

  // accepting sends reset:true -- never a names list, which would BANK them
  window.confirm = () => true;
  byId['unkeep'].onclick(); await flush();
  ck(body && body.reset === true, 't15: did not send reset:true, sent ' +
     JSON.stringify(body));
  ck(!body.names, 't15: sent a names list on reset -- that would re-bank them');
  ck(API.st().seenN === 0, 't15: kept total not cleared after restore');

  // with nothing kept it must warn instead of prompting to restore nothing
  window.alert = m => { alerted = m; };
  window.confirm = () => { throw new Error('should not prompt with 0 kept'); };
  byId['unkeep'].onclick(); await flush();
  ck(alerted && /Nothing to restore/.test(alerted),
     't15: no guard when there is nothing to restore');
}

// ── 16. a new page starts at the top ─────────────────────────────────────
async function t16() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6), []),
           '/api/review/seen': () => ({ ok: true, seen_total: 1 }),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  scrolls = [];
  byId['next'].onclick(); await flush();
  ck(scrolls.length >= 1, 't16: paging did not scroll to the top');
  ck(scrolls[scrolls.length - 1] &&
     (scrolls[scrolls.length - 1].top === 0 || scrolls[scrolls.length - 1] === 0),
     't16: scrolled somewhere other than the top: ' +
     JSON.stringify(scrolls[scrolls.length - 1]));
  // flagging must NOT jump the page -- the user is mid-grid judging crops
  scrolls = [];
  await API.flag(0); await flush();
  ck(scrolls.length === 0, 't16: a flag scrolled the page');
  // undo must not either
  scrolls = [];
  await API.undo(); await flush();
  ck(scrolls.length === 0, 't16: an undo scrolled the page');
}

// ── 17. box editing keeps ORIGINAL pixels, whatever the render scale ─────
async function t17() {
  let posted = null;
  const BOX = { ok: true, image_id: 'img1', w: 4000, h: 3000, has_file: true,
                boxes: [{det_idx: 0, x1: 1000, y1: 800, x2: 1400, y2: 1200,
                         conf: 0.5}], saved: null };
  RESP = { '/api/review/box': (u, o) => { if (o) { posted = JSON.parse(o.body);
                                                   return { ok: true }; }
                                          return BOX; },
           '/api/review': () => payload([crop0()], []),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  // openLb rebuilds the overlay, so size the <img> AFTER it exists:
  // a 4000px image rendered at 800px is scale 0.2
  async function open0(){
    API.openLb(0); await flush();
    const im = byId['lbi'];
    im.naturalWidth = 4000; im.naturalHeight = 3000;
    byId['lbw'].clientWidth = 800; byId['lbw'].clientHeight = 600;
    API.fitImage();            // 800/4000 == 600/3000 == 0.2
  }
  await open0();
  ck(byId['lbbox'].hidden === false, 't17: box overlay never shown');
  ck(Math.abs(API.imgScale() - 0.2) < 1e-9, 't17: scale=' + API.imgScale());
  // overlay is placed in DISPLAY px
  ck(byId['lbbox'].style.left === '200px',
     't17: left=' + byId['lbbox'].style.left + ' want 200px (1000*0.2)');
  ck(byId['lbbox'].style.width === '80px',
     't17: width=' + byId['lbbox'].style.width + ' want 80px (400*0.2)');

  // drag the whole box 100 display px right = 500 ORIGINAL px
  byId['lbbox']._listeners.mousedown[0]({ target: { getAttribute: () => null },
    clientX: 0, clientY: 0, preventDefault(){}, stopPropagation(){} });
  for (const f of (docL['mousemove'] || [])) f({ clientX: 100, clientY: 0 });
  for (const f of (docL['mouseup'] || [])) f({});
  await API.saveBox(); await flush();
  ck(posted && Math.round(posted.box[0]) === 1500,
     't17: moved box x1=' + (posted && posted.box[0]) + ' want 1500 ORIGINAL px');
  ck(Math.round(posted.box[2]) === 1900, 't17: x2=' + posted.box[2]);
  ck(Math.round(posted.box[1]) === 800 && Math.round(posted.box[3]) === 1200,
     't17: vertical drifted on a horizontal drag');
  ck(posted.det_idx === 0, 't17: wrong det_idx ' + posted.det_idx);

  // resizing by a corner must move only that corner
  await open0();
  byId['lbbox']._listeners.mousedown[0]({ target: { getAttribute: () => 'se' },
    clientX: 0, clientY: 0, preventDefault(){}, stopPropagation(){} });
  for (const f of (docL['mousemove'] || [])) f({ clientX: 20, clientY: 20 });
  for (const f of (docL['mouseup'] || [])) f({});
  await API.saveBox(); await flush();
  ck(Math.round(posted.box[0]) === 1000 && Math.round(posted.box[1]) === 800,
     't17: SE handle moved the NW corner');
  ck(Math.round(posted.box[2]) === 1500 && Math.round(posted.box[3]) === 1300,
     't17: SE corner went to ' + posted.box[2] + ',' + posted.box[3]);

  // dragging far off-image must clamp inside the picture, not save negatives
  await open0();
  byId['lbbox']._listeners.mousedown[0]({ target: { getAttribute: () => null },
    clientX: 0, clientY: 0, preventDefault(){}, stopPropagation(){} });
  for (const f of (docL['mousemove'] || [])) f({ clientX: -99999, clientY: -99999 });
  for (const f of (docL['mouseup'] || [])) f({});
  await API.saveBox(); await flush();
  ck(posted.box.every(v => v >= 0), 't17: saved a negative coordinate: ' +
     JSON.stringify(posted.box));
  ck(posted.box[2] <= 4000 && posted.box[3] <= 3000,
     't17: saved past the image bounds');

  // a SMALL object must open zoomed in, not fitted to the whole frame --
  // a 30px box on a 4000px image is 6 screen px at fit, which is the
  // complaint this whole zoom model exists to answer
  BOX.boxes = [{det_idx: 0, x1: 2000, y1: 1500, x2: 2030, y2: 1530, conf: 0.5}];
  await open0();
  API.fitBox();
  const zBox = API.imgScale(), zFit = 0.2;
  ck(zBox > zFit * 5, 't17: fitBox barely zoomed: ' + zBox + ' vs fit ' + zFit);
  const px = 30 * zBox;
  ck(px > 150, 't17: a 30px object renders at only ' + Math.round(px) +
     ' screen px after Fit box');
  // and the handles must NOT have grown with it -- they are plain px in CSS,
  // so assert the box itself is what scaled
  ck(byId['lbbox'].style.width === px + 'px',
     't17: box width ' + byId['lbbox'].style.width + ' want ' + px + 'px');
  // one-pixel nudge stays one ORIGINAL pixel however deep the zoom
  const x0 = 2000;
  for (const f of (docL['keydown'] || []))
    f({ key: 'ArrowRight', shiftKey: true, preventDefault(){},
        target: { tagName: 'BODY' } });
  await API.saveBox(); await flush();
  ck(Math.round(posted.box[0]) === x0 + 1,
     't17: Shift+Arrow moved ' + (posted.box[0] - x0) + 'px, want exactly 1');
}
function crop0(){ return CROPS.normal[0]; }

// ── 18. box edits save themselves, and always before the verdict ─────────
async function t18() {
  const order = [];
  const BOX = { ok: true, image_id: 'img1', w: 4000, h: 3000, has_file: true,
                boxes: [{det_idx: 0, x1: 100, y1: 100, x2: 500, y2: 500,
                         conf: 0.5}], saved: null };
  RESP = { '/api/review/box': (u, o) => { if (o) { order.push('box');
                                                   return { ok: true }; }
                                          return BOX; },
           // three crops: stepping away must have somewhere to go
           '/api/review': () => payload(CROPS.normal.slice(0, 3), []),
           '/api/detect/flag': () => { order.push('flag'); return { ok: true }; } };
  await API.load(); await flush();
  API.openLb(0); await flush();
  byId['lbi'].naturalWidth = 4000; byId['lbi'].naturalHeight = 3000;
  byId['lbw'].clientWidth = 800; byId['lbw'].clientHeight = 600;
  API.fitImage();

  // there is no Save button any more
  ck(!byId['lbsave'], 't18: a Save box button still exists');

  // an edit schedules a save on its own -- no click
  order.length = 0;
  API.dirty(true);
  ck(order.length === 0, 't18: saved instantly, losing the debounce');
  runTimers(); await flush();
  ck(order[0] === 'box', 't18: an edit did not autosave, order=' + order);

  // a verdict must not reach the server before the pending box does
  order.length = 0;
  API.dirty(true);                     // dirty again, still debouncing
  byId['lbf'].onclick(); await flush(); await flush();
  ck(order.join(',') === 'box,flag',
     't18: verdict raced the box save, order=' + order.join(','));

  // stepping away also flushes first
  API.openLb(0); await flush();
  order.length = 0;
  API.dirty(true);
  await API.stepLb(1); await flush();
  ck(order[0] === 'box', 't18: stepping away dropped the pending edit');
}

// ── 19. the collapse count is surfaced, not silently swallowed ──────────
async function t19() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 4), [],
                                        {collapsed: 669, total_unflagged: 1881}) };
  await API.load(); await flush();
  ck(byId['dups'].textContent === '669',
     't19: hidden-repeat count not shown, got ' + byId['dups'].textContent);
  ck(byId['left'].textContent === '1,881', 't19: left=' + byId['left'].textContent);
  // a payload without the field must not print undefined/NaN
  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 4), []);
  await API.load(); await flush();
  ck(/^[0-9,]+$/.test(byId['dups'].textContent),
     't19: non-numeric when the server omits collapsed: ' + byId['dups'].textContent);
}

// ── 20. the country filter reaches the server and repaints its options ──
async function t20() {
  const LIST = [{iso:'DEU',name:'Germany',n:904},{iso:'JPN',name:'Japan',n:838}];
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
                                        {countries: LIST, country: ''}),
           '/api/review/seen': () => ({ ok: true, seen_total: 1 }) };
  await API.load(); await flush();
  const opts = byId['country'].innerHTML;
  ck(/All countries/.test(opts), 't20: no "All countries" option');
  ck(/Germany \(904\)/.test(opts), 't20: option text lacks name+count: ' + opts);
  ck(/value="DEU"/.test(opts), 't20: option value is not the ISO code');

  // choosing one must send ?country= and reset to page 0
  CALLS.length = 0;
  byId['country'].value = 'DEU';
  await byId['country'].onchange.call(byId['country']);
  await flush(); await flush();
  const req = CALLS.map(c => c.url).filter(u => /\/api\/review\?/.test(u)).pop();
  ck(/country=DEU/.test(req || ''), 't20: filter not sent, url=' + req);
  ck(/page=0/.test(req || ''), 't20: filter did not reset to page 1: ' + req);

  // an unchanged option set must NOT be rewritten -- doing so on every page
  // turn drops an open dropdown mid-click
  const before = byId['country'].innerHTML;
  byId['country'].innerHTML = before + '<!--sentinel-->';
  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 3), [],
                                      {countries: LIST, country: 'DEU'});
  await API.load(); await flush();
  ck(/sentinel/.test(byId['country'].innerHTML),
     't20: options rebuilt although the list was identical');

  // a payload with no countries key must not blow up or wipe the control
  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 3), []);
  await API.load(); await flush();
  ck(byId['country'].innerHTML.length > 0, 't20: control emptied when the '
     + 'server omitted countries');
}

// ── 21. a filtered count must not read as a global one ──────────────────
// 'left to review' is scoped to the country filter while 'flagged' and 'kept'
// stay all-time. Side by side with no marker, 198 next to 1,166 reads as
// "198 left in total".
async function t21() {
  const LIST = [{iso:'DEU',name:'Germany',n:904},{iso:'JPN',name:'Japan',n:838}];
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
             {countries: LIST, country: '', total_unflagged: 2100,
              flagged_total: 1166}),
           '/api/review/seen': () => ({ ok: true, seen_total: 1 }) };
  await API.load(); await flush();
  ck(byId['leftlab'].textContent === 'left to review',
     't21: unfiltered label changed: ' + byId['leftlab'].textContent);

  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 3), [],
        {countries: LIST, country: 'DEU', total_unflagged: 198,
         flagged_total: 1166});
  await API.load(); await flush();
  ck(byId['left'].textContent === '198', 't21: left=' + byId['left'].textContent);
  ck(/Germany/.test(byId['leftlab'].textContent),
     't21: filtered count not scoped to the country, label=' +
     byId['leftlab'].textContent);
  ck(byId['done'].textContent === '1,166',
     't21: global flagged count changed under a filter: ' + byId['done'].textContent);

  // clearing the filter restores the global wording
  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 3), [],
        {countries: LIST, country: '', total_unflagged: 2100});
  await API.load(); await flush();
  ck(byId['leftlab'].textContent === 'left to review',
     't21: label stuck on a country after clearing: ' + byId['leftlab'].textContent);
}

// ── 22. an option's count must equal what selecting it returns ──────────
// The first cut tallied the dropdown from the country INDEX, which spans the
// rolling pool plus both flag ledgers, while the queue excludes everything
// judged/kept/collapsed. Measured on the live server: 60 of 60 options were
// dead, promising 4,090 crops that did not exist.
async function t22() {
  const LIST = [{iso:'BRA',name:'Brazil',n:1073},{iso:'JPN',name:'Japan',n:312}];
  RESP = { '/api/review': (url) => {
             const iso = /country=([A-Z]*)/.exec(url);
             const sel = iso && iso[1];
             const hit = LIST.filter(c => c.iso === sel)[0];
             return payload(CROPS.normal.slice(0, 3), [],
               {countries: LIST, country: sel || '',
                total_unflagged: hit ? hit.n : 1385});
           },
           '/api/review/seen': () => ({ ok: true, seen_total: 1 }) };
  await API.load(); await flush();
  // pick each option and check the queue size matches what it advertised
  for (const c of LIST) {
    byId['country'].value = c.iso;
    await byId['country'].onchange.call(byId['country']);
    await flush(); await flush();
    const shown = byId['left'].textContent.replace(/,/g, '');
    ck(shown === String(c.n),
       't22: ' + c.iso + ' advertised ' + c.n + ' but the queue shows ' +
       byId['left'].textContent);
  }
  // and an option that would return nothing must not be offered at all
  RESP['/api/review'] = () => payload([], [], {countries: [], country: ''});
  await API.load(); await flush();
  ck(!/value="[A-Z]{3}"/.test(byId['country'].innerHTML),
     't22: a country was still offered with an empty queue');
}

// ── 23. a search that cannot work has to say so ─────────────────────────
// The vectors belong to crop FILES and the pool rotates hourly, so coverage
// decays to nothing whenever the embedder is stopped. Measured on the live
// box: 4,513 vectors, 3,010 crops in the pool, zero in both -- and the page
// reported the search as working while the queue did not move, which reads
// as the model returning nonsense. Every state that fails to reorder the
// queue must put a sentence on screen -- and none of them may point at the
// guesser controls, which no longer exist on this page: the empty states
// name the tool (triage_crops.py) instead.
async function t23() {
  const FIND = {find: 'a cat', find_terms: ['a cat'], find_cover: [0, 3010]};
  for (const [state, want] of [['cold', /embedded/],
                               // mismatch now clears itself: the words are
                               // re-encoded under whichever model embedded the
                               // crops, so the message must promise that and
                               // not send the reader off to run a tool
                               ['mismatch', /re-encoding the search words/],
                               ['learning', /moment/], ['unknown', /encoded/],
                               ['failed', /crop_search\.log/],
                               ['novectors', /triage_crops\.py/]]) {
    RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
               Object.assign({}, FIND, {find_state: state})) };
    await API.load(); await flush();
    ck(!byId['findmsg'].hidden,
       't23: ' + state + ' said nothing on screen');
    ck(want.test(byId['findmsg'].textContent),
       't23: ' + state + ' message unhelpful: ' + byId['findmsg'].textContent);
    ck(/\bwarn\b/.test(byId['find'].className || ''),
       't23: ' + state + ' left the box looking healthy');
    ck(!/guesser above|start the guesser/.test(byId['findmsg'].textContent),
       't23: ' + state + ' points at a control that no longer exists: ' +
       byId['findmsg'].textContent);
  }
  // and a search that DID order the queue must not nag
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
             Object.assign({}, FIND, {find_state: 'on', find_hits: 2663,
                                      find_cover: [2663, 3010]})) };
  await API.load(); await flush();
  ck(byId['findmsg'].hidden, 't23: a working search still warned');
  ck(!/\bwarn\b/.test(byId['find'].className || ''),
     't23: a working search kept the warning border');

  // 'cold' with most of the pool embedded is a FILTER problem, not a stopped
  // embedder -- naming only the pool size sends the reviewer after the wrong
  // thing when 4,014 of 5,018 crops already carry vectors.
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
             Object.assign({}, FIND, {find_state: 'cold',
                                      find_cover: [4014, 5018]})) };
  await API.load(); await flush();
  ck(/4,014/.test(byId['findmsg'].textContent),
     't23: cold ignored how much IS embedded: ' + byId['findmsg'].textContent);
  ck(!/triage_crops\.py/.test(byId['findmsg'].textContent),
     't23: cold blamed the embedder with the pool mostly embedded');

  // the term is written with textContent, so it must not arrive pre-escaped
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
             {find: 'cats & dogs', find_terms: ['a cat'],
              find_state: 'learning', find_cover: [0, 10]}) };
  await API.load(); await flush();
  ck(/cats & dogs/.test(byId['findmsg'].textContent),
     't23: term double-escaped in the message: ' + byId['findmsg'].textContent);
  ck(!/&amp;/.test(byId['findmsg'].textContent),
     't23: entities shown literally: ' + byId['findmsg'].textContent);
}

// ── 24. the guesses feature is gone, and stays gone ─────────────────────
// "remove the guesses feature from review" -- the suggest filter, the gate
// filter, the backend toggle and the run strip. Gone means gone from the
// wire too: the queue request names none of them, nothing on this page
// talks to /api/triage, and a payload still carrying the old keys (the
// server keeps serving other callers) must not conjure a control, a chip,
// or a tile badge out of them.
async function t24() {
  const before = CALLS.length;
  RESP = { '/api/review': () => payload(
             CROPS.normal.slice(0, 3).map(
               c => Object.assign({}, c, {sg: 'dog', sgl: 'terrier',
                                          sgp: 0.9})), [],
             {suggest: 'dog', suggest_ready: true,
              suggest_counts: {dog: 5, none: 2}, backend: 'siglip',
              buckets: [{key: 'dog', label: 'Looks like a dog'}],
              gate: 'dog', gate_ready: true, gate_label: 'Dog-bin gate',
              gate_counts: {all: 9, dog: 5, not_dog: 3, none: 1},
              pool_unfiltered: 300}) };
  await API.load(); await flush(); await flush();
  const asked = CALLS.filter(c => /\/api\/review\?/.test(c.url)).pop();
  for (const tok of ['suggest=', 'backend=', 'gate='])
    ck(asked.url.indexOf(tok) < 0,
       't24: the queue request still carries the removed filter ' + tok +
       ': ' + asked.url);
  ck(!CALLS.slice(before).some(c => /\/api\/triage/.test(c.url)),
     't24: something on this page still talks to /api/triage');
  ck(byId['chips'].hidden ||
     (!/Looks like a dog/.test(byId['chips'].innerHTML) &&
      !/Gate says/.test(byId['chips'].innerHTML)),
     't24: a chip appeared for a filter the page no longer offers: ' +
     byId['chips'].innerHTML);
  ck(!document.querySelector('.sg'),
     't24: a tile still wears a guess badge');
  ck(byId['cap'].textContent.indexOf('narrowed') < 0,
     't24: the caption claims a narrowing no control applied: ' +
     byId['cap'].textContent);
}

// ── 25. "Check my annotations" can be narrowed to when I judged ─────────
// The filter is server-side -- the ledger rows carry flagged_at, and a hide
// on the client would leave the counts describing rows nobody can see -- so
// the page's whole job is to send it, show it as a chip, and not pretend an
// empty week is an empty ledger. The words are the audit pages' own.
async function t25() {
  const fire = (id, v) => { const e = byId[id]; e.value = v;
    if (e.onchange) e.onchange.call(e);
    (e._listeners.change || []).forEach(f => f.call(e)); };
  RESP = {'/api/review': () => payload(CROPS.normal.slice(0, 2), []),
          '/api/review/annotated': () => ({items: [], page: 0, pages: 1,
              total: 0, pool_unfiltered: 7,
              n_false_positive: 0, n_true_positive: 0})};
  fire('mode', 'audit'); await flush(); await flush();
  // the queue request never carried it; the audit request always does
  const q = CALLS.filter(c => /\/api\/review\?/.test(c.url)).pop();
  ck(q.url.indexOf('period=') < 0,
     't25: the queue request carries a filter that means nothing there: ' +
     q.url);
  // the two calendars spell the value; the hidden input IS the filter
  byId['pfrom'].value = '2026-08-12';
  byId['pto'].value = '2026-08-19';
  (byId['pfrom']._listeners.change || []).forEach(f =>
    f.call(byId['pfrom']));
  await flush(); await flush();
  let sent = CALLS.filter(c => /annotated/.test(c.url)).pop().url;
  ck(/period=2026-08-12\.\.2026-08-19/.test(sent),
     't25: the dates never reached the request: ' + sent);
  ck(byId['period'].value === '2026-08-12..2026-08-19',
     't25: the calendars did not spell the filter: ' + byId['period'].value);
  ck(/judged 2026-08-12/.test(byId['chips'].textContent),
     't25: no chip for the window narrowing the list: ' +
     byId['chips'].textContent);
  ck(byId['pclr'].hidden === false,
     't25: nothing offers to clear a window that is set');
  // an empty week must not read as an empty ledger
  ck(!/nothing annotated yet/.test(byId['pg'].textContent),
     't25: an empty period claims the ledger is empty: ' +
     byId['pg'].textContent);
  ck(/period/.test(byId['pg'].textContent),
     't25: the empty state does not name the period: ' +
     byId['pg'].textContent);
  // clearing the chip clears the filter on the wire
  const x = byId['chips'].querySelector('.chipx');
  ck(!!x, 't25: the period chip cannot be cleared where it is read');
  (byId['chips']._listeners.click || []).forEach(f =>
    f.call(byId['chips'], {target: x}));
  await flush(); await flush();
  sent = CALLS.filter(c => /annotated/.test(c.url)).pop().url;
  ck(/period=(&|$)/.test(sent),
     't25: clearing the chip did not clear the period: ' + sent);
  // ...and it clears the CALENDARS, not just the value behind them: a chip
  // dismissed while the fields still read 12 Aug is a control disagreeing
  // with the list it is over.
  ck(byId['pfrom'].value === '' && byId['pto'].value === '',
     't25: the chip cleared the filter and left the dates on screen: ' +
     byId['pfrom'].value + '/' + byId['pto'].value);
  ck(byId['pclr'].hidden === true,
     't25: the clear button stands over two empty fields');
  // one open end is a window too, and the x is the way back from it
  byId['pfrom'].value = '2026-08-12';
  (byId['pfrom']._listeners.change || []).forEach(f =>
    f.call(byId['pfrom']));
  await flush(); await flush();
  ck(/period=2026-08-12\.\.(&|$)/.test(
       CALLS.filter(c => /annotated/.test(c.url)).pop().url),
     't25: an open far end is not sent as one');
  ck(/judged since 2026-08-12/.test(byId['chips'].textContent),
     't25: an open window is not named as one: ' +
     byId['chips'].textContent);
  (byId['pclr']._listeners.click || []).forEach(f => f.call(byId['pclr']));
  await flush(); await flush();
  ck(/period=(&|$)/.test(
       CALLS.filter(c => /annotated/.test(c.url)).pop().url) &&
     byId['period'].value === '',
     't25: the clear button did not clear the window');
  // A value that is not a window -- a preference saved when this control
  // offered "last 7 days" -- names nothing. The server reads it the same way
  // and narrows nothing, so a chip announcing it would be the page claiming
  // a narrowing the list does not have.
  fire('period', '7d'); await flush(); await flush();
  ck(!/7d/.test(byId['chips'].textContent),
     't25: a value that is not a window still claims a chip: ' +
     byId['chips'].textContent);
  fire('period', ''); await flush(); await flush();
  fire('mode', 'queue'); await flush(); await flush();
}

// ── 26. the audit view's two filters are two chips, not one ─────────────
// Verdict and period narrow the same list on different axes, so each gets
// its own chip and clearing one must leave the other on the wire -- the
// same rule every queue filter already follows.
async function t26() {
  const fire = (id, v) => { const e = byId[id]; e.value = v;
    if (e.onchange) e.onchange.call(e);
    (e._listeners.change || []).forEach(f => f.call(e)); };
  RESP = {'/api/review': () => payload(CROPS.normal.slice(0, 2), []),
          '/api/review/annotated': () => ({items: [], page: 0, pages: 1,
              total: 1, pool_unfiltered: 7,
              n_false_positive: 1, n_true_positive: 0})};
  fire('mode', 'audit'); await flush(); await flush();
  fire('verdict', 'false_positive'); await flush(); await flush();
  fire('period', '2026-07-01..'); await flush(); await flush();
  ck(/not a dog/.test(byId['chips'].textContent) &&
     /judged since 2026-07-01/.test(byId['chips'].textContent),
     't26: the two audit filters do not both show as chips: ' +
     byId['chips'].textContent);
  const xs = byId['chips'].querySelectorAll('.chipx')
    .filter(b => (b.dataset || {}).f === 'period');
  ck(xs.length === 1, 't26: no chip cross belongs to the period');
  (byId['chips']._listeners.click || []).forEach(f =>
    f.call(byId['chips'], {target: xs[0]}));
  await flush(); await flush();
  const sent = CALLS.filter(c => /annotated/.test(c.url)).pop().url;
  ck(/label=false_positive/.test(sent) && /period=(&|$)/.test(sent),
     't26: clearing the period chip did not leave the verdict alone: ' +
     sent);
  fire('verdict', 'all'); await flush(); await flush();
  fire('mode', 'queue'); await flush(); await flush();
}

// ── 27. the caption, the chips, and the one disclosure ──────────────────
// Nine controls sat in one row holding four different kinds of thing. The
// block is now a caption over a fold: it says what the queue is, shows only
// the filters actually applied, and keeps the rest behind one button. Each of
// those three claims is checked, because each replaced something visible.
async function t27() {
  const fire = (id, v) => { const e = byId[id]; e.value = v;
    if (e.onchange) e.onchange.call(e);
    (e._listeners.change || []).forEach(f => f.call(e)); };
  const FULL = {total_unflagged: 2157, pool_unfiltered: 2157,
                countries: [{iso: 'JPN', name: 'Japan', n: 838}], country: ''};
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [], FULL) };
  await API.load(); await flush();
  ck(/2,157/.test(byId['cap'].textContent),
     't27: the caption does not say what the queue holds: ' +
     byId['cap'].textContent);
  ck(!/narrowed from/.test(byId['cap'].textContent),
     't27: claimed a narrowing with no filter applied: ' +
     byId['cap'].textContent);
  ck(byId['chips'].hidden, 't27: an empty chip row still took a line');
  ck(byId['npanel'].hidden, 't27: the panel is open before it is asked for');

  // apply one: the chip appears, and the caption says what it narrowed from
  RESP['/api/review'] = () => Object.assign({}, FULL,
        {total_unflagged: 838, country: 'JPN', pool_unfiltered: 2157});
  fire('country', 'JPN');
  await flush(); await flush();
  ck(!byId['chips'].hidden && /Japan/.test(byId['chips'].innerHTML),
     't27: the applied filter is not shown as a chip: ' +
     byId['chips'].innerHTML);
  ck(/narrowed from/.test(byId['cap'].textContent) &&
     /2,157/.test(byId['cap'].textContent) &&
     /838/.test(byId['cap'].textContent),
     't27: the caption does not report the narrowing: ' +
     byId['cap'].textContent);
  ck(/1/.test(byId['narrow'].textContent),
     't27: the shut panel does not say how many filters are on: ' +
     byId['narrow'].textContent);

  // clearing from the chip resets the control it came from
  RESP['/api/review'] = () => Object.assign({}, FULL, {country: ''});
  const x = byId['chips'].querySelector('.chipx');
  ck(!!x, 't27: the chip cannot be cleared where it is read');
  (byId['chips']._listeners.click || []).forEach(f => f.call(byId['chips'],
      {target: x}));
  await flush(); await flush();
  // Asserted on the REQUEST, not on the control's value afterwards: the next
  // payload echoes `country` back and paintCountries writes it into the
  // select, so a chip that cleared nothing would still look cleared a moment
  // later. The URL is the only observable the echo cannot fake.
  const after = CALLS.filter(c => /\/api\/review\?/.test(c.url)).pop();
  ck(/country=(&|$)/.test(after.url),
     't27: clearing the chip did not clear the filter it names: ' + after.url);
  ck(byId['chips'].hidden, 't27: the chip outlived the filter');

  // the disclosure holds the rest, and says so
  const nb = byId['narrow'];
  (nb._listeners.click || []).forEach(f => f.call(nb));
  ck(!byId['npanel'].hidden, 't27: Filter does not open the panel');
  ck(nb.getAttribute && nb.getAttribute('aria-expanded') === 'true',
     't27: the disclosure does not report its state to a screen reader');
  (nb._listeners.click || []).forEach(f => f.call(nb));
  ck(byId['npanel'].hidden, 't27: Filter does not shut the panel again');
}

// ── 28. the training-set balance row stays gone ─────────────────────────
// It used to paint "N crops left to judge" over a track, and 28 drove the
// three exits of its painter. The row answered a question about the DATASET
// on a page about the crop in front of you, and took a line of the viewport
// on every visit to say something that only moves at a rebuild. Removed --
// so the check is that it does not come back, ids and all, because the
// element and its painter were wired to each other in four places.
async function t28() {
  for (const id of ['bal', 'balNum', 'balNumU', 'balFill', 'balLeft',
                    'balMain']) {
    ck(!byId[id], 't28: #' + id + ' is back on the review page');
  }
  ck(typeof API.paintBal === 'undefined',
     't28: the balance painter is back');
}

// ── 29. the chips describe the request that was actually sent ───────────
// The two views fetch different things. The audit list is fetched with
// label=, leash= and period= and nothing else, so a country left set from
// the queue narrows nothing there. The chip row advertised it anyway and hid
// the verdict filter -- the one that does apply -- so it explained an empty
// list with a cause that was not the cause, and offered no way to undo the
// real one.
async function t29() {
  const Q = () => ({items: [], reserve: [], page: 0, size: 50, pages: 1,
      total_unflagged: 100, pool_unfiltered: 100,
      countries: [{iso: 'JPN', name: 'Japan', n: 9}], country: 'JPN'});
  const A = () => ({items: [], page: 0, pages: 1, total: 5,
      pool_unfiltered: 7, n_false_positive: 5, n_true_positive: 2});
  RESP = {'/api/review': Q, '/api/review/annotated': A};
  const fire = (id, v) => { const e = byId[id]; e.value = v;
    if (e.onchange) e.onchange.call(e);
    (e._listeners.change || []).forEach(f => f.call(e)); };

  fire('country', 'JPN'); await flush(); await flush();
  ck(/Japan/.test(byId['chips'].innerHTML),
     't29: a queue filter produced no chip: ' + byId['chips'].innerHTML);

  fire('mode', 'audit'); await flush(); await flush();
  const sent = (CALLS.filter(c => /annotated/.test(c.url)).pop() || {}).url;
  ck(!/Japan/.test(byId['chips'].innerHTML),
     't29: the audit view kept a chip for a filter its request does not ' +
     'carry (' + sent + '): ' + byId['chips'].innerHTML);

  fire('verdict', 'false_positive'); await flush(); await flush();
  ck(/not a dog/.test(byId['chips'].textContent),
     't29: the audit view shows no chip for the one filter it applies: ' +
     byId['chips'].textContent);
  ck(!byId['npanel'].hidden || !byId['verdict'].hidden,
     't29: the audit view\'s only filter is behind a fold');

  // and clearing it has to reach the audit request, not the queue's
  const x = byId['chips'].querySelector('.chipx');
  (byId['chips']._listeners.click || []).forEach(f =>
    f.call(byId['chips'], {target: x}));
  await flush(); await flush();
  const after = (CALLS.filter(c => /annotated/.test(c.url)).pop() || {}).url;
  ck(/label=all/.test(after),
     't29: clearing the verdict chip did not clear the verdict: ' + after);

  fire('mode', 'queue'); await flush(); await flush();
  fire('country', ''); await flush(); await flush();
}

// ── 30. a preference outliving its control must not narrow anything ─────
// The guess feature's keys -- suggest, backend, gatef -- still sit in
// localStorage on every box that used the old build (the harness seeds
// them), and the page's very first request is the one restorePrefs feeds.
// If any of them reaches the wire, a filter with no control, no chip and no
// cross is narrowing the queue again -- the exact silence the chip row was
// built to end.
async function t30() {
  const first = CALLS.find(c => /\/api\/review\?/.test(c.url));
  ck(!!first, 't30: no initial queue request was recorded');
  for (const tok of ['suggest=', 'backend=', 'gate='])
    ck(!first || first.url.indexOf(tok) < 0,
       't30: the stale ' + tok.slice(0, -1) +
       ' preference reached the wire: ' + (first || {}).url);
}

// ── 31. the panel offers no control that does nothing ───────────────────
// A group whose every control is hidden rendered as an uppercase heading over
// an empty row. The leash filter -- the group that used to empty out this way,
// and the reason this case exists -- has been removed from the page entirely,
// so this now asserts BOTH halves: that it is gone, and that no surviving
// group can render a heading over nothing. Written against whatever groups the
// page ships rather than a named one, so the next group added is covered
// without anybody remembering to come back here.
async function t31() {
  ck(!byId['leashf'] && !byId['ngrpLeash'],
     't31: the leash filter is back on the review page');
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 2), []) };
  await API.load(); await flush();
  Object.keys(byId).forEach(function (id) {
    if (id.indexOf('ngrp') !== 0) return;
    const g = byId[id];
    if (!g || g.hidden) return;
    const live = (g.querySelectorAll ? [].slice.call(
      g.querySelectorAll('select,input,button')) : []).filter(function (e) {
        return !e.hidden;
      });
    ck(live.length > 0,
       't31: ' + id + ' shows a heading with no control under it');
  });
}

// ── 32. the header stops taking a third of the screen once you scroll ───
// Everything up there is setup, and setup is read once. Working through crops
// with a running tally, two progress rows and an open panel pinned above them
// is the complaint that started this.
async function t32() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), []) };
  await API.load(); await flush();
  ck(!document.body.classList.contains('compact'),
     't32: the header starts compact, before anything has scrolled');
  scrolled(true);
  ck(document.body.classList.contains('compact'),
     't32: scrolling past the top did not compact the header');
  scrolled(false);
  ck(!document.body.classList.contains('compact'),
     't32: scrolling back to the top left the header compact');

  // A sticky header that sheds height moves everything under it, and that
  // settling can cross the sentinel again and ask for the opposite. Two
  // crossings inside the hold window are one change, not a flutter.
  scrolled(true);
  ck(document.body.classList.contains('compact'), 't32: did not compact');
  scrolled(false, 40);
  ck(document.body.classList.contains('compact'),
     't32: a reversal 40ms later undid it — that is the flutter');
  scrolled(false, 900);
  ck(!document.body.classList.contains('compact'),
     't32: a real scroll back to the top was refused too');
}

// ── 33. the id can be copied, but only from the enlarged view ───────────
// The id is wanted while looking hard at ONE crop. A copy button on every
// tile in a 50-crop grid would be fifty controls nobody asked for, so it
// lives in the lightbox — and it has to copy the crop currently shown, not
// the one that was open before you stepped.
async function t33() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), []),
           '/api/review/box': () => ({ok: false, pending: true}) };
  await API.load(); await flush();
  API.closeLb();                 // a previous case may have left one open
  ck(!document.getElementById('lbcopy'),
     't33: a copy button with nothing enlarged');
  ck(!byId['grid'].querySelector('.lbcopy'),
     't33: a copy button on every tile in the grid');

  API.openLb(0); await flush();
  const btn = byId['lbcopy'];
  ck(!!btn, 't33: no copy button in the enlarged view');
  ck(btn.dataset.id === API.st().items[0].image_id,
     't33: the button is armed with the wrong id: ' + btn.dataset.id);

  COPIED = null; EXEC_OK = true;
  btn.onclick.call(btn, {stopPropagation() {}});
  await flush(); await flush();
  ck(COPIED === API.st().items[0].image_id,
     't33: the id never reached the clipboard: ' + COPIED);
  ck(/Copied/.test(btn.textContent),
     't33: the button did not say it worked: ' + btn.textContent);

  // stepping must re-arm it, or it copies the crop you just left
  API.stepLb(1); await flush();
  ck(btn.dataset.id === API.st().items[1].image_id,
     't33: stepping left the button on the previous crop: ' + btn.dataset.id);
  ck(btn.textContent === 'Copy ID',
     't33: the button kept its "Copied" state across crops: ' + btn.textContent);

  // and a refusal has to say so rather than claim success
  COPIED = null; EXEC_OK = false;
  btn.onclick.call(btn, {stopPropagation() {}});
  await flush(); await flush();
  ck(!/Copied/.test(btn.textContent),
     't33: claimed a copy that the browser refused: ' + btn.textContent);
  API.closeLb();
}

// ── 34. auditing a verdict: changing it, taking it back, being told ─────
// flag() has a second branch for "Check my annotations" and nothing drove it.
// It is not a variation on the queue path: the crop keeps its place instead of
// being consumed, re-clicking the verdict a crop already has REMOVES the
// annotation, and the only thing that says any of that happened is a notice
// the branch puts on screen. An undefined identifier there throws into the
// chain's own .catch, so the ledger is rewritten and the reviewer is told
// nothing -- with every other case still green.
async function t34() {
  const posts = [];
  const ann = () => ({ items: JSON.parse(JSON.stringify(CROPS.annotated)),
      page: 0, pages: 1, total: 4, pool_unfiltered: 4,
      n_false_positive: 2, n_true_positive: 2 });
  RESP = { '/api/review/annotated': ann,
           '/api/review': () => payload(CROPS.normal.slice(0, 3), []),
           '/api/detect/flag': (u, o) => { posts.push(JSON.parse(o.body));
             return { ok: true, flagged_total: 41, positive_total: 12 } } };
  byId['mode'].value = 'audit';
  byId['mode'].onchange.call(byId['mode']);
  await flush(); await flush();
  ck(document.querySelectorAll('.card').length === 4,
     't34: the audit view rendered ' + document.querySelectorAll('.card').length +
     ' tiles, want 4');
  const name = API.st().items[0].name;
  const card = document.querySelector('.card[data-name="' + name + '"]');
  ck(card && card.querySelector('.fbtn.yes').classList.contains('on'),
     't34: the tile does not restate the verdict already on record');
  // An undo staged by an earlier case would post under this case's stub and
  // read as an audit action. Let the 5 s window expire the way it does on the
  // page rather than reaching in, so this starts from a state the page can be
  // in on its own.
  runTimers();
  ck(API.st().lastUndo === null,
     't34: an earlier case left an undo staged; this case cannot tell them apart');

  // change it
  API.hideToast();
  let threw = await watchThrows(async () => {
    await API.flag(0, false, 'false_positive'); await flush();
  });
  ck(!threw.length, 't34: changing a verdict in audit mode threw ' + why(threw[0]));
  ck(posts.length === 1 && posts[0].name === name &&
     posts[0].label === 'false_positive' && posts[0].undo === false,
     't34: wrong POST for a verdict change: ' + JSON.stringify(posts));
  ck(API.st().items.length === 4,
     't34: auditing consumed the crop -- the list must not resequence under the reader');
  ck(API.st().items[0].label === 'false_positive',
     't34: the crop still carries the old verdict: ' + API.st().items[0].label);
  ck(card.querySelector('.fbtn.no').classList.contains('on') &&
     !card.querySelector('.fbtn.yes').classList.contains('on'),
     't34: the buttons do not restate the new verdict');
  ck(card.classList.contains('changed'),
     't34: nothing marks the tile as differing from what the ledger held');
  ck(noticed().length > 0, 't34: changing a verdict said nothing on screen');

  // take it back: clicking the verdict a crop already has removes it
  API.hideToast();
  posts.length = 0;
  threw = await watchThrows(async () => {
    await API.flag(0, false, 'false_positive'); await flush();
  });
  ck(!threw.length, 't34: removing an annotation threw ' + why(threw[0]));
  ck(posts.length === 1 && posts[0].undo === true,
     't34: re-clicking the verdict on record did not take it back: ' +
     JSON.stringify(posts));
  ck(API.st().items[0].label === null,
     't34: the crop still carries the annotation that was removed: ' +
     API.st().items[0].label);
  ck(!card.querySelector('.fbtn.no').classList.contains('on') &&
     !card.querySelector('.fbtn.yes').classList.contains('on'),
     't34: a removed annotation is still lit on the tile');
  ck(card.classList.contains('unjudged'),
     't34: nothing marks the crop as carrying no verdict any more');
  ck(noticed().length > 0,
     't34: removing an annotation said nothing on screen -- the sentence that ' +
     'promises it is back in the queue is the only thing that says so');

  // Auditing stages no undo toast: the crop never left, and re-clicking is how
  // it is taken back. So U must do nothing -- a staged undo here would post a
  // verdict against a crop that no longer has one.
  ck(API.st().lastUndo === null, 't34: an audit verdict staged an undo toast');
  posts.length = 0;
  await API.undo(); await flush();
  key('u'); await flush();
  ck(posts.length === 0,
     't34: U posted a verdict in audit mode: ' + JSON.stringify(posts));
}

// ── 35. auditing banks nothing as "reviewed and kept" ───────────────────
// markSeen() records "I looked at these and kept them", and the server retires
// those image_ids from the queue for good. An audit screen is not a review: the
// reviewer may have just REMOVED an annotation, which the button promises puts
// the crop back in the queue, and banking it retires the very crop that was
// handed back. Reload and the pagehide beacon are the two callers with no view
// of their own, which is why they were the two that were missed.
async function t35() {
  const banked = [];
  const ann = () => ({ items: JSON.parse(JSON.stringify(CROPS.annotated)),
      page: 0, pages: 1, total: 4, pool_unfiltered: 4,
      n_false_positive: 2, n_true_positive: 2 });
  RESP = { '/api/review/annotated': ann,
           '/api/review': () => payload(CROPS.normal.slice(0, 4), []),
           '/api/review/seen': (u, o) => { banked.push(JSON.parse(o.body));
             return { ok: true, seen_total: 77 } },
           '/api/detect/flag': () => ({ ok: true }) };
  byId['mode'].value = 'audit';
  byId['mode'].onchange.call(byId['mode']);
  await flush(); await flush();
  const name = API.st().items[0].name;
  await API.flag(0, false, API.st().items[0].label); await flush();
  ck(API.st().items[0].label === null,
     't35: the annotation was not removed, so nothing here is at risk');

  banked.length = 0; CALLS.length = 0;
  byId['reload'].onclick(); await flush(); await flush();
  ck(banked.length === 0,
     't35: Reload banked the audit screen as reviewed and kept, including ' +
     name + ': ' + JSON.stringify(banked));
  ck(CALLS.some(c => /annotated/.test(String(c.url))),
     't35: Reload did not reload the annotations either');
  await API.markSeen(); await flush();
  ck(banked.length === 0,
     't35: markSeen banked an audit screen when called directly: ' +
     JSON.stringify(banked));
  beacons.length = 0;
  for (const f of (winL['pagehide'] || [])) f({});
  ck(!beacons.some(u => /seen/.test(String(u))),
     't35: closing the tab banked the audit screen: ' + beacons.join(','));

  // ...and all three still bank in the queue, where paging away IS the
  // decision. Without this half the case passes just as well against a
  // markSeen that banks nothing anywhere.
  byId['mode'].value = 'queue';
  byId['mode'].onchange.call(byId['mode']);
  await flush(); await flush();
  banked.length = 0;
  byId['reload'].onclick(); await flush(); await flush();
  ck(banked.length === 1 && banked[0].names && banked[0].names.length === 4,
     't35: Reload no longer banks the screen in the queue: ' +
     JSON.stringify(banked));
  beacons.length = 0;
  for (const f of (winL['pagehide'] || [])) f({});
  ck(beacons.some(u => /seen/.test(String(u))),
     't35: closing the tab no longer banks the queue screen');
}

// ── 36. undo after a leash-held "Is a dog" ──────────────────────────────
// With the leash store on, "Is a dog" does NOT consume the crop: the tile is
// held on screen because a leash call has just become askable on it. undo()
// was written for the consuming path and re-inserts unconditionally, so taking
// that verdict back put the crop on the grid twice, both tiles lit "Is a dog",
// for a kept record that had just been deleted -- a tile asserting an
// annotation that is not on disk, over a crop that is in fact unjudged.
async function t36() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6),
             CROPS.normal.slice(6, 9),
             { leash_totals: { leashed: 8, unleashed: 5 }, leash: {} }),
           '/api/detect/flag': () => ({ ok: true }),
           '/api/review/seen': () => ({ ok: true, seen_total: 1 }) };
  await API.load(); await flush();
  const name = API.st().items[2].name;
  const n0 = API.st().items.length, r0 = API.st().reserve.length;
  await API.flag(2, false, 'true_positive'); await flush();
  const held = document.querySelector('.card[data-name="' + name + '"]');
  ck(!!held && held.classList.contains('awaitleash'),
     't36: the crop was not held for a leash call, so this case tests nothing');
  ck(API.st().items.length === n0 && API.st().reserve.length === r0,
     't36: a held flag consumed the crop after all');

  await API.undo(); await flush();
  const names = API.st().items.map(c => c.name);
  const dom = document.querySelectorAll('.card').map(e => e.dataset.name);
  ck(names.filter(x => x === name).length === 1,
     't36: undo left ' + names.filter(x => x === name).length +
     ' copies of the crop in items');
  ck(dom.filter(x => x === name).length === 1,
     't36: undo left ' + dom.filter(x => x === name).length +
     ' tiles on the grid for one crop');
  ck(names.length === n0,
     't36: the page grew ' + n0 + ' -> ' + names.length + ' over one flag/undo');
  ck(JSON.stringify(dom) === JSON.stringify(names),
     't36: DOM order diverged from items order');
  const it = API.st().items.filter(c => c.name === name)[0];
  ck(it && !it.label,
     't36: the crop still carries the verdict the undo deleted: ' +
     (it && it.label));
  const lit = document.querySelectorAll('.card[data-name="' + name + '"]')
    .filter(e => { const y = e.querySelector('.fbtn.yes');
                   return y && y.classList.contains('on') });
  ck(lit.length === 0, 't36: ' + lit.length + ' tile(s) still lit "Is a dog" ' +
     'for a record the undo deleted');
  const asking = document.querySelectorAll('.card[data-name="' + name + '"]')
    .filter(e => e.classList.contains('awaitleash'));
  ck(asking.length === 0,
     't36: the tile still asks for a leash call on a verdict that was undone');
  ck(API.st().reserve.length === r0,
     't36: reserve drifted to ' + API.st().reserve.length + ', want ' + r0);
}

// ── 37. a zero-match FILTER does not claim the pool is judged ───────────
// With a narrowing filter active the grid can empty while thousands of crops
// remain unjudged, and the empty state read "Queue is clear -- Every
// detection in the pool has been judged" directly under a header saying
// "narrowed from 2,716". Two facts share that empty grid; the page has to
// say the one that is true.
async function t37() {
  const fire = (id, v) => { const e = byId[id]; e.value = v;
    if (e.onchange) e.onchange.call(e);
    (e._listeners.change || []).forEach(f => f.call(e)); };
  RESP = { '/api/review': () => payload([], [], { total_unflagged: 2716,
             pool_unfiltered: 2716, pages: 1, country: 'JPN',
             countries: [{iso: 'JPN', name: 'Japan', n: 9}] }) };
  fire('country', 'JPN'); await flush(); await flush();
  ck(/Nothing matches these filters/.test(byId['state'].innerHTML),
     't37: the filtered empty state says: ' + byId['state'].innerHTML);
  ck(!/Every detection in the pool has been judged/
       .test(byId['state'].innerHTML),
     't37: an empty SLICE still claims the whole pool is judged');
  // and with every filter cleared, the honest clear-queue sentence returns
  RESP = { '/api/review': () => payload([], [], { total_unflagged: 0,
             pages: 1 }) };
  fire('country', ''); await flush(); await flush();
  ck(/Queue is clear/.test(byId['state'].innerHTML),
     't37: the unfiltered empty queue lost its own sentence: ' +
     byId['state'].innerHTML);
}

// ── 38. following a link off this page is not "I kept these" ────────────
// The pagehide beacon banks every crop on screen as reviewed-and-kept, and
// there is no per-crop undo: the only recovery restores the whole ledger.
// cf1523c9 put a three-tab strip at the top of this page -- one of whose tabs
// points at THIS page -- so a click that changes nothing at all retired fifty
// unjudged crops, and a reader tapping over to an audit and back retired
// fifty more. An explicit page turn still banks; so does a real departure.
async function t38() {
  const banked = [];
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 4), []),
           '/api/review/seen': (u, o) => { banked.push(JSON.parse(o.body));
             return { ok: true, seen_total: 7 } } };
  byId['mode'].value = 'queue';
  byId['mode'].onchange.call(byId['mode']);
  await flush(); await flush();
  ck(API.st().items.length === 4, 't38: no queue on screen to bank');

  // first, the departure that IS a decision: the tab closed, the address bar
  beacons.length = 0;
  for (const f of (winL['pagehide'] || [])) f({});
  ck(beacons.some(u => /seen/.test(String(u))),
     't38: leaving with no link clicked no longer banks the screen — that ' +
     'is the whole "paging away IS the decision" contract');

  // now a click on the strip. Every one of its three tabs is a navigation,
  // including the one already marked aria-current.
  for (const href of ['/audit/review', '/audit/gate', '/']) {
    const a = new El('a');
    a.setAttribute('href', href);
    beacons.length = 0;
    for (const f of (docL['click'] || [])) f({ target: a });
    for (const f of (winL['pagehide'] || [])) f({});
    ck(!beacons.some(u => /seen/.test(String(u))),
       't38: clicking the link to ' + href + ' banked the fifty crops on ' +
       'screen as judged dogs: ' + beacons.join(','));
  }
}

// ── 39. the address the page was opened at is read ──────────────────────
// /review answers 301 to /audit/review and carefully carries the query string
// along -- which does nothing at all unless the destination reads one. A
// bookmark made on ?country=ECU used to land here unfiltered, because the
// country came from localStorage and nowhere else.
async function t39() {
  ck(BOOT_CALLS.some(c => /country=ECU/.test(String(c.url))),
     't39: the first request ignored ?country=ECU from the address: ' +
     JSON.stringify(BOOT_CALLS.map(c => String(c.url))));
  ck(!/ECU/.test(String(global.localStorage._o)),
     't39: the URL\'s country was written into the stored preferences — a ' +
     'link is where you were sent, not a preference you set: ' +
     global.localStorage._o);
}

// ── 40. the login gate's refusal must not read as a saved crop ──────────
// The gate answers a request whose session has ended with 401 and
// {"error":"sign in"} -- a body with no `ok` key in it at all. Every write on
// this page was guarded with `j.ok===false`, which is FALSE for that body, so
// the success branch ran: the crop left the grid, the counters counted it,
// the toast said "Flagged as not a dog" and the ledger gained nothing. Fifty
// crops can be lost that way while the strip counts up, and a session ends on
// its own -- a 7-day expiry, an admin disabling the account, a password
// change. Every case above answers 200, so none of them could see it.
async function t40() {
  const refuse = { '/api/detect/flag': 401, '/api/review/seen': 401 };
  const signin = () => ({ error: 'sign in' });      // the gate's own body

  // ── the queue: a flag that was refused ────────────────────────────────
  byId['mode'].value = 'queue';
  byId['mode'].onchange.call(byId['mode']);
  await flush();
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6), [],
                                        { seen_total: 7 }),
           '/api/detect/flag': signin, '/api/review/seen': signin };
  STATUS = refuse;
  await API.load(); await flush();
  const n = API.st().items.length, s0 = API.st().session;
  const left = byId['left'].textContent, done = byId['done'].textContent;
  const name = API.st().items[1].name;
  API.hideToast();
  await API.flag(1); await flush();
  ck(API.st().items.length === n,
     't40: a 401 took the crop out of the queue — it was never written');
  ck(API.st().items[1].name === name, 't40: a 401 resequenced the queue');
  ck(API.st().session === s0,
     't40: a 401 counted as a crop reviewed this session');
  ck(byId['left'].textContent === left && byId['done'].textContent === done,
     't40: the counters moved on a 401 — left ' + left + '->' +
     byId['left'].textContent + ', flagged ' + done + '->' +
     byId['done'].textContent);
  ck(API.st().lastUndo === null,
     't40: a 401 staged an Undo for a flag the server never took');
  // getElementById, not byId: the raw map keeps detached nodes findable, and
  // an Undo control from an earlier case would read as this one's
  ck(!document.getElementById('undoB'), 't40: a 401 drew the Undo control');
  ck(/session has ended/.test(noticed()),
     't40: nothing on screen said the session had ended, it said: "' +
     noticed() + '"');
  const card = document.querySelector('.card[data-name="' + name + '"]');
  ck(card && !card.classList.contains('go'),
     't40: the tile was left mid-exit after a 401');

  // ── undo: the same refusal, on the way back ──────────────────────────
  RESP['/api/detect/flag'] = () => ({ ok: true });
  STATUS = {};
  API.hideToast();
  await API.flag(1); await flush();
  ck(API.st().items.length === n - 1, 't40: the 200 flag did not land');
  RESP['/api/detect/flag'] = signin;
  STATUS = refuse;
  const sBefore = API.st().session, lenBefore = API.st().items.length;
  await API.undo(); await flush();
  ck(API.st().items.length === lenBefore,
     't40: a refused undo put the crop back on a grid the server still holds');
  ck(API.st().session === sBefore,
     't40: a refused undo decremented the session count');
  ck(/session has ended/.test(noticed()),
     't40: a refused undo said nothing: "' + noticed() + '"');

  // ── Restore kept: the count must not zero on a refusal ───────────────
  let alerted = '';
  window.alert = m => { alerted = String(m); };
  window.confirm = () => true;
  const kept = API.st().seenN;
  ck(kept > 0, 't40: no kept crops staged, the restore path cannot be driven');
  byId['unkeep'].onclick(); await flush();
  ck(API.st().seenN === kept,
     't40: a 401 zeroed the kept counter — it says ' + API.st().seenN +
     ' crops came back into the queue and none did');
  ck(/session has ended/.test(alerted),
     't40: the restore refusal did not name the ended session: "' +
     alerted + '"');
  ck(byId['unkeep'].disabled === false,
     't40: the restore button was left disabled after a refusal');

  // ── auditing: re-deciding a verdict the server refused ───────────────
  // the stub answers before the switch, not after: onchange loads straight
  // away and an unstubbed URL rejects
  RESP['/api/review/annotated'] = () => ({
    items: JSON.parse(JSON.stringify(CROPS.annotated)), page: 0, pages: 1,
    total: 4, pool_unfiltered: 4, n_false_positive: 2, n_true_positive: 2 });
  byId['mode'].value = 'audit';
  byId['mode'].onchange.call(byId['mode']);
  await flush(); await flush();
  const was = API.st().items[0] && API.st().items[0].label;
  ck(!!was, 't40: the audit view staged no annotated crop to re-decide');
  const acard = document.querySelector(
    '.card[data-name="' + API.st().items[0].name + '"]');
  API.hideToast();
  await API.flag(0, false, was === 'true_positive' ? 'false_positive'
                                                   : 'true_positive');
  await flush();
  ck(API.st().items[0].label === was,
     't40: a 401 rewrote the verdict on screen — the tile now says ' +
     API.st().items[0].label + ' and the ledger still says ' + was);
  ck(acard && !acard.classList.contains('changed'),
     't40: a 401 marked the tile as differing from the ledger it never changed');
  ck(/session has ended/.test(noticed()),
     't40: a refused audit verdict said nothing: "' + noticed() + '"');
  STATUS = {};
}

(async () => {
  const tests = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31,t32,t33,t34,t35,t36,t37,t38,t39,t40];
  for (const t of tests) {
    try { await t(); console.log('ok   ' + t.name); }
    catch (e) {
      failures.push(t.name + ': THREW ' + (e && e.stack || e));
      console.log('FAIL ' + t.name + ' — ' + e);
    }
  }
  if (failures.length) {
    console.log('FAILURES: ' + failures.join(' | '));
    process.exit(1);
  }
  console.log('all review cases passed');
})();
"""


def main():
    if not shutil.which('node'):
        print('SKIP: node not on PATH — cannot execute the review page JS')
        return 0
    mod = load_dashboard()
    html = mod.REVIEW_HTML
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]

    # parse the whole block first: one syntax error kills every handler, and
    # driving the functions below would report a confusing cascade instead
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(script)
        probe = f.name
    try:
        p = subprocess.run(['node', '--check', probe],
                           capture_output=True, text=True)
    finally:
        os.unlink(probe)
    if p.returncode:
        print('FAIL: /review script does not parse:\n' + p.stderr.strip()[:900])
        return 1
    print('ok   whole review script parses (%d chars)' % len(script))

    # The filter panel must survive the header compacting. A body.compact rule
    # once shed it with display:none, which read as tidy until somebody
    # scrolled with it open: the filters they were using vanished, and the
    # Filter button went dead for the scrolled life of the page -- its toggle
    # flips [hidden], and the shed rule overruled it either way. The check is
    # on the CSS because that is where the defect lived: any body.compact rule
    # that touches .npanel is the bug coming back, whatever it sets.
    css = html[html.index('<style>') + 7:html.index('</style>')]
    shed = [m for m in re.findall(r'body\.compact[^{}]*\{[^}]*\}', css)
            if '.npanel' in m.split('{')[0]]
    if shed:
        print('FAIL a body.compact rule touches the filter panel again -- an '
              'open panel vanishes on scroll and the Filter button goes dead '
              'while scrolled: ' + shed[0][:120])
        return 1
    print('ok   compacting the header leaves the open filter panel alone')

    failed = False
    for name, bad in (('the shared tab strip', tab_strip_checks(html)),
                      ('the removed guess feature',
                       guess_absence_checks(html, script)),
                      ('the annotated-date filter',
                       period_payload_checks(mod)),
                      ('the always-on score', score_checks(html, script)),
                      ('the annotated-date window',
                       period_markup_checks(html)),
                      ('the /audit/review routes', route_checks(mod))):
        for b in bad:
            print('FAIL %s: %s' % (name, b))
            failed = True
        if not bad:
            print('ok   ' + name)
    if failed:
        return 1

    fixtures = {
        'normal': [crop(i, conf=round(0.95 - i * 0.05, 2)) for i in range(9)],
        'mixed': [crop(0, full=False), crop(1, full=True), crop(2, full=False),
                  crop(3, full=True), crop(4, full=True)],
        # crops that already carry a verdict, for "Check my annotations".
        # Both verdicts are present because the audit branch decides what a
        # click means by comparing it with the one on record.
        'annotated': [dict(crop(i, conf=round(0.9 - i * 0.05, 2)),
                           label='true_positive' if i in (0, 3)
                                 else 'false_positive')
                      for i in range(4)],
        'hostile': [{
            'name': '1700000000000_x_090.jpg',
            'image_id': '"><script>alert(1)</script>',
            'ts': 1_700_000_000_000, 'conf': 0.9, 'has_full': True,
        }, {
            'name': '1700000000001_y_080.jpg',
            'image_id': '<img src=x onerror=alert(1)>',
            'ts': 1_700_000_000_001, 'conf': 0.8, 'has_full': False,
        }],
    }

    with tempfile.TemporaryDirectory() as tmp:
        js = os.path.join(tmp, 'review.js')
        fx = os.path.join(tmp, 'crops.json')
        run = os.path.join(tmp, 'run.js')
        with open(js, 'w') as f:
            f.write(script)
        # Which ids the real markup ships hidden, read off the markup so the
        # stub starts where the page does. Asserting a panel is shut against a
        # stub that starts everything visible proves nothing.
        hidden = re.findall(r'<[a-z]+[^>]*\bid="(\w+)"[^>]*\bhidden\b',
                            html)
        hidden += re.findall(r'<[a-z]+[^>]*\bhidden\b[^>]*\bid="(\w+)"',
                             html)
        # The options each <select> ships in the markup, so the stub starts
        # with the same choices the page does.
        opts = dict(re.findall(
            r'<select id="(\w+)"[^>]*>(.*?)</select>', html, re.S))
        opts = {k: v.strip() for k, v in opts.items() if '<option' in v}
        # Which controls each panel group holds, so the stub has the tree
        # trimGroups() walks.
        # Split the panel at each group start rather than trying to match
        # balanced divs: the nesting differs per group (one carries a
        # <details>), and a regex that assumed otherwise silently dropped the
        # Run button from its group and made a test fail for the wrong reason.
        groups = {}
        panel = html.split('<div class="npanel"', 1)[-1]
        parts = re.split(r'<div class="ngrp"', panel)[1:]
        for part in parts:
            # the id must be in the group's OWN opening tag. Searching the
            # whole block found the first control's id instead for a group
            # that has none, which wired that control as its own parent --
            # a cycle the stub's descendant walk never returned from.
            gid = re.match(r'[^>]*id="(\w+)"', part)
            if not gid:
                continue
            groups[gid.group(1)] = re.findall(
                r'<(?:select|button|input)[^>]*\bid="(\w+)"', part)
        with open(fx, 'w') as f:
            json.dump({'crops': fixtures, 'hidden': sorted(set(hidden)),
                       'options': opts, 'groups': groups}, f)
        with open(run, 'w') as f:
            f.write(HARNESS)
        p = subprocess.run(['node', run, js, fx],
                           capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.stderr.strip():
        sys.stderr.write(p.stderr)
    return p.returncode


if __name__ == '__main__':
    sys.exit(main())
