#!/usr/bin/env python3
"""The Duel page is blind, signed, durable and closed to anyone signed out.

    python tools/detect/tests/adv_duel.py

The Duel page (tools/dashboard/duel_page.py, /duel) shows the dogs only one
of two detectors counts, shuffled together, and asks a person whether each
box holds a dog. The answer is only worth anything if the person cannot tell
which detector drew the box, so the first thing checked is that nothing the
browser can fetch says so:

  d1  BLINDING. Every reply of every duel route, and the page itself, is
      scanned for everything in key.json beyond the item numbers, the cyan
      items, the set name and the count: sides, reasons, confidences, image
      ids (whole and as runs of their digits), box and cyan-box coordinates,
      per-side totals, the seed. Every JSON key in a reply is on a short
      allow-list. Pictures are asked for by set and number, never by id, and
      key.json / the set's old index.html are never served. Run against the
      real review set too when it is on this machine (read-only).
  d2  TRAVERSAL. The picture route serves the right bytes for a real item and
      an empty 404 for '..', %2e%2e, '#', encoded slashes, padded or odd
      numbers, an item the key does not list, another review dir's name, a
      set dir that is a symlink, an img/ that is a symlink, and a picture
      that is a symlink into another set or out of the tree
  d3  AUTH. Over real HTTP through the real gate, every duel route refuses a
      request with no session: the APIs 401, the picture 401 JSON (not a
      redirect into the login page), the page a 302 to /login, and a POST
      with no cookie writes nothing. No duel route is admin-only, the nav is
      drawn for a member and nobody signed out, and the source watcher
      restarts on duel_page.py
  d4  POST VALIDATION. An unknown answer, "same dog" on an item with no cyan
      box, an unknown set or item, a number that is not a whole number, a
      body that is not an object, a non-JSON content type, and a request with
      no session are refused and write nothing; the body cannot name the
      annotator
  d5  STORE. The latest answer wins, counts are of items not lines, two
      annotators never see each other's answers, a torn last line costs
      nothing, the answers survive a new process, and a set rebuilt under the
      same name starts fresh
  d6  EXPORT. The download is {"store": "<set>_<annotator>", "answers":
      {"<n>": answer}} and review.py's own score() reads it (run against a
      temp copy of the key, never the kit)
  d7  UNSET. With duel_dir unset -- environment AND config file -- the page
      says which key to set, every API answers configured:false, the
      picture 404s and a POST writes nothing; a duel_dir that does not exist
      says so without echoing the path; an exception's text never reaches
      the browser

EVERY PATH IS REDIRECTED to a temp directory: the ledger, the review sets,
the accounts database and its key, and the config file for the unset case.
The live ledger is fingerprinted before and compared after.
"""
import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DASH = os.path.join(REPO, 'tools', 'dashboard')
sys.path.insert(0, DASH)
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

try:
    from PIL import Image
except ImportError:
    print('SKIP: PIL is not importable here -- nothing checked the Duel page')
    sys.exit(0)

import duel_page as dp  # noqa: E402

MARK = 'zzduelguard'
WHO = MARK + '_alice'
WHO2 = MARK + '_bob'
_fails = []


def check(label, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + label +
          ('' if ok else '  ' + str(detail)[:600]))
    if not ok:
        _fails.append(label)


def stamp(path):
    try:
        st = os.stat(path)
        return (st.st_size, st.st_mtime_ns)
    except OSError:
        return None


LIVE = os.path.join(REPO, 'data', 'duel_answers', 'answers.jsonl')
live_before = stamp(LIVE)
REAL_KIT = os.path.join(REPO, 'data', 'detector_diagnosis', 'nov2026',
                        'random')

# ── the fixture ─────────────────────────────────────────────────────────────
# Values nobody would write by accident, so a scan that finds one has found a
# leak and not a coincidence.
IDS = [731958264017350 + k for k in range(10)]
SEED = 31337
TOTALS = {'cand_dog_only': 4242, 'prod_dog_only': 1717}
WHY = 'zzwhy_gamma'
CONF, PDOG = 0.123456789, 0.987654321
DET = 7777
BOX = [1234.5678, 2345.6789, 3456.7891, 4567.8912]
TWIN = [8111.1111, 8222.2222, 8333.3333, 8444.4444]
CYAN = (2, 5)


def make_key(name, ns, ids=IDS):
    items = []
    for k, n in enumerate(ns):
        items.append({
            'n': n,
            'image_id': ids[k],
            'side': 'cand_dog_only' if k % 2 else 'prod_dog_only',
            'why': WHY,
            'det_idx': DET,
            'conf': CONF,
            'p_dog': PDOG,
            'box': BOX,
            'earlier_dogs': [[3, TWIN]] if n in CYAN else [],
        })
    return {
        'cand':
        name,
        'seed':
        SEED,
        'totals':
        TOTALS,
        'sampled': {
            'cand_dog_only': 3,
            'prod_dog_only': 3
        },
        'items':
        items,
        'skipped': [{
            'image_id': 731958264017399,
            'side': 'cand_dog_only',
            'error': 'zzskipped_error'
        }],
        'dup_rule': {
            'iou': 0.1,
            'containment': 0.5
        },
    }


def jpg(path, seed):
    # a colour of its own, so a route serving the wrong picture is caught
    c = seed * 37
    Image.new('RGB', (24, 16),
              (c % 251, c // 7 % 241, c // 13 % 239)).save(path, quality=95)


def make_set(root, name, ns, colour_base):
    d = os.path.join(root, 'review_' + name)
    os.makedirs(os.path.join(d, 'img'))
    with open(os.path.join(d, 'key.json'), 'w') as fh:
        json.dump(make_key(name, ns), fh)
    with open(os.path.join(d, 'index.html'), 'w') as fh:
        fh.write('<html>old standalone page %s</html>' % IDS[0])
    for n in ns:
        for v, off in (('c', 0), ('f', 500)):
            jpg(os.path.join(d, 'img', '%03d%s.jpg' % (n, v)),
                colour_base + n + off)
    return d


def forbidden_values():
    """Every string from the fixture's key a reply must never carry."""
    out = {
        'cand_dog_only', 'prod_dog_only', WHY, 'zzskipped_error',
        str(SEED), '4242', '1717', '123456789', '987654321',
        str(DET), 'earlier_dogs', 'det_idx', 'p_dog', 'image_id', 'dup_rule',
        'containment', 'sampled', 'totals'
    }
    for x in BOX + TWIN:
        out.add('%.4f' % x)
        out.add(str(int(x)))
    for i in IDS + [731958264017399]:
        s = str(i)
        out.add(s)
        # a run of nine digits out of an id is enough to find the frame
        for k in range(0, len(s) - 8):
            out.add(s[k:k + 9])
    return out


ALLOWED_KEYS = {
    'configured', 'key', 'error', 'sets', 'set', 'total', 'answered', 'items',
    'cyan', 'answers', 'store', 'ok', 'n', 'answer'
}


def json_keys(o, path=''):
    """Every dict key in a reply that is not on the allow-list (the keys of
    'answers' are item numbers and are checked as digits)."""
    bad = []
    if isinstance(o, dict):
        for k, v in o.items():
            if path.endswith('.answers'):
                if not re.match(r'^[0-9]{1,5}\Z', str(k)):
                    bad.append(path + '.' + str(k))
            elif k not in ALLOWED_KEYS:
                bad.append(path + '.' + str(k))
            bad += json_keys(v, path + '.' + str(k))
    elif isinstance(o, list):
        for v in o:
            bad += json_keys(v, path)
    return bad


def leaks(text, forbid):
    return sorted(f for f in forbid if f in text)


tmp = tempfile.mkdtemp(prefix='duel-guard-')
real_ledger = (dp.LEDGER_DIR, dp.LEDGER)
old_env = os.environ.get('DUEL_DIR')
dash = None
real_cfg = None
srv = None
try:
    # ── the gate first: auth bootstraps lazily against the LIVE accounts
    # database unless it is pointed somewhere else before anything asks ──
    import accounts as A  # noqa: E402
    import auth  # noqa: E402
    got = auth.bootstrap(db_path=os.path.join(tmp, 'a.db'),
                         key_path=os.path.join(tmp, 'k'),
                         env={
                             'DASHBOARD_USER': MARK + '_boss',
                             'DASHBOARD_PASSWORD': 'a-password-long-enough'
                         })
    if not got.get('ok'):
        print('FAIL the gate would not start on a temp store: %r' % (got, ))
        sys.exit(1)

    spec = importlib.util.spec_from_file_location(
        'dashboard', os.path.join(DASH, 'dashboard.py'))
    dash = importlib.util.module_from_spec(spec)
    sys.modules['dashboard'] = dash
    spec.loader.exec_module(dash)
    real_cfg = dash.CFG_PATH
    check(
        'd0 the dashboard loads the same duel_page module this test '
        'redirects',
        dash._duel_page() is dp, dash._duel_page())

    dp.LEDGER_DIR = os.path.join(tmp, 'duel_answers')
    dp.LEDGER = os.path.join(dp.LEDGER_DIR, 'answers.jsonl')
    dp._cache.clear()

    root = os.path.join(tmp, 'sets')
    os.makedirs(root)
    NS = [1, 2, 3, 4, 5, 6]
    alpha = make_set(root, 'zzalpha', NS, 0)
    beta = make_set(root, 'zzbeta', [1, 2, 3], 100)
    # a picture with no item behind it in the key
    jpg(os.path.join(alpha, 'img', '007c.jpg'), 7)
    # an img/ picture that is a link into another set
    os.remove(os.path.join(beta, 'img', '003c.jpg'))
    os.symlink(os.path.join(alpha, 'img', '003c.jpg'),
               os.path.join(beta, 'img', '003c.jpg'))
    # an img/ picture that is a link out of the tree
    outside = os.path.join(tmp, 'outside')
    os.makedirs(outside)
    jpg(os.path.join(outside, 'secret.jpg'), 99)
    os.remove(os.path.join(alpha, 'img', '004f.jpg'))
    os.symlink(os.path.join(outside, 'secret.jpg'),
               os.path.join(alpha, 'img', '004f.jpg'))
    # a set directory that is a link to somewhere outside
    ext = make_set(outside, 'zzext', [1, 2], 200)
    os.symlink(ext, os.path.join(root, 'review_zzlinked'))
    # a set whose img/ is a link
    gamma = os.path.join(root, 'review_zzgamma')
    os.makedirs(gamma)
    with open(os.path.join(gamma, 'key.json'), 'w') as fh:
        json.dump(make_key('zzgamma', [1, 2]), fh)
    os.symlink(os.path.join(ext, 'img'), os.path.join(gamma, 'img'))
    # things that are not sets
    os.makedirs(os.path.join(root, 'notreview', 'img'))
    with open(os.path.join(root, 'notreview', 'key.json'), 'w') as fh:
        json.dump(make_key('notreview', [1]), fh)
    jpg(os.path.join(root, 'notreview', 'img', '001c.jpg'), 1)
    os.makedirs(os.path.join(root, 'review_nokey', 'img'))
    jpg(os.path.join(root, 'review_nokey', 'img', '001c.jpg'), 1)
    with open(os.path.join(root, 'key.json'), 'w') as fh:
        json.dump(make_key('top', [1]), fh)
    os.environ['DUEL_DIR'] = root

    FORBID = forbidden_values()
    ALPHA_BYTES = open(os.path.join(alpha, 'img', '002c.jpg'), 'rb').read()

    def ledger():
        out = []
        try:
            with open(dp.LEDGER) as fh:
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        return out

    MEMBER = {'id': 2, 'username': WHO, 'role': 'member', 'csrf': 'x'}
    MEMBER2 = {'id': 3, 'username': WHO2, 'role': 'member', 'csrf': 'y'}

    def handler(path, session, body=None, ctype='application/json', raw=None):
        h = dash.BoardHandler.__new__(dash.BoardHandler)
        if raw is None:
            raw = json.dumps(body if body is not None else {}).encode()
        h.path = path
        h.command = 'POST' if (body is not None or raw) and ctype else 'GET'
        h.headers = {'Content-Length': str(len(raw))}
        if ctype:
            h.headers['Content-Type'] = ctype
        h.rfile = io.BytesIO(raw)
        h.wfile = io.BytesIO()
        h.session = session
        h.client_address = ('127.0.0.1', 0)
        h.request_version = 'HTTP/1.1'
        out = {'json': [], 'status': [], 'errors': [], 'headers': {}}
        h._json = lambda o, c=200: out['json'].append((o, c))
        h._gate = lambda: False  # the gate has already said yes
        h.send_response = lambda c, m=None: out['status'].append(c)
        h.send_header = lambda k, v: out['headers'].__setitem__(k, v)
        h.end_headers = lambda: None
        h.send_error = lambda c, m=None, e=None: out['errors'].append(c)
        return h, out

    def get(path, session=MEMBER):
        h, out = handler(path, session, raw=b'', ctype=None)
        h.do_GET()
        out['body'] = h.wfile.getvalue()
        return out

    def post(body, session=MEMBER, ctype='application/json', raw=None):
        h, out = handler('/api/duel/answer',
                         session,
                         body,
                         ctype=ctype,
                         raw=raw)
        h.do_POST()
        return out

    replies = []  # (label, text) of everything the browser could get

    def seen(label, out):
        for o, _c in out['json']:
            replies.append((label, json.dumps(o)))
            bad = json_keys(o)
            if bad:
                replies.append((label, 'BADKEYS ' + ' '.join(bad)))
        if out.get('body'):
            replies.append((label, out['body'].decode('latin-1')))
        return out

    # ── d1: the page and every read ─────────────────────────────────────
    page = seen('page', get('/duel'))
    html = page['body'].decode('utf-8')
    check(
        'd1a /duel serves the page to a member', page['status'] == [200]
        and '<h1>Duel</h1>' in html and 'Is the boxed thing a dog?' in html
        and 'Same dog as cyan' in html and '__ACCOUNT__' not in html,
        page['status'])
    check(
        'd1b the page asks for pictures by set and item number, never by '
        'image id or a relative img/ path', "'/duel/img?set='" in html
        and "'img/'" not in html and '"img/"' not in html
        and 'image_id' not in html)
    lst = seen('sets', get('/api/duel/sets'))
    names = [s['set'] for s in lst['json'][0][0].get('sets', [])]
    check(
        'd1c the set list is the real review_* directories with a key, '
        'and no link, no linked img/, no keyless dir, no non-review dir',
        names == ['zzalpha', 'zzbeta'], names)
    v = seen('set', get('/api/duel/set?set=zzalpha'))['json'][0][0]
    check(
        'd1d a set says its items in ascending order, which show cyan, and '
        'its count -- and nothing more',
        v.get('items') == NS and v.get('cyan') == list(CYAN)
        and v.get('total') == 6 and v.get('set') == 'zzalpha', v)
    seen('set beta', get('/api/duel/set?set=zzbeta'))
    seen('export', get('/api/duel/export?set=zzalpha'))
    for p in ('/duel/review_zzalpha/key.json', '/duel/key.json',
              '/duel/review_zzalpha/index.html', '/duel/img/key.json',
              '/api/duel/key?set=zzalpha', '/api/duel/key.json', '/duel/',
              '/duel/review_zzalpha/'):
        o = seen('static ' + p, get(p))
        check(
            'd1e %s is not served' % p, o['errors'] == [404]
            or (o['json'] and 'error' in o['json'][0][0]), o)

    # ── d2: traversal on the picture route ──────────────────────────────
    ok = get('/duel/img?set=zzalpha&n=2&v=c')
    check(
        'd2a a real item\'s close-up is served byte for byte as a jpeg',
        ok['status'] == [200] and ok['body'] == ALPHA_BYTES
        and ok['headers'].get('Content-Type') == 'image/jpeg', ok['status'])
    f = get('/duel/img?set=zzalpha&n=2&v=f')
    check(
        'd2b and its whole frame is the other picture', f['status'] == [200]
        and f['body'] != ALPHA_BYTES and f['body'] == open(
            os.path.join(alpha, 'img', '002f.jpg'), 'rb').read())
    nasty = [
        'set=../review_zzbeta&n=1&v=c',
        'set=..&n=1&v=c',
        'set=zzalpha/../zzbeta&n=1&v=c',
        'set=%2e%2e%2fzzbeta&n=1&v=c',
        'set=zzalpha%2F..%2Fzzbeta&n=1&v=c',
        'set=zzalpha%23&n=1&v=c',
        'set=review_zzalpha&n=1&v=c',
        'set=ZZALPHA&n=1&v=c',
        'set=zzalpha%0A&n=1&v=c',
        'set=zzalpha%00&n=1&v=c',
        'set=notreview&n=1&v=c',
        'set=nokey&n=1&v=c',
        'set=zzlinked&n=1&v=c',
        'set=zzext&n=1&v=c',
        'set=zzgamma&n=1&v=c',
        'set=zzbeta&n=3&v=c',
        'set=zzalpha&n=4&v=f',
        'set=zzalpha&n=7&v=c',
        'set=zzalpha&n=0&v=c',
        'set=zzalpha&n=-1&v=c',
        'set=zzalpha&n=001&v=x',
        'set=zzalpha&n=1.0&v=c',
        'set=zzalpha&n=true&v=c',
        'set=zzalpha&n=%2e%2e%2fkey.json&v=c',
        'set=zzalpha&n=../../key&v=c',
        'set=zzalpha&n=1&v=c%2F..%2F..',
        'set=zzalpha&n=1&v=..%2Fkey.json',
        'set=zzalpha&n=1&v=',
        'set=zzalpha&n=1',
        'set=zzalpha&v=c',
        'n=1&v=c',
        'set=zzalpha&n=1%23&v=c',
        'set=zzalpha&n=%201&v=c',
        'set=zzalpha&n=99999999999&v=c',
        'set=zzalpha&n=1&v=C',
    ]
    for q in nasty:
        o = get('/duel/img?' + q)
        # %00 is turned away earlier still, by _sane_path, with a 400
        check(
            'd2c refused, empty: /duel/img?' + q, o['errors'] in ([404], [400])
            and not o['body'] and not o['json'],
            (o['status'], o['errors'], o['json']))
    o = get('/duel/img#x?set=zzalpha&n=1&v=c')
    check('d2d a fragment on the route name falls to the static 404',
          o['errors'] == [404] and not o['body'], o)
    # n=001 on a real item is a number, and the same picture -- not a hole,
    # since the file name is built from the parsed number
    o = get('/duel/img?set=zzalpha&n=002&v=c')
    check(
        'd2e a padded number is the same item, never a second path to '
        'another file', o['body'] == ALPHA_BYTES, o['status'])

    # ── d4: POST validation ─────────────────────────────────────────────
    def refused(label, out, n_before):
        js = out['json']
        check(
            label,
            len(ledger()) == n_before and js and not js[0][0].get('ok')
            and js[0][1] in (400, 413, 415), (js, len(ledger()), n_before))
        for o, _c in js:
            replies.append((label, json.dumps(o)))

    n0 = len(ledger())
    refused('d4a an unknown answer is refused',
            post({
                'set': 'zzalpha',
                'n': 1,
                'answer': 'cat'
            }), n0)
    refused('d4b a missing answer is refused', post({
        'set': 'zzalpha',
        'n': 1
    }), n0)
    refused('d4c "same dog" on an item with no cyan box is refused',
            post({
                'set': 'zzalpha',
                'n': 1,
                'answer': 'dup'
            }), n0)
    refused('d4d an unknown set is refused',
            post({
                'set': 'zzdelta',
                'n': 1,
                'answer': 'dog'
            }), n0)
    refused('d4e a set named by its directory is refused',
            post({
                'set': 'review_zzalpha',
                'n': 1,
                'answer': 'dog'
            }), n0)
    refused('d4f a linked set dir is refused',
            post({
                'set': 'zzlinked',
                'n': 1,
                'answer': 'dog'
            }), n0)
    refused('d4g an item the key does not list is refused',
            post({
                'set': 'zzalpha',
                'n': 7,
                'answer': 'dog'
            }), n0)
    for bad_n in (True, 1.5, '1.5', 'abc', -1, 0, None, [1], '1 ', ''):
        refused('d4h n=%r is refused' % (bad_n, ),
                post({
                    'set': 'zzalpha',
                    'n': bad_n,
                    'answer': 'dog'
                }), n0)
    refused('d4i a body that is a list is refused', post(None, raw=b'[1, 2]'),
            n0)
    refused('d4j a body that is not JSON is refused',
            post(None, raw=b'set=zzalpha&n=1&answer=dog'), n0)
    refused(
        'd4k text/plain is refused even when it is JSON',
        post({
            'set': 'zzalpha',
            'n': 1,
            'answer': 'dog'
        }, ctype='text/plain'), n0)
    refused(
        'd4l a form content type is refused',
        post({
            'set': 'zzalpha',
            'n': 1,
            'answer': 'dog'
        },
             ctype='application/x-www-form-urlencoded'), n0)
    out = post({
        'set': 'zzalpha',
        'n': 1,
        'answer': 'dog',
        'by': 'mallory'
    },
               session=None)
    check(
        'd4m with no session nothing is written',
        len(ledger()) == n0
        and not (out['json'] and out['json'][0][0].get('ok')), out['json'])
    out = post({'set': 'zzalpha', 'n': 1, 'answer': 'dog', 'by': 'mallory'})
    L = ledger()
    check(
        'd4n a member\'s answer is signed with the session\'s name, not '
        'the one in the body', out['json'][0][1] == 200
        and out['json'][0][0].get('ok') and len(L) == n0 + 1
        and L[-1].get('by') == WHO and L[-1].get('n') == 1
        and L[-1].get('answer') == 'dog', (out['json'], L[-1:]))
    seen('save reply', out)
    out = post({
        'set': 'zzalpha',
        'n': '2',
        'answer': 'dup'
    },
               ctype='application/json; charset=utf-8')
    check(
        'd4o "same dog" on a cyan item is taken (n as digits, JSON with a '
        'charset)', out['json'][0][0].get('ok'), out['json'])
    seen('save dup reply', out)
    # 'at' is left out of the scan: it is the server's own unix clock, ten
    # digits that sooner or later contain one of FORBID's short coordinate
    # runs ('4242', '1234' ...), which failed this check at random while
    # nothing leaked. It is checked instead to be a bare whole number, so it
    # cannot be smuggling a key value either.
    _lines = [{k: v for k, v in r.items() if k != 'at'} for r in ledger()]
    check(
        'd4p a saved line records the item, never anything of the key',
        not leaks(json.dumps(_lines), FORBID - {'image_id'})
        and all(type(r.get('at')) is int for r in ledger()),
        leaks(json.dumps(_lines), FORBID))

    # ── d5: the store ───────────────────────────────────────────────────
    post({'set': 'zzalpha', 'n': 1, 'answer': 'not_dog'})
    post({'set': 'zzalpha', 'n': 3, 'answer': 'unsure'})
    v = seen('set after', get('/api/duel/set?set=zzalpha'))['json'][0][0]
    check(
        'd5a the latest answer wins, and progress counts items not lines',
        v.get('answers') == {
            '1': 'not_dog',
            '2': 'dup',
            '3': 'unsure'
        } and v.get('answered') == 3 and len(ledger()) == 4,
        (v, len(ledger())))
    out = post({'set': 'zzalpha', 'n': 1, 'answer': 'dog'}, session=MEMBER2)
    v2 = seen('set bob', get('/api/duel/set?set=zzalpha',
                             MEMBER2))['json'][0][0]
    v1 = get('/api/duel/set?set=zzalpha')['json'][0][0]
    check(
        'd5b two annotators keep their own answers and their own progress',
        v2.get('answers') == {'1': 'dog'} and v2.get('answered') == 1
        and v1.get('answers', {}).get('1') == 'not_dog'
        and v1.get('answered') == 3, (v1, v2))
    s1 = get('/api/duel/sets')['json'][0][0]['sets']
    check('d5c the set list shows this annotator\'s progress per set',
          [(s['set'], s['answered'], s['total'])
           for s in s1] == [('zzalpha', 3, 6), ('zzbeta', 0, 3)], s1)
    post({'set': 'zzbeta', 'n': 1, 'answer': 'dog'})
    v1 = get('/api/duel/set?set=zzalpha')['json'][0][0]
    check(
        'd5d an answer in one set is not an answer in another',
        v1.get('answered') == 3 and '1' in v1['answers']
        and get('/api/duel/set?set=zzbeta')['json'][0][0]['answers']
        == {'1': 'dog'})
    with open(dp.LEDGER, 'ab') as fh:
        fh.write(b'{"set": "zzalpha", "n": 4, "ans')  # a torn line
    out = post({'set': 'zzalpha', 'n': 4, 'answer': 'dog'})
    v1 = get('/api/duel/set?set=zzalpha')['json'][0][0]
    check(
        'd5e a torn last line costs nothing: the next answer lands whole',
        out['json'][0][0].get('ok') and v1['answers'].get('4') == 'dog'
        and v1.get('answered') == 4, (out['json'], v1))
    # a new process: nothing held in memory
    code = ('import sys,json;sys.path.insert(0,%r);import duel_page as d;'
            'd.LEDGER=%r;print(json.dumps(d.export(%r,"zzalpha",%r)))' %
            (DASH, dp.LEDGER, root, WHO))
    r = subprocess.run([sys.executable, '-c', code],
                       capture_output=True,
                       text=True,
                       timeout=60)
    try:
        fresh = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        fresh = {'stderr': r.stderr[-400:]}
    check(
        'd5f the answers survive a restart: a new process reads them back',
        fresh.get('answers') == {
            '1': 'not_dog',
            '2': 'dup',
            '3': 'unsure',
            '4': 'dog'
        }, fresh)
    # rebuilt in place: same name, different boxes behind the numbers
    keyp = os.path.join(alpha, 'key.json')
    orig_key = open(keyp).read()
    with open(keyp, 'w') as fh:
        json.dump(make_key('zzalpha', NS, ids=[i + 1000 for i in IDS]), fh)
    v1 = get('/api/duel/set?set=zzalpha')['json'][0][0]
    check('d5g a set rebuilt under the same name starts fresh',
          v1.get('answers') == {} and v1.get('answered') == 0, v1)
    with open(keyp, 'w') as fh:
        fh.write(orig_key)
    os.utime(keyp, ns=(1, 1))  # a stamp of its own, so the cache re-reads
    v1 = get('/api/duel/set?set=zzalpha')['json'][0][0]
    check('d5h and the old build\'s answers come back with it',
          v1.get('answered') == 4, v1)

    # ── d6: export, read by review.py's own score() ─────────────────────
    ex = seen('export after', get('/api/duel/export?set=zzalpha'))
    ex = ex['json'][0][0]
    check(
        'd6a the export is {"store": "<set>_<annotator>", "answers": '
        '{"<n>": answer}} with plain digits for keys',
        ex.get('store') == 'zzalpha_' + WHO
        and set(ex) == {'store', 'answers'} and ex['answers'] == {
            '1': 'not_dog',
            '2': 'dup',
            '3': 'unsure',
            '4': 'dog'
        }, ex)
    ex2 = get('/api/duel/export?set=zzalpha', MEMBER2)['json'][0][0]
    check('d6b each annotator exports only their own answers', ex2 == {
        'store': 'zzalpha_' + WHO2,
        'answers': {
            '1': 'dog'
        }
    }, ex2)
    review_py = os.path.join(REAL_KIT, 'review.py')
    if not os.path.isfile(review_py):
        print('skip d6c: review.py is not on this machine')
    else:
        kit = os.path.join(tmp, 'kit')
        os.makedirs(os.path.join(kit, 'review_zzalpha'))
        shutil.copy(keyp, os.path.join(kit, 'review_zzalpha', 'key.json'))
        ansp = os.path.join(kit, 'answers.json')
        with open(ansp, 'w') as fh:
            json.dump(ex, fh)
        sys.path.insert(0, REAL_KIT)  # review.py imports its sibling compare
        rspec = importlib.util.spec_from_file_location('duel_review_py',
                                                       review_py)
        rv = importlib.util.module_from_spec(rspec)
        rspec.loader.exec_module(rv)
        rv.HERE = kit  # score.json is written here, never into the kit
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rv.score(types.SimpleNamespace(cand='zzalpha', answers=ansp))
            sc = json.load(
                open(os.path.join(kit, 'review_zzalpha', 'score.json')))
            err = None
        except SystemExit as e:
            sc, err = {}, 'SystemExit: %s' % (e, )
        check(
            'd6c review.py score() reads the export: 2 unanswered, one '
            '"can\'t tell", one "same dog"', err is None
            and sc.get('missing_answers') == 2
            and sum(sc.get('unsure', {}).values()) == 1
            and sum(sc.get('same_dog_as_cyan', {}).values()) == 1, err or sc)
        check('d6d and nothing was written next to the real kit',
              not os.path.exists(os.path.join(REAL_KIT, 'review_zzalpha')))

    # ── d1 again: the scan over everything collected ───────────────────
    hits = [(lab, leaks(t, FORBID)) for lab, t in replies]
    hits = [(lab, h) for lab, h in hits if h]
    badkeys = [(lab, t) for lab, t in replies if t.startswith('BADKEYS')]
    check(
        'd1f no reply of any duel route carries a side, reason, '
        'confidence, image id, coordinate, total or seed (%d replies '
        'scanned)' % len(replies), not hits, hits[:5])
    check('d1g every JSON key in every reply is on the allow-list',
          not badkeys, badkeys[:5])
    check('d1h the page itself carries no fixture value either',
          not leaks(html, FORBID), leaks(html, FORBID))

    # the real set, read-only, when it is here
    real_set = os.path.join(REAL_KIT, 'review_finetune_e2e', 'key.json')
    if os.path.isfile(real_set):
        rk = json.load(open(real_set))
        rforbid = {
            'cand_dog_only', 'prod_dog_only', 'earlier_dogs', 'image_id',
            'det_idx', 'p_dog', 'totals', 'sampled'
        }
        for it in rk['items']:
            rforbid.add(str(it['image_id']))
            rforbid.add(str(it['why']))
            rforbid.add(repr(it['conf']))
            rforbid.add(repr(it['p_dog']))
            for x in it['box']:
                rforbid.add(repr(float(x)))
        for s, t in rk['totals'].items():
            rforbid.add(str(t))
        rforbid.discard(rk['cand'])  # the set name is allowed
        os.environ['DUEL_DIR'] = REAL_KIT
        got = [
            get('/api/duel/sets'),
            get('/api/duel/set?set=finetune_e2e'),
            get('/api/duel/export?set=finetune_e2e')
        ]
        os.environ['DUEL_DIR'] = root
        texts = [json.dumps(o['json'][0][0]) for o in got]
        rv_ = got[1]['json'][0][0]
        expect_cyan = sorted(i['n'] for i in rk['items'] if i['earlier_dogs'])
        # short totals like '744' could appear inside a longer number by
        # chance; only whole JSON tokens count for those
        toks = set(re.findall(r'[A-Za-z0-9_.]+', ' '.join(texts)))
        rhits = sorted(
            f for f in rforbid
            if (f in toks) or (len(f) >= 8 and f in ' '.join(texts)))
        check(
            'd1i the real review set (read-only): items, cyan and count '
            'only -- %d items, cyan %s' % (len(rk['items']), expect_cyan),
            rv_.get('items') == sorted(i['n'] for i in rk['items'])
            and rv_.get('cyan') == expect_cyan and not rhits
            and not any(json_keys(o['json'][0][0])
                        for o in got), (rhits[:5], rv_.get('cyan')))
    else:
        print('skip d1i: the real review set is not on this machine')

    # ── d3: wiring, then the real gate over HTTP ────────────────────────
    BH = dash.BoardHandler
    check(
        'd3a the picture route answers a signed-out <img> the way the '
        'crops do, and no duel route is admin-only',
        '/duel/img' in BH.AUTH_IMAGE_PATHS and not any(
            p.startswith('/duel') or p.startswith('/api/duel')
            for p in BH.ADMIN_GET | BH.ADMIN_POST))
    nav = dash.render_duel_nav({'username': WHO, 'role': 'member'})
    check(
        'd3b the nav link is drawn for a member and for nobody signed out',
        'href="/duel"' in nav and '>Duel</a>' in nav
        and dash.render_duel_nav(None) == '', nav)
    src = open(os.path.join(DASH, 'dashboard.py')).read()
    at = src.find("for _m in ('audit.py'")
    check(
        'd3c the source watcher restarts on duel_page.py, and annotate_page'
        '.py is still where its own check looks', at >= 0
        and 'duel_page.py' in src[at:at + 240]
        and 'annotate_page.py' in src[at:at + 200], src[at:at + 240])
    check(
        'd3d the nav is spliced into the front page beside Annotate',
        "render_annotate_nav(getattr(self, 'session', None))\n"
        "                     + render_duel_nav(" in src)

    p = dash._accounts_db()
    key = os.path.join(tmp, 'k')
    check('d3e the gate is on the temp store, not the live one', p
          and os.path.dirname(os.path.realpath(p)) == os.path.realpath(tmp), p)
    A.create_user(WHO, 'another-good-password', path=p)

    def cookie(name):
        value, _ = auth.mint(A.get_user(name, path=p), key_path=key)
        return {'Cookie': auth.COOKIE + '=' + value}

    class Quiet(dash.BoardHandler):

        def log_message(self, *a):
            pass

    class NoRedirect(urllib.request.HTTPRedirectHandler):

        def redirect_request(self, *a, **k):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    srv = ThreadingHTTPServer(('127.0.0.1', 0), Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = 'http://127.0.0.1:%d' % srv.server_port

    def http(path, headers=None, data=None, method=None):
        req = urllib.request.Request(base + path,
                                     headers=headers or {},
                                     data=data,
                                     method=method)
        try:
            r = opener.open(req, timeout=30)
            return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    for path in ('/api/duel/sets', '/api/duel/set?set=zzalpha',
                 '/api/duel/export?set=zzalpha'):
        st, _h, body = http(path)
        check('d3f signed out, %s answers 401' % path, st == 401
              and b'zzalpha_' not in body and b'answers' not in body,
              (st, body[:200]))
    st, hd, body = http('/duel/img?set=zzalpha&n=2&v=c')
    check(
        'd3g signed out, the picture answers 401 JSON, not the login page',
        st == 401 and 'json' in hd.get('Content-Type', '')
        and body != ALPHA_BYTES, (st, hd.get('Content-Type')))
    st, hd, body = http('/duel')
    check('d3h signed out, /duel is a redirect to the login page', st == 302
          and hd.get('Location', '').startswith('/login'),
          (st, hd.get('Location')))
    n0 = len(ledger())
    st, _h, body = http(
        '/api/duel/answer', {'Content-Type': 'application/json'},
        json.dumps({
            'set': 'zzalpha',
            'n': 5,
            'answer': 'dog'
        }).encode(), 'POST')
    check('d3i signed out, a POST answers 401 and writes nothing', st == 401
          and len(ledger()) == n0, (st, body[:200]))
    ck = cookie(WHO)
    st, _h, body = http('/duel', ck)
    check('d3j signed in, the page is served', st == 200
          and b'<h1>Duel</h1>' in body, st)
    st, hd, body = http('/duel/img?set=zzalpha&n=2&v=c', ck)
    check('d3k signed in, the picture is served', st == 200
          and body == ALPHA_BYTES, st)
    st, _h, body = http(
        '/api/duel/answer', dict(ck, **{'Content-Type': 'application/json'}),
        json.dumps({
            'set': 'zzalpha',
            'n': 5,
            'answer': 'dup'
        }).encode(), 'POST')
    check(
        'd3l signed in, a POST through the real gate is saved under the '
        'session\'s name', st == 200 and json.loads(body).get('ok')
        and ledger()[-1].get('by') == WHO and ledger()[-1].get('n') == 5,
        (st, body[:200]))
    st, _h, body = http(
        '/api/duel/answer', dict(ck, **{'Content-Type': 'text/plain'}),
        json.dumps({
            'set': 'zzalpha',
            'n': 6,
            'answer': 'dog'
        }).encode(), 'POST')
    check(
        'd3m signed in, a text/plain POST (one a foreign page can send '
        'without asking) is refused', st == 415 and ledger()[-1].get('n') == 5,
        (st, body[:200]))
    st, _h, body = http('/duel/img?set=zzalpha&n=2&v=c', ck, method='HEAD')
    check('d3n HEAD on the picture serves nothing', st == 404, st)

    # ── d7: unset, missing, and an exception's text ────────────────────
    dash.CFG_PATH = os.path.join(tmp, 'no-such-config.json')
    dash._cfg_cache.update(mtime=None, data={})
    os.environ.pop('DUEL_DIR', None)
    check('d7a the config is really unset for this part',
          dash.duel_dir() == '', dash.duel_dir())
    pg = get('/duel')
    check('d7b with duel_dir unset the page still loads (it says why itself)',
          pg['status'] == [200] and b'duel_dir' in pg['body'], pg['status'])
    for path in ('/api/duel/sets', '/api/duel/set?set=zzalpha',
                 '/api/duel/export?set=zzalpha'):
        o = seen('unset ' + path, get(path))
        j = o['json'][0][0] if o['json'] else {}
        check(
            'd7c unset, %s answers configured:false naming duel_dir' % path,
            j.get('configured') is False and j.get('key') == 'duel_dir'
            and 'duel_dir' in j.get('error', '')
            and 'DUEL_DIR' in j.get('error', ''), j)
    o = get('/duel/img?set=zzalpha&n=2&v=c')
    check('d7d unset, the picture 404s', o['errors'] == [404]
          and not o['body'], o)
    n0 = len(ledger())
    o = post({'set': 'zzalpha', 'n': 6, 'answer': 'dog'})
    check(
        'd7e unset, a POST is refused and writes nothing', o['json']
        and o['json'][0][1] == 400 and not o['json'][0][0].get('ok')
        and len(ledger()) == n0, o['json'])
    os.environ['DUEL_DIR'] = os.path.join(tmp, 'zzsecretpath', 'nowhere')
    texts = [
        json.dumps(get(p)['json'][0][0])
        for p in ('/api/duel/sets', '/api/duel/set?set=zzalpha')
    ]
    texts.append(
        json.dumps(
            post({
                'set': 'zzalpha',
                'n': 6,
                'answer': 'dog'
            })['json'][0][0]))
    check(
        'd7f a duel_dir that is not there says so, without the path',
        all('not a directory' in t for t in texts)
        and not any('zzsecretpath' in t or tmp in t for t in texts), texts)
    os.environ['DUEL_DIR'] = root
    real_view = dp.set_view

    def boom(*a, **k):
        raise RuntimeError('cand_dog_only %s %s' % (IDS[0], root))

    dp.set_view = boom
    try:
        o = get('/api/duel/set?set=zzalpha')
    finally:
        dp.set_view = real_view
    t = json.dumps(o['json'])
    check(
        'd7g an exception answers JSON with its type, never its text',
        o['json'] and 'RuntimeError' in t and str(IDS[0]) not in t
        and 'cand_dog_only' not in t and root not in t, t)
finally:
    if srv is not None:
        srv.shutdown()
        srv.server_close()
    dp.LEDGER_DIR, dp.LEDGER = real_ledger
    dp._cache.clear()
    if dash is not None and real_cfg is not None:
        dash.CFG_PATH = real_cfg
        dash._cfg_cache.update(mtime=None, data={})
    os.environ.pop('DUEL_DIR', None)
    if old_env is not None:
        os.environ['DUEL_DIR'] = old_env
    shutil.rmtree(tmp, ignore_errors=True)

check('d0 the live answer ledger is untouched',
      stamp(LIVE) == live_before, (live_before, stamp(LIVE)))

if _fails:
    print('\n%d FAILED:' % len(_fails))
    for f in _fails:
        print('  ' + f)
    sys.exit(1)
print('\nall Duel page checks passed')
