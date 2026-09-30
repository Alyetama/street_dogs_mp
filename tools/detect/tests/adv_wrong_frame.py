#!/usr/bin/env python3
"""The "what it got wrong" tiles open the frame the mistake came from.

    python tools/detect/tests/adv_wrong_frame.py

A detector's mistake is about a REGION of a street photo, and the tile shows
a crop around it. That answers "what did it see" and hides the two things a
diagnosis needs: where in the frame the box sits, and what the labels say is
really there. Clicking a tile opens the whole frame with the model's box
drawn in it, every labelled box beside it, and one line saying which kind of
wrong this is. Those three kinds look identical as a crop and are fixed in
three different ways:

    invented, IoU >= 0.5 with a labelled box   a duplicate: the dog was
                                               already taken by a surer box
    invented, 0 < IoU < 0.5                    on a dog, too loose to count --
                                               a localisation failure
    invented, touching nothing                 nothing labelled there at all

Graded here against a dataset built for the purpose, so every expectation is
known exactly rather than read off whatever run happens to be on disk:

  w1  the frame's size and name, and how many OTHER mistakes share it
  w2  labelled boxes come out of the label file in pixels; a labelled box the
      model missed is drawn once, as the miss, not twice
  w3  the note names the right kind of wrong for each of the four cases
  w4  a cache that kept only its surest mistakes does not claim "found"
  w5  a value just under the 0.5 line never prints as 0.50
  w6  the index is the only thing the client controls, and a bad one is None
  w7  the label file is found under the LAST /images/ of the path
  w8  the frame is served as-is when it fits, re-encoded when it is too big
      or carries an EXIF rotation that would put every box in the wrong place
  w9  every tile's button carries the index of its own picture
  w10 the page wires the tiles to the viewer, and the viewer to the routes
  w11 over HTTP: signed out is refused, signed in is served, a bad index 404s
"""
import inspect
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))

import dashboard as d  # noqa: E402

try:
    from PIL import Image
except ImportError:
    print('SKIP: PIL is not importable here -- nothing checked that a frame '
          'is served or measured')
    sys.exit(0)

_fails = []


def check(label, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + label + ('' if ok else '  ' + detail))
    if not ok:
        _fails.append(label)


def jpeg(path, size, exif_rotation=None):
    im = Image.new('RGB', size, (90, 110, 70))
    kw = {}
    if exif_rotation:
        ex = Image.Exif()
        ex[0x0112] = exif_rotation
        kw['exif'] = ex.tobytes()
    im.save(path, 'JPEG', quality=80, **kw)


def yolo(box, W, H):
    x1, y1, x2, y2 = box
    return (f'0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} '
            f'{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')


tmp = tempfile.mkdtemp(prefix='wrongframe-')
try:
    # A dataset that itself lives under a directory called `images`, so the
    # label lookup has to take the LAST /images/ in the path (w7).
    ds = os.path.join(tmp, 'images', 'ds')
    for sub in ('images/val', 'labels/val'):
        os.makedirs(os.path.join(ds, sub))
    W, H = 200, 100
    jpeg(os.path.join(ds, 'images/val/a.jpg'), (W, H))
    jpeg(os.path.join(ds, 'images/val/b.jpg'), (W, H))
    jpeg(os.path.join(ds, 'images/val/big.jpg'), (5000, 200))
    jpeg(os.path.join(ds, 'images/val/turned.jpg'), (W, H), exif_rotation=6)
    FOUND, MISSED = [10, 10, 50, 50], [120, 40, 160, 80]
    with open(os.path.join(ds, 'labels/val/a.txt'), 'w') as fh:
        fh.write(yolo(FOUND, W, H) + '\n' + yolo(MISSED, W, H) + '\n')
    with open(os.path.join(ds, 'labels/val/b.txt'), 'w') as fh:
        fh.write(yolo(MISSED, W, H) + '\n')
    items = [
        # 0: on the missed dog, but loose -> misplaced
        {
            'file': 'images/val/a.jpg',
            'kind': 'invented',
            'p': 0.9,
            'cls': 'target',
            'box': [118, 38, 190, 90]
        },
        # 1: the dog it missed
        {
            'file': 'images/val/a.jpg',
            'kind': 'missed',
            'p': None,
            'cls': 'target',
            'box': list(MISSED)
        },
        # 2: a second box on the dog it found -> duplicate
        {
            'file': 'images/val/a.jpg',
            'kind': 'invented',
            'p': 0.8,
            'cls': 'target',
            'box': [12, 12, 50, 52]
        },
        # 3: over grass -> nothing labelled there
        {
            'file': 'images/val/a.jpg',
            'kind': 'invented',
            'p': 0.7,
            'cls': 'target',
            'box': [60, 60, 80, 90]
        },
        # 4: another frame, a miss with nothing drawn near it
        {
            'file': 'images/val/b.jpg',
            'kind': 'missed',
            'p': None,
            'cls': 'target',
            'box': list(MISSED)
        },
        # 5, 6: frames for the byte checks
        {
            'file': 'images/val/big.jpg',
            'kind': 'invented',
            'p': 0.6,
            'cls': 'target',
            'box': [10, 10, 40, 40]
        },
        {
            'file': 'images/val/turned.jpg',
            'kind': 'invented',
            'p': 0.6,
            'cls': 'target',
            'box': [10, 10, 40, 40]
        },
    ]
    mdir = os.path.join(tmp, 'mistakes')
    os.makedirs(mdir)
    d.MISTAKE_DIR = mdir
    d._MISS.clear()

    def cache(key, **extra):
        doc = {
            'task': 'detect',
            'dataset': ds,
            'n': 3,
            'hit': 1,
            'wrong': len(items),
            'items': items
        }
        doc.update(extra)
        with open(os.path.join(mdir,
                               key.replace('/', '__') + '.json'), 'w') as fh:
            json.dump(doc, fh)

    KEY = 'proj/run'
    cache(KEY)

    # ── w1 ────────────────────────────────────────────────────────────────
    f0 = d.mistake_frame(KEY, 0)
    check(
        'w1 the frame reports its own size, name and the other mistakes '
        'in it',
        bool(f0)
        and (f0['w'], f0['h'], f0['name'], f0['others']) == (W, H, 'a.jpg', 3),
        f'got {f0 and (f0["w"], f0["h"], f0["name"], f0["others"])}')

    # ── w2 ────────────────────────────────────────────────────────────────
    kinds = sorted((b['kind'], tuple(round(v) for v in b['box']))
                   for b in (f0 or {}).get('boxes', []))
    found = [b for b in kinds if b[0] == 'found']
    missed = [b for b in kinds if b[0] == 'missed']
    check(
        'w2 labelled boxes come out in pixels, and a missed one is drawn '
        'once, as the miss', found == [('found', tuple(FOUND))]
        and missed == [('missed', tuple(MISSED))], f'boxes were {kinds}')
    sel = [b for b in (f0 or {}).get('boxes', []) if b.get('sel')]
    check('w2b exactly the clicked mistake is marked selected',
          len(sel) == 1 and sel[0].get('i') == 0, f'selected: {sel}')

    # ── w3 ────────────────────────────────────────────────────────────────
    want = {
        0: 'misplaced',
        1: 'it did put a box here',
        2: 'a second box on a dog it already found',
        3: 'no labelled dog anywhere under it',
        4: 'it drew nothing on this dog'
    }
    got = {i: (d.mistake_frame(KEY, i) or {}).get('note', '') for i in want}
    wrong = {i: got[i] for i in want if want[i] not in got[i]}
    check(
        'w3 the note names the kind of wrong: misplaced, duplicate, '
        'nothing there, near miss, untouched miss', not wrong,
        f'these said the wrong thing: {wrong}')

    # ── w4 ────────────────────────────────────────────────────────────────
    cache('proj/cut', truncated=True)
    ft = d.mistake_frame('proj/cut', 0) or {}
    ks = {b['kind'] for b in ft.get('boxes', [])}
    check(
        'w4 a truncated cache calls an unlisted labelled box "label", '
        'never "found"', 'found' not in ks and 'label' in ks,
        f'kinds were {sorted(ks)}')

    # ── w5 ────────────────────────────────────────────────────────────────
    it = {'kind': 'invented', 'box': [0, 0, 100, 100]}
    near = d._frame_note(it, [{'kind': 'found', 'box': [0, 0, 100, 200.2]}])
    check('w5 an IoU just under 0.5 never prints as 0.50 beside "too loose"',
          'too loose' in near and '0.50' not in near and '0.49' in near,
          f'said: {near!r}')

    # ── w6 ────────────────────────────────────────────────────────────────
    bad = {
        v: d.mistake_frame(KEY, v)
        for v in (-1, len(items), '1; drop', 'x', None)
    }
    bad['unknown run'] = d.mistake_frame('proj/nope', 0)
    leaked = {k: v for k, v in bad.items() if v is not None}
    check(
        'w6 a negative, past-the-end or non-numeric index, or an unknown '
        'run, answers None', not leaked,
        f'answered for {sorted(map(str, leaked))}')

    # ── w7 ────────────────────────────────────────────────────────────────
    lp = d._label_path(os.path.join(ds, 'images', 'val', 'a.jpg'))
    check('w7 the label is found under the LAST /images/ of the path',
          lp == os.path.join(ds, 'labels', 'val', 'a.txt'), f'got {lp}')

    # ── w8 ────────────────────────────────────────────────────────────────
    small = d.mistake_frame_bytes(KEY, 0)
    with open(os.path.join(ds, 'images/val/a.jpg'), 'rb') as fh:
        orig = fh.read()
    check('w8 a frame that fits is served byte for byte', small == orig)
    big = d.mistake_frame_bytes(KEY, 5)
    bs = Image.open(io.BytesIO(big)).size if big else None
    check(
        'w8b a frame over FRAME_MAX is shrunk to fit, shape kept',
        bool(bs) and max(bs) <= d.FRAME_MAX
        and abs(bs[0] / bs[1] - 5000 / 200) < 0.2, f'served {bs}')
    turned = d.mistake_frame_bytes(KEY, 6)
    rot = (Image.open(io.BytesIO(turned)).getexif().get(0x0112, 1)
           if turned else 'none')
    with open(os.path.join(ds, 'images/val/turned.jpg'), 'rb') as fh:
        torig = fh.read()
    check(
        'w8c a frame with an EXIF rotation is re-encoded without it, so the '
        'browser shows the raw pixels the boxes are in',
        bool(turned) and turned != torig and rot in (0, 1),
        f'orientation served: {rot}')

    # ── w9 ────────────────────────────────────────────────────────────────
    r = {
        'project': 'proj',
        'name': 'run',
        'task': 'detect',
        'status': 'finished'
    }
    html = d.render_mistakes(r)
    import re
    tiles = re.findall(
        r'<figure class="wrtile[^"]*"[^>]*data-i="(\d+)">'
        r'<button type="button" class="wropen"[^>]*>'
        r'<img [^>]*src="/api/training/wrong\?key=[^"&]*'
        r'&amp;i=(\d+)"', html)
    check(
        f'w9 every tile opens its OWN picture ({len(tiles)} of '
        f'{len(items)} tiles)',
        len(tiles) == len(items) and all(a == b for a, b in tiles),
        f'tile/picture index pairs: {tiles}')

    # ── w10 ───────────────────────────────────────────────────────────────
    page = d.TEMPLATE
    body = page[page.index('function wireWrong(){'):]
    body = body[:body.index('\n  function wireDetail(')]
    check('w10 the panel wires each tile\'s button to the viewer',
          "querySelectorAll('.wropen')" in body and 'wfv.open(' in body)
    viewer = page[page.index('var wfv=(function(){'):]
    viewer = viewer[:viewer.index('})();')]
    check(
        'w10b the viewer asks the two routes for the frame and its boxes',
        "'/api/training/wrongframe'+q" in viewer
        and "'/api/training/wrongframe.jpg'+q" in viewer)
    get = inspect.getsource(d.BoardHandler.do_GET)
    check(
        'w10c do_GET answers both routes',
        "== '/api/training/wrongframe'" in get
        and "== '/api/training/wrongframe.jpg'" in get)

    # ── w11 ───────────────────────────────────────────────────────────────
    try:
        import functools
        import http.client
        import threading
        from http.server import ThreadingHTTPServer

        import auth
    except Exception as e:  # noqa: BLE001
        print(f'SKIP w11: the gate would not import ({e}) -- nobody checked '
              f'these routes over HTTP')
    else:
        boot = auth.bootstrap(db_path=os.path.join(tmp, 'accounts.db'),
                              key_path=os.path.join(tmp, 'session.key'),
                              env={
                                  'DASHBOARD_USER': 'admin',
                                  'DASHBOARD_PASSWORD':
                                  'a-password-long-enough'
                              })
        if not boot.get('ok'):
            print(f'SKIP w11: the throwaway store would not bootstrap '
                  f'({boot.get("detail")})')
        else:
            srv = ThreadingHTTPServer(('127.0.0.1', 0),
                                      functools.partial(d.BoardHandler,
                                                        directory=tmp))
            threading.Thread(target=srv.serve_forever, daemon=True).start()

            def hit(path, cookie=None, method='GET', body=None, ctype=None):
                c = http.client.HTTPConnection('127.0.0.1',
                                               srv.server_port,
                                               timeout=60)
                head = {}
                if cookie:
                    head['Cookie'] = cookie
                if ctype:
                    head['Content-Type'] = ctype
                c.request(method, path, body=body, headers=head)
                rr = c.getresponse()
                return rr.status, dict(rr.getheaders()), rr.read()

            q = '?key=proj/run&i=0'
            out = [
                hit('/api/training/wrongframe' + q)[0],
                hit('/api/training/wrongframe.jpg' + q)[0]
            ]
            check('w11 signed out, neither route serves a frame or its boxes',
                  all(s in (302, 303, 401, 403) for s in out),
                  f'answered {out}')
            s, h, _ = hit(
                '/login',
                method='POST',
                body='username=admin&password=a-password-long-enough',
                ctype='application/x-www-form-urlencoded')
            cookie = h.get('Set-Cookie', '').split(';', 1)[0]
            s1, h1, b1 = hit('/api/training/wrongframe' + q, cookie)
            s2, h2, b2 = hit('/api/training/wrongframe.jpg' + q, cookie)
            s3 = hit('/api/training/wrongframe?key=proj/run&i=999', cookie)[0]
            s4 = hit('/api/training/wrongframe.jpg?key=proj/run&i=-1',
                     cookie)[0]
            ok_json = False
            try:
                ok_json = json.loads(b1)['sel']['box'] == items[0]['box']
            except Exception:  # noqa: BLE001
                pass
            check(
                'w11b signed in: the boxes come back as JSON and the frame '
                'as a JPEG', s1 == 200 and ok_json and s2 == 200 and b2 == orig
                and h2.get('Content-Type') == 'image/jpeg',
                f'json {s1} (sel ok: {ok_json}), jpg {s2} '
                f'{h2.get("Content-Type")}')
            check('w11c signed in: an index past the end or below zero 404s',
                  (s3, s4) == (404, 404), f'answered {(s3, s4)}')
            srv.shutdown()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if _fails:
    print(f'\n{len(_fails)} wrong-frame check(s) FAILED: ' + '; '.join(_fails))
    sys.exit(1)
print('\nall clear: a tile opens its own frame, with the box where the model '
      'put it and the labels beside it')
