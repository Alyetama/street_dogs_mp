#!/usr/bin/env python3
"""The Annotate page records what a person drew, signed with their name, and
the detector build takes it as a whole-frame label and nothing else.

    python tools/detect/tests/adv_annotate.py

A frame label is the strongest thing a person can say about a frame -- "these
are ALL the dogs in it" -- so the build trusts the silence around its boxes and
takes an empty one as a background. That is only safe if every link holds:

  n1  a save that cannot name its annotator is refused and writes nothing
  n2  the route signs the row with the SESSION's name, whatever the body
      says, and a request with no session writes nothing; any member may
      annotate, and nav and picture routes are wired the way the gate expects
  n3  "no dog" with a box drawn, or nothing drawn without "no dog", is refused
  n4  a box is clamped to the frame, and one outside it, too small to be a
      dog, not a number, or one too many, is refused
  n5  a frame is found by its batch and its index in that batch's manifest,
      and nothing the client sends reaches a path -- nor can a manifest line
      point the page at a file that is not the frame it names
  n6  the latest save is the label, and a torn last line is not one
  n7  a labelled frame is not served again; skipping walks on and wraps;
      two annotators walk a batch in different orders; a finished batch
      says so
  n8  the ledger row holds what was drawn and nothing else -- the seed dog
      shown as a hint never becomes a label
  n9  the hint is the seed box scaled from its frame to this one
  n10 a frame stored on its side is refused rather than labelled wrong
  n11 the build reads only rows the page wrote and a person signed
  n12 a row becomes YOLO text exactly, an empty frame an empty label, and an
      inconsistent row nothing
  n13 the build leaves out frames already in the set, frames of a val
      sequence, missing files and files that are not the size labelled, and
      puts an unresolved frame in train
  n14 build_dogdet_v3, run on a set small enough to read whole, puts a
      frame label in train as its boxes and an empty one as a background,
      leaves out a frame of a val sequence, lets a frame label replace the
      confirmed-box addition of the same frame, and counts what it did; and
      the ledger is a store every dogdet build digests
  n15 what the page says over a frame is true of that frame: a hidden-dog
      batch never claims the detector drew nothing or that the hint is the
      gate's dog, a frame with a confirmed dog asks for that dog too, a
      rejected frame offers "no dog", a neighbour batch keeps its own
      sentence, summary.json `kind` wins over the manifest, and the page
      draws the frame's sentence and tooltip rather than a fixed header
  n16 a change to the page reaches the running dashboard: annotate_page.py
      is one of the files its source watcher restarts on

EVERY PATH IS REDIRECTED to a temp directory, and the live ledger is
fingerprinted before and compared after.
"""
import io
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

try:
    from PIL import Image
except ImportError:
    print('SKIP: PIL is not importable here -- nothing checked the Annotate '
          'page')
    sys.exit(0)

import annotate_page as ap  # noqa: E402
import frame_labels as fl  # noqa: E402

MARK = 'zzannotateguard'
WHO = MARK + '_alice'
_fails = []


def check(label, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + label + ('' if ok else '  ' + detail))
    if not ok:
        _fails.append(label)


def stamp(path):
    try:
        st = os.stat(path)
        return (st.st_size, st.st_mtime_ns)
    except OSError:
        return None


LIVE = os.path.join(REPO, 'data', 'frame_labels', 'labels.jsonl')
live_before = stamp(LIVE)

tmp = tempfile.mkdtemp(prefix='annotate-guard-')
real = (ap.BATCHES, ap.LEDGER_DIR, ap.LEDGER)
try:
    ap.BATCHES = os.path.join(tmp, 'batches')
    ap.LEDGER_DIR = os.path.join(tmp, 'frame_labels')
    ap.LEDGER = os.path.join(ap.LEDGER_DIR, 'labels.jsonl')
    ap._cache.clear()
    frames = os.path.join(tmp, 'frames')
    os.makedirs(frames)

    def jpg(iid, w=400, h=300, orient=None, where=frames):
        p = os.path.join(where, iid + '.jpg')
        # a colour of its own, so a route serving the wrong frame is caught
        c = sum(map(ord, iid)) * 37
        im = Image.new('RGB', (w, h), (c % 251, c // 7 % 241, c // 13 % 239))
        if orient:
            ex = Image.Exif()
            ex[0x0112] = orient
            im.save(p, exif=ex)
        else:
            im.save(p)
        return p

    ids = ['7%012d' % k for k in range(10)]
    rows = [{
        'image_id': i,
        'jpg': jpg(i),
        'seed_id': '8%012d' % k
    } for k, i in enumerate(ids)]
    # a manifest line that names one frame and points at another
    other = jpg('9000000000001')
    rows.append({'image_id': '9000000000000', 'jpg': other})
    # a link that carries the right name to the wrong file
    os.makedirs(os.path.join(tmp, 'links'))
    link = os.path.join(tmp, 'links', '9000000000002.jpg')
    os.symlink(other, link)
    rows.append({'image_id': '9000000000002', 'jpg': link})
    # a frame stored on its side
    rot = jpg('9000000000003', orient=6)
    rows.append({'image_id': '9000000000003', 'jpg': rot})
    B = 'guard_batch'
    bdir = os.path.join(ap.BATCHES, B)
    os.makedirs(bdir)
    with open(os.path.join(bdir, 'manifest.jsonl'), 'w') as fh:
        fh.write(''.join(json.dumps(r) + '\n' for r in rows))
    with open(os.path.join(bdir, 'seeds.json'), 'w') as fh:
        json.dump(
            {
                ids[0]: {
                    'seed_id': '8' + '0' * 12,
                    'box': [100, 50, 200, 150],
                    'w': 1000,
                    'h': 500
                }
            }, fh)

    def ledger():
        out = []
        try:
            with open(ap.LEDGER) as fh:
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        return out

    # ── n1 ──────────────────────────────────────────────────────────────
    got = ap.save(B, 0, [[10, 10, 50, 50]], False, by='')
    check('n1 a save with no annotator is refused and writes nothing',
          not got.get('ok') and ledger() == [], f'{got}, {ledger()}')

    # ── n2 ──────────────────────────────────────────────────────────────
    import dashboard as dash

    def handler(path, session, body=None):
        h = dash.BoardHandler.__new__(dash.BoardHandler)
        raw = json.dumps(body or {}).encode()
        h.path = path
        h.command = 'POST' if body is not None else 'GET'
        h.headers = {'Content-Length': str(len(raw))}
        h.rfile = io.BytesIO(raw)
        h.wfile = io.BytesIO()
        h.session = session
        out = {'json': [], 'status': [], 'errors': []}
        h._json = lambda o, c=200: out['json'].append((o, c))
        h._gate = lambda: False  # the gate has already said yes
        h.send_response = lambda c, m=None: out['status'].append(c)
        h.send_header = lambda k, v: None
        h.end_headers = lambda: None
        h.send_error = lambda c, m=None, e=None: out['errors'].append(c)
        return h, out

    h, out = handler('/api/annotate/save', {
        'id': 2,
        'username': WHO,
        'role': 'member'
    }, {
        'batch': B,
        'i': 1,
        'boxes': [[10, 10, 60, 70]],
        'empty': False,
        'by': 'mallory'
    })
    h.do_POST()
    L = ledger()
    check(
        'n2 a member\'s save is signed with the session\'s name, not the '
        'one in the body',
        len(L) == 1 and L[-1].get('by') == WHO,
        f'reply {out["json"]}, rows {L}')
    n = len(L)
    h, out = handler(
        '/api/annotate/save', None, {
            'batch': B,
            'i': 2,
            'boxes': [[10, 10, 60, 70]],
            'empty': False,
            'by': 'mallory'
        })
    h.do_POST()
    check(
        'n2b with no session nothing is written',
        len(ledger()) == n
        and not (out['json'] and out['json'][0][0].get('ok')),
        f'{out["json"]}, {ledger()[n:]}')
    check(
        'n2c saving is not an admin-only route, and the frame picture '
        'answers a signed-out <img> the way the crops do',
        '/api/annotate/save' not in dash.BoardHandler.ADMIN_POST
        and '/annotate/frame.jpg' in dash.BoardHandler.AUTH_IMAGE_PATHS
        and not any(
            p.startswith('/annotate') or p.startswith('/api/annotate')
            for p in dash.BoardHandler.ADMIN_GET))
    nav_in = dash.render_annotate_nav({'username': WHO, 'role': 'member'})
    check(
        'n2d the nav link is drawn for a member and for nobody signed out',
        'href="/annotate"' in nav_in and dash.render_annotate_nav(None) == '',
        nav_in)
    h, out = handler('/annotate/frame.jpg?batch=%s&i=3' % B, {
        'id': 2,
        'username': WHO,
        'role': 'member'
    })
    h.do_GET()
    with open(rows[3]['jpg'], 'rb') as fh:
        want = fh.read()
    check('n2e the picture route serves the manifest\'s frame, byte for byte',
          out['status'] == [200] and h.wfile.getvalue() == want,
          f'status {out["status"]} errors {out["errors"]}')
    h, out = handler('/annotate/frame.jpg?batch=%s&i=../../x' % B, {
        'id': 2,
        'username': WHO,
        'role': 'member'
    })
    h.do_GET()
    check('n2f and a 404 for an index that is not one', out['errors'] == [404]
          and not h.wfile.getvalue(), f'{out}')

    # ── n3 ──────────────────────────────────────────────────────────────
    n = len(ledger())
    a = ap.save(B, 2, [[10, 10, 60, 70]], True, by=WHO)
    b = ap.save(B, 2, [], False, by=WHO)
    check(
        'n3 "no dog" with a box, and nothing drawn without "no dog", are '
        'both refused', not a.get('ok') and not b.get('ok')
        and len(ledger()) == n, f'{a} / {b}')

    # ── n4 ──────────────────────────────────────────────────────────────
    got = ap.save(B,
                  2, [[390, 280, 420, 330], [60, 70, 10, 10]],
                  False,
                  by=WHO)
    check(
        'n4 a box past the edge is clamped and reversed corners are put '
        'right',
        got.get('ok')
        and ledger()[-1]['boxes'] == [[390.0, 280.0, 400.0, 300.0],
                                      [10.0, 10.0, 60.0, 70.0]], f'{got}')
    n = len(ledger())
    tries = {
        'outside': [[500, 10, 600, 50]],
        'too small': [[10, 10, 11, 40]],
        'NaN': [[float('nan'), 10, 40, 40]],
        'inf': [[0, 0, float('inf'), 40]],
        'three numbers': [[1, 2, 3]],
        'a string': [['a', 2, 3, 4]],
        'too many': [[1, 1, 10, 10]] * (ap.MAX_BOXES + 1),
        'not a list': {
            'x1': 1
        },
    }
    said = {k: ap.save(B, 2, v, False, by=WHO) for k, v in tries.items()}
    let_in = [k for k, v in said.items() if v.get('ok')]
    check(
        'n4b a box outside the frame, too small, not a number, or one too '
        'many is refused', not let_in and len(ledger()) == n,
        f'accepted: {let_in}')
    check('n4c and one outside the frame is told so, not that it is small',
          'outside' in str(said['outside'].get('error')), f'{said["outside"]}')

    # ── n5 ──────────────────────────────────────────────────────────────
    stray = [
        i for i in (-1, len(rows), 'x', '1/../2', None, 1.5, '', True, '-1',
                    ' 1', '\u0661') if ap.frame_path(B, i)
    ]
    check('n5 only a real index in the manifest names a frame', not stray,
          f'these resolved: {stray}')
    check(
        'n5b a batch name that is a path resolves nothing',
        all(
            ap.frame_path(b, 0) is None
            for b in ('../batches/' + B, '/etc', B + '/..', '..', '')))
    k_mis = next(k for k, r in enumerate(rows)
                 if r['image_id'] == '9000000000000')
    k_link = next(k for k, r in enumerate(rows)
                  if r['image_id'] == '9000000000002')
    check(
        'n5c a manifest line naming one frame and pointing at another is '
        'refused',
        ap.frame_path(B, k_mis) is None)
    check('n5d so is a link that carries the right name to the wrong file',
          ap.frame_path(B, k_link) is None)

    # ── n6 ──────────────────────────────────────────────────────────────
    ap.save(B, 4, [[10, 10, 60, 70]], False, by=WHO)
    ap.save(B, 4, [], True, by=WHO)
    with open(ap.LEDGER, 'a') as fh:
        fh.write('{"image_id": "%s", "boxes": [[1,' % ids[5])
    lab = ap.labels()
    check('n6 the latest save is the label, and a torn last line is not one',
          lab.get(ids[4], {}).get('empty') is True and ids[5] not in lab,
          f'{lab.get(ids[4])}, {lab.get(ids[5])}')
    ap.save(B, 6, [[10, 10, 60, 70]], False, by=WHO)
    check('n6b the save after a torn line is not swallowed by it',
          ap.labels().get(ids[6], {}).get('by') == WHO,
          f'{ap.labels().get(ids[6])}')

    # ── n7 ──────────────────────────────────────────────────────────────
    done = set(ap.labels())
    served = []
    after = None
    for _ in range(len(rows) + 2):
        f = ap.next_frame(B, WHO, after)
        if not f or f.get('done') or f.get('error') \
                or f['image_id'] in served:
            break
        served.append(f['image_id'])
        after = f['image_id']
    walkable = {r['image_id'] for r in rows} - done
    check(
        'n7 a labelled frame is never served, and skipping reaches every '
        'other frame once', not (set(served) & done) and set(served)
        >= walkable - {'9000000000000', '9000000000002', '9000000000003'},
        f'served {served}, labelled {sorted(done)}')
    first = ap.next_frame(B, WHO)
    wrap = ap.next_frame(B, WHO, served[-1]) if served else None
    check('n7b skipping past the last frame wraps to the first', wrap and first
          and wrap.get('image_id') == first.get('image_id'),
          f'{wrap and wrap.get("image_id")} vs {first.get("image_id")}')
    orders = {
        tuple(sorted(ids, key=lambda i: ap._order(u, i)))
        for u in ('ana', 'ben', 'cai', 'dev')
    }
    check('n7c different annotators walk a batch in different orders',
          len(orders) > 1)
    small = 'guard_small'
    os.makedirs(os.path.join(ap.BATCHES, small))
    with open(os.path.join(ap.BATCHES, small, 'manifest.jsonl'), 'w') as fh:
        fh.write(json.dumps(rows[0]) + '\n')
    ap.save(small, 0, [], True, by=WHO)
    fin = ap.next_frame(small, WHO)
    check('n7d a batch with every frame labelled says it is done', fin
          and fin.get('done') is True, f'{fin}')

    # ── n8 ──────────────────────────────────────────────────────────────
    got = ap.save(B, 0, [[20, 20, 90, 80]], False, by=WHO)
    rec = ledger()[-1]
    check(
        'n8 the row holds what was drawn and nothing else; the hint is not '
        'in it',
        got.get('ok') and set(rec) == {
            'image_id', 'batch', 'boxes', 'empty', 'w', 'h', 'by', 'at',
            'source'
        } and rec['boxes'] == [[20.0, 20.0, 90.0, 80.0]]
        and rec['source'] == ap.SOURCE, f'{rec}')

    # ── n9 ──────────────────────────────────────────────────────────────
    f0 = ap.frame(B, 0, WHO)
    check('n9 the hint is the seed box scaled from its frame to this one', f0
          and f0.get('hint') == [40.0, 30.0, 80.0, 90.0],
          f'{f0 and f0.get("hint")}')

    # ── n10 ─────────────────────────────────────────────────────────────
    k_rot = next(k for k, r in enumerate(rows)
                 if r['image_id'] == '9000000000003')
    got = ap.save(B, k_rot, [[10, 10, 60, 70]], False, by=WHO)
    check(
        'n10 a frame stored on its side is not served and cannot be '
        'labelled',
        ap.frame(B, k_rot, WHO) is None and not got.get('ok'), f'{got}')

    # ── n15 ─────────────────────────────────────────────────────────────
    H = 'guard_hidden'
    hdir = os.path.join(ap.BATCHES, H)
    os.makedirs(hdir)
    hrows = [dict(rows[0], role='addition'), dict(rows[1], role='background')]
    with open(os.path.join(hdir, 'manifest.jsonl'), 'w') as fh:
        fh.write(''.join(json.dumps(r) + '\n' for r in hrows))
    fa, fb, fn = ap.frame(H, 0, WHO), ap.frame(H, 1, WHO), ap.frame(B, 0, WHO)
    said = ' '.join(
        str((x or {}).get(k)) for x in (fa, fb)
        for k in ('intro', 'hint_title')).lower()
    check(
        'n15 a hidden-dog batch never says the detector drew nothing or that '
        'the hint is the gate\'s dog', fa and fb and 'drew nothing' not in said
        and 'second earlier' not in said and 'gate saw' not in said,
        said[:300])
    check(
        'n15b a frame with a confirmed dog asks for that dog too; a rejected '
        'frame offers "no dog"', fa and 'confirmed' in fa['intro'].lower()
        and fb and 'no dog' in fb['intro'].lower()
        and fa['intro'] != fb['intro'],
        f'{(fa or {}).get("intro")!r} / {(fb or {}).get("intro")!r}')
    check(
        'n15c a neighbour batch keeps its own sentence and tooltip', fn
        and 'drew nothing' in fn['intro'].lower()
        and 'second earlier' in fn['hint_title'].lower(),
        f'{(fn or {}).get("intro")!r}')
    with open(os.path.join(hdir, 'summary.json'), 'w') as fh:
        json.dump({'kind': 'neighbours'}, fh)
    check('n15d summary.json kind wins over what the manifest suggests',
          ap.batch_kind(H) == 'neighbours', f'{ap.batch_kind(H)}')
    os.remove(os.path.join(hdir, 'summary.json'))
    page = ap.page_html()
    check(
        'n15e the page shows the frame\'s own sentence and tooltip, and its '
        'fixed header claims nothing about the detector',
        'textContent=d.intro' in page and 'F.hint_title' in page
        and 'drew <b>nothing</b>' not in page
        and 'gate saw a dog about here' not in page,
        'the page still draws a fixed header or tooltip')

    # ── n16 ─────────────────────────────────────────────────────────────
    dsrc = open(os.path.join(REPO, 'tools', 'dashboard',
                             'dashboard.py')).read()
    at = dsrc.find("for _m in ('audit.py'")
    check(
        'n16 the dashboard restarts when annotate_page.py changes, so a fix to '
        'the page reaches annotators', at >= 0
        and "'annotate_page.py'" in dsrc[at:at + 200],
        'annotate_page.py is not in the source watcher list')

    # ── n11 ─────────────────────────────────────────────────────────────
    led = os.path.join(tmp, 'build_ledger.jsonl')
    base = {
        'boxes': [[10, 10, 60, 70]],
        'empty': False,
        'w': 400,
        'h': 300,
        'at': 1
    }
    with open(led, 'w') as fh:
        for r in (dict(base, image_id='1', by=WHO, source=fl.SOURCE),
                  dict(base, image_id='2', by='', source=fl.SOURCE),
                  dict(base, image_id='3', by=WHO, source='model_suggestion'),
                  dict(base, image_id='4', by=WHO, source=fl.SOURCE),
                  dict(base,
                       image_id='4',
                       by=WHO,
                       source=fl.SOURCE,
                       boxes=[],
                       empty=True)):
            fh.write(json.dumps(r) + '\n')
        fh.write('{"image_id": "5", "by"')
    got = fl.read(led)
    check(
        'n11 the build reads only rows the page wrote and a person signed, '
        'the latest for each frame',
        sorted(got) == ['1', '4'] and got['4']['empty'] is True,
        f'{sorted(got)}')
    check('n11b and page and build agree on what the page is called',
          fl.SOURCE == ap.SOURCE, f'{fl.SOURCE} vs {ap.SOURCE}')

    # ── n12 ─────────────────────────────────────────────────────────────
    t = fl.yolo({
        'w': 400,
        'h': 300,
        'empty': False,
        'boxes': [[100, 60, 200, 180], [0, 0, 40, 30]]
    })
    want = ('0 0.375000 0.400000 0.250000 0.400000\n'
            '0 0.050000 0.050000 0.100000 0.100000\n')
    check('n12 a row becomes YOLO text exactly', t == want, repr(t))
    check('n12b an empty frame becomes an empty label',
          fl.yolo({
              'w': 400,
              'h': 300,
              'empty': True,
              'boxes': []
          }) == '')
    bad_rows = {
        'empty with a box': {
            'w': 400,
            'h': 300,
            'empty': True,
            'boxes': [[1, 1, 9, 9]]
        },
        'nothing, not empty': {
            'w': 400,
            'h': 300,
            'empty': False,
            'boxes': []
        },
        'no size': {
            'empty': False,
            'boxes': [[1, 1, 9, 9]]
        },
        'zero size': {
            'w': 0,
            'h': 300,
            'empty': False,
            'boxes': [[1, 1, 9, 9]]
        },
        'NaN box': {
            'w': 400,
            'h': 300,
            'empty': False,
            'boxes': [[math.nan, 1, 9, 9]]
        },
        'flat box': {
            'w': 400,
            'h': 300,
            'empty': False,
            'boxes': [[5, 1, 5, 9]]
        },
    }
    let_in = [k for k, r in bad_rows.items() if fl.yolo(r) is not None]
    check('n12c an inconsistent row is nothing, not a label', not let_in,
          f'made labels of: {let_in}')

    # ── n13 ─────────────────────────────────────────────────────────────
    pj = {}
    for iid in ('a1', 'a2', 'a3', 'a4', 'a6'):
        pj[iid] = jpg(iid)
    pj['a6'] = jpg('a6', w=800)
    recs = {
        i: dict(base, image_id=i, by=WHO, source=fl.SOURCE)
        for i in ('a1', 'a2', 'a3', 'a4', 'a5', 'a6')
    }
    seq = {
        'a1': 's_train',
        'a2': 's_val',
        'a3': 's_train',
        'a5': 's_x',
        'a6': 's_x'
    }
    where = dict(pj, a5=os.path.join(tmp, 'gone', 'a5.jpg'))
    added, skipped = fl.plan(recs,
                             seq, {'s_val'}, {'a3'},
                             where,
                             size_of=fl.image_size)
    got = sorted(i for i, *_ in added)
    check(
        'n13 only a new, non-val frame whose file is the one labelled goes '
        'in, and an unresolved one goes to train', got == ['a1', 'a4']
        and skipped == {
            'already_in_dataset': 1,
            'val_sequence': 1,
            'no_file': 1,
            'bad_record': 0,
            'size_changed': 1
        }, f'added {got}, skipped {skipped}')

    # ── n14 ─────────────────────────────────────────────────────────────
    # build_dogdet_v3 run for real on a set small enough to read whole:
    # only what it asks of the harvest (sequences, sizes, where a jpg is)
    # and of the other ledgers is stood in for.
    import build_dogdet_v3 as v3
    v2, out = os.path.join(tmp, 'v2'), os.path.join(tmp, 'v3out')
    roots = os.path.join(tmp, 'grid')
    for split, members in (('train', ('t1', 't2')), ('val', ('v1', ))):
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(v2, d, split), exist_ok=True)
        for iid in members:
            jpg(iid, where=os.path.join(v2, 'images', split))
            with open(os.path.join(v2, 'labels', split, iid + '.txt'),
                      'w') as fh:
                fh.write('0 0.5 0.5 0.1 0.1\n')
    cell = os.path.join(roots, 'cellA', 'ground_animal_images')
    os.makedirs(cell)
    for iid in ('f1', 'f2', 'f3', 'f4', 'c1'):
        jpg(iid, where=cell)
    seqmap = {
        't1': 'S_t',
        't2': 'S_t',
        'v1': 'S_v',
        'f1': 'S_1',
        'f2': 'S_2',
        'f3': 'S_v',
        'f4': 'S_4',
        'c1': 'S_c'
    }
    two = [[100, 60, 200, 180], [0, 0, 40, 30]]
    recs = {
        'f1':
        dict(base, image_id='f1', by=WHO, source=fl.SOURCE),
        'f2':
        dict(base,
             image_id='f2',
             by=WHO,
             source=fl.SOURCE,
             boxes=[],
             empty=True),
        'f3':
        dict(base, image_id='f3', by=WHO, source=fl.SOURCE),
        'f4':
        dict(base, image_id='f4', by=WHO, source=fl.SOURCE, boxes=two),
    }
    stood_in = {
        k: getattr(v3, k)
        for k in ('resolve_sequences', 'frame_meta', '_grid_roots',
                  'confirmed_positive_ids', 'corrections')
    }
    real_read, real_argv = fl.read, sys.argv
    try:
        v3.resolve_sequences = lambda ids, con: {
            i: seqmap[i]
            for i in ids if i in seqmap
        }
        v3.frame_meta = lambda ids, con: ({
            i: (1, 400, 300, 'cellA', 'drv')
            for i in ids
        }, {
            (i, 0): (10.0, 10.0, 50.0, 50.0)
            for i in ids
        })
        v3._grid_roots = lambda: {'drv': roots}
        v3.confirmed_positive_ids = lambda: {'f4', 'c1'}
        v3.corrections = lambda: {}
        fl.read = lambda path=None: dict(recs)
        sys.argv = [
            'build_dogdet_v3', '--src', v2, '--out', out, '--val-target', '0'
        ]
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            v3.main()
        with open(os.path.join(out, 'manifest.json')) as fh:
            man = json.load(fh)
    except Exception as e:  # noqa: BLE001 -- a raise is a failed build
        man = {'raised': repr(e)}
    finally:
        for k, v in stood_in.items():
            setattr(v3, k, v)
        fl.read, sys.argv = real_read, real_argv

    def label(split, iid):
        try:
            with open(os.path.join(out, 'labels', split, iid + '.txt')) as fh:
                return fh.read()
        except OSError:
            return None

    check(
        'n14 a frame label goes into train as its boxes, and an empty one '
        'as an empty label',
        label('train', 'f1') == fl.yolo(recs['f1'])
        and label('train', 'f2') == ''
        and os.path.isfile(os.path.join(out, 'images', 'train', 'f1.jpg')),
        f'{man.get("raised") or (label("train", "f1"), label("train", "f2"))}')
    check(
        'n14b a frame of a val sequence is left out of both splits',
        label('train', 'f3') is None and label('val', 'f3') is None
        and man.get('frame_labels_skipped', {}).get('val_sequence') == 1,
        f'{man.get("frame_labels_skipped")}')
    check(
        'n14c a frame label replaces the confirmed-box addition of the same '
        'frame',
        label('train', 'f4') == fl.yolo(recs['f4'])
        and man.get('added_ids') == ['c1']
        and man.get('added_frame_label_ids') == ['f1', 'f2', 'f4'],
        f'added {man.get("added_ids")}, framed '
        f'{man.get("added_frame_label_ids")}, f4 {label("train", "f4")!r}')
    check(
        'n14e and the counts say so',
        man.get('added_frame_labels') == 3
        and man.get('frame_labels_empty') == 1
        and man.get('train') == 2 + 1 + 3
        and man.get('backgrounds_train') == 1, f'{man}'[:300])
    import build_dataset as bd
    spec = bd.STORES.get('frame_labels') or {}
    check(
        'n14d the ledger is a store every dogdet build digests',
        os.path.realpath(spec.get('ledger') or '') == os.path.realpath(
            fl.LEDGER) and 'dogdet' in spec.get('for', ()), f'{spec}')
finally:
    ap.BATCHES, ap.LEDGER_DIR, ap.LEDGER = real
    ap._cache.clear()
    shutil.rmtree(tmp, ignore_errors=True)

check('n0 the live ledger was not touched',
      stamp(LIVE) == live_before, f'{live_before} -> {stamp(LIVE)}')
try:
    with open(LIVE) as fh:
        leaked = MARK in fh.read()
except OSError:
    leaked = False
check('n0b and nothing this file invented is in it', not leaked)

if _fails:
    print(f'\n{len(_fails)} annotate check(s) FAILED: ' + '; '.join(_fails))
    sys.exit(1)
print('\nall clear: what a person drew is what the build takes, signed with '
      'their name')
