#!/usr/bin/env python3
"""pick_label_frames.py offers only frames that are safe to label for training.

    python tools/detect/tests/adv_pick_label_frames.py

The tool chooses street frames for people to label whole. A wrong choice is
not a crash, it is a quiet defect in the next model: a frame from a val
sequence labelled for training leaks val into train, a frame already labelled
is time spent twice, a squashed panorama is a label the detector cannot learn
from. So each exclusion is driven here against tables built by hand, where
the right answer is known exactly:

  k1  the closest undetected neighbour of a sure dog is picked, one per
      sequence
  k2  a sequence that holds a held-out frame gives up nothing
  k3  a frame the detector drew on, a panorama, a spherical or over-wide
      frame, or one already known elsewhere, is never offered
  k4  a neighbour outside --window places or --seconds is not a neighbour
  k5  --per-cell caps how many frames one cell gives up
  k6  the same batch name picks the same frames; a new one picks afresh
  k7  plan() refuses to rebuild over an existing batch
  k8  stage() copies nothing if any file would clash, and otherwise carries
      the capture time onto the copy
"""
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:
    import duckdb
except ImportError:
    print('SKIP: duckdb is not importable here -- nothing checked which '
          'frames get offered for labelling')
    sys.exit(0)

import pick_label_frames as plf  # noqa: E402

_fails = []


def check(label, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + label + ('' if ok else '  ' + detail))
    if not ok:
        _fails.append(label)


def world(extra_seqs=0):
    """A hand-built harvest. Every frame's fate is decided below."""
    meta, img, seeds = [], [], []

    def frame(iid,
              seq,
              ts,
              n_det=0,
              w=1280,
              cell='c_A',
              pano=False,
              ctype='perspective'):
        meta.append((iid, seq, ts, pano, ctype))
        img.append((iid, n_det, w, 960, cell, 'drv'))

    # A: seed at ts 10; neighbours at 1s, 2s, 3s either side
    for k, ts in enumerate((7, 8, 9, 10, 11, 12, 13)):
        frame('10000000000%02d' % k,
              'A',
              ts,
              n_det=1 if ts == 10 else 0,
              cell='c_A')
    seeds.append(('1000000000003', 40.0))
    # B: a held sequence -- its perfect neighbour must never be offered
    frame('2000000000000', 'B', 50, n_det=1, cell='c_B')
    frame('2000000000001', 'B', 51, cell='c_B')
    seeds.append(('2000000000000', 40.0))
    # C: every neighbour disqualified, each for a different reason
    frame('3000000000000', 'C', 100, n_det=1, cell='c_C')
    frame('3000000000001', 'C', 101, n_det=2, cell='c_C')  # detector drew
    frame('3000000000002', 'C', 102, pano=True, cell='c_C')  # panorama
    frame('3000000000003', 'C', 99, w=8000, cell='c_C')  # too wide
    frame('3000000000004', 'C', 98, ctype='spherical', cell='c_C')
    frame('3000000000005', 'C', 103, cell='c_C')  # known already
    seeds.append(('3000000000000', 40.0))
    # D: neighbours too far in time or in place
    frame('4000000000000', 'D', 200, n_det=1, cell='c_D')
    frame('4000000000001', 'D', 210, cell='c_D')  # 10 s away
    for k in range(2, 6):  # 5 places away
        frame('40000000000%02d' % k, 'D', 200 - k * 0.1, n_det=1, cell='c_D')
    frame('4000000000009', 'D', 199.3, cell='c_D')
    seeds.append(('4000000000000', 40.0))
    # E, F: two good sequences in one cell -- the cap keeps one
    for s in 'EF':
        frame('5%s00000000000' % ('0' if s == 'E' else '1'),
              s,
              300,
              n_det=1,
              cell='c_EF')
        frame('5%s00000000001' % ('0' if s == 'E' else '1'),
              s,
              301,
              cell='c_EF')
        seeds.append(('5%s00000000000' % ('0' if s == 'E' else '1'), 40.0))
    # many one-candidate sequences in their own cells, for the salt check
    for k in range(extra_seqs):
        seed, nb = '6%012d' % (2 * k), '6%012d' % (2 * k + 1)
        frame(seed, 'X%d' % k, 1000 + k, n_det=1, cell='cx%d' % k)
        frame(nb, 'X%d' % k, 1000.5 + k, cell='cx%d' % k)
        seeds.append((seed, 40.0))
    con = duckdb.connect()
    con.execute('create table meta(image_id varchar, seq varchar, ts double, '
                'pano boolean, ctype varchar)')
    con.executemany('insert into meta values (?,?,?,?,?)', meta)
    con.execute('create table img(image_id varchar, n_det int, w int, h int, '
                'cell varchar, drive varchar)')
    con.executemany('insert into img values (?,?,?,?,?,?)', img)
    con.execute('create table seeds(image_id varchar, bw double)')
    con.executemany('insert into seeds values (?,?)', seeds)
    con.execute("create table excl as select '3000000000005' image_id")
    con.execute("create table held_seq as select 'B' seq")
    return con


con = world()
picks, stats = plf.select(con,
                          50,
                          'guard',
                          window=3,
                          seconds=5,
                          max_width=4096,
                          per_cell=1)
got = {p['image_id'] for p in picks}
by_seq = {}
for p in picks:
    by_seq.setdefault(p['seq'], []).append(p['image_id'])

# ── k1 ────────────────────────────────────────────────────────────────────
a = by_seq.get('A', [])
check(
    'k1 one frame per sequence, and it is a neighbour a second from the '
    'seed',
    len(a) == 1 and a[0] in ('1000000000002', '1000000000004'),
    f'sequence A gave {a}')

# ── k2 ────────────────────────────────────────────────────────────────────
check('k2 a held-out sequence gives up nothing, even a perfect neighbour', 'B'
      not in by_seq, f'offered {by_seq.get("B")}')

# ── k3 ────────────────────────────────────────────────────────────────────
check(
    'k3 drawn-on, panorama, over-wide, spherical and already-known frames '
    'are never offered', 'C' not in by_seq, f'offered {by_seq.get("C")}')

# ── k4 ────────────────────────────────────────────────────────────────────
check('k4 a frame 10 s away or 5 places away is not a neighbour', 'D'
      not in by_seq, f'offered {by_seq.get("D")}')
wide = plf.select(world(),
                  50,
                  'guard',
                  window=5,
                  seconds=5,
                  max_width=4096,
                  per_cell=5)[0]
check('k4b widening --window lets the 5-places-away frame in', '4000000000009'
      in {p['image_id']
          for p in wide})

# ── k5 ────────────────────────────────────────────────────────────────────
ef = [s for s in ('E', 'F') if s in by_seq]
check('k5 --per-cell 1 takes one of two good sequences in the same cell',
      len(ef) == 1, f'took {ef}')
both = plf.select(world(), 50, 'guard', per_cell=2)[0]
check('k5b --per-cell 2 takes both', {'E', 'F'} <= {p['seq'] for p in both})

# ── k6 ────────────────────────────────────────────────────────────────────
one = [p['image_id'] for p in plf.select(world(40), 10, 'batch_one')[0]]
again = [p['image_id'] for p in plf.select(world(40), 10, 'batch_one')[0]]
other = [p['image_id'] for p in plf.select(world(40), 10, 'batch_two')[0]]
check('k6 the same batch name picks exactly the same frames', one == again
      and len(one) == 10)
check('k6b a new batch name picks a different set',
      set(one) != set(other),
      'two names drew identical sets from 40+ candidates')

# ── k7, k8 ────────────────────────────────────────────────────────────────
tmp = tempfile.mkdtemp(prefix='pickframes-')
try:
    plf.BATCHES = os.path.join(tmp, 'batches')
    os.makedirs(os.path.join(plf.BATCHES, 'old_batch'))
    try:
        plf.plan(types.SimpleNamespace(batch='old_batch'))
        refused = 'it went ahead'
    except SystemExit as e:
        refused = str(e)
    except Exception as e:  # noqa: BLE001 -- going ahead and then falling
        # over is still going ahead: the refusal has to come FIRST
        refused = 'it went ahead and then raised %r' % (e, )
    check('k7 plan refuses to rebuild over an existing batch', 'never rebuilt'
          in refused, f'said {refused!r}')

    ls = os.path.join(tmp, 'ls')
    os.makedirs(os.path.join(ls, 'new_images'))
    src = os.path.join(tmp, 'src')
    os.makedirs(src)
    b = os.path.join(plf.BATCHES, 'b1')
    os.makedirs(b)
    rows = []
    for k in range(3):
        f = os.path.join(src, '70000000000%02d.jpg' % k)
        with open(f, 'wb') as fh:
            fh.write(b'\xff\xd8' + bytes([k]) * 50)
        rows.append({
            'image_id': '70000000000%02d' % k,
            'jpg': f,
            'captured': 1500000000 + k
        })
    with open(os.path.join(b, 'manifest.jsonl'), 'w') as fh:
        fh.write(''.join(json.dumps(r) + '\n' for r in rows))
    # a clash: one of the three is already in the intake directory
    shutil.copyfile(
        rows[1]['jpg'],
        os.path.join(ls, 'new_images', rows[1]['image_id'] + '.jpg'))
    try:
        plf.stage(types.SimpleNamespace(batch='b1', ls_dir=ls))
        clash = ''
    except SystemExit as e:
        clash = str(e)
    staged = sorted(os.listdir(os.path.join(ls, 'new_images')))
    check(
        'k8 one clash and stage copies NOTHING', 'nothing was copied' in clash
        and staged == [rows[1]['image_id'] + '.jpg'],
        f'said {clash!r}, dir {staged}')
    os.remove(os.path.join(ls, 'new_images', rows[1]['image_id'] + '.jpg'))
    plf.stage(types.SimpleNamespace(batch='b1', ls_dir=ls))
    mt = {
        r['image_id']:
        int(
            os.path.getmtime(
                os.path.join(ls, 'new_images', r['image_id'] + '.jpg')))
        for r in rows
    }
    same = all(
        open(r['jpg'], 'rb').read() == open(
            os.path.join(ls, 'new_images', r['image_id'] +
                         '.jpg'), 'rb').read() for r in rows)
    check(
        'k8b a clean stage copies every frame byte for byte, stamped with '
        'its capture time', same and all(mt[r['image_id']] == r['captured']
                                         for r in rows), f'mtimes {mt}')
    check(
        'k8c no .part file is left behind', not [
            f for f in os.listdir(os.path.join(ls, 'new_images'))
            if f.endswith('.part')
        ])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if _fails:
    print(f'\n{len(_fails)} frame-picking check(s) FAILED: ' +
          '; '.join(_fails))
    sys.exit(1)
print('\nall clear: only safe, unlabelled, undetected neighbours are offered')
