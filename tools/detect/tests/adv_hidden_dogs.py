#!/usr/bin/env python3
"""A frame where an independent model sees a dog nobody labelled is never
taught as silence, and the flag that says so is never a label.

    python tools/detect/tests/adv_hidden_dogs.py

  h1  a missing store flags nothing; a store that is there but unreadable, or
      not an `unverified` store, raises instead of flagging nothing
  h2  an import is stamped unverified, keeps the boxes in the frame's own
      pixels with the frame's size, the role and where it came from, and a
      second import of the same frame replaces it
  h3  an import keeps the store it replaces, byte for byte, under superseded/
  h4  an import refuses a role it does not know, a frame with no size and a
      flag with no source -- and writes nothing when it refuses
  h5  screen splits by the image id it is told to read
  h6  a batch takes only frames nobody has labelled whole, of the role asked,
      whose jpg is on disk; its seed is the most confident box, stamped
      unverified, and the Annotate page shows it as the hint
  h7  a batch is never written into a directory that exists
  h8  build_dogdet_v3 leaves a flagged addition out and lists it; ignores a
      flagged BACKGROUND there; --no-hidden-dogs keeps it; --leashed-
      additions drop leaves out a leashed addition and keep is the default,
      and the leash calls are read off the stage's shards and nothing else
  h9  build_detector_negatives drops a flagged frame from the draw, does not
      refill it, and lists it; --no-hidden-dogs keeps it

EVERY PATH IS REDIRECTED to a temp directory; the live store is
fingerprinted before and compared after.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

import hidden_dogs as hd  # noqa: E402

_fails = []


def check(label, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + label + ('' if ok else '  ' + detail))
    if not ok:
        _fails.append(label.split()[0])


def stamp(path):
    try:
        st = os.stat(path)
        return (st.st_size, st.st_mtime_ns)
    except OSError:
        return None


def raises(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    return None


LIVE = hd.STORE
live_before = stamp(LIVE)
tmp = tempfile.mkdtemp(prefix='hidden-dogs-')
try:
    store = os.path.join(tmp, 'hidden_dogs', 'flagged.json')

    # ── h1 ──────────────────────────────────────────────────────────────────
    try:
        empty = hd.read(store)
    except Exception as e:  # noqa: BLE001
        empty = repr(e)
    check('h1 a missing store flags nothing', empty == {}, f'{empty}')
    bad = os.path.join(tmp, 'bad.json')
    with open(bad, 'w') as fh:
        fh.write('{"frames": {"1": {}}')
    other = os.path.join(tmp, 'other.json')
    with open(other, 'w') as fh:
        json.dump({'tier': 'human', 'frames': {'1': {}}}, fh)
    check(
        'h1b an unreadable store, or one that is not unverified, raises',
        raises(lambda: hd.read(bad)) and raises(lambda: hd.read(other)),
        f'torn {raises(lambda: hd.read(bad))}, human '
        f'{raises(lambda: hd.read(other))}')

    # ── h2 ──────────────────────────────────────────────────────────────────
    sizes = {
        '3000001': (4000, 3000),
        '3000002': (8000, 4000),
        '3000003': (1920, 1080)
    }
    hd.import_flags([['3000001', [[10.04, 20.0, 30.0, 40.0, 0.81234, 16]]],
                     ['3000002', [[1, 2, 3, 4, 0.6], [5, 6, 7, 8, 0.9]]]],
                    sizes,
                    'addition',
                    'probe audit',
                    store,
                    now=100)
    with open(store, 'rb') as fh:
        first = fh.read()
    doc = json.loads(first)
    a1 = doc['frames'].get('3000001', {})
    check(
        'h2 an import is stamped unverified, with boxes, size, role, source',
        doc.get('tier') == 'unverified' and a1 == {
            'role': 'addition',
            'boxes': [[10.0, 20.0, 30.0, 40.0, 0.8123]],
            'w': 4000,
            'h': 3000,
            'source': 'probe audit',
            'at': 100
        }, f'{doc}'[:300])
    hd.import_flags([['3000001', [[11, 21, 31, 41, 0.7]]],
                     ['3000003', [[1, 1, 9, 9, 0.5]]]],
                    sizes,
                    'background',
                    'second audit',
                    store,
                    now=200)
    got = hd.read(store)
    check(
        'h2b a second import of the same frame replaces it, others stay',
        sorted(got) == ['3000001', '3000002', '3000003']
        and got['3000001']['boxes'] == [[11.0, 21.0, 31.0, 41.0, 0.7]]
        and got['3000001']['role'] == 'background'
        and got['3000002']['source'] == 'probe audit', f'{got}'[:300])

    # ── h3 ──────────────────────────────────────────────────────────────────
    sup = os.path.join(os.path.dirname(store), 'superseded')
    kept = sorted(os.listdir(sup)) if os.path.isdir(sup) else []
    same = False
    if len(kept) == 1:
        with open(os.path.join(sup, kept[0]), 'rb') as fh:
            same = fh.read() == first
    check(
        'h3 an import keeps the store it replaces, byte for byte',
        len(kept) == 1 and kept[0].startswith('flagged.')
        and kept[0].endswith('.json') and same, f'kept {kept}, same {same}')

    # ── h4 ──────────────────────────────────────────────────────────────────
    before = stamp(store)
    why = [
        raises(lambda: hd.import_flags([['3000001', [[1, 1, 2, 2, .5]]]],
                                       sizes, 'label', 'x', store)),
        raises(lambda: hd.import_flags([['zz', [[1, 1, 2, 2, .5]]]], sizes,
                                       'addition', 'x', store)),
        raises(lambda: hd.import_flags([['3000001', [[1, 1, 2, 2, .5]]]],
                                       sizes, 'addition', '', store)),
    ]
    check(
        'h4 an unknown role, a frame with no size and a flag with no source '
        'are refused, and nothing is written',
        all(why) and stamp(store) == before, f'raised {why}, store '
        f'{"unchanged" if stamp(store) == before else "CHANGED"}')

    # ── h5 ──────────────────────────────────────────────────────────────────
    items = [('3000001', 'x'), ('c9', 'y'), ('3000003', 'z')]
    kept_items, out = hd.screen(items, got, key=lambda x: x[0])
    check(
        'h5 screen splits by the id it is told to read',
        kept_items == [('c9', 'y')] and out == [('3000001', 'x'),
                                                ('3000003', 'z')],
        f'kept {kept_items}, out {out}')

    # ── h6 ──────────────────────────────────────────────────────────────────
    grid = os.path.join(tmp, 'grid')
    cell = os.path.join(grid, 'cellA', 'ground_animal_images')
    os.makedirs(cell)
    for iid in ('3000001', '3000002', '3000003', 'done'):
        open(os.path.join(cell, iid + '.jpg'), 'wb').close()
    flagged = {
        '3000002': got['3000002'],
        '3000003': got['3000003'],
        '3000001': dict(got['3000001'], role='addition'),
        'gone': dict(got['3000002']),
        'done': dict(got['3000002']),
    }
    meta = {
        i: (1, 0, 0, 'cellA', 'drv')
        for i in ('3000001', '3000002', '3000003', 'gone', 'done')
    }
    rows, seeds = hd.plan_batch(flagged, {'done': {}},
                                meta, {'3000002': 'S2'}, {'drv': grid},
                                role='addition')
    ids = sorted(r['image_id'] for r in rows)
    check(
        'h6 a batch takes unlabelled frames of the role asked whose jpg is '
        'on disk', ids == ['3000001', '3000002'] and all(
            os.path.basename(r['jpg']) == r['image_id'] + '.jpg'
            for r in rows), f'picked {ids}')
    check(
        'h6b its seed is the most confident box, stamped unverified',
        seeds.get('3000002', {}).get('box') == [5.0, 6.0, 7.0, 8.0]
        and seeds['3000002'].get('tier') == 'unverified'
        and (seeds['3000002']['w'], seeds['3000002']['h']) == (8000, 4000),
        f'{seeds.get("a2")}')
    import annotate_page as ap
    batches = os.path.join(tmp, 'label_batches')
    d = hd.write_batch('hidden_probe_001', rows, seeds, {'probe': 1}, batches)
    real_batches = ap.BATCHES
    try:
        ap.BATCHES = batches
        ap._cache.clear()
        man = ap.manifest('hidden_probe_001')
        row = next(r for r in man if r['image_id'] == '3000002')
        hint = ap._hint('hidden_probe_001', row, 4000, 2000)
    except Exception as e:  # noqa: BLE001
        man, hint = None, repr(e)
    finally:
        ap.BATCHES = real_batches
        ap._cache.clear()
    check(
        'h6c the Annotate page reads the batch and shows the seed as its '
        'hint', man is not None and len(man) == 2
        and hint == [2.5, 3.0, 3.5, 4.0], f'manifest {man}, hint {hint}')

    # ── h7 ──────────────────────────────────────────────────────────────────
    check(
        'h7 a batch is never written into a directory that exists',
        raises(lambda: hd.write_batch('hidden_probe_001', [], {}, {}, batches))
        == 'FileExistsError'
        and os.path.getsize(os.path.join(d, 'manifest.jsonl')) > 0,
        'an existing batch was written over')

    # ── h8 ──────────────────────────────────────────────────────────────────
    import build_dogdet_v3 as v3
    import frame_labels as fl
    v2 = os.path.join(tmp, 'v2')
    for split, members in (('train', ('t1', )), ('val', ('v1', ))):
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(v2, sub, split), exist_ok=True)
        for iid in members:
            open(os.path.join(v2, 'images', split, iid + '.jpg'), 'wb').close()
            with open(os.path.join(v2, 'labels', split, iid + '.txt'),
                      'w') as fh:
                fh.write('0 0.5 0.5 0.1 0.1\n')
    for iid in ('p1', 'p2', 'p3', 'p4'):
        open(os.path.join(cell, iid + '.jpg'), 'wb').close()
    v3store = os.path.join(tmp, 'v3store.json')
    hd.import_flags([['p1', [[1, 1, 9, 9, .9]]]], {'p1': (400, 300)},
                    'addition', 'probe', v3store)
    hd.import_flags([['p2', [[1, 1, 9, 9, .9]]]], {'p2': (400, 300)},
                    'background', 'probe', v3store)
    seqmap = {
        't1': 'S_t',
        'v1': 'S_v',
        'p1': 'S1',
        'p2': 'S2',
        'p3': 'S3',
        'p4': 'S4'
    }
    stood_in = {
        k: getattr(v3, k)
        for k in ('resolve_sequences', 'frame_meta', '_grid_roots',
                  'confirmed_positive_ids', 'corrections', 'leashed_ids')
    }
    real_fl = fl.read

    def build(*extra):
        out = os.path.join(tmp, 'v3out_' + str(len(os.listdir(tmp))))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                v3.main([
                    '--src', v2, '--out', out, '--val-target', '0',
                    '--hidden-dogs', v3store
                ] + list(extra))
            with open(os.path.join(out, 'manifest.json')) as fh:
                return json.load(fh)
        except Exception as e:  # noqa: BLE001 -- a raise is a failed build
            return {'raised': repr(e)}

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
        v3._grid_roots = lambda: {'drv': grid}
        v3.confirmed_positive_ids = lambda: {'p1', 'p2', 'p3', 'p4'}
        v3.corrections = lambda: {}
        v3.leashed_ids = lambda ids, con: {'p3'} & set(ids)
        fl.read = lambda path=None: {}
        on = build()
        off = build('--no-hidden-dogs')
        drop = build('--leashed-additions', 'drop')
    finally:
        for k, v in stood_in.items():
            setattr(v3, k, v)
        fl.read = real_fl
    check(
        'h8 a flagged addition is left out of the build and listed',
        on.get('added_ids') == ['p2', 'p3', 'p4']
        and on.get('hidden_dogs', {}).get('additions_left_out') == ['p1']
        and on.get('additions_skipped', {}).get('hidden_dog') == 1,
        f'{on.get("raised") or (on.get("added_ids"), on.get("hidden_dogs"))}')
    check(
        'h8b --no-hidden-dogs keeps it, and says no store was read',
        off.get('added_ids') == ['p1', 'p2', 'p3', 'p4']
        and off.get('hidden_dogs', {}).get('store') is None,
        f'{off.get("raised") or off.get("added_ids")}')
    check(
        'h8c leashed additions are kept unless asked, and dropped when asked',
        on.get('leashed_additions') == {
            'mode': 'keep',
            'left_out': []
        } and drop.get('added_ids') == ['p2', 'p4']
        and drop.get('leashed_additions') == {
            'mode': 'drop',
            'left_out': ['p3']
        }, f'keep {on.get("leashed_additions")}, drop '
        f'{drop.get("raised") or drop.get("added_ids")} '
        f'{drop.get("leashed_additions")}')

    # ── h8d ─────────────────────────────────────────────────────────────────
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    lrepo = os.path.join(tmp, 'leashrepo')
    ldir = os.path.join(lrepo, 'data', 'leash')
    os.makedirs(os.path.join(ldir, 'archive', 'old'))

    def lrow(path, iid):
        pq.write_table(
            pa.table({
                'image_id': [iid],
                'det_idx': [0],
                'label': ['leashed']
            }), path)

    lrow(os.path.join(ldir, 'leash-00000.parquet'), 'L1')
    lrow(os.path.join(ldir, 'work.parquet'), 'W1')
    lrow(os.path.join(ldir, 'archive', 'old', 'leash-00000.parquet'), 'A1')
    real_repo = v3.REPO
    try:
        v3.REPO = lrepo
        got_l = v3.leashed_ids(['L1', 'W1', 'A1'], duckdb.connect())
    except Exception as e:  # noqa: BLE001
        got_l = repr(e)
    finally:
        v3.REPO = real_repo
    check(
        'h8d the leash calls come from the stage\'s shards only -- not its '
        'work list, not an archived run', got_l == {'L1'}, f'{got_l}')

    # ── h9 ──────────────────────────────────────────────────────────────────
    import build_detector_negatives as neg
    import rebuild_crop_dataset as rcd
    src = os.path.join(tmp, 'negsrc')
    for split in ('train', 'val'):
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(src, sub, split))
    for k in range(10):
        iid = f'{1000000 + k}'
        open(os.path.join(src, 'images', 'train', iid + '.jpg'), 'wb').close()
        with open(os.path.join(src, 'labels', 'train', iid + '.txt'),
                  'w') as fh:
            fh.write('0 0.5 0.5 0.1 0.1\n')
    cands = [f'{2000000 + k}' for k in range(4)]
    origs = {}
    for iid in cands:
        origs[iid] = os.path.join(cell, iid + '.jpg')
        open(origs[iid], 'wb').close()
    negstore = os.path.join(tmp, 'negstore.json')
    saved = {
        k: getattr(neg, k)
        for k in ('flagged_ids', 'detections_per_image', 'resolve_originals')
    }
    real_seq = rcd.resolve_sequences

    def negbuild(*extra):
        out = os.path.join(tmp, 'negout_' + str(len(os.listdir(tmp))))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                neg.main([
                    '--src', src, '--out', out, '--target-frac', '0.23',
                    '--hidden-dogs', negstore, '--execute'
                ] + list(extra))
            with open(os.path.join(out, 'negatives_manifest.json')) as fh:
                return json.load(fh), sorted(
                    os.listdir(os.path.join(out, 'labels', 'train')))
        except Exception as e:  # noqa: BLE001
            return {'raised': repr(e)}, []

    try:
        neg.flagged_ids = lambda ledger: {i: {} for i in cands}
        neg.detections_per_image = lambda ids, repo: {i: 1 for i in ids}
        neg.resolve_originals = lambda ids, rf, repo: {
            i: origs[i]
            for i in ids
        }
        rcd.resolve_sequences = lambda ids, repo, py: {i: 'N' + i for i in ids}
        # 10 train frames, no background: 0.23 draws 3 of the 4 candidates
        oman, _ = negbuild('--no-hidden-dogs')
        drawn = oman.get('added_image_ids') or []
        flag = drawn[0] if drawn else cands[0]
        hd.import_flags([[flag, [[1, 1, 9, 9, .9]]]], {flag: (10, 10)},
                        'background', 'probe', negstore)
        nman, nfiles = negbuild()
    finally:
        for k, v in saved.items():
            setattr(neg, k, v)
        rcd.resolve_sequences = real_seq
    want = [i for i in drawn if i != flag]
    check(
        'h9 a flagged frame is dropped from the draw, not refilled, and '
        'listed',
        len(drawn) == 3 and nman.get('added_image_ids') == want
        and nman.get('hidden_dogs', {}).get('left_out') == [flag]
        and flag + '.txt' not in nfiles and all(i + '.txt' in nfiles
                                                for i in want),
        f'drawn {drawn}, {nman.get("raised") or nman.get("added_image_ids")} '
        f'{nman.get("hidden_dogs")}')
    check('h9b --no-hidden-dogs keeps it, and says no store was read',
          flag in drawn and oman.get('hidden_dogs', {}).get('store') is None,
          f'{oman.get("raised") or oman.get("added_image_ids")}')

finally:
    shutil.rmtree(tmp, ignore_errors=True)

check('h0 the live store was not touched',
      stamp(LIVE) == live_before, f'{live_before} -> {stamp(LIVE)}')

if _fails:
    print(f'\n{len(_fails)} hidden-dog check(s) FAILED: ' + '; '.join(_fails))
    sys.exit(1)
print('\nall clear: a frame with a dog nobody labelled is never taught as '
      'silence, and the flag is never a label')
