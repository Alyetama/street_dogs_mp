#!/usr/bin/env python3
"""The audit pages sample from gen 2 and never attach anything to a box of
another generation.

A key is '<image_id>#<det_idx>' and det_idx is numbered per detector
generation: of the gen-1 gate verdict keys, three in four exist in the gen-2
store and most of those name another box. So the fixtures here are built round
ONE key, 100#0, that is a different box in each generation -- (10,10,60,60)
over a RED patch of the frame in gen 1, (300,300,380,380) over a BLUE patch in
gen 2 -- and every check of a picture reads the colour of what it got back, so
a crop, frame, dataset file or redraw of the wrong generation cannot pass by
having the right file name.

Two fake repos, both in temp dirs (fn_audit.REPO, audit.REPO, crop_sources.REPO
and build_dataset.REPO point there and the box corrections file is a temp
file), so nothing here can reach data/:

  the BUILD repo   a gen-2 detect store whose det shards hold one key twice
                   (agreeing) in two files of different schemas, a gen-2 gate
                   store over it, and a legacy gen-1 audit dir -- for what
                   `fn_audit.py build/export/verify` refuse and prove
  the PAGE repo    gen-1 and gen-2 stores, frames, a legacy gen-1 audit of
                   each stage and a ready gen-2 pool of each -- driven through
                   the real request handler (BoardHandler._audit_get /
                   do_POST, the gate stubbed as already passed)

  G1  paths(s, gen=1) is today's dict, key for key; every (stage, gen) dir is
      its own; none matches crop_sources.FORBIDDEN_PATH_RE; paths('gate')
      with no gen is a TypeError and a bad gen a ValueError
  G2  build refuses: no --gen, gen 1, a plan of another gen, a plan with no
      detect_root (and the ledger pin is never opened), a store that says
      another gen, a non-empty out dir, two models, a row count that is not
      the plan's, an unknown protocol, a work.parquet holding a key twice --
      each before a pool is visible
  G3  the full geometry check: a box moved by 1 px, or a key missing from the
      det shards, fails the build and leaves no pool; duplicate det rows that
      agree, in det files of different schemas, pass
  G4  a stop between the two renames leaves pool.json and no parquet: "not
      ready", unlisted, and a rebuild refused; a gen-2 dir with no pool.json
      (or one of another gen) is no pool, never gen 1; a parquet of another
      size/mtime is refused on every read; `verify` catches a sha mismatch
  G5  a verdict posted to gen 2 lands only in the gen-2 ledger and dataset,
      tagged gen and pool, with the gen-2 box and the PAGE's p_dog (the
      client's is ignored); the dataset file is the gen-2 (blue) cut. A
      wrong pool, a key on no page ("None" included), a missing gen or pool
      write nothing (409 / 400), nor does a pool replaced mid-request
  G6  a redraw posted with the gen-1 frame's box_sig under gen 2 is refused;
      with the gen-2 one it lands tagged gen 2 + pool; corrections() of the
      gen-1 ctx never sees it, nor does crop_sources' reader of gen 1, and
      neither gen-2 reader takes another gen-2 pool's row
  G7  read_verdicts raises -- never skips -- on an untagged, gen-1 or
      other-pool row in a gen-2 ledger, and on a gen-2 row in gen 1's
  G8  /audit/crop/gate/g2/100_0.jpg is the blue cut, the legacy key-only URL
      the red gen-1 one; the frame meta names gen, pool and box_sig, and a
      frame asked for under another pool is a 409; the page's cropSrc puts
      the generation in the path (run under node when there is one) and
      every fetch it makes names gen and pool
  G9  the gen-2 leash tile (crops/) is cut at pad050 (leash_vit.rect_pad),
      its full/ at cut_crops' pad012 -- and crop_sources.framing_check_audit
      passes on that full/; `build` gives a pad050@448 leash store's pool
      the pad050 view_frame
  G10 a draw on the frozen gen 1 is a 409 and writes nothing, as is one on a
      gen-2 pool whose store is now judged by another model; api_page past
      the last gen-1 page answers "closed" and draws nothing; prefetch on a
      frozen gen queues nothing; an unknown stage on a POST is a 400; locks
      and prefetch are keyed (stage, gen); the page is READONLY on gen 1 and
      live on gen 2, and with no gen-2 pool the page says what to build
      instead of falling back
  G11 `export --gen 2` writes only audit_finds_gen2 (the gen-1 dataset is
      byte-identical), under a marker naming the pool, every manifest row
      naming gen, pool, detect_root, box and the pool's model; a marker of
      another pool, a marked-less dir with files, and `export --gen 1` are
      refused; so are a verdict whose key is not in the pool and one whose
      recorded box is not its pool box
  G12 crop_sources.audit_candidates(detect_root=gen-1 store) reads only the
      gen-1 audit, (gen-2 store) only the gen-2 one -- each row's crop, box
      and provenance its own generation's -- and a store the pool was not
      cut from is SystemExit; rows of two generations stop apply_box_vetoes
      and fail rebuild_crop_dataset.check_rows; build_dataset.stores(g)
      digests exactly fa.paths(..., gen=g)
  G13 done_by(user, 'gate') counts that person's answers in gen 1 AND gen 2
  G14 source scan: every fn_audit call in audit.py that is per generation
      names it, every audit.py function that reads or writes a pool takes a
      ctx, and audit.py never resolves fa.paths itself; no path to an audit
      dir or dataset is spelled out in llm_annotate, eval_leash,
      build_dataset, crop_sources or build_dogdet_v3
  G1' every gen-1 file of the page repo is byte-identical after all the
      gen-2 traffic; a verdict on the frozen gen 1 then still lands there,
      tagged gen 1, with the gen-1 box
  W1-W11 what the reviews of the first version found (tail_checks): no
      redraw or re-cut of a key on no page, foreign ledger/page/drawn rows
      refused before anything is written, a dataset dir of another pool
      left alone, "None" refused on gen 1, the page index incremental, the
      work count per generation inside _counting(), vetoes per generation,
      v3's redraws only on their pool's own store, and the defaults and
      install-window behaviour of the routes
  W12-W16 guards the fixture never made matter (gap_checks): a draw on a
      store whose model cannot be read is a 409; a dataset marker of this
      pool_id but another detect_root/stage/gen is refused by place() and
      export; pool.json swapped for one of the same size and mtime_ns is
      read afresh; the page-index memo is per generation even when two
      page dirs share a stamp; a gen-2 crop is cut from its pool.json
      roots, not the plan's
  m   mutants: each guard removed (a textual replace that must apply exactly
      once to the source under test) is caught

AUDIT_PY, DASH_PY and FA_PY point the test at edited copies.

    python tools/detect/tests/adv_audit_gen2.py      (mp14 env)
"""

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DASHD = os.path.join(REPO, 'tools', 'dashboard')
DETECT = os.path.join(REPO, 'tools', 'detect')
AUDIT_PY = os.environ.get('AUDIT_PY') or os.path.join(DASHD, 'audit.py')
DASH_PY = os.environ.get('DASH_PY') or os.path.join(DASHD, 'dashboard.py')
FA_PY = os.environ.get('FA_PY') or os.path.join(DETECT, 'fn_audit.py')
CS_PY = os.path.join(DETECT, 'crop_sources.py')
BD_PY = os.path.join(DETECT, 'build_dataset.py')
RB_PY = os.path.join(DETECT, 'rebuild_crop_dataset.py')
LLM_PY = os.path.join(DETECT, 'llm_annotate.py')
EL_PY = os.path.join(DETECT, 'eval_leash.py')
V3_PY = os.path.join(DETECT, 'build_dogdet_v3.py')
sys.path.insert(0, DASHD)
sys.path.insert(0, DETECT)

# every source a mutant can edit, by the name MUTANTS uses
SOURCES = {'fa': FA_PY, 'audit': AUDIT_PY, 'dash': DASH_PY, 'cs': CS_PY,
           'bd': BD_PY, 'rb': RB_PY, 'llm': LLM_PY, 'el': EL_PY, 'v3': V3_PY}

FAILS = []
TEMPS = []          # the mutated sources' temp dirs, removed at the end
RED, BLUE = (220, 20, 20), (20, 20, 220)
G1_BOX = (10.0, 10.0, 60.0, 60.0)
G2_BOX = (300.0, 300.0, 380.0, 380.0)
SESSION = {'id': 1, 'username': 'gen2tester', 'role': 'admin'}
MEM = '1GB'


def check(name, cond, detail=''):
    detail = ' | '.join(str(detail).split('\n')) if detail else ''
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail[:400]}' if detail else ''))
    if not cond:
        FAILS.append(name)


def _q(p):
    return str(p).replace("'", "''")


def _parquet(path, rows, cols):
    import duckdb
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{MEM}'")
        con.execute('CREATE TABLE t(%s)' % ', '.join(cols))
        con.executemany('INSERT INTO t VALUES (%s)'
                        % ', '.join('?' * len(cols)), rows)
        con.execute(f"COPY t TO '{_q(path)}' (FORMAT PARQUET)")
    finally:
        con.close()


def tree_sha(root):
    """{relpath: sha256} of every file under `root` ({} if there is none)."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            with open(p, 'rb') as fh:
                out[os.path.relpath(p, root)] = hashlib.sha256(
                    fh.read()).hexdigest()
    return out


def colour(path_or_bytes):
    """'red', 'blue' or 'other' -- the mean colour of a picture."""
    from PIL import Image, ImageStat
    try:
        if isinstance(path_or_bytes, (bytes, bytearray)):
            im = Image.open(io.BytesIO(path_or_bytes))
        else:
            im = Image.open(path_or_bytes)
    except Exception:                         # noqa: BLE001 - no picture
        return 'missing'
    r, g, b = ImageStat.Stat(im.convert('RGB')).mean
    if r > 150 and b < 90:
        return 'red'
    if b > 150 and r < 90:
        return 'blue'
    return 'other'


def lines(path):
    try:
        with open(path) as fh:
            return [json.loads(x) for x in fh if x.strip()]
    except OSError:
        return []


def _status(root, gen):
    os.makedirs(os.path.join(root, 'shards'), exist_ok=True)
    with open(os.path.join(root, 'status.json'), 'w') as fh:
        json.dump({'gen': gen}, fh)


# ── the BUILD repo ──────────────────────────────────────────────────────────
# 100#0 is (300,300,380,380) in gen 2 and (10,10,60,60) in gen 1, as on the
# page repo; 100#1 is gen 1's box again under gen 2's numbering.
DET2 = [('100', 0, 300., 300., 380., 380., 0.9),
        ('100', 1, 10., 10., 60., 60., 0.8),
        ('200', 0, 50., 50., 150., 150., 0.7),
        ('300', 0, 5., 5., 25., 45., 0.6)]
DET1 = [('100', 0, 10., 10., 60., 60., 0.85)]


def build_repo(stage='gate', protocol=('full', 640), models=('m2',),
               rows_delta=0, plan_gen=2, detect_root=True, store_gen=2,
               moved=None, drop_det=None, extra_work_dup=False,
               shard_dup=False):
    """A fake repo for `fn_audit.py build`: (repo, {'det1', 'det2'}).

    `moved` is (image_id, det_idx, delta) -- x1 of every det copy of that key
    moved by delta -- or (image_id, det_idx, field, delta, which), `which`
    being 'a', 'b' (the one det file) or 'both'. 100#0 is in both det files,
    so moving it in one is a disagreeing duplicate behind an agreeing one.
    The gen-2 store also holds a gen=0001 det file whose 200#0 is elsewhere:
    a check that read every gen=* partition would see it disagree.
    `shard_dup` judges 100#0 twice in the stage store (and plans for it)."""
    import duckdb
    repo = tempfile.mkdtemp(prefix='adv_audit_gen2_build_')
    d2 = os.path.join(repo, 'stores', 'det2')
    d1 = os.path.join(repo, 'stores', 'det1')
    _status(d2, store_gen)
    _status(d1, 1)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEM}'")
    try:
        # gen-2 det shards: two files of DIFFERENT schemas (the second has no
        # cell/drive, as some real ones do), and the first key in both
        det = [r for r in DET2 if r[:2] != drop_det]
        if moved and len(moved) == 3:
            moved = (moved[0], moved[1], 'x1', moved[2], 'both')
        cols = ('x1', 'y1', 'x2', 'y2', 'conf')

        def move(rs, part):
            if not moved or moved[4] not in (part, 'both'):
                return rs
            out = []
            for i, k, x1, y1, x2, y2, c in rs:
                v = dict(zip(cols, (x1, y1, x2, y2, c)))
                if (i, k) == tuple(moved[:2]):
                    v[moved[2]] += moved[3]
                out.append((i, k) + tuple(v[c_] for c_ in cols))
            return out
        p = os.path.join(d2, 'shards', 'gen=0002', 'region=R', 'cell=C',
                         'drive=d')
        os.makedirs(p)

        def vals(rs, extra):
            return ','.join(
                f"({i}::UBIGINT, {k}::UTINYINT, {c}::FLOAT, {x1}::FLOAT, "
                f"{y1}::FLOAT, {x2}::FLOAT, {y2}::FLOAT{extra})"
                for i, k, x1, y1, x2, y2, c in rs)
        con.execute(
            f"COPY (SELECT * FROM (VALUES "
            f"{vals(move(det[:2], 'a'), ", 'C', 'd', '0002'")}"
            f") t(image_id, det_idx, conf, x1, y1, x2, y2, cell, drive, gen)) "
            f"TO '{_q(os.path.join(p, 'a.det.parquet'))}' (FORMAT PARQUET)")
        con.execute(
            f"COPY (SELECT * FROM (VALUES "
            f"{vals(move(det[2:] + det[:1], 'b'), ", '0002'")}) "
            f"t(image_id, det_idx, conf, x1, y1, x2, y2, gen)) "
            f"TO '{_q(os.path.join(p, 'b.det.parquet'))}' (FORMAT PARQUET)")
        # another generation's partition in the same store: never this
        # pool's boxes, and 200#0 is somewhere else in it
        pg1 = os.path.join(d2, 'shards', 'gen=0001', 'x')
        os.makedirs(pg1)
        con.execute(
            f"COPY (SELECT * FROM (VALUES "
            f"{vals([('200', 0, 400., 10., 500., 90., 0.2)], ", '0001'")}) "
            f"t(image_id, det_idx, conf, x1, y1, x2, y2, gen)) "
            f"TO '{_q(os.path.join(pg1, 'g1.det.parquet'))}' (FORMAT PARQUET)")
        p1 = os.path.join(d1, 'shards', 'gen=0001', 'x')
        os.makedirs(p1)
        con.execute(
            f"COPY (SELECT * FROM (VALUES {vals(DET1, '')}) "
            f"t(image_id, det_idx, conf, x1, y1, x2, y2)) "
            f"TO '{_q(os.path.join(p1, 'a.det.parquet'))}' (FORMAT PARQUET)")
        # the stage store, judged on gen 2
        pcol = {'gate': 'p_dog', 'leash': 'p_leashed'}[stage]
        store = os.path.join(repo, 'data', stage)
        os.makedirs(store)
        grid = os.path.join(repo, 'grid')
        os.makedirs(os.path.join(grid, 'C'))
        plan = {'gen': plan_gen,
                'rows': len(DET2) + rows_delta + (1 if shard_dup else 0),
                'roots': {'d': grid}, 'created': 'now'}
        if detect_root:
            plan['detect_root'] = d2
        with open(os.path.join(store, 'plan.json'), 'w') as fh:
            json.dump(plan, fh)
        with open(os.path.join(store, 'protocol.json'), 'w') as fh:
            json.dump({'crop': protocol[0], 'imgsz': protocol[1]}, fh)
        judged = DET2 + (DET2[:1] if shard_dup else [])
        ms = list(models) * len(judged)
        v = ','.join(f"('{i}', {k}, {0.1 + 0.2 * (n % 4)}, '{ms[n]}')"
                     for n, (i, k, *_) in enumerate(judged))
        con.execute(
            f"COPY (SELECT * FROM (VALUES {v}) t(image_id, det_idx, {pcol}, "
            f"model)) TO '{_q(os.path.join(store, stage + '-00000.parquet'))}'"
            f" (FORMAT PARQUET)")
        work = DET2 + (DET2[:1] if extra_work_dup else [])
        v = ','.join(f"('{i}', {k}, {x1}::FLOAT, {y1}::FLOAT, {x2}::FLOAT, "
                     f"{y2}::FLOAT, {c}::FLOAT, 'C', 'd')"
                     for i, k, x1, y1, x2, y2, c in work)
        con.execute(
            f"COPY (SELECT * FROM (VALUES {v}) t(image_id, det_idx, x1, y1, "
            f"x2, y2, conf, cell, drive)) TO "
            f"'{_q(os.path.join(store, 'work.parquet'))}' (FORMAT PARQUET)")
        con.execute(
            f"COPY (SELECT * FROM (VALUES (100, 'sA'), (200, 'sB')) "
            f"t(image_id, \"sequence\")) TO "
            f"'{_q(os.path.join(grid, 'C', 'all_data_0.parquet'))}' "
            f"(FORMAT PARQUET)")
    finally:
        con.close()
    # a legacy gen-1 audit dir, and a gen-1 dataset that must never move
    a1 = os.path.join(repo, 'data', {'gate': 'fn_audit',
                                     'leash': 'leash_audit'}[stage])
    os.makedirs(a1)
    with open(os.path.join(a1, 'pool.json'), 'w') as fh:
        json.dump({'stage': stage, 'rows': 1, 'shards': 1}, fh)
    with open(os.path.join(repo, 'data', 'ledger_detect_root.txt'), 'w') as fh:
        fh.write(d1 + '\n')
    return repo, {'det1': d1, 'det2': d2}


def do_build(fa, repo, stage='gate', gen=2):
    """fn_audit.build on `repo` -> (why it stopped or '', stdout). A crash
    that is not a refusal is returned as 'CRASH ...', which no check
    accepts."""
    fa.REPO = repo
    fa._META.clear()
    fa._MODELS.clear()
    a = types.SimpleNamespace(stage=stage, gen=gen, memory=MEM,
                              tmp=os.path.join(repo, 'spill'))
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            fa.build(a)
        return '', out.getvalue()
    except SystemExit as e:
        return str(e), out.getvalue()
    except Exception as e:                    # noqa: BLE001 - a crash is a FAIL
        return f'CRASH {type(e).__name__}: {e}', out.getvalue()


def do_export(fa, stage, gen):
    a = types.SimpleNamespace(stage=stage, gen=gen)
    fa._META.clear()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fa.export(a)
        return ''
    except SystemExit as e:
        return str(e)
    except Exception as e:                    # noqa: BLE001
        return f'CRASH {type(e).__name__}: {e}'


LEGACY = {
    'gate': {'out': 'data/fn_audit', 'pool': 'data/fn_audit/pool.parquet',
             'verdicts': 'data/fn_audit/verdicts.jsonl',
             'drawn': 'data/fn_audit/drawn.jsonl',
             'crops': 'data/fn_audit/crops', 'full': 'data/fn_audit/full',
             'pages': 'data/fn_audit/pages', 'dataset': 'data/audit_finds',
             'store': 'data/gate', 'shards': 'data/gate/gate-*.parquet',
             'work': 'data/gate/work.parquet',
             # the joins audit.py and fn_audit made by hand before paths()
             # named them -- the same files
             'pool_json': 'data/fn_audit/pool.json',
             'edited': 'data/fn_audit/edited',
             'edited_thumbs': 'data/fn_audit/edited_thumbs'},
    'leash': {'out': 'data/leash_audit',
              'pool': 'data/leash_audit/pool.parquet',
              'verdicts': 'data/leash_audit/verdicts.jsonl',
              'drawn': 'data/leash_audit/drawn.jsonl',
              'crops': 'data/leash_audit/crops',
              'full': 'data/leash_audit/full',
              'pages': 'data/leash_audit/pages',
              'dataset': 'data/audit_finds_leash', 'store': 'data/leash',
              'shards': 'data/leash/leash-*.parquet',
              'work': 'data/leash/work.parquet',
              'pool_json': 'data/leash_audit/pool.json',
              'edited': 'data/leash_audit/edited',
              'edited_thumbs': 'data/leash_audit/edited_thumbs'},
}
# the keys that are the audit's own (store/shards/work are the model's)
OWN = ('out', 'pool', 'pool_json', 'verdicts', 'drawn', 'crops', 'full',
       'edited', 'edited_thumbs', 'pages', 'dataset')


def layout_checks(fa, cs, pre=''):
    """G1."""
    fa.REPO = tempfile.gettempdir()
    rel = {}
    for s in ('gate', 'leash'):
        for g in (1, 2, 3):
            P = fa.paths(s, gen=g)
            rel[(s, g)] = {k: os.path.relpath(v, fa.REPO) for k, v in
                           P.items() if isinstance(v, str)}
    for s in ('gate', 'leash'):
        got = {k: rel[(s, 1)].get(k) for k in LEGACY[s]}
        check(pre + f'G1 paths({s!r}, gen=1) is the legacy layout, key for '
              f'key', got == LEGACY[s],
              {k: (got[k], LEGACY[s][k]) for k in LEGACY[s]
               if got[k] != LEGACY[s][k]})
        check(pre + f'G1 paths({s!r}, gen=2) is data/*_gen2',
              rel[(s, 2)]['out'] == LEGACY[s]['out'] + '_gen2'
              and rel[(s, 2)]['dataset'] == LEGACY[s]['dataset'] + '_gen2',
              (rel[(s, 2)]['out'], rel[(s, 2)]['dataset']))
    seen = {}
    clash = []
    for sg, d in rel.items():
        for k in OWN:
            if k in ('pool', 'pool_json', 'verdicts', 'drawn', 'crops',
                     'full', 'edited', 'edited_thumbs', 'pages'):
                continue        # under `out`, which is compared itself
            if d[k] in seen:
                clash.append((sg, k, seen[d[k]]))
            seen[d[k]] = (sg, k)
    check(pre + 'G1 no two (stage, gen) share an audit dir or dataset dir',
          not clash, clash)
    hit = sorted({d[k] for d in rel.values() for k in OWN
                  if cs.FORBIDDEN_PATH_RE.search(d[k])})
    check(pre + 'G1 no audit path looks like a model\'s store '
          '(FORBIDDEN_PATH_RE)', not hit, hit)
    try:
        fa.paths('gate')
        ok, why = False, 'returned'
    except TypeError:
        ok, why = True, ''
    except Exception as e:                    # noqa: BLE001
        ok, why = False, type(e).__name__
    check(pre + 'G1 paths() with no gen is a TypeError (no default '
          'generation)', ok, why)
    bad = []
    for g in (None, '2', True, 0, 2.0, -1):
        try:
            fa.paths('gate', gen=g)
            bad.append(g)
        except ValueError:
            pass
        except Exception as e:                # noqa: BLE001
            bad.append((g, type(e).__name__))
    check(pre + 'G1 paths(gen=None/"2"/True/0/2.0/-1) is a ValueError',
          not bad, bad)


def build_checks(fa, cs, pre=''):
    """G2, G3, G4, G7, G11 on fresh build repos. Returns nothing; checks."""
    made = []

    def fresh(**kw):
        r, st = build_repo(**kw)
        made.append(r)
        return r, st
    try:
        # ── a clean build ──
        repo, st = fresh()
        why, out = do_build(fa, repo)
        check(pre + 'G3 a clean gen-2 build passes the geometry check, with '
              'a key twice in the det shards (agreeing) and det files of two '
              'schemas', why == '' and '0 unmatched, 0 disagree' in out
              and '4 pool keys joined to 5' in out, why or out[-400:])
        P = fa.paths('gate', gen=2)
        try:
            m = fa.pool_meta('gate', 2)
        except Exception as e:                # noqa: BLE001
            m = {'error': repr(e)}
        check(pre + 'G2 the pool names its generation, store, model, '
              'framing and roots',
              m.get('gen') == 2 and m.get('state') == 'ready'
              and m.get('model') == 'm2' and m.get('view_frame') == 'pad012'
              and m.get('protocol') == 'full@640' and m.get('rows') == 4
              and m.get('detect_root') == os.path.realpath(st['det2'])
              and m.get('roots') and len(m.get('pool_id') or '') >= 16, m)
        check(pre + 'G4 verify passes on the pool as built',
              fa.verify_pool('gate', 2) is None, fa.verify_pool('gate', 2))
        check(pre + 'G4 audit_gens lists the built generation',
              fa.audit_gens('gate') == [1, 2], fa.audit_gens('gate'))
        before = open(P['pool_json'], 'rb').read()
        why, _ = do_build(fa, repo)
        check(pre + 'G2 a second build into the (non-empty) gen-2 dir is '
              'refused and changes nothing',
              'already holds' in why
              and open(P['pool_json'], 'rb').read() == before, why)

        # ── G4: what the pool's identity holds against ──
        pq, pj = P['pool'], P['pool_json']
        raw = open(pq, 'rb').read()
        s0 = os.stat(pq)
        b = bytearray(raw)
        b[len(b) // 2] ^= 0xff
        with open(pq, 'wb') as fh:
            fh.write(bytes(b))
        os.utime(pq, ns=(s0.st_atime_ns, s0.st_mtime_ns))
        fa._META.clear()
        got = fa.verify_pool('gate', 2) or ''
        check(pre + 'G4 verify catches a parquet with the recorded size and '
              'mtime but other bytes (sha256)', 'sha256' in got, got)
        with open(pq, 'wb') as fh:
            fh.write(raw + b'x')
        try:
            fa.pool_meta('gate', 2)
            ok, why = False, 'read as ready'
        except fa.PoolIdentityError as e:
            ok, why = 'not the file' in str(e), str(e)
        check(pre + 'G4 a parquet of another size/mtime is refused on every '
              'read', ok, why)
        with open(pq, 'wb') as fh:
            fh.write(raw)
        os.utime(pq, ns=(s0.st_atime_ns, s0.st_mtime_ns))
        doc = json.load(open(pj))
        for what, edit in (('gen 3', {'gen': 3}), ('no gen', {'gen': None}),
                           ('stage leash', {'stage': 'leash'}),
                           ('state building', {'state': 'building'}),
                           ('a view_frame that is not its protocol\'s',
                            {'view_frame': 'pad050'})):
            d = dict(doc, **edit)
            if edit.get('gen', 0) is None:
                d.pop('gen')
            with open(pj, 'w') as fh:
                json.dump(d, fh)
            try:
                fa.pool_meta('gate', 2)
                ok = False
            except fa.PoolIdentityError:
                ok = True
            check(pre + f'G4 a gen-2 pool.json saying {what} is no pool',
                  ok and fa.pool_gen('gate', gen=2) is None)
        with open(pj, 'w') as fh:
            json.dump(doc, fh)
        os.rename(pj, pj + '.away')
        try:
            fa.pool_meta('gate', 2)
            ok, why = False, 'read as a pool'
        except fa.PoolIdentityError as e:
            ok, why = True, str(e)
        check(pre + 'G4 a gen-2 dir with no pool.json is no pool -- never '
              'gen 1', ok and fa.pool_gen('gate', gen=2) is None
              and fa.audit_gens('gate') == [1], why)
        os.rename(pj + '.away', pj)
        fa._META.clear()

        # ── G7: the gen-2 ledger holds this pool's rows and nothing else ──
        pid = fa.pool_meta('gate', 2)['pool_id']

        def ledger(rows, path=P['verdicts']):
            with open(path, 'w') as fh:
                for x in rows:
                    fh.write(json.dumps(x) + '\n')
        good = {'key': '200#0', 'verdict': 'dog', 'gen': 2, 'pool': pid}
        ledger([good])
        check(pre + 'G7 a tagged row of this pool reads back',
              [v['key'] for v in fa.read_verdicts(stage='gate', gen=2)]
              == ['200#0'])
        for name, bad in (
                ('an untagged', {'key': '100#0', 'verdict': 'dog',
                                 'pool': pid}),
                ('a gen-1', {'key': '100#0', 'verdict': 'dog', 'gen': 1,
                             'pool': pid}),
                ('a pool-less', {'key': '100#0', 'verdict': 'dog',
                                 'gen': 2}),
                ('another pool\'s', {'key': '100#0', 'verdict': 'dog',
                                     'gen': 2, 'pool': 'other'})):
            ledger([good, bad])
            try:
                got = fa.read_verdicts(stage='gate', gen=2)
                ok, why = False, f'read {len(got)} row(s)'
            except fa.GenError as e:
                ok, why = ':2:' in str(e), str(e)
            check(pre + f'G7 {name} row in the gen-2 ledger raises, naming '
                  f'its line', ok, why)
        g1v = fa.paths('gate', gen=1)['verdicts']
        ledger([{'key': '1#0', 'verdict': 'dog'}], g1v)
        check(pre + 'G7 an untagged row of the gen-1 ledger reads as gen 1',
              len(fa.read_verdicts(stage='gate', gen=1)) == 1)
        ledger([{'key': '1#0', 'verdict': 'dog'},
                {'key': '2#0', 'verdict': 'dog', 'gen': 2, 'pool': pid}], g1v)
        try:
            fa.read_verdicts(stage='gate', gen=1)
            ok = False
        except fa.GenError:
            ok = True
        check(pre + 'G7 a gen-2 row in the gen-1 ledger raises', ok)
        ledger([{'key': '1#0', 'verdict': 'dog'},
                {'key': '2#0', 'verdict': 'dog', 'pool': pid}], g1v)
        try:
            fa.read_verdicts(stage='gate', gen=1)
            ok, why = False, 'read'
        except fa.GenError as e:
            ok, why = ':2:' in str(e), str(e)
        check(pre + 'G7 an untagged row naming another pool in the gen-1 '
              'ledger raises (untagged is gen 1, and gen 1 is g1-legacy)',
              ok, why)
        os.remove(g1v)

        # ── G11: export ──
        ledger([{'key': '100#0', 'verdict': 'dog', 'gen': 2, 'pool': pid,
                 'by': 'a', 'p_dog': 0.1, 'band': 1, 'seq': 'sA',
                 'box': list(G2_BOX)},
                {'key': '200#0', 'verdict': 'not_dog', 'gen': 2, 'pool': pid,
                 'by': 'a', 'p_dog': 0.5, 'band': 5, 'seq': 'sB'}])
        os.makedirs(P['full'])
        for k in ('100_0', '200_0'):
            with open(os.path.join(P['full'], k + '.jpg'), 'w') as fh:
                fh.write(k)
        d1 = fa.paths('gate', gen=1)['dataset']
        os.makedirs(os.path.join(d1, 'dog'))
        with open(os.path.join(d1, 'dog', '100_0.jpg'), 'w') as fh:
            fh.write('GEN1')
        with open(os.path.join(d1, 'dog', '555_0.jpg'), 'w') as fh:
            fh.write('GEN1 only')
        g1_ds = tree_sha(d1)
        why = do_export(fa, 'gate', 2)
        ds2 = P['dataset']
        man = lines(os.path.join(ds2, 'manifest.jsonl'))
        check(pre + 'G11 export --gen 2 writes audit_finds_gen2',
              why == '' and os.path.basename(ds2) == 'audit_finds_gen2'
              and sorted(os.listdir(os.path.join(ds2, 'dog'))) == ['100_0.jpg']
              and sorted(os.listdir(os.path.join(ds2, 'not_dog')))
              == ['200_0.jpg'], why)
        check(pre + 'G11 ...and leaves the gen-1 dataset byte for byte (no '
              'marker, nothing removed)', tree_sha(d1) == g1_ds)
        check(pre + 'G11 every manifest row names gen, pool, detect_root, '
              'its pool box and the pool\'s model',
              len(man) == 2 and all(
                  x.get('gen') == 2 and x.get('pool') == pid
                  and x.get('judged_model') == 'm2'
                  and x.get('detect_root') == os.path.realpath(st['det2'])
                  for x in man)
              and man[0].get('box') == list(G2_BOX)
              and man[1].get('box') == [50.0, 50.0, 150.0, 150.0], man)
        mk = os.path.join(ds2, fa.DATASET_MARKER)
        check(pre + 'G11 the dataset dir carries a marker naming the pool',
              os.path.exists(mk) and json.load(open(mk)).get('pool_id') == pid)
        why = do_export(fa, 'gate', 1)
        check(pre + 'G11 export --gen 1 is refused (frozen) and gen 1\'s '
              'dataset is untouched', 'frozen' in why
              and tree_sha(d1) == g1_ds, why)
        with open(mk, 'w') as fh:
            json.dump({'stage': 'gate', 'gen': 2, 'pool_id': 'zzz',
                       'detect_root': 'x'}, fh)
        keep = tree_sha(ds2)
        ledger([{'key': '200#0', 'verdict': 'not_dog', 'gen': 2,
                 'pool': pid}])
        why = do_export(fa, 'gate', 2)
        check(pre + 'G11 a dataset dir marked for another pool is refused, '
              'nothing deleted', 'belongs to' in why
              and tree_sha(ds2) == keep, why)
        os.remove(mk)
        why = do_export(fa, 'gate', 2)
        check(pre + 'G11 a dataset dir with files and no marker is refused',
              'no ' + fa.DATASET_MARKER in why
              and tree_sha(ds2) == {k: v for k, v in keep.items()
                                    if k != fa.DATASET_MARKER}, why)
        fa.write_dataset_marker(fa.ctx('gate', 2))
        ledger([{'key': '100#0', 'verdict': 'dog', 'gen': 2, 'pool': pid,
                 'box': list(G1_BOX)}])
        why = do_export(fa, 'gate', 2)
        check(pre + 'G11 a verdict that recorded another box than its pool '
              'box is refused', 'not their pool box' in why, why)
        ledger([{'key': '999#0', 'verdict': 'dog', 'gen': 2, 'pool': pid}])
        why = do_export(fa, 'gate', 2)
        check(pre + 'G11 a verdict whose key is not in the pool is refused',
              'not in pool' in why, why)

        # ── G2: each refusal, on its own fresh repo ──
        def refused(name, kw, want, gen=2):
            r, _ = fresh(**kw)
            why, out = do_build(fa, r, gen=gen)
            p2 = os.path.join(r, 'data', 'fn_audit_gen2')
            check(pre + f'G2 build refuses {name}', want in why
                  and not os.path.exists(os.path.join(p2, 'pool.parquet'))
                  and not os.path.exists(os.path.join(p2, 'pool.json')),
                  why or out[-300:])
        refused('no --gen', {}, '--gen is required', gen=None)
        refused('gen 1 (frozen)', {}, 'frozen', gen=1)
        refused('a plan of another generation', {'plan_gen': 3}, 'is gen 3')
        # the pin names a gen-2 store here: a build that fell back to it
        # would find a "matching" store
        r, s = fresh(detect_root=False)
        with open(os.path.join(r, 'data', 'ledger_detect_root.txt'), 'w') \
                as fh:
            fh.write(s['det2'] + '\n')
        real = fa._ledger_pin
        calls = []
        fa._ledger_pin = lambda: calls.append(1) or real()
        try:
            why, out = do_build(fa, r)
        finally:
            fa._ledger_pin = real
        check(pre + 'G2 build refuses a plan with no detect_root, and never '
              'opens the ledger pin', 'names no detect_root' in why
              and not calls and not os.path.exists(os.path.join(
                  r, 'data', 'fn_audit_gen2', 'pool.json')), why or out[-300:])
        refused('a store that says another generation', {'store_gen': 3},
                'says it is gen 3')
        refused('shards judged by two models', {'models': ('m2', 'm3')},
                'model value(s)')
        refused('a row count that is not the plan\'s', {'rows_delta': 1},
                'planned')
        refused('an unknown protocol', {'protocol': ('pad999', 224)},
                'cannot frame')
        refused('a work.parquet holding a key twice',
                {'extra_work_dup': True}, 'judged boxes')
        refused('shards that judge a key twice (row count as planned)',
                {'shard_dup': True}, 'more than once')
        # ── G3: geometry ──
        refused('a pool box 1 px off the det shards',
                {'moved': ('200', 0, 1.0)}, 'disagree')
        refused('a pool key missing from the det shards',
                {'drop_det': ('300', 0)}, 'not in')
        # every edge and conf, and every copy -- not x1 of a key seen once
        refused('a pool box whose y2 is 1 px off the det shards',
                {'moved': ('200', 0, 'y2', 1.0, 'both')}, 'disagree')
        refused('a pool box whose conf is 1e-3 off the det shards',
                {'moved': ('200', 0, 'conf', 1e-3, 'both')}, 'disagree')
        refused('a key whose second det copy is moved, behind an agreeing '
                'first', {'moved': ('100', 0, 'x1', 1.0, 'b')}, 'disagree')

        # live_gen: a plan saying gen 2 over a store that says gen 1 makes
        # no generation live (and so freezes every one), never gen 2
        r, s2 = fresh()
        pj_ = os.path.join(r, 'data', 'gate', 'plan.json')
        doc_ = json.load(open(pj_))
        with open(pj_, 'w') as fh:
            json.dump(dict(doc_, detect_root=s2['det1']), fh)
        fa.REPO = r
        fa._META.clear()
        try:
            fa.live_gen('gate')
            ok, why = False, 'returned'
        except fa.GenError as e:
            ok, why = 'says gen 1' in str(e), str(e)
        check(pre + 'G2 live_gen refuses a gen-2 plan whose detect_root is a '
              'gen-1 store, and every generation is then frozen',
              ok and fa.is_frozen('gate', 1) and fa.is_frozen('gate', 2),
              why)
        r, s3 = fresh(moved=('300', 0, 1.0))
        do_build(fa, r)
        try:
            fa.pool_meta('gate', 2)
            ok = False
        except fa.PoolIdentityError:
            ok = True
        check(pre + 'G3 ...and after a geometry failure there is no pool to '
              'read', ok and fa.audit_gens('gate') == [1])
        # ...and the dir it left (pool.parquet.tmp, nothing ever drawn) does
        # not stop every dataset build on the gen-2 store; one that holds a
        # ledger does
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                got = cs.audit_ctx_for('gate', s3['det2'])
        except (SystemExit, Exception) as e:  # noqa: BLE001
            got = repr(e)
        check(pre + 'G3 ...and audit_ctx_for reads that never-drawn dir as no '
              'audit (None, said), not SystemExit', got is None
              and 'nothing was ever drawn' in out.getvalue(), got)
        with open(fa.paths('gate', gen=2)['verdicts'], 'w') as fh:
            fh.write(json.dumps({'key': '100#0', 'verdict': 'dog', 'gen': 2,
                                 'pool': 'x'}) + '\n')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                got = cs.audit_ctx_for('gate', s3['det2'])
        except SystemExit as e:
            got = 'SystemExit ' + str(e)
        except Exception as e:                # noqa: BLE001
            got = repr(e)
        check(pre + 'G3 ...but a dir with a ledger and no readable pool is '
              'SystemExit (answers are never dropped quietly)',
              str(got).startswith('SystemExit'), got)

        # the CLI: build needs a spill dir (no tmpfs default), and stats with
        # no --gen on a live generation with no pool says where gen 1 is
        r, _ = fresh(plan_gen=3)
        fa.REPO = r
        err = io.StringIO()
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(err):
                fa.main(['build', '--stage', 'gate', '--gen', '2'])
            code = 'ran'
        except SystemExit as e:
            code = e.code
        check(pre + 'G2 `build` with no --tmp is a usage error (the old '
              'default spilled to tmpfs)', code == 2
              and '--tmp' in err.getvalue(), (code, err.getvalue()[-200:]))
        r, _ = fresh()
        fa.REPO = r
        fa._META.clear()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fa.stats(types.SimpleNamespace(stage='gate', gen=None))
            why = 'printed'
        except SystemExit as e:
            why = str(e)
        check(pre + 'G2 `stats` with no --gen and no live pool yet points at '
              '--gen 1', '--gen 1' in why, why)

        # ── G4: a stop between the two renames ──
        r, _ = fresh()
        real_replace = os.replace

        def stop_at_parquet(src, dst, *a, **k):
            if str(dst).endswith('pool.parquet'):
                raise KeyboardInterrupt('stopped between the renames')
            return real_replace(src, dst, *a, **k)
        os.replace = stop_at_parquet
        try:
            try:
                do_build(fa, r)
            except KeyboardInterrupt:
                pass
        finally:
            os.replace = real_replace
        P = fa.paths('gate', gen=2)
        try:
            fa.pool_meta('gate', 2)
            ok, why = False, 'read as ready'
        except fa.PoolIdentityError as e:
            ok, why = 'missing' in str(e), str(e)
        check(pre + 'G4 a build stopped between its renames leaves pool.json '
              'and no pool.parquet, which is "not ready"',
              ok and os.path.exists(P['pool_json'])
              and not os.path.exists(P['pool']), why)
        why, _ = do_build(fa, r)
        check(pre + 'G4 ...is not listed, and a rebuild over it is refused',
              fa.audit_gens('gate') == [1] and 'already holds' in why, why)

        # ── the leash protocol frames its tiles at pad050 ──
        r, _ = fresh(stage='leash', protocol=('pad050', 448))
        why, _ = do_build(fa, r, stage='leash')
        try:
            vf = fa.pool_meta('leash', 2)['view_frame']
        except Exception as e:                # noqa: BLE001
            vf = repr(e)
        check(pre + 'G9 a pad050@448 leash store builds a pool whose tiles '
              'are pad050', why == '' and vf == 'pad050', why or vf)
    finally:
        for r in made:
            shutil.rmtree(r, ignore_errors=True)


# ── the PAGE repo ───────────────────────────────────────────────────────────

POOL_COLS = ('image_id VARCHAR', 'det_idx INTEGER', 'p_dog DOUBLE',
             'x1 DOUBLE', 'y1 DOUBLE', 'x2 DOUBLE', 'y2 DOUBLE', 'conf DOUBLE',
             'cell VARCHAR', 'drive VARCHAR', 'seq VARCHAR', 'band INTEGER')


def _pool_rows(box, p):
    return [('100', 0, p) + box + (0.8, 'c', 'd', 's100', int(p * 10))]


def _store_shard(path, model):
    _parquet(path, [('100', 0, model)],
             ('image_id VARCHAR', 'det_idx INTEGER', 'model VARCHAR'))


def fixture():
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix='adv_audit_gen2_')
    repo = os.path.join(tmp, 'repo')
    data = os.path.join(repo, 'data')
    frames = os.path.join(tmp, 'frames')
    roots = {'d': frames}
    fdir = os.path.join(frames, 'c', 'ground_animal_images')
    os.makedirs(fdir)
    im = Image.new('RGB', (800, 600), (128, 128, 128))
    im.paste(RED, (0, 0, 120, 120))
    im.paste(BLUE, (240, 240, 440, 440))
    frame = os.path.join(fdir, '100.jpg')
    im.save(frame, quality=98)
    stores = {}
    for g in (1, 2):
        root = os.path.join(tmp, f'det{g}')
        os.makedirs(os.path.join(root, 'shards', 'gen=%04d' % g))
        _status(root, g)
        stores[g] = root
    # a second gen-2 store: the right generation, not the pool's store
    other2 = os.path.join(tmp, 'det2b')
    _status(other2, 2)
    os.makedirs(data)
    with open(os.path.join(data, 'ledger_detect_root.txt'), 'w') as fh:
        fh.write(stores[1] + '\n')
    models = {'gate': 'dogbin_m2', 'leash': 'leash_m2'}
    protos = {'gate': ('full', 640), 'leash': ('pad050', 448)}
    for stage in ('gate', 'leash'):
        sd = os.path.join(data, stage)
        os.makedirs(sd)
        with open(os.path.join(sd, 'plan.json'), 'w') as fh:
            json.dump({'gen': 2, 'detect_root': stores[2], 'roots': roots,
                       'rows': 1}, fh)
        with open(os.path.join(sd, 'protocol.json'), 'w') as fh:
            json.dump({'crop': protos[stage][0],
                       'imgsz': protos[stage][1]}, fh)
        _store_shard(os.path.join(sd, f'{stage}-00000.parquet'),
                     models[stage])
    # gen 1: the legacy layout -- pool.json with no gen, a page drawn before
    # pages carried tags or boxes, a crop and a full/ already cut (red), and
    # the literal "None" key the real gate ledger holds
    for stage, base in (('gate', 'fn_audit'), ('leash', 'leash_audit')):
        out = os.path.join(data, base)
        os.makedirs(os.path.join(out, 'pages'))
        _parquet(os.path.join(out, 'pool.parquet'), _pool_rows(G1_BOX, 0.31),
                 POOL_COLS)
        with open(os.path.join(out, 'pool.json'), 'w') as fh:
            json.dump({'stage': stage, 'rows': 1, 'shards': 1}, fh)
        with open(os.path.join(out, 'pages', '00000.json'), 'w') as fh:
            json.dump({'index': 0, 'band': 'all', 'items': [
                {'key': '100#0', 'image_id': '100', 'det_idx': 0,
                 'p_dog': 0.31, 'conf': 0.8, 'band': 3, 'seq': 's100',
                 'drive': 'd', 'cell': 'c'}]}, fh)
        with open(os.path.join(out, 'drawn.jsonl'), 'w') as fh:
            fh.write(json.dumps({'key': '100#0', 'seq': 's100'}) + '\n')
        with open(os.path.join(out, 'verdicts.jsonl'), 'w') as fh:
            fh.write(json.dumps({'key': 'None', 'verdict': 'unsure'}) + '\n')
        for sub, size in (('crops', 70), ('full', 106)):
            os.makedirs(os.path.join(out, sub))
            Image.new('RGB', (size, size), RED).save(
                os.path.join(out, sub, '100_0.jpg'))
    os.makedirs(os.path.join(data, 'box_corrections'))
    return {'tmp': tmp, 'repo': repo, 'data': data, 'roots': roots,
            'stores': stores, 'other2': other2, 'models': models,
            'frame': frame}


def build_gen2(fx, stage, pool_id):
    """A ready gen-2 pool for `stage`: 100#0 is the BLUE box."""
    base = {'gate': 'fn_audit', 'leash': 'leash_audit'}[stage]
    out = os.path.join(fx['data'], base + '_gen2')
    os.makedirs(out)
    pq = os.path.join(out, 'pool.parquet')
    _parquet(pq, _pool_rows(G2_BOX, 0.72), POOL_COLS)
    st = os.stat(pq)
    proto = {'gate': 'full@640', 'leash': 'pad050@448'}[stage]
    with open(os.path.join(out, 'pool.json'), 'w') as fh:
        json.dump({'stage': stage, 'gen': 2, 'pool_id': pool_id,
                   'state': 'ready', 'detect_root': fx['stores'][2],
                   'model': fx['models'][stage], 'protocol': proto,
                   'view_frame': {'full@640': 'pad012',
                                  'pad050@448': 'pad050'}[proto],
                   'roots': fx['roots'], 'rows': 1, 'shards': 1,
                   'parquet_size': st.st_size,
                   'parquet_mtime_ns': st.st_mtime_ns,
                   'sha256': hashlib.sha256(open(pq, 'rb').read())
                   .hexdigest()}, fh)
    return out


# ── modules ─────────────────────────────────────────────────────────────────

def load(path, name, src=None, register=False):
    """Exec `path` (or `src` written beside nothing else) as module `name`.
    `register` puts it in sys.modules first, for the modules the others
    import by name (fn_audit, crop_sources, build_dataset)."""
    if src is not None:
        d = tempfile.mkdtemp(prefix='adv_audit_gen2_mut_')
        TEMPS.append(d)
        p2 = os.path.join(d, os.path.basename(path))
        with open(p2, 'w') as fh:
            fh.write(src)
        path = p2
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


class World:
    """Every module under test, fresh, with the named ones from (mutated)
    source; fn_audit, crop_sources and build_dataset stand in sys.modules
    while it lives (the others import them by name). close() puts back
    sys.modules and sys.path as they were."""

    NAMED = {'fa': 'fn_audit', 'cs': 'crop_sources', 'bd': 'build_dataset'}

    def __init__(self, srcs):
        self._mods = {n: sys.modules.get(n) for n in self.NAMED.values()}
        self._path = list(sys.path)
        self.srcs = srcs
        try:
            self.fa = load(FA_PY, 'fn_audit', srcs.get('fa'), register=True)
            self.cs = load(CS_PY, 'crop_sources', srcs.get('cs'),
                           register=True)
            self.bd = load(BD_PY, 'build_dataset', srcs.get('bd'),
                           register=True)
            self.rb = load(RB_PY, 'rebuild_crop_dataset_g2t', srcs.get('rb'))
            self.audit = load(AUDIT_PY, 'audit_gen2t', srcs.get('audit'))
            self.dash = load(DASH_PY, 'dash_gen2t', srcs.get('dash'))
            # TRAINING_ROOT set for the import, or build_dogdet_v3 imports
            # the dashboard to ask it; nothing here reads that root
            keep = os.environ.get('TRAINING_ROOT')
            os.environ['TRAINING_ROOT'] = tempfile.gettempdir()
            try:
                self.v3 = load(V3_PY, 'build_dogdet_v3_g2t', srcs.get('v3'))
            finally:
                if keep is None:
                    os.environ.pop('TRAINING_ROOT', None)
                else:
                    os.environ['TRAINING_ROOT'] = keep
        except BaseException:
            self.close()
            raise

    def point(self, fx):
        """Aim every module at the page repo."""
        self.fa.REPO = self.cs.REPO = self.bd.REPO = fx['repo']
        self.fa._META.clear()
        self.fa._MODELS.clear()
        self.audit.REPO = fx['repo']
        self.audit.BOX_FILE = os.path.join(fx['data'], 'box_corrections',
                                           'boxes.jsonl')
        self.dash._AUDIT = {'mod': self.audit, 'tried': True}
        self.dash._fn_audit = self.fa

    def close(self):
        for n, m in self._mods.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
        sys.path[:] = self._path


class Reply:
    def __init__(self):
        self.status, self.headers, self.body, self.json = None, {}, b'', None


def request(dash, method, path, body=None, session=SESSION):
    """One request through the real handler: (Reply)."""
    h = dash.BoardHandler.__new__(dash.BoardHandler)
    rep = Reply()
    raw = json.dumps(body if body is not None else {}).encode()
    h.path = path
    h.command = method
    h.headers = {'Content-Length': str(len(raw))}
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.session = session
    h.close_connection = False
    h.request_version = 'HTTP/1.1'

    def send_response(code, msg=None):
        rep.status = code
    h.send_response = send_response
    h.send_header = lambda k, v: rep.headers.__setitem__(k, v)
    h.end_headers = lambda: None
    h.send_error = lambda code, msg=None: send_response(code)
    h._gate = lambda: False
    h._admin_only = lambda p: False
    if method == 'GET':
        handled = h._audit_get()
        if not handled:
            rep.status = 'unhandled'
    else:
        h.do_POST()
    rep.body = h.wfile.getvalue()
    ctype = rep.headers.get('Content-Type', '')
    if 'json' in ctype:
        try:
            rep.json = json.loads(rep.body)
        except ValueError:
            pass
    return rep


# ── the page checks ─────────────────────────────────────────────────────────

def _script(html):
    i = html.find('var GEN=')
    return html[html.rfind('<script', 0, i):] if i >= 0 else ''


def js_checks(txt, pre):
    """G8, client side: cropSrc names the generation in the path; every
    fetch() the page makes names gen and pool."""
    js = _script(txt)
    i = js.find('function cropSrc(')
    body = js[i:js.find('\n}', i)] if i >= 0 else ''
    check(pre + 'G8 the page\'s cropSrc puts the generation in the crop path',
          "'/g'+GEN+'/'" in body.replace(' ', ''), body[:300])
    bad, n = [], 0
    at = js.find('fetch(')
    while at >= 0:
        # the URL expression: up to the options object or the first .then
        end = min(x for x in (js.find(',{', at), js.find('.then', at),
                              len(js)) if x >= 0)
        url = js[at:end].replace(' ', '').replace('\n', '')
        # the audit's own endpoints; the shared work strip's /api/assignment
        # is no audit's and names none
        if url.startswith(("fetch('/api/audit/", "fetch('/audit/")):
            n += 1
            if not ('GQ()' in url or ("'/g'+GEN" in url and 'pool=' in url)):
                bad.append(url[:90])
        at = js.find('fetch(', at + 6)
    gq = js[js.find('function GQ('):][:120]
    check(pre + 'G8 every fetch() of the page names gen and pool (GQ() or '
          'the /g<GEN>/ path with pool=)', n >= 6 and not bad
          and "'&gen='+GEN" in gq and 'pool=' in gq, (n, bad, gq))
    node = shutil.which('node')
    if not node or i < 0:
        print('skip G8 node run of cropSrc (no node)')
        return
    prog = ("var STAGE='gate',GEN=2;function esc(s){return String(s)}\n"
            + body + '\n}\nprocess.stdout.write(JSON.stringify(['
            "cropSrc({key:'100#0'}),cropSrc({key:'100#0',v:7})]))")
    try:
        out = subprocess.run([node, '-e', prog], capture_output=True,
                             text=True, timeout=30).stdout
        got = json.loads(out)
    except Exception as e:                    # noqa: BLE001
        got = repr(e)
    check(pre + 'G8 under node, cropSrc of gen 2 is /audit/crop/gate/g2/...',
          got == ['/audit/crop/gate/g2/100_0.jpg',
                  '/audit/crop/gate/g2/100_0.jpg?v=7'], got)


def run(fx, w, pre=''):
    """G5, G6, G8, G9, G10, G12, G13 and G1' on the page repo, through the
    real handler. True when every check passed."""
    fa, audit, dash, cs = w.fa, w.audit, w.dash, w.cs
    n0 = len(FAILS)
    data = fx['data']
    g1 = {s: os.path.join(data, b) for s, b in
          (('gate', 'fn_audit'), ('leash', 'leash_audit'))}
    g1_ds = [os.path.join(data, 'audit_finds'),
             os.path.join(data, 'audit_finds_leash')]
    before = {d: tree_sha(d) for d in list(g1.values())}

    # G10: with no gen-2 pool yet, the page says what to build -- never a
    # silent gen-1 page
    r = request(dash, 'GET', '/audit/gate')
    txt = r.body.decode('utf-8', 'replace')
    check(pre + 'G10 no gen-2 pool: the page names the build, links gen 1',
          'fn_audit.py build --stage gate --gen 2' in txt
          and '?gen=1' in txt and 'var GEN=' not in txt, txt[:300])
    g2 = build_gen2(fx, 'gate', 'P2-gate')
    l2 = build_gen2(fx, 'leash', 'P2-leash')
    P2 = 'P2-gate'

    # the page of each generation
    r = request(dash, 'GET', '/audit/gate')
    txt = r.body.decode('utf-8', 'replace')
    check(pre + 'G10 the default page is the live gen 2, drawable',
          'var GEN=2,POOL="P2-gate",READONLY=false' in txt
          and '?gen=1' in txt, txt[txt.find('var GEN'):][:120])
    js_checks(txt, pre)
    r = request(dash, 'GET', '/audit/gate?gen=1')
    txt = r.body.decode('utf-8', 'replace')
    check(pre + 'G10 the gen-1 page is READONLY and says why',
          'var GEN=1,POOL="g1-legacy",READONLY=true' in txt
          and 'closed for new pages' in txt,
          txt[txt.find('var GEN'):][:160])

    # G10: writes that cannot be placed are refused before anything is read
    r = request(dash, 'POST', '/api/audit/draw?stage=gate',
                {'band': 'all', 'n': 25, 'gen': 1, 'pool': 'g1-legacy'})
    check(pre + 'G10 a draw on the frozen gen 1 is a 409',
          r.status == 409 and (r.json or {}).get('closed'), (r.status, r.json))
    r = request(dash, 'GET',
                '/api/audit/page?stage=gate&gen=1&pool=g1-legacy&i=1&n=25')
    c1 = fa.ctx('gate', 1)
    audit.prefetch(c1)
    check(pre + 'G10 api_page one past the last gen-1 page answers "closed" '
          'and draws nothing; prefetch queues nothing for a frozen gen',
          r.status == 200 and (r.json or {}).get('closed') is True
          and (r.json or {}).get('exhausted') is True
          and ('gate', 1) not in audit._PREFETCH
          and tree_sha(g1['gate']) == before[g1['gate']],
          (r.status, r.json, list(audit._PREFETCH)))
    r = request(dash, 'POST', '/api/audit/draw?stage=nope',
                {'band': 'all', 'n': 25, 'gen': 2, 'pool': P2})
    check(pre + 'G10 an unknown stage on a POST is a 400', r.status == 400,
          (r.status, r.json))
    r = request(dash, 'POST', '/api/audit/draw?stage=gate',
                {'band': 'all', 'n': 25})
    check(pre + 'G10 a POST that names no generation is a 400 "reload"',
          r.status == 400 and 'reload' in str(r.json), (r.status, r.json))

    # draw a gen-2 page
    r = request(dash, 'POST', '/api/audit/draw?stage=gate&gen=2&pool=' + P2,
                {'band': 'all', 'n': 25, 'gen': 2, 'pool': P2})
    page = (r.json or {}).get('page') or {}
    c2 = fa.ctx('gate', 2)
    audit.await_prefetch(c2)
    check(pre + 'G10 a gen-2 draw serves 100#0 off the gen-2 pool',
          r.status == 200 and [it.get('key') for it in page.get('items', [])]
          == ['100#0'] and page.get('gen') == 2 and page.get('pool') == P2
          and page['items'][0].get('box') == list(G2_BOX),
          (r.status, r.json))
    drawn = lines(os.path.join(g2, 'drawn.jsonl'))
    check(pre + 'G10 the gen-2 drawn log names gen and pool on every row',
          drawn and all(d.get('gen') == 2 and d.get('pool') == P2
                        for d in drawn), drawn)
    check(pre + 'G10 locks and prefetch are keyed (stage, gen)',
          ('gate', 2) in audit._LOCKS
          and all(isinstance(k, tuple) for k in audit._LOCKS)
          and all(isinstance(k, tuple) for k in audit._PREFETCH),
          (list(audit._LOCKS), list(audit._PREFETCH)))

    # G8: the crops, by URL
    r2 = request(dash, 'GET', '/audit/crop/gate/g2/100_0.jpg')
    r1 = request(dash, 'GET', '/audit/crop/gate/100_0.jpg')
    rg1 = request(dash, 'GET', '/audit/crop/gate/g1/100_0.jpg')
    check(pre + 'G8 /audit/crop/gate/g2/... is the gen-2 (blue) cut, the '
          'key-only URL and /g1/ the gen-1 (red) one',
          r2.status == 200 and colour(r2.body) == 'blue'
          and r1.status == 200 and colour(r1.body) == 'red'
          and rg1.status == 200 and colour(rg1.body) == 'red',
          (r2.status, r1.status, rg1.status))
    check(pre + 'G8 the gen-2 crop file itself is blue, in the gen-2 dir',
          colour(os.path.join(g2, 'crops', '100_0.jpg')) == 'blue'
          and colour(os.path.join(g2, 'full', '100_0.jpg')) == 'blue')
    r = request(dash, 'GET', '/audit/crop/gate/g7/100_0.jpg')
    check(pre + 'G8 a crop of a generation with no pool is no picture',
          r.status in (404, 409), r.status)

    # G8: the frames
    f2 = request(dash, 'GET', '/audit/frame/gate/g2/100_0.jpg?pool=' + P2)
    m2 = json.loads(f2.headers.get('X-Audit-Meta') or 'null') or {}
    f1 = request(dash, 'GET', '/audit/frame/gate/g1/100_0.jpg?pool=g1-legacy')
    m1 = json.loads(f1.headers.get('X-Audit-Meta') or 'null') or {}
    check(pre + 'G8 the frame meta names gen, pool and the pool box in frame '
          'pixels (box_sig)',
          (m2.get('gen'), m2.get('pool'), m2.get('box_sig'))
          == (2, P2, list(G2_BOX))
          and (m1.get('gen'), m1.get('pool'), m1.get('box_sig'))
          == (1, 'g1-legacy', list(G1_BOX)), (m2, m1))
    r = request(dash, 'GET', '/audit/frame/gate/g2/100_0.jpg?pool=g1-legacy')
    check(pre + 'G8 a frame asked for under another pool is a 409',
          r.status == 409, r.status)

    # G5: verdicts
    vf2 = os.path.join(g2, 'verdicts.jsonl')
    for what, body, want in (
            ('a wrong pool', {'key': '100#0', 'verdict': 'dog', 'gen': 2,
                              'pool': 'g1-legacy'}, 409),
            ('no gen', {'key': '100#0', 'verdict': 'dog', 'pool': P2}, 400),
            ('no pool', {'key': '100#0', 'verdict': 'dog', 'gen': 2}, 400),
            ('a key on no page', {'key': '999#0', 'verdict': 'dog', 'gen': 2,
                                  'pool': P2}, 409),
            ('the key "None"', {'key': 'None', 'verdict': 'dog', 'gen': 2,
                                'pool': P2}, 409),
            ('a gen-1 page\'s pool under gen 2', {
                'key': '100#0', 'verdict': 'dog', 'gen': 1,
                'pool': P2}, 409)):
        r = request(dash, 'POST', '/api/audit/verdict?stage=gate', body)
        check(pre + f'G5 a verdict with {what} is refused ({want}) and '
              f'writes nothing',
              r.status == want and not (r.json or {}).get('ok')
              and not lines(vf2), (r.status, r.json, lines(vf2)))
    r = request(dash, 'POST', '/api/audit/verdict?stage=gate&gen=2&pool=' + P2,
                {'key': '100#0', 'verdict': 'dog', 'gen': 2, 'pool': P2,
                 'p_dog': 0.01, 'band': 0, 'seq': 'forged'})
    rows = lines(vf2)
    want = {'key': '100#0', 'verdict': 'dog', 'gen': 2, 'pool': P2,
            'box': list(G2_BOX), 'p_dog': 0.72, 'band': 7, 'seq': 's100',
            'by': SESSION['username']}
    check(pre + 'G5 the gen-2 verdict lands in the gen-2 ledger, tagged, '
          'with the page\'s meta (not the client\'s)',
          r.status == 200 and (r.json or {}).get('ok') and len(rows) == 1
          and {k: rows[0].get(k) for k in want} == want, (r.json, rows))
    ds2 = os.path.join(data, 'audit_finds_gen2')
    got = os.path.join(ds2, 'dog', '100_0.jpg')
    mk = os.path.join(ds2, fa.DATASET_MARKER)
    check(pre + 'G5 ...and in the gen-2 dataset, as the blue cut, under a '
          'marker naming the pool',
          os.path.exists(got) and colour(got) == 'blue'
          and os.path.exists(mk) and json.load(open(mk)).get('pool_id') == P2,
          os.listdir(ds2) if os.path.isdir(ds2) else 'no dir')
    check(pre + 'G5 the gen-1 dataset dirs were never created',
          not any(os.path.exists(d) for d in g1_ds))
    check(pre + 'G7 read_verdicts reads the gen-2 ledger back',
          [v['key'] for v in fa.read_verdicts(stage='gate', gen=2)]
          == ['100#0'])

    # G6: redraws
    bf = audit.BOX_FILE
    r = request(dash, 'POST', '/api/audit/box?stage=gate',
                {'key': '100#0', 'gen': 2, 'pool': P2,
                 'box_sig': m1.get('box_sig'), 'box': [290, 290, 390, 390]})
    check(pre + 'G6 a redraw carrying the gen-1 frame\'s box_sig is refused '
          'under gen 2 and writes nothing',
          r.status == 409 and not lines(bf), (r.status, r.json, lines(bf)))
    r = request(dash, 'POST', '/api/audit/box?stage=gate',
                {'key': '100#0', 'gen': 2, 'pool': P2,
                 'box_sig': m2.get('box_sig'), 'box': [290, 290, 390, 390]})
    rows = lines(bf)
    check(pre + 'G6 the gen-2 redraw lands tagged gen 2 and its pool',
          r.status == 200 and (r.json or {}).get('ok') and len(rows) == 1
          and (rows[0].get('crop'), rows[0].get('gen'), rows[0].get('pool'))
          == ('audit:gate', 2, P2), (r.json, rows))
    check(pre + 'G6 ...re-cut into the gen-2 edited/ (blue), and re-filed',
          colour(os.path.join(g2, 'edited', '100_0.jpg')) == 'blue'
          and (r.json or {}).get('refiled') is True, r.json)
    check(pre + 'G6 corrections(gen-1 ctx) never sees the gen-2 redraw, and '
          'corrections(gen-2 ctx) does',
          audit.corrections(c1) == {} and ('100', 0) in audit.corrections(c2))
    check(pre + 'G6 crop_sources.read_audit_corrections reads it for gen 2 '
          'only', ('100', 0) in cs.read_audit_corrections('gate', gen=2)
          and cs.read_audit_corrections('gate', gen=1) == {})
    # a gen-2 row of ANOTHER gen-2 pool (a later model's audit dir keys the
    # same det_idx) is not this pool's redraw
    with open(bf, 'a') as fh:
        fh.write(json.dumps({'crop': 'audit:gate', 'image_id': '101',
                             'det_idx': 0, 'x1': 1, 'y1': 1, 'x2': 9,
                             'y2': 9, 'gen': 2, 'pool': 'P-other'}) + '\n')
    check(pre + 'G6 corrections(gen-2 ctx) skips another gen-2 pool\'s row',
          ('101', 0) not in audit.corrections(c2))
    import collections
    cnt = collections.Counter()
    got = cs.read_audit_corrections('gate', gen=2, counts=cnt)
    check(pre + 'G6 ...and so does crop_sources\' reader, and counts it',
          ('101', 0) not in got and cnt['other_pool'] == 1, (got, cnt))
    # and record() checks the pool itself, not only the route in front of it
    n2 = len(lines(vf2))
    out = audit.record('100#0', 'dog', c2, pool='g1-legacy', by='x')
    check(pre + 'G5 record() refuses another pool\'s page by itself',
          not out.get('ok') and len(lines(vf2)) == n2, out)
    # a pool replaced while the request was in flight: c2 was resolved
    # against P2, and pool.json now names another pool
    pj = os.path.join(g2, 'pool.json')
    keep = open(pj, 'rb').read()
    with open(pj, 'w') as fh:
        json.dump(dict(json.loads(keep), pool_id='P2-swapped'), fh)
    nb = len(lines(bf))
    try:
        try:
            out = audit.record('100#0', 'not_dog', c2, pool=P2, by='x')
        except Exception as e:                # noqa: BLE001
            out = {'raised': repr(e)}
        try:
            out2 = audit.save_correction('100#0', [285, 285, 395, 395], c2,
                                         pool=P2, box_sig=list(G2_BOX),
                                         by='x')
        except Exception as e:                # noqa: BLE001
            out2 = {'raised': repr(e)}
    finally:
        with open(pj, 'wb') as fh:
            fh.write(keep)
        fa._META.clear()
    check(pre + 'G5 a verdict against a pool replaced mid-request is '
          'refused (409) and writes nothing',
          not out.get('ok') and out.get('status') == 409
          and len(lines(vf2)) == n2, out)
    check(pre + 'G6 so is a redraw', not out2.get('ok')
          and out2.get('status') == 409 and len(lines(bf)) == nb, out2)

    # G9: the gen-2 leash tile is the DINOv3 framing, full/ cut_crops'
    r = request(dash, 'POST', '/api/audit/draw?stage=leash',
                {'band': 'all', 'n': 25, 'gen': 2, 'pool': 'P2-leash'})
    audit.await_prefetch(fa.ctx('leash', 2))
    from PIL import Image
    import leash_vit
    tile = os.path.join(l2, 'crops', '100_0.jpg')
    full = os.path.join(l2, 'full', '100_0.jpg')
    ok = r.status == 200 and os.path.exists(tile) and os.path.exists(full)
    if ok:
        a, b, c, d = leash_vit.rect_pad(G2_BOX, 800, 600, 0.5)
        pad = int(audit.PAD_FRAC * 80) + audit.PAD_PX
        tw, th = Image.open(tile).size
        fw, fh_ = Image.open(full).size
        ok = ((tw, th) == (c - a, d - b) and (fw, fh_) == (80 + 2 * pad,
                                                          80 + 2 * pad)
              and (tw, th) != (fw, fh_))
        detail = ((tw, th), (c - a, d - b), (fw, fh_))
    else:
        detail = (r.status, r.json)
    check(pre + 'G9 the gen-2 leash tile is pad050, its full/ pad012', ok,
          detail)
    try:
        got = cs.framing_check_audit(
            [{'uid': 'la_100_d0', 'framing': 'audit_full',
              'box': list(G2_BOX), 'path': full,
              'frame': {'path': fx['frame']}}], minimum=1,
            say=lambda s: None)
        ok, detail = got.get('checked') == 1, got
    except (SystemExit, Exception) as e:      # noqa: BLE001
        ok, detail = False, repr(e)
    check(pre + 'G9 crop_sources.framing_check_audit passes on the gen-2 '
          'leash full/', ok, detail)

    # G10: a store re-judged by another model closes the live pool to draws
    shard = os.path.join(data, 'gate', 'gate-00000.parquet')
    keep_shard = open(shard, 'rb').read()
    _store_shard(shard, 'dogbin_m3')
    fa._MODELS.clear()
    nd = len(lines(os.path.join(g2, 'drawn.jsonl')))
    r = request(dash, 'POST', '/api/audit/draw?stage=gate',
                {'band': 'all', 'n': 25, 'gen': 2, 'pool': P2})
    try:
        mm = fa.pool_info('gate', gen=2).get('model_moved')
    except Exception as e:                    # noqa: BLE001
        mm = repr(e)
    check(pre + 'G10 a gen-2 pool whose store is now judged by another '
          'model reports model_moved and takes no draw (409)',
          r.status == 409 and mm is True
          and len(lines(os.path.join(g2, 'drawn.jsonl'))) == nd,
          (r.status, r.json, mm))
    with open(shard, 'wb') as fh:
        fh.write(keep_shard)
    fa._MODELS.clear()

    # G1': gen 1 untouched by all of it
    after = {d: tree_sha(d) for d in g1.values()}
    check(pre + 'G1\' every gen-1 file is byte-identical after the gen-2 '
          'traffic', after == before,
          [d for d in g1.values() if after[d] != before[d]])
    # ...and the frozen gen 1 still takes an answer on a key it showed
    r = request(dash, 'POST', '/api/audit/verdict?stage=gate',
                {'key': '100#0', 'verdict': 'not_dog', 'gen': 1,
                 'pool': 'g1-legacy'})
    rows = lines(os.path.join(g1['gate'], 'verdicts.jsonl'))
    check(pre + 'G5 a verdict on the frozen gen 1 lands there, tagged gen 1, '
          'with the gen-1 box',
          r.status == 200 and (r.json or {}).get('ok')
          and (rows[-1].get('gen'), rows[-1].get('pool'), rows[-1].get('box'))
          == (1, 'g1-legacy', list(G1_BOX))
          and len(lines(vf2)) == 1, (r.json, rows[-1:]))

    # G13: a target counts both generations' answers, once each
    got = (dash.done_by(SESSION['username'], 'gate'),
           dash.done_by(SESSION['username'], 'leash'))
    check(pre + 'G13 done_by counts the gen-1 and the gen-2 answer (2), and '
          'nothing on the leash', got == (2, 0), got)

    g12_checks(fx, w, pre)
    tail_checks(fx, w, pre)
    gap_checks(fx, w, pre)
    return len(FAILS) == n0


def _restamp(pool_dir):
    """Re-record a rewritten gen >= 2 pool.parquet's size, mtime and sha in
    its pool.json (the pool id stays): the fixture's way to add a key."""
    pq = os.path.join(pool_dir, 'pool.parquet')
    pj = os.path.join(pool_dir, 'pool.json')
    st = os.stat(pq)
    doc = json.load(open(pj))
    doc.update(parquet_size=st.st_size, parquet_mtime_ns=st.st_mtime_ns,
               sha256=hashlib.sha256(open(pq, 'rb').read()).hexdigest())
    with open(pj, 'w') as fh:
        json.dump(doc, fh)


def tail_checks(fx, w, pre):
    """What the reviews of the first version found, each pinned (after G1'
    and G12, so the gen-1 files they add change nothing checked before):

      W1  a redraw of a pool key on no page is refused -- gen 2 and the
          frozen gen 1 -- and writes neither boxes.jsonl nor edited/
      W2  a crop GET of a pool key on no page cuts nothing (404), while a
          drawn key whose crop went missing is cut again
      W3  a row of another generation in the gen-1 ledger is not taken into
          the page index (a verdict on its key is a 409, nothing written);
          a foreign row in the gen-2 ledger stops a redraw BEFORE it lands
      W4  a page document or drawn row of another generation in the gen-2
          dir is a 409 for the page, the verdict and the draw
      W5  place() leaves a dataset dir another pool marked alone
      W6  "None" posted to the frozen gen 1, whose ledger holds that key,
          is refused
      W7  a page drawn after the index was built is in it (incremental),
          and done_by inside dash._counting() counts gen 1 (1) and gen 2
          (2) as 3 -- the per-generation memo key
      W8  a redraw against a pool replaced mid-request is refused even with
          an empty ledger (save_correction's own re-read of the pool)
      W9  other_family_vetoes reads the build's own generation, every veto
          tagged gen and pool; vetoes of two generations stop
          apply_box_vetoes
      W10 build_dogdet_v3.corrections() takes a gen-2 audit redraw only on
          the store its pool was cut from
      W11 an API GET with no gen is gen 1; a crop URL whose token and
          ?gen= disagree is a 400; the dashboard watches fn_audit.py; the
          POST handler answers even when audit.py has no Refused (an
          install half done)
    """
    fa, audit, dash, cs = w.fa, w.audit, w.dash, w.cs
    data = fx['data']
    g1 = os.path.join(data, 'fn_audit')
    g2 = os.path.join(data, 'fn_audit_gen2')
    l2 = os.path.join(data, 'leash_audit_gen2')
    P2, L2 = 'P2-gate', 'P2-leash'
    bf = audit.BOX_FILE
    vf1 = os.path.join(g1, 'verdicts.jsonl')
    vf2 = os.path.join(g2, 'verdicts.jsonl')
    audit.await_prefetch(fa.ctx('gate', 2))

    # 100#1 joins both gate pools, on no page of either: (240,240,440,440)
    # over the blue patch in gen 2, its own gen-1 box over grey in gen 1
    _parquet(os.path.join(g2, 'pool.parquet'), _pool_rows(G2_BOX, 0.72) + [
        ('100', 1, 0.66, 240., 240., 440., 440., 0.7, 'c', 'd', 's102', 6)],
        POOL_COLS)
    _restamp(g2)
    _parquet(os.path.join(g1, 'pool.parquet'), _pool_rows(G1_BOX, 0.31) + [
        ('100', 1, 0.2, 500., 300., 700., 500., 0.7, 'c', 'd', 's101', 2)],
        POOL_COLS)
    fa._META.clear()

    # ── W1: redraws of a key nobody was shown ──
    for gen, pool, d in ((2, P2, g2), (1, 'g1-legacy', g1)):
        f = request(dash, 'GET', f'/audit/frame/gate/g{gen}/100_1.jpg?pool='
                    + pool)
        sig = (json.loads(f.headers.get('X-Audit-Meta') or 'null')
               or {}).get('box_sig')
        nb = lines(bf)
        ed = sorted(os.listdir(os.path.join(d, 'edited'))) \
            if os.path.isdir(os.path.join(d, 'edited')) else []
        r = request(dash, 'POST', '/api/audit/box?stage=gate',
                    {'key': '100#1', 'gen': gen, 'pool': pool,
                     'box_sig': sig, 'box': [250, 250, 430, 430]})
        ed2 = sorted(os.listdir(os.path.join(d, 'edited'))) \
            if os.path.isdir(os.path.join(d, 'edited')) else []
        check(pre + f'W1 a gen-{gen} redraw of a pool key on no page is '
              f'refused (409) and writes neither boxes.jsonl nor edited/',
              sig is not None and r.status == 409 and lines(bf) == nb
              and ed2 == ed, (sig, r.status, r.json, ed2))

    # ── W2: a crop GET cuts only what was shown ──
    before = {sub: sorted(os.listdir(os.path.join(g1, sub)))
              for sub in ('crops', 'full')}
    r = request(dash, 'GET', '/audit/crop/gate/g1/100_1.jpg')
    after = {sub: sorted(os.listdir(os.path.join(g1, sub)))
             for sub in ('crops', 'full')}
    check(pre + 'W2 a crop GET of a gen-1 pool key on no page is a 404 and '
          'cuts nothing into the frozen dir', r.status == 404
          and after == before, (r.status, after))
    os.remove(os.path.join(g1, 'crops', '100_0.jpg'))
    r = request(dash, 'GET', '/audit/crop/gate/g1/100_0.jpg')
    check(pre + 'W2 ...while a drawn key whose crop went missing is cut '
          'again (red, gen 1)', r.status == 200 and colour(r.body) == 'red',
          r.status)

    # ── W7: a page drawn after the index was built, and the work count ──
    c2 = fa.ctx('gate', 2)
    audit._page_index(c2)               # built: 100#0 only
    r = request(dash, 'POST', '/api/audit/draw?stage=gate',
                {'band': 'all', 'n': 25, 'gen': 2, 'pool': P2})
    audit.await_prefetch(c2)
    keys = [it.get('key') for it in ((r.json or {}).get('page') or {})
            .get('items', [])]
    r = request(dash, 'POST', '/api/audit/verdict?stage=gate',
                {'key': '100#1', 'verdict': 'dog', 'gen': 2, 'pool': P2})
    check(pre + 'W7 a key on a page drawn after the index was built is in '
          'it: its verdict lands', keys == ['100#1'] and r.status == 200
          and (r.json or {}).get('ok')
          and lines(vf2)[-1].get('box') == [240.0, 240.0, 440.0, 440.0],
          (keys, r.status, r.json))
    with dash._counting():
        got = dash.done_by(SESSION['username'], 'gate')
    check(pre + 'W7 inside dash._counting(), done_by counts gen 1 (1) and '
          'gen 2 (2) answers as 3', got == 3, got)

    # ── W3: foreign rows in a ledger ──
    keep1 = open(vf1, 'rb').read()
    with open(vf1, 'a') as fh:
        fh.write(json.dumps({'key': '100#7', 'verdict': 'dog', 'gen': 2,
                             'pool': P2, 'box': list(G2_BOX), 'p_dog': 0.72,
                             'band': 7, 'seq': 's100'}) + '\n')
    mid = open(vf1, 'rb').read()
    audit._INDEX.clear()                # as after a restart
    r = request(dash, 'POST', '/api/audit/verdict?stage=gate',
                {'key': '100#7', 'verdict': 'not_dog', 'gen': 1,
                 'pool': 'g1-legacy'})
    check(pre + 'W3 a gen-2 row in the gen-1 ledger is not indexed: a gen-1 '
          'verdict on its key is a 409 and appends nothing',
          r.status == 409 and open(vf1, 'rb').read() == mid,
          (r.status, r.json))
    with open(vf1, 'wb') as fh:
        fh.write(keep1)
    audit._INDEX.clear()
    keep2 = open(vf2, 'rb').read()
    with open(vf2, 'a') as fh:
        fh.write(json.dumps({'key': '100#0', 'verdict': 'dog'}) + '\n')
    nb = lines(bf)
    ed = tree_sha(os.path.join(g2, 'edited'))
    r = request(dash, 'POST', '/api/audit/box?stage=gate',
                {'key': '100#0', 'gen': 2, 'pool': P2,
                 'box_sig': list(G2_BOX), 'box': [292, 292, 392, 392]})
    check(pre + 'W3 an untagged row in the gen-2 ledger stops a redraw '
          'before it lands (409; boxes.jsonl and edited/ unchanged)',
          r.status == 409 and lines(bf) == nb
          and tree_sha(os.path.join(g2, 'edited')) == ed, (r.status, r.json))
    with open(vf2, 'wb') as fh:
        fh.write(keep2)

    # ── W4: a page document / drawn row of another generation ──
    n_pages = audit.page_count(c2)
    bad_page = os.path.join(g2, 'pages', '%05d.json' % n_pages)
    with open(bad_page, 'w') as fh:
        json.dump({'index': n_pages, 'items': [
            {'key': '777#0', 'image_id': '777', 'det_idx': 0, 'p_dog': 0.5,
             'band': 5, 'seq': 's777'}]}, fh)
    try:
        r = request(dash, 'GET', f'/api/audit/page?stage=gate&gen=2&pool={P2}'
                    f'&i={n_pages}')
        check(pre + 'W4 an untagged page document in the gen-2 dir is a 409 '
              'for the page', r.status == 409, (r.status, r.json))
        n2 = len(lines(vf2))
        r = request(dash, 'POST', '/api/audit/verdict?stage=gate',
                    {'key': '777#0', 'verdict': 'dog', 'gen': 2, 'pool': P2})
        check(pre + 'W4 ...and for a verdict on its key, which writes '
              'nothing', r.status == 409 and len(lines(vf2)) == n2,
              (r.status, r.json))
    finally:
        os.remove(bad_page)
    dl = os.path.join(g2, 'drawn.jsonl')
    keepd = open(dl, 'rb').read()
    with open(dl, 'a') as fh:
        fh.write(json.dumps({'key': '555#0', 'seq': 's555'}) + '\n')
    mid = open(dl, 'rb').read()
    np_ = audit.page_count(c2)
    r = request(dash, 'POST', '/api/audit/draw?stage=gate',
                {'band': 'all', 'n': 25, 'gen': 2, 'pool': P2})
    audit.await_prefetch(c2)
    check(pre + 'W4 an untagged row in the gen-2 drawn log is a 409 for a '
          'draw, which reserves and writes nothing',
          r.status == 409 and open(dl, 'rb').read() == mid
          and audit.page_count(c2) == np_, (r.status, r.json))
    with open(dl, 'wb') as fh:
        fh.write(keepd)

    # ── W5: a dataset dir another pool marked ──
    ds2 = os.path.join(data, 'audit_finds_gen2')
    mk = os.path.join(ds2, fa.DATASET_MARKER)
    keepm = open(mk, 'rb').read()
    with open(mk, 'w') as fh:
        json.dump({'stage': 'gate', 'gen': 2, 'pool_id': 'P-other',
                   'detect_root': fx['stores'][2]}, fh)
    ds_before = tree_sha(ds2)
    n2 = len(lines(vf2))
    r = request(dash, 'POST', '/api/audit/verdict?stage=gate',
                {'key': '100#0', 'verdict': 'not_dog', 'gen': 2, 'pool': P2})
    check(pre + 'W5 a verdict whose dataset dir is marked for another pool '
          'is recorded, not filed: the dir is untouched',
          r.status == 200 and (r.json or {}).get('place_error')
          and len(lines(vf2)) == n2 + 1 and tree_sha(ds2) == ds_before,
          (r.status, r.json))
    with open(mk, 'wb') as fh:
        fh.write(keepm)

    # ── W6: "None" on gen 1 ──
    keep1 = open(vf1, 'rb').read()
    r = request(dash, 'POST', '/api/audit/verdict?stage=gate',
                {'key': 'None', 'verdict': 'dog', 'gen': 1,
                 'pool': 'g1-legacy'})
    check(pre + 'W6 the key "None" -- in the gen-1 ledger, on no page -- '
          'is refused on gen 1 (409) and appends nothing',
          r.status == 409 and open(vf1, 'rb').read() == keep1,
          (r.status, r.json))

    # ── W8: a pool swapped mid-request, empty ledger (leash) ──
    cl2 = fa.ctx('leash', 2)
    pj = os.path.join(l2, 'pool.json')
    keep = open(pj, 'rb').read()
    with open(pj, 'w') as fh:
        json.dump(dict(json.loads(keep), pool_id='P2-swapped'), fh)
    nb = len(lines(bf))
    try:
        try:
            out = audit.save_correction('100#0', [285, 285, 395, 395], cl2,
                                        pool=L2, box_sig=list(G2_BOX),
                                        by='x')
        except Exception as e:                # noqa: BLE001
            out = {'raised': repr(e)}
    finally:
        with open(pj, 'wb') as fh:
            fh.write(keep)
        fa._META.clear()
    check(pre + 'W8 a redraw against a pool replaced mid-request is refused '
          '(409) with nothing written, the ledger empty',
          not lines(os.path.join(l2, 'verdicts.jsonl'))
          and not out.get('ok') and out.get('status') == 409
          and len(lines(bf)) == nb, out)

    # ── W9: the other family's vetoes, per generation ──
    for gen, pool, v in ((2, L2, 'leashed'), (1, 'g1-legacy', 'unleashed')):
        r = request(dash, 'POST', '/api/audit/verdict?stage=leash',
                    {'key': '100#0', 'verdict': v, 'gen': gen, 'pool': pool})
    stubs = {'leash_call_candidates': lambda *a, **k: ([], None, None),
             'flag_candidates': lambda *a, **k: ([], None)}
    real = {k: getattr(cs, k) for k in stubs}
    try:
        for k, f in stubs.items():
            setattr(cs, k, f)
        got = {}
        for g in (1, 2):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    vs, _ = cs.other_family_vetoes('dogbin', fx['stores'][g])
                got[g] = [(x['label'], x.get('gen'), x.get('pool'), x['box'])
                          for x in vs]
            except (SystemExit, Exception) as e:  # noqa: BLE001
                got[g] = repr(e)
    finally:
        for k, f in real.items():
            setattr(cs, k, f)
    check(pre + 'W9 other_family_vetoes reads the build\'s own generation of '
          'the leash audit, each veto tagged with it and its pool box',
          got == {1: [('unleashed', 1, 'g1-legacy', list(G1_BOX))],
                  2: [('leashed', 2, L2, list(G2_BOX))]}, got)
    try:
        mixed = 'kept'
        if isinstance(got.get(1), list) and isinstance(got.get(2), list):
            vs = [{'image_id': '100', 'det_key': 'd0', 'box': b,
                   'label': lab, 'source': 'leash_audit', 'gen': g_,
                   'pool': p_} for lab, g_, p_, b in got[1] + got[2]]
            cs.apply_box_vetoes([], vs, {'dog', 'not_dog'}, 'w9')
    except SystemExit as e:
        mixed = str(e)
    except Exception as e:                    # noqa: BLE001
        mixed = 'CRASH ' + repr(e)
    check(pre + 'W9 vetoes of two generations stop apply_box_vetoes',
          'generations' in mixed, mixed)

    # ── W10: v3 takes a gen-2 audit redraw only on its pool's store ──
    v3 = w.v3
    v3.CORRECTIONS = bf
    got = {}
    for name, root in (('own', fx['stores'][2]), ('other', fx['other2'])):
        v3.DETECT_ROOT = root
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                got[name] = ('100', 0) in v3.corrections()
        except (SystemExit, Exception) as e:  # noqa: BLE001
            got[name] = repr(e)
    check(pre + 'W10 build_dogdet_v3.corrections() takes the gen-2 gate '
          'redraw on the store its pool was cut from, and not on another '
          'gen-2 store', got == {'own': True, 'other': False}, got)

    # ── W11: defaults and the install window ──
    r = request(dash, 'GET', '/api/audit/page?stage=gate&i=0')
    pg = (r.json or {}).get('page') or {}
    check(pre + 'W11 an API GET that names no gen is the gen-1 audit (a tab '
          'from before the switch)', r.status == 200
          and pg.get('gen') is None and pg.get('pool') is None
          and [it.get('key') for it in pg.get('items', [])] == ['100#0'],
          (r.status, pg))
    r = request(dash, 'GET', '/audit/crop/gate/g2/100_0.jpg?gen=1')
    check(pre + 'W11 a crop URL whose g<N> token and ?gen= disagree is a 400',
          r.status == 400, r.status)
    check(pre + 'W11 the dashboard re-execs when fn_audit.py is saved',
          "'detect', 'fn_audit.py')\n    if os.path.exists(_q):\n"
          "        _watched.add(_q)\n" in w.srcs.get('dash', ORIG['dash']))
    nd = len(lines(os.path.join(g1, 'drawn.jsonl')))
    refused = audit.Refused
    del audit.Refused
    try:
        try:
            r = request(dash, 'POST', '/api/audit/draw?stage=gate',
                        {'band': 'all', 'n': 25, 'gen': 1,
                         'pool': 'g1-legacy'})
            ok, why = (r.json or {}).get('ok') is False, (r.status, r.json)
        except Exception as e:                # noqa: BLE001
            ok, why = False, repr(e)
    finally:
        audit.Refused = refused
    check(pre + 'W11 with an audit.py that has no Refused (half an '
          'install), a refused draw is still a JSON refusal, nothing drawn',
          ok and len(lines(os.path.join(g1, 'drawn.jsonl'))) == nd, why)


def gap_checks(fx, w, pre):
    """Guards no earlier check could tell from their absence, because the
    fixture never made them matter (last, after tail_checks; each puts back
    what it changed):

      W12 a draw on a gen-2 pool whose store's model cannot be read (a shard
          that is not a parquet) is a 409 that says so, and draws nothing --
          while the pool still has an undrawn key, which the same draw
          serves once the store is readable again
      W13 a dataset marker naming this pool_id but another detect_root,
          stage or gen is not this pool's: place() files nothing and
          export refuses, the dir unchanged
      W14 pool.json replaced by one of the same size and mtime_ns (another
          pool_id) is read afresh -- the pool_meta cache is keyed on inode
          and ctime too -- and a POST carrying the old pool is a 409
      W15 the page-index memo is per generation even when the gen-1 and
          gen-2 page dirs carry the same stamp (page count, newest mtime):
          a gen-2 verdict on a key only gen 1 showed is a 409
      W16 a gen >= 2 pool cuts from the roots its pool.json recorded, not
          from data/gate/plan.json's: a missing gen-2 crop is cut again from
          the right (blue) frame while the plan names a wrong (green) one
    """
    from PIL import Image
    fa, audit, dash = w.fa, w.audit, w.dash
    data = fx['data']
    g2 = os.path.join(data, 'fn_audit_gen2')
    l1 = os.path.join(data, 'leash_audit')
    l2 = os.path.join(data, 'leash_audit_gen2')
    P2, L2 = 'P2-gate', 'P2-leash'
    vf2 = os.path.join(g2, 'verdicts.jsonl')
    c2 = fa.ctx('gate', 2)
    audit.await_prefetch(c2)

    # ── W12: a store whose model cannot be read ──
    # 100#2 joins the gate pool on no page, so a draw that went ahead would
    # have something to draw: "nothing drawn" is then the refusal's doing
    _parquet(os.path.join(g2, 'pool.parquet'), _pool_rows(G2_BOX, 0.72) + [
        ('100', 1, 0.66, 240., 240., 440., 440., 0.7, 'c', 'd', 's102', 6),
        ('100', 2, 0.55, 600., 100., 700., 200., 0.7, 'c', 'd', 's103', 5)],
        POOL_COLS)
    _restamp(g2)
    fa._META.clear()
    shard = os.path.join(data, 'gate', 'gate-00000.parquet')
    keep_shard = open(shard, 'rb').read()
    dl = os.path.join(g2, 'drawn.jsonl')
    keepd = open(dl, 'rb').read()
    pages = sorted(os.listdir(os.path.join(g2, 'pages')))
    with open(shard, 'wb') as fh:
        fh.write(b'not a parquet file')
    fa._MODELS.clear()
    try:
        r = request(dash, 'POST', '/api/audit/draw?stage=gate',
                    {'band': 'all', 'n': 25, 'gen': 2, 'pool': P2})
        audit.await_prefetch(c2)
    finally:
        with open(shard, 'wb') as fh:
            fh.write(keep_shard)
        fa._MODELS.clear()
    check(pre + 'W12 a draw on a gen-2 pool whose store\'s model cannot be '
          'read is a 409 that says so, and draws nothing',
          r.status == 409 and 'cannot be read' in str((r.json or {}).get('msg'))
          and open(dl, 'rb').read() == keepd
          and sorted(os.listdir(os.path.join(g2, 'pages'))) == pages,
          (r.status, r.json))
    r = request(dash, 'POST', '/api/audit/draw?stage=gate',
                {'band': 'all', 'n': 25, 'gen': 2, 'pool': P2})
    audit.await_prefetch(c2)
    check(pre + 'W12 ...and with the store readable again the same draw '
          'serves the undrawn key', r.status == 200
          and [it.get('key') for it in ((r.json or {}).get('page') or {})
               .get('items', [])] == ['100#2'], (r.status, r.json))

    # ── W13: a marker of this pool_id that is not this pool ──
    ds2 = os.path.join(data, 'audit_finds_gen2')
    mk = os.path.join(ds2, fa.DATASET_MARKER)
    keepm = open(mk, 'rb').read()
    try:
        for what, edit in (('another detect_root',
                            {'detect_root': fx['other2']}),
                           ('another stage', {'stage': 'leash'}),
                           ('another gen', {'gen': 3})):
            doc = dict(json.loads(keepm), **edit)
            with open(mk, 'w') as fh:
                json.dump(doc, fh)
            ds_before = tree_sha(ds2)
            n2 = len(lines(vf2))
            r = request(dash, 'POST', '/api/audit/verdict?stage=gate',
                        {'key': '100#0', 'verdict': 'not_dog', 'gen': 2,
                         'pool': P2})
            check(pre + f'W13 a marker naming this pool_id but {what} is not '
                  f'this pool\'s: the verdict is recorded, nothing filed',
                  doc.get('pool_id') == P2 and r.status == 200
                  and (r.json or {}).get('place_error')
                  and len(lines(vf2)) == n2 + 1
                  and tree_sha(ds2) == ds_before, (r.status, r.json))
            if what == 'another detect_root':
                why = do_export(fa, 'gate', 2)
                check(pre + 'W13 ...and export refuses that dir, deleting '
                      'nothing', 'belongs to' in why
                      and tree_sha(ds2) == ds_before, why)
    finally:
        with open(mk, 'wb') as fh:
            fh.write(keepm)

    # ── W14: pool.json replaced by one of the same size and mtime ──
    # os.replace of a temp file gives a new inode (and ctime); the size and
    # mtime_ns are made the old file's, so only inode/ctime tell the two
    # apart -- what a rewrite within one mtime tick looks like
    pj = os.path.join(g2, 'pool.json')
    keep = open(pj, 'rb').read()
    st0 = os.stat(pj)
    fa._META.clear()
    first = fa.pool_meta('gate', 2)['pool_id']     # cached now
    swapped = keep.replace(b'"P2-gate"', b'"Q2-gate"')
    n2 = len(lines(vf2))
    try:
        with open(pj + '.swap', 'wb') as fh:
            fh.write(swapped)
        os.utime(pj + '.swap', ns=(st0.st_atime_ns, st0.st_mtime_ns))
        os.replace(pj + '.swap', pj)
        st1 = os.stat(pj)
        try:
            now = fa.pool_meta('gate', 2)['pool_id']
        except Exception as e:                # noqa: BLE001
            now = repr(e)
        r = request(dash, 'POST', '/api/audit/verdict?stage=gate',
                    {'key': '100#0', 'verdict': 'dog', 'gen': 2, 'pool': P2})
    finally:
        with open(pj + '.swap', 'wb') as fh:
            fh.write(keep)
        os.replace(pj + '.swap', pj)
        fa._META.clear()
    check(pre + 'W14 a pool.json replaced by another pool\'s of the same size '
          'and mtime_ns is read afresh (the cache is keyed on inode and '
          'ctime too)', keep.count(b'"P2-gate"') == 1
          and len(swapped) == len(keep)
          and (st1.st_size, st1.st_mtime_ns) == (st0.st_size, st0.st_mtime_ns)
          and first == P2 and now == 'Q2-gate', (first, now))
    check(pre + 'W14 ...and a verdict carrying the old pool is a 409 that '
          'writes nothing', r.status == 409 and len(lines(vf2)) == n2,
          (r.status, r.json))

    # ── W15: two generations' page dirs with the same stamp ──
    # the leash: one page in each generation. Gen 1's page gains 100#3, a key
    # on no gen-2 page; the page counts are evened out (empty pages, each
    # tagged as its dir's) and every page file gets one mtime, so the two
    # dirs' (count, newest mtime) stamps are equal
    cl1, cl2 = fa.ctx('leash', 1), fa.ctx('leash', 2)
    pd1, pd2 = os.path.join(l1, 'pages'), os.path.join(l2, 'pages')
    p1 = os.path.join(pd1, '00000.json')
    keep_p1 = open(p1, 'rb').read()
    added = []
    lv2 = os.path.join(l2, 'verdicts.jsonl')
    try:
        doc = json.loads(keep_p1)
        doc['items'] = doc['items'] + [
            {'key': '100#3', 'image_id': '100', 'det_idx': 3, 'p_dog': 0.4,
             'conf': 0.8, 'band': 4, 'seq': 's103', 'drive': 'd',
             'cell': 'c'}]
        with open(p1, 'w') as fh:
            json.dump(doc, fh)
        for d, tags in ((pd1, {}), (pd2, {'gen': 2, 'pool': L2})):
            other = pd2 if d == pd1 else pd1
            while len(glob_json(d)) < len(glob_json(other)):
                f = os.path.join(d, '%05d.json' % len(glob_json(d)))
                with open(f, 'w') as fh:
                    json.dump(dict(tags, index=len(glob_json(d)), items=[]),
                              fh)
                added.append(f)
        t = 1_700_000_000_000_000_000
        for f in glob_json(pd1) + glob_json(pd2):
            os.utime(f, ns=(t, t))

        # the stamp _page_index keys its memo on
        m1 = [os.path.getmtime(f) for f in glob_json(pd1)]
        m2 = [os.path.getmtime(f) for f in glob_json(pd2)]
        stamp_eq = (len(m1), max(m1, default=0)) == (len(m2),
                                                     max(m2, default=0))
        audit._INDEX.clear()
        on1 = '100#3' in audit._page_index(cl1)    # gen 1 memoized first
        nl2 = len(lines(lv2))
        r = request(dash, 'POST', '/api/audit/verdict?stage=leash',
                    {'key': '100#3', 'verdict': 'leashed', 'gen': 2,
                     'pool': L2})
        on2 = '100#3' in audit._page_index(cl2)
    finally:
        with open(p1, 'wb') as fh:
            fh.write(keep_p1)
        for f in added:
            os.remove(f)
        audit._INDEX.clear()
    check(pre + 'W15 with gen 1 and gen 2 page dirs of one stamp, a gen-2 '
          'verdict on a key only gen 1 showed is a 409 that writes nothing '
          '(the index memo is per generation)',
          stamp_eq and on1 and not on2 and r.status == 409
          and len(lines(lv2)) == nl2, (stamp_eq, on1, on2, r.status, r.json))

    # ── W16: gen 2 cuts from its pool's own roots ──
    # the plan now names a grid whose frame 100 is all green; gen 2's
    # pool.json still names the right one. A missing gen-2 leash tile (100#0
    # is on a gen-2 leash page) is cut again on the next GET
    wrong = os.path.join(fx['tmp'], 'frames_wrong')
    wdir = os.path.join(wrong, 'c', 'ground_animal_images')
    os.makedirs(wdir, exist_ok=True)
    Image.new('RGB', (800, 600), (20, 200, 20)).save(
        os.path.join(wdir, '100.jpg'), quality=98)
    plan = os.path.join(data, 'gate', 'plan.json')
    keep_plan = open(plan, 'rb').read()
    tile = os.path.join(l2, 'crops', '100_0.jpg')
    try:
        with open(plan, 'w') as fh:
            json.dump(dict(json.loads(keep_plan), roots={'d': wrong}), fh)
        os.remove(tile)
        r = request(dash, 'GET', '/audit/crop/leash/g2/100_0.jpg')
    finally:
        with open(plan, 'wb') as fh:
            fh.write(keep_plan)
    check(pre + 'W16 a gen-2 crop cut again reads the frame from its '
          'pool.json roots, not the plan\'s (blue, not green)',
          r.status == 200 and colour(r.body) == 'blue'
          and colour(tile) == 'blue', (r.status, colour(r.body)))


def glob_json(d):
    """The page files of pages dir `d`, sorted."""
    return sorted(f for f in (os.path.join(d, n) for n in os.listdir(d))
                  if f.endswith('.json'))


def g12_checks(fx, w, pre):
    """G12: the builders read their own generation's audit and no other."""
    import duckdb
    fa, cs, bd, rb = w.fa, w.cs, w.bd, w.rb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEM}'")
    out = {}
    try:
        for g in (1, 2):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    out[g] = cs.audit_candidates(
                        'gate', con, detect_root=fx['stores'][g])
            except (SystemExit, Exception) as e:  # noqa: BLE001
                out[g] = ([], {'error': repr(e)})
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cs.audit_candidates('gate', con, detect_root=fx['other2'])
            stray = 'read'
        except SystemExit as e:
            stray = str(e)
        except Exception as e:                # noqa: BLE001
            stray = 'CRASH ' + repr(e)
    finally:
        con.close()
    rows1, c1 = out[1]
    rows2, c2 = out[2]
    a1 = os.path.join(fx['data'], 'fn_audit') + os.sep
    a2 = os.path.join(fx['data'], 'fn_audit_gen2') + os.sep
    check(pre + 'G12 audit_candidates on the gen-1 store reads the gen-1 '
          'audit only: its crop, its box, provenance gen 1',
          [(r['label'], r['src_path'].startswith(a1), r['pool_box'],
            r['provenance'].get('gen'), r['provenance'].get('pool'))
           for r in rows1] == [('not_dog', True, list(G1_BOX), 1,
                                'g1-legacy')]
          and c1.get('other_gen_verdicts') == 1, (rows1, dict(c1)))
    check(pre + 'G12 ...on the gen-2 store, the gen-2 audit only: the '
          'redrawn gen-2 crop over the gen-2 pool box, provenance gen 2',
          [(r['label'], r['src_path'].startswith(a2), r['pool_box'],
            r['box'], r['provenance'].get('gen'),
            r['provenance'].get('pool')) for r in rows2]
          == [('dog', True, list(G2_BOX), [290.0, 290.0, 390.0, 390.0], 2,
               'P2-gate')]
          and c2.get('other_gen_verdicts') == 2, (rows2, dict(c2)))
    check(pre + 'G12 a gen-2 store the pool was not cut from is SystemExit',
          'was cut from' in stray, stray)
    try:
        cs.apply_box_vetoes(rows2, [], {'dog', 'not_dog'}, 'g12')
        one = True
    except (SystemExit, Exception):           # noqa: BLE001
        one = False
    try:
        cs.apply_box_vetoes(rows1 + rows2, [], {'dog', 'not_dog'}, 'g12')
        mixed = 'kept'
    except SystemExit as e:
        mixed = str(e)
    except Exception as e:                    # noqa: BLE001
        mixed = 'CRASH ' + repr(e)
    check(pre + 'G12 apply_box_vetoes takes one generation, and stops on '
          'two', one and 'generations' in mixed, mixed)

    def plan_rows(rows):
        return [{'uid': 'ga_%s_d0_%d' % (r['image_id'], n), 'split': 'train',
                 'label': r['label'], 'image_id': r['image_id'],
                 'det_key': r['det_key'], 'sequence': 's%d' % n,
                 'seq_status': 'ok', 'source': r['source'],
                 'provenance': r['provenance'], 'framing': r['framing'],
                 'sha256': 'x%d' % n} for n, r in enumerate(rows)]
    try:
        f1 = rb.check_rows(plan_rows(rows2))
        f2 = rb.check_rows(plan_rows(rows1 + rows2))
    except Exception as e:                    # noqa: BLE001
        f1, f2 = ['CRASH ' + repr(e)], []
    check(pre + 'G12 rebuild_crop_dataset.check_rows fails a plan of two '
          'generations, and only that one',
          not any('generation' in x for x in f1)
          and any('detector generations' in x for x in f2), (f1, f2))
    bad = []
    for g in (1, 2):
        s = bd.stores(g)
        for name, stage in (('gate_audit', 'gate'), ('leash_audit', 'leash')):
            P = fa.paths(stage, gen=g)
            want = {name: (P['verdicts'], P['full'], g),
                    name + '_edited': (P['pool'], P['edited'], g)}
            for k, v in want.items():
                got = (s[k]['ledger'], s[k]['crops'], s[k].get('gen'))
                if got != v:
                    bad.append((g, k, got, v))
    try:
        same = bd.stores_for(fx['stores'][2]) == bd.stores(2)
    except (SystemExit, Exception) as e:      # noqa: BLE001
        same = repr(e)
    check(pre + 'G12 build_dataset.stores(g) digests exactly fa.paths(..., '
          'gen=g), and stores_for(gen-2 store) is stores(2)',
          not bad and same is True, (bad, same))


# ── G14: what the sources say ───────────────────────────────────────────────
# fn_audit calls that read or write ONE generation, and must name it
GEN_KW = {'paths', 'read_verdicts', 'summarise', 'pool_info', 'band_totals',
          'sheets', 'targeted', 'pool_gen'}
GEN_POS = {'pool_meta', 'ctx', 'is_frozen'}
CTX_FUNCS = ('sample', 'materialise', 'draw_page', 'prefetch',
             'await_prefetch', 'api_draw', 'api_page', 'record', 'place',
             'save_correction', 'corrections', 'frame_view', 'crop_path',
             'judged', 'stats', 'class_split', 'page_count', 'get_page',
             'pool_ready', 'with_verdicts', 'page_html')
AUDIT_NAMES = ('fn_audit', 'leash_audit', 'audit_finds', 'audit_finds_leash')


def _docstrings(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef,
                          ast.AsyncFunctionDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) \
                and isinstance(n.body[0].value, ast.Constant):
            out.add(id(n.body[0].value))
    return out


def scan_audit(src):
    """[problem] in audit.py's source."""
    tree = ast.parse(src)
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and isinstance(n.func.value, ast.Name) \
                and n.func.value.id == 'fa':
            a = n.func.attr
            if a == 'paths':
                bad.append(f'L{n.lineno} fa.paths() -- audit.py reads ctx.P')
            elif a in GEN_KW and not any(k.arg == 'gen' for k in n.keywords):
                bad.append(f'L{n.lineno} fa.{a}() names no gen')
            elif a in GEN_POS and len(n.args) + len(n.keywords) < 2:
                bad.append(f'L{n.lineno} fa.{a}() names no gen')
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for f in CTX_FUNCS:
        d = defs.get(f)
        if d is None:
            bad.append(f'no {f}()')
        elif 'ctx' not in [a.arg for a in d.args.args + d.args.kwonlyargs]:
            bad.append(f'{f}() takes no ctx')
    return bad


def scan_paths(name, src):
    """[problem]: an audit dir or dataset spelled out as a path."""
    tree = ast.parse(src)
    docs = _docstrings(tree)
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            for a in n.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and (a.value in AUDIT_NAMES or any(
                            a.value.startswith(b + '_gen')
                            for b in AUDIT_NAMES)):
                    bad.append(f'{name} L{a.lineno} {a.value!r} as a path '
                               f'part')
        if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                and id(n) not in docs \
                and any(f'{b}/' in n.value or f'data/{b}' in n.value
                        for b in AUDIT_NAMES):
            bad.append(f'{name} L{n.lineno} {n.value[:60]!r}')
    return bad


def scan_checks(srcs, pre=''):
    bad = scan_audit(srcs['audit'])
    check(pre + 'G14 audit.py: every per-generation fn_audit call names its '
          'gen, every pool function takes a ctx, no fa.paths()', not bad, bad)
    bad = []
    for k in ('llm', 'el', 'bd', 'cs', 'v3'):
        bad += scan_paths(os.path.basename(SOURCES[k]), srcs[k])
    check(pre + 'G14 no audit dir or dataset path is spelled out in '
          'llm_annotate, eval_leash, build_dataset, crop_sources or '
          'build_dogdet_v3 (fa.paths names them)', not bad, bad)


# ── one whole pass ──────────────────────────────────────────────────────────

def run_all(srcs, pre=''):
    """Every check against the sources `srcs`; True when all passed."""
    n0 = len(FAILS)
    scan_checks(srcs, pre)
    w = World({k: v for k, v in srcs.items() if v is not ORIG.get(k)})
    fx = None
    try:
        layout_checks(w.fa, w.cs, pre)
        build_checks(w.fa, w.cs, pre)
        fx = fixture()
        w.point(fx)
        run(fx, w, pre)
    finally:
        try:
            w.audit.await_prefetch(w.fa.ctx('gate', 2), timeout=30)
        except Exception:                     # noqa: BLE001
            pass
        w.close()
        if fx is not None:
            shutil.rmtree(fx['tmp'], ignore_errors=True)
    return len(FAILS) == n0


ORIG = {}


def S(kind, old, new):
    return (kind, old, new)


# Each mutant removes or loosens ONE guard. A tuple of edits applies them all;
# every `old` must occur exactly once in the source it names.
MUTANTS = [
    # G1
    ('paths has a default generation again', [S(
        'fa', "def paths(stage=DEFAULT_STAGE, *, gen):",
        "def paths(stage=DEFAULT_STAGE, *, gen=1):")]),
    ('every generation shares the legacy dir names', [S(
        'fa', "    return base if g == 1 else f'{base}_gen{g}'",
        "    return base")]),
    # G2
    ('build takes gen 1', [S(
        'fa', "    if gen == 1:\n        return (f\"refusing to build a gen-1 ",
        "    if False:\n        return (f\"refusing to build a gen-1 ")]),
    ('build takes a plan of another generation', [S(
        'fa', "    if plan.get('gen') != gen:\n        return (f\"refusing",
        "    if False:\n        return (f\"refusing")]),
    ('build falls back to the ledger pin for a plan with no detect_root',
     [S('fa', "    root = plan.get('detect_root')\n    if not root:\n",
        "    root = plan.get('detect_root') or _ledger_pin()\n"
        "    if not root:\n")]),
    ('build takes a store that says another generation', [S(
        'fa', "    got = _store_gen_of(root)\n    if got != gen:\n",
        "    got = _store_gen_of(root)\n    if False:\n")]),
    ('build re-cuts a non-empty out dir', [S(
        'fa', "    other = sorted(n for n in names if not n.endswith('.tmp'))",
        "    other = []")]),
    ('build takes shards of two models', [S(
        'fa', "    if len(models) != 1 or not models[0]:",
        "    if not models[0]:")]),
    ('build takes a row count that is not the plan\'s', [S(
        'fa', "    if n_shard != plan.get('rows'):", "    if False:")]),
    ('build takes an unknown protocol', [S(
        'fa', "    if proto not in VIEW_FRAMES:\n        return",
        "    if False:\n        return")]),
    # G3
    ('the geometry check is not run', [S(
        'fa', "    _geometry_check(con, tmp, detect_root, gen)\n", "")]),
    ('the geometry check ignores a moved edge', [S(
        'fa', "abs(p.x1 - d.x1) < 0.5 AND", "true AND")]),
    ('the geometry check misses a key the det shards lack', [S(
        'fa', "            SELECT count(*) FILTER (WHERE d.di IS NULL),",
        "            SELECT 0 * count(*),")]),
    ('the geometry check compares counts, not rows', [S(
        'fa', "    if unmatched or moved:\n        raise SystemExit(\n"
              "            f'refusing to swap the pool in: {unmatched:,}",
        "    if joined < keys:\n        raise SystemExit(\n"
        "            f'refusing to swap the pool in: {unmatched:,}")]),
    ('the geometry check samples one frame', [S(
        'fa', "            FROM read_parquet('{_q(pool_path)}') p\n",
        "            FROM (SELECT * FROM read_parquet('{_q(pool_path)}') "
        "WHERE image_id = '100') p\n")]),
    # (No mutant drops union_by_name: duckdb 1.5 reads the columns the check
    # projects by name across det files of different schemas either way, so
    # that edit changes nothing here. The fixture's two schemas still prove
    # the check passes over them.)
    # G4
    ('build renames the parquet before pool.json', [S(
        'fa', "    os.replace(jtmp, P['pool_json'])\n"
              "    os.replace(tmp, P['pool'])\n",
        "    os.replace(tmp, P['pool'])\n"
        "    os.replace(jtmp, P['pool_json'])\n")]),
    ('a gen-2 dir with no pool.json reads as gen 1', [S(
        'fa', "        if g == 1:\n            return _gen1_meta(stage, {})\n",
        "        return _gen1_meta(stage, {})\n")]),
    ('pool_meta ignores the parquet\'s size and mtime', [S(
        'fa', "    if (pst.st_size, pst.st_mtime_ns) != (doc['parquet_size'],",
        "    if False and (pst.st_size, pst.st_mtime_ns) != "
        "(doc['parquet_size'],")]),
    ('verify ignores the sha256', [S(
        'fa', "    if got != meta['sha256']:", "    if False:")]),
    ('a gen-2 pool.json of another generation is taken', [S(
        'fa', "    if doc.get('gen') != gen or gen_of(doc.get('gen')) != gen:",
        "    if False:")]),
    # G5
    ('record takes another pool\'s page', [S(
        'audit', "    if pool != ctx.pool_id:\n        return {'ok': False, "
                 "'msg': STALE_PAGE, 'status': 409}\n    key = str(key)\n",
        "    key = str(key)\n")]),
    ('record takes a key on no page', [S(
        'audit', "    item = _page_index(ctx).get(key)\n    if item is None:\n",
        "    item = _page_index(ctx).get(key) or {'key': key}\n"
        "    if False:\n")]),
    ('record keeps no box', [S(
        'audit', "    if box is not None:\n        rec['box']",
        "    if False:\n        rec['box']")]),
    ('record does not re-read the pool before writing', [S(
        'audit', "        why = _same_pool(ctx)\n        if why:\n"
                 "            return {'ok': False, 'msg': why.replace(",
        "        why = None\n        if why:\n"
        "            return {'ok': False, 'msg': why.replace(")]),
    ('a POST with no gen reads as gen 1', [S(
        'dash', "            gen, err = self._audit_gen(stage, q, default=None, "
                "body=data)\n",
        "            gen, err = self._audit_gen(stage, q, default=1, "
        "body=data)\n")]),
    ('the pool of a request is not compared', [S(
        'dash', "        if p is None or p == ctx.pool_id:\n",
        "        if True:\n")]),
    # G6
    ('save_correction ignores box_sig', [S(
        'audit', "    if _sig_of(box_sig) != _box_sig(cand):", "    if False:")]),
    ('save_correction does not re-read the pool before writing', [S(
        'audit', "        why = _same_pool(ctx)\n        if why:\n"
                 "            return {'ok': False, 'msg': why, 'status': 409}",
        "        why = None\n        if why:\n"
        "            return {'ok': False, 'msg': why, 'status': 409}")]),
    ('audit corrections ignore the pool', [S(
        'audit', "                if (gen >= 2 or rp is not None) and rp != "
                 "pool:\n                    continue\n", "")]),
    ('crop_sources\' audit redraws ignore the pool', [S(
        'cs', "                if (gen >= 2 or rp is not None) and rp != "
              "pool_id:\n", "                if False:\n")]),
    # G7
    ('read_verdicts skips an untagged gen-2 row', [S(
        'fa', "            if g >= 2 and tag is None:\n                raise "
              "GenError(", "            if g >= 2 and tag is None:\n"
                           "                continue\n                raise "
                           "GenError(")]),
    ('read_verdicts skips a row of another generation', [S(
        'fa', "            if gen_of(tag) != g:\n                raise GenError(",
        "            if gen_of(tag) != g:\n                continue\n"
        "                raise GenError(")]),
    ('read_verdicts skips a row of another pool', [S(
        'fa', "            if (g >= 2 or got is not None) and got != want_pool:"
              "\n                raise GenError(",
        "            if (g >= 2 or got is not None) and got != want_pool:\n"
        "                continue\n                raise GenError(")]),
    # G8
    ('the crop route drops the g<N> token', [S(
        'dash', "            gen, err = self._audit_gen(stage, q, token=token, "
                "default=1)\n",
        "            gen, err = self._audit_gen(stage, q, token=None, "
        "default=1)\n")]),
    ('cropSrc leaves the generation out of the path', [S(
        'audit', "  return '/audit/crop/'+STAGE+'/g'+GEN+'/'+",
        "  return '/audit/crop/'+STAGE+'/'+")]),
    ('the stats fetch names no gen or pool', [S(
        'audit', "fetch('/api/audit/stats?stage='+STAGE+GQ())",
        "fetch('/api/audit/stats?stage='+STAGE)")]),
    # G9
    ('the leash tile cut at pad012', [S(
        'audit', "        tile = _view_rect(view, w, h, box)\n",
        "        tile = _cut_rect(w, h, box)\n")]),
    ('full/ cut in the model\'s framing', [S(
        'audit', "        rect = _cut_rect(w, h, box)\n        tile",
        "        rect = _view_rect(view, w, h, box)\n        tile")]),
    # G10
    ('a frozen generation draws', [S(
        'audit', "    why = fa.draw_refusal(ctx)\n    if why:\n"
                 "        raise Refused(why)\n", "    why = None\n")]),
    ('api_page draws past the last page of a frozen generation', [S(
        'audit', "    if i == total:\n        if closed:\n"
                 "            return _closed_reply(ctx, closed)\n",
        "    if i == total:\n")]),
    ('prefetch queues a frozen generation', [S(
        'audit', "    if fa.draw_refusal(ctx):\n        return\n"
                 "    k = (ctx.stage, ctx.gen)\n",
        "    k = (ctx.stage, ctx.gen)\n")]),
    ('locks are keyed by stage alone', [S(
        'audit', "def _locks(ctx):\n    k = (ctx.stage, ctx.gen)\n",
        "def _locks(ctx):\n    k = ctx.stage\n")]),
    ('a moved model still draws', [S(
        'fa', "    if models != [c.meta['model']]:", "    if False:")]),
    ('a POST falls back to the gate', [S(
        'dash', "            return v if v in a.STAGES else None\n",
        "            return v if v in a.STAGES else a.DEFAULT_STAGE\n")]),
    ('the page falls back to gen 1 when gen 2 has no pool', [S(
        'dash', "            ctx, err = self._audit_ctx(stage, gen)\n"
                "            if ctx is None or not a.pool_ready(ctx):\n",
        "            ctx, err = self._audit_ctx(stage, gen)\n"
        "            if ctx is None:\n                ctx, err = "
        "self._audit_ctx(stage, 1)\n"
        "            if ctx is None or not a.pool_ready(ctx):\n")]),
    # G11
    ('export ignores the dataset marker', [S(
        'fa', "    why = dataset_marker_refusal(c)\n    if why:\n"
              "        raise SystemExit(f'refusing to export: {why}')\n", "")]),
    ('export rebuilds a frozen generation', [S(
        'fa', "    if c.frozen:\n        raise SystemExit(f\"refusing to "
              "export the gen-{c.gen} \"",
        "    if False:\n        raise SystemExit(f\"refusing to "
        "export the gen-{c.gen} \"")]),
    ('export names the legacy model', [S(
        'fa', "    model = c.meta['model']\n", "    model = sp['model']\n")]),
    ('export takes a verdict of another box', [S(
        'fa', "    if off:\n        raise SystemExit(", "    if False:\n"
                                                         "        raise "
                                                         "SystemExit(")]),
    # G12
    ('audit_ctx_for takes a pool cut from another store', [S(
        'cs', "    if not have or os.path.realpath(have) != "
              "os.path.realpath(detect_root):", "    if False:")]),
    ('audit_ctx_for reads the live generation', [S(
        'cs', "    import fn_audit as fa\n    gen = join_gen(detect_root)\n"
              "    P = fa.paths(stage, gen=gen)\n",
        "    import fn_audit as fa\n    gen = fa.live_gen(stage)[0]\n"
        "    P = fa.paths(stage, gen=gen)\n")]),
    ('apply_box_vetoes takes two generations', [S(
        'cs', "    if len(gens) > 1:\n        raise SystemExit('apply_box_",
        "    if False:\n        raise SystemExit('apply_box_")]),
    ('rebuild takes two generations', [S(
        'rb', "    if len(gens) > 1:\n        fail('crops of %d detector",
        "    if False:\n        fail('crops of %d detector")]),
    ('build_dataset digests the gen-1 audit whatever the build', [S(
        'bd', "    return fn_audit.audit_name(fn_audit.spec(stage)['audit_dir'],"
              " gen)", "    return fn_audit.spec(stage)['audit_dir']")]),
    # G13
    ('done_by counts gen 1 only', [S(
        'dash', "    for g in _memo('audit:' + str(stage) + ':gens', gens):",
        "    for g in [1]:")]),
    # G14
    ('eval_leash spells out data/leash_audit', [S(
        'el', "AUDIT_DIR = _fa.paths('leash', gen=1)['out']",
        "AUDIT_DIR = os.path.join(REPO, 'data', 'leash_audit')")]),
    ('llm_annotate spells out data/fn_audit/full', [S(
        'llm', "    root = fa.paths('gate', gen=AUDIT_FINDS_GEN)['full']",
        "    root = os.path.join(REPO, 'data', 'fn_audit', 'full')")]),
    ('audit.py summarises with no generation', [S(
        'audit', "    s = fa.summarise(verdicts=_verdicts(ctx), "
                 "stage=ctx.stage, gen=ctx.gen)",
        "    s = fa.summarise(verdicts=_verdicts(ctx), stage=ctx.stage)")]),
    # ── what the reviews of the first version found ──
    # G2/G3: every edge, conf and copy; the generation's own partition
    ('the geometry check ignores y2', [S(
        'fa', " AND abs(p.y2 - d.y2) < 0.5", "")]),
    ('the geometry check ignores conf', [S(
        'fa', "\n                       AND abs(p.conf - d.conf) < 1e-4, false)),",
        ", false)),")]),
    ('the geometry check lets an agreeing copy hide a moved one', [S(
        'fa', "                CAST(det_idx AS INTEGER) AS di, x1, y1, x2, y2, "
              "conf\n",
        "                CAST(det_idx AS INTEGER) AS di, min(x1) OVER "
        "(PARTITION BY image_id, det_idx) AS x1, y1, x2, y2, conf\n")]),
    ('the geometry check reads every gen=* partition', [S(
        'fa', "'gen=%04d' % gen, '**'", "'gen=*', '**'")]),
    ('build takes shards that judge a key twice', [S(
        'fa', "    if n_keys != n_shard:", "    if False:")]),
    ('pool_meta takes a view_frame that is not its protocol\'s', [S(
        'fa', "            doc.get('view_frame') != VIEW_FRAMES[doc['protocol']]:",
        "            False:")]),
    ('live_gen takes a plan whose store says another generation', [S(
        'fa', "    got = _store_gen_of(root)\n    if got != g:\n",
        "    got = _store_gen_of(root)\n    if False:\n")]),
    ('read_verdicts takes an untagged row of another pool in gen 1', [S(
        'fa', "            if (g >= 2 or got is not None) and got != want_pool:",
        "            if g >= 2 and got != want_pool:")]),
    ('build spills to tmpfs by default again', [S(
        'fa', "    b.add_argument('--tmp', required=True,",
        "    b.add_argument('--tmp', default='/tmp/fn_audit_spill',")]),
    ('stats with no --gen does not say where gen 1 is', [S(
        'fa', "str(e) + ('' if asked or gen == 1 else",
        "str(e) + ('' if True else")]),
    # W1/W2/W3: nothing keyed to a box nobody was shown, or to a foreign row
    ('save_correction takes a key on no page', [S(
        'audit', "    if cand['key'] not in _page_index(ctx):\n",
        "    if False:\n")]),
    ('crop_path cuts a key on no page', [S(
        'audit', "    if not shown:\n        return None\n", "")]),
    ('the gen-1 page index takes ledger rows of another generation', [S(
        'audit', "                            if d.get('key'):\n"
                 "                                _row_tags(ctx, d, "
                 "f'{path}:{no}')\n", "")]),
    ('save_correction reads the ledger only after writing', [S(
        'audit', "        seen = {v['key']: v.get('verdict')\n"
                 "                for v in fa.read_verdicts(ctx.P['verdicts'], "
                 "stage=ctx.stage,\n"
                 "                                          gen=ctx.gen)}\n",
        "        seen = {}\n")]),
    # W4: foreign page documents and drawn rows
    ('_page_index takes a page of another generation', [S(
        'audit', "    _doc_tags(ctx, doc, n)\n", "")]),
    ('get_page serves a page of another generation', [S(
        'audit', "    _doc_tags(ctx, doc, f)\n", "")]),
    ('_drawn_keys takes rows of another generation', [S(
        'audit', "                    if d.get('key') or d.get('seq'):\n"
                 "                        _row_tags(ctx, d, f'{path}:{n}')\n",
        "")]),
    # W5, W6, W7
    ('place files into a dataset dir another pool marked', [S(
        'audit', "    why = fa.dataset_marker_refusal(ctx)\n    if why:\n"
                 "        raise Refused(why)\n", "")]),
    ('the gen-1 ledger lends the index any key, "None" too', [S(
        'audit', "                            if not _KEY_RE.fullmatch("
                 "str(d.get('key') or '')):",
        "                            if not d.get('key'):")]),
    ('the page index never reads a page drawn after it was built', [S(
        'audit', "            for n in sorted(set(mtimes) - set(got[2])):\n"
                 "                if not _page_items(ctx, n, idx):\n"
                 "                    mtimes.pop(n)\n", "")]),
    ('done_by memoizes every generation under one key', [S(
        'dash', "        rows = _memo(f'audit:{stage}:g{g}', read)",
        "        rows = _memo(f'audit:{stage}', read)")]),
    # W9, W10, G3 (audit_ctx_for)
    ('other_family_vetoes reads gen 1 whatever the build', [S(
        'cs', "        ac = audit_ctx_for(stage, detect_root)\n"
              "        if ac is None:\n            c['no_audit_for_gen'] += 1",
        "        ac = fa.ctx(stage, 1)\n"
        "        if ac is None:\n            c['no_audit_for_gen'] += 1")]),
    ('vetoes carry no generation or pool', [S(
        'cs', "                        'gen': ac.gen, 'pool': ac.pool_id})",
        "                        })")]),
    ('audit_ctx_for stops a build over a never-drawn dir', [S(
        'cs', "        if gen >= 2 and not shown:", "        if False:")]),
    ('audit_ctx_for drops a ledger whose pool is unreadable', [S(
        'cs', "        shown = [k for k in ('drawn', 'verdicts', 'pages')",
        "        shown = [k for k in ('drawn', 'pages')")]),
    ('v3 takes a gen-2 redraw of a pool cut from another store', [S(
        'v3', "                pools[stage] = (meta['pool_id'] if cut and "
              "os.path.realpath(\n                    cut) == "
              "os.path.realpath(detect_root()) else None)",
        "                pools[stage] = meta['pool_id']")]),
    # W11
    ('an API GET with no gen is the live generation', [S(
        'dash', "            # whose pages, keys and stats were all gen 1's\n"
                "            gen, err = self._audit_gen(stage, q, default=1)\n",
        "            # whose pages, keys and stats were all gen 1's\n"
        "            gen, err = self._audit_gen(stage, q, default='live')\n")]),
    ('two generations in one request: one is picked', [S(
        'dash', "        if len(vals) > 1:\n", "        if False:\n")]),
    ('the dashboard does not watch fn_audit.py', [S(
        'dash', "    _q = os.path.join(os.path.dirname(_src_dir), 'detect', "
                "'fn_audit.py')\n    if os.path.exists(_q):\n"
                "        _watched.add(_q)\n",
        "    _q = os.path.join(os.path.dirname(_src_dir), 'detect', "
        "'fn_audit.py')\n")]),
    ('the POST handler reads a.Refused unguarded', [S(
        'dash', "            refused = isinstance(e, getattr(a, 'Refused', ()))",
        "            refused = a is not None and isinstance(e, a.Refused)")]),
    # W12-W16: guards the fixture never made matter before gap_checks
    ('a store whose model cannot be read still draws', [S(
        'fa', "    models = _store_models(c.P)\n    if models is None:\n",
        "    models = _store_models(c.P)\n    if False:\n")]),
    ('a dataset marker is compared on pool_id alone', [S(
        'fa', "    want = dataset_marker(c)\n    if got != want:\n",
        "    want = dataset_marker(c)\n    if not isinstance(got, dict) or "
        "got.get('pool_id') != want['pool_id']:\n")]),
    ('the pool_meta cache is keyed on size and mtime only', [S(
        'fa', "    key = (st.st_ino, st.st_mtime_ns, st.st_ctime_ns, "
              "st.st_size)\n",
        "    key = (st.st_mtime_ns, st.st_size)\n")]),
    ('the page-index memo is keyed by stage alone', [S(
        'audit', "    k = (ctx.stage, ctx.gen, ctx.P['pages'])\n"
                 "    with _INDEX_LOCK:\n",
        "    k = ctx.stage\n    with _INDEX_LOCK:\n")]),
    ('gen 2 cuts from the plan\'s roots', [S(
        'audit', "    if ctx.gen >= 2:\n        return dict(ctx.meta.get("
                 "'roots') or {})\n", "")]),
]


def main():
    try:
        import duckdb  # noqa: F401
        import PIL  # noqa: F401
    except ImportError as e:
        print(f'SKIP: {e} -- the gen-2 audit fixtures need duckdb and PIL')
        return 0
    for k, p in SOURCES.items():
        with open(p) as fh:
            ORIG[k] = fh.read()
    try:
        return _main()
    finally:
        for d in TEMPS:
            shutil.rmtree(d, ignore_errors=True)


def _main():
    run_all(dict(ORIG))
    # A mutant is caught only by a check that PASSED on the sources as they
    # are: with any check already failing, "some check failed" would catch
    # every mutant there is.
    clean = set(FAILS)
    for name, edits in MUTANTS:
        srcs = dict(ORIG)
        bad = None
        for kind, old, new in edits:
            n = srcs[kind].count(old)
            if n != 1:
                bad = f'{kind}: target found {n} times'
                break
            srcs[kind] = srcs[kind].replace(old, new)
        if bad:
            check(f'm  mutant "{name}" applies', False, bad)
            continue
        saved = list(FAILS)
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                run_all(srcs, 'mut: ')
            new_fails = [f[len('mut: '):] for f in FAILS[len(saved):]]
            killers = [f for f in new_fails if f not in clean]
            why = killers[0] if killers else 'no check failed'
        # a module that will not load, or a check that crashes, is caught --
        # but said, so a mutant "caught" by a typo can be told apart
        except (Exception, SystemExit) as e:  # noqa: BLE001
            new_fails = [f[len('mut: '):] for f in FAILS[len(saved):]]
            killers = [f for f in new_fails if f not in clean]
            why = (killers[0] + f' (then raised {type(e).__name__})'
                   if killers else f'raised {type(e).__name__}: {e}')
            killers = killers or [e]
        FAILS[:] = saved
        check(f'm  mutant "{name}" is caught', bool(killers), why)
    if FAILS:
        print(f'{len(FAILS)} failed')
        return 1
    print(f'the audit pages sample gen 2 and attach every verdict, redraw, '
          f'crop and dataset file to a box of its own generation '
          f'({len(MUTANTS)} mutants caught)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
