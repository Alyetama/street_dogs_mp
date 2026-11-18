#!/usr/bin/env python3
"""Every new human-ledger row names the detector generation its det_idx,
conf and crop belong to, and every reader keeps only the rows of the
generation of the store it joins against.

A det_idx (and a conf to two decimals) names a box only within one detector:
gen 2 numbers the same frame's boxes differently. Until rows carried the
generation, a gen-2 review queue or audit would have written rows nothing
could tell from the gen-1 ones, and every build would have joined both to
whichever store it read. A row with no `gen` is gen 1 (every row written
before the tag existed was); a row whose `gen` is not the store's is skipped
and counted.

  w1  flag_crop (both labels), save_box and mark_seen write `gen` = the
      generation of the pinned ledger store the review crop is resolved
      against -- 1 under a gen-1 pin, 2 under a gen-2 pin, never the live
      store's ($DETECT_ROOT is a gen-5 store throughout)
  w2  ...and the /api/review/leash route records it in leash.db's `gen`
  w3  no pin: each of the four refuses and writes nothing
  w4  audit save_correction writes crop 'audit:<stage>' and the pool's gen
      (pool.json `gen`; none -> 1), and refuses a pool.json naming none
  w5  a crop from a review set (review_extra_dir) cut from another
      generation than the pin's is refused by all four; its own gen is taken
  l1  leash_store: an old database is migrated once (ADD COLUMN gen, NULLs
      kept, idempotent); record() refuses a missing or malformed gen, and a
      re-decision takes the new call's gen; row_dict reads NULL as 1
  l3  record() with no gen at all (a dashboard that predates the argument)
      takes the pinned store's gen, and refuses when there is no pin
  l4  leash_store imports in a process that loaded fn_audit before it had
      GEN_FIELD (the running dashboard), instead of failing for good
  l2  read_calls reads an old database without migrating it (bytes and
      columns unchanged) and gives gen None
  g1  fn_audit.gen_of: None -> 1, an int -> itself, '2'/True/-1/2.0 -> None;
      pool_gen(stage, gen=N) says whether gen N's pool can vouch for itself:
      gen 1 reads a missing or untagged pool.json as 1 and anything else
      (unreadable, a malformed gen, any gen but 1) as None; gen >= 2 has no
      legacy default -- a missing pool.json is None
  s1  sweep.preview_refusal: no preview crops from a sweep of another
      generation than the pinned store's (recent_crops is the review queue)
  g2  crop_sources.store_gen_of: status.json, else the only shards/gen=*;
      join_gen refuses a store that names none
  r1  harvest_flagged.read_flags / read_corrections keep only `gen` rows
      (last write wins within it) and count the rest as other_gen
  r2  crop_sources.flag_candidates joins a gen-1 store with gen-1 flags and a
      gen-2 store with gen-2 ones, counting other_gen; a store naming no
      generation is refused
  r3  crop_sources.leash_call_candidates likewise, from leash.db's column
  r4  read_audit_corrections: `gen` required; only 'audit:<stage>' rows of
      that gen, and for gen >= 2 of that gen's pool (`pool`), counting
      other_gen / other_pool
  r5  build_dogdet_v3 confirmed_positive_ids / corrections and
      build_detector_negatives.flagged_ids read only detect_root()'s gen;
      v3 reads the gate audit of THAT generation's dir (a gen-1 build keeps
      its gen-1 confirmations however far the page has moved on), and a
      gen >= 2 audit redraw only of that generation's pool; v3 corrections
      never take a leash-audit redraw (framed with the leash, not a dog-only
      box) and say how many they skipped
  a1  audit.corrections: only 'audit:<stage>' rows of the pool's gen
      -- not the review page's, not the other audit's (the old bug); and for
      gen >= 2 only its own pool's
  m   mutants: each guard removed, or a generation hardcoded, is caught

Everything runs in a temp tree: the modules are loaded from copies whose
REPO is a fake repo, so no default path can reach data/.

    DASH_PY=<copy> / AUDIT_PY=<copy> check edited copies instead of the live
    tools/dashboard files (an edit to those restarts the running dashboard).

    python tools/detect/tests/adv_ledger_gen.py      (mp14 env)
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DETECT = os.path.join(REPO, 'tools', 'detect')
DASHD = os.path.join(REPO, 'tools', 'dashboard')
DASH_PY = os.environ.get('DASH_PY') or os.path.join(DASHD, 'dashboard.py')
AUDIT_PY = os.environ.get('AUDIT_PY') or os.path.join(DASHD, 'audit.py')
sys.path.insert(0, DASHD)
sys.path.insert(0, DETECT)

FAILS = []
TRACKED = ('fn_audit', 'leash_store', 'harvest_flagged', 'crop_sources',
           'build_dogdet_v3', 'build_detector_negatives')
WHO = 'gentester'
CROP_A = '1786153241730_1234567890_091.jpg'
CROP_B = '1786153241731_2234567890_055.jpg'
CROP_C = '1786153241732_3234567890_040.jpg'
CROP_L = '1786153241733_4234567890_070.jpg'
CROP_X = '1786153241734_5234567890_065.jpg'
CROP_Y = '1786153241735_6234567890_075.jpg'

LEASH_SCHEMA_BY = """
CREATE TABLE IF NOT EXISTS leash (
    crop        TEXT PRIMARY KEY,
    image_id    TEXT NOT NULL,
    label       TEXT NOT NULL CHECK (label IN ('leashed', 'unleashed')),
    conf        REAL,
    ts          INTEGER,
    labelled_at INTEGER NOT NULL,
    source      TEXT NOT NULL DEFAULT 'review_page',
    note        TEXT,
    "by"        TEXT
);
"""


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          ('' if cond else f' -- {detail}'))
    if not cond:
        FAILS.append(name)


def lines(path):
    try:
        with open(path) as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]
    except OSError:
        return []


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')


# ── fixtures ────────────────────────────────────────────────────────────────

def make_store(root, shard_gens, status_gen=None):
    for g in shard_gens:
        os.makedirs(os.path.join(root, 'shards', 'gen=%04d' % g),
                    exist_ok=True)
    if status_gen is not None:
        with open(os.path.join(root, 'status.json'), 'w') as fh:
            json.dump({'gen': status_gen}, fh)
    return root


def fixture():
    tmp = tempfile.mkdtemp(prefix='adv-ledger-gen-')
    fx = types.SimpleNamespace(tmp=tmp)
    fx.g1 = make_store(os.path.join(tmp, 'detect'), [1], 1)
    fx.g2 = make_store(os.path.join(tmp, 'detect_g2'), [2], 2)
    fx.g5 = make_store(os.path.join(tmp, 'detect_live'), [5], 5)
    fx.shard3 = make_store(os.path.join(tmp, 'detect_shard3'), [3])
    fx.nogen = make_store(os.path.join(tmp, 'detect_nogen'), [1, 2])
    # the fake repo every loaded module resolves its REPO to
    fx.repo = os.path.join(tmp, 'repo')
    for d in ('data', os.path.join('tools', 'detect'),
              os.path.join('tools', 'dashboard')):
        os.makedirs(os.path.join(fx.repo, d), exist_ok=True)
    fx.pin = os.path.join(fx.repo, 'data', 'ledger_detect_root.txt')
    fx.n = 0
    return fx


def fresh(fx, name):
    fx.n += 1
    d = os.path.join(fx.tmp, 'run%04d_%s' % (fx.n, name))
    os.makedirs(d)
    return d


def set_pin(fx, root):
    if root is None:
        if os.path.exists(fx.pin):
            os.remove(fx.pin)
        return
    with open(fx.pin, 'w') as fh:
        fh.write(root + '\n')


def audit_layout(d, pool_gen=None, gen=1, stage='gate', detect_root=None,
                 pool_id='P2'):
    """fn_audit.paths(stage, gen=gen) for an audit dir `d`.

    Gen 1: a legacy pool.json ({'stage': 'x'}, `gen` = pool_gen when given).
    Gen >= 2: a pool.parquet stand-in and a pool.json that pool_meta accepts
    -- pool `pool_id`, cut from `detect_root`."""
    lay = {'out': d, 'gen': gen, 'pool': os.path.join(d, 'pool.parquet'),
           'pool_json': os.path.join(d, 'pool.json'),
           'verdicts': os.path.join(d, 'verdicts.jsonl'),
           'drawn': os.path.join(d, 'drawn.jsonl'),
           'crops': os.path.join(d, 'crops'), 'full': os.path.join(d, 'full'),
           'edited': os.path.join(d, 'edited'),
           'edited_thumbs': os.path.join(d, 'edited_thumbs'),
           'pages': os.path.join(d, 'pages'),
           'dataset': os.path.join(d, 'dataset'),
           'store': os.path.join(d, 'store'),
           'shards': os.path.join(d, 'store', 'x-*.parquet'),
           'work': os.path.join(d, 'store', 'work.parquet')}
    os.makedirs(d, exist_ok=True)
    if gen == 1:
        doc = {'stage': 'x'}
        if pool_gen is not None:
            doc['gen'] = pool_gen
    else:
        import hashlib
        with open(lay['pool'], 'wb') as fh:
            fh.write(b'POOL%d' % gen)
        st = os.stat(lay['pool'])
        doc = {'stage': stage, 'gen': gen, 'pool_id': pool_id,
               'state': 'ready', 'detect_root': detect_root, 'model': 'm',
               'protocol': 'full@640', 'view_frame': 'pad012',
               'roots': {'d': d}, 'parquet_size': st.st_size,
               'parquet_mtime_ns': st.st_mtime_ns,
               'sha256': hashlib.sha256(b'POOL%d' % gen).hexdigest()}
    with open(lay['pool_json'], 'w') as fh:
        json.dump(doc, fh)
    return lay


# ── loading: every module from a copy whose REPO is the fake repo ──────────

def load_world(fx, edits):
    """Fresh modules, mutated by `edits` {module: [(old, new)]}. The tracked
    detect modules go into sys.modules under their own names (they import
    one another by name); the dashboard and audit copies under private
    names. Returns (world, restore)."""
    saved = {n: sys.modules.get(n) for n in TRACKED}
    w = types.SimpleNamespace()
    srcs = {n: open(os.path.join(DETECT, n + '.py')).read() for n in TRACKED}
    srcs['dashboard'] = open(DASH_PY).read()
    srcs['audit'] = open(AUDIT_PY).read()
    srcs['sweep'] = open(os.path.join(DETECT, 'sweep.py')).read()
    for mod, pairs in edits.items():
        for old, new in pairs:
            if srcs[mod].count(old) != 1:
                raise LookupError('mutant target not found once in %s: %r'
                                  % (mod, old[:60]))
            srcs[mod] = srcs[mod].replace(old, new)

    def put(name, sub, src, register):
        path = os.path.join(fx.repo, 'tools', sub, name + '.py')
        with open(path, 'w') as fh:
            fh.write(src)
        modname = name if register else 'gentest_' + name
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        if register:
            sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    os.environ['TRAINING_ROOT'] = fx.tmp      # v3 never imports dashboard
    for n in TRACKED:
        setattr(w, n, put(n, 'detect', srcs[n], True))
    w.fa, w.ls, w.hf, w.cs = w.fn_audit, w.leash_store, \
        w.harvest_flagged, w.crop_sources
    w.v3, w.bn = w.build_dogdet_v3, w.build_detector_negatives
    with contextlib.redirect_stdout(io.StringIO()):
        w.dash = put('dashboard', 'dashboard', srcs['dashboard'], False)
        w.audit = put('audit', 'dashboard', srcs['audit'], False)
        w.sweep = put('sweep', 'detect', srcs['sweep'], False)

    def restore():
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    return w, restore


# ── the writers ─────────────────────────────────────────────────────────────

def dash_layout(dash, d):
    dash.CROPS = os.path.join(d, 'recent_crops')
    os.makedirs(os.path.join(dash.CROPS, 'full'))
    for n in (CROP_A, CROP_B, CROP_C, CROP_L):
        with open(os.path.join(dash.CROPS, n), 'wb') as fh:
            fh.write(b'\xff\xd8crop')
    dash.HN_DIR = os.path.join(d, 'hard_negatives')
    dash.HN_CROPS = os.path.join(dash.HN_DIR, 'crops')
    dash.HN_FULL = os.path.join(dash.HN_DIR, 'full')
    dash.HN_LABELS = os.path.join(dash.HN_DIR, 'labels.jsonl')
    dash.HP_DIR = os.path.join(d, 'hard_positives')
    dash.BOX_DIR = os.path.join(d, 'box_corrections')
    dash.BOX_LABELS = os.path.join(dash.BOX_DIR, 'boxes.jsonl')
    dash.SEEN_FILE = os.path.join(dash.HN_DIR, 'reviewed.jsonl')
    dash.review_extra_dir = lambda: None
    dash._flagged = None
    dash._seen = None


def post(dash, path, payload):
    """One request through the real dispatch, signed in. (reply, status)."""
    h = dash.BoardHandler.__new__(dash.BoardHandler)
    body = json.dumps(payload).encode()
    h.path = path
    h.command = 'POST'
    h.headers = {'Content-Length': str(len(body))}
    h.rfile = io.BytesIO(body)
    h.session = {'id': 1, 'username': WHO, 'role': 'member'}
    got = []
    h._json = lambda o, c=200: got.append((o, c))
    h._gate = lambda: False
    h.do_POST()
    return got[0] if got else (None, None)


def leash_rows(db):
    if not os.path.exists(db):
        return []
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute('SELECT * FROM leash')]
    finally:
        con.close()


def writer_checks(w, fx, tag):
    dash = w.dash
    dash.REPO = fx.repo
    dash.leash_store = lambda: w.ls
    for pin, want in ((fx.g1, 1), (fx.g2, 2), (fx.shard3, 3)):
        set_pin(fx, pin)
        d = fresh(fx, 'writers')
        dash_layout(dash, d)
        w.ls.DB_PATH = os.path.join(d, 'leash.db')
        w.ls.CROPS_OUT = os.path.join(d, 'leash_crops')
        w.ls.FULL_OUT = os.path.join(d, 'leash_full')
        b1, _ = dash.flag_crop(CROP_A, dash.FLAG_LABEL, by=WHO)
        b2, _ = dash.flag_crop(CROP_B, dash.POS_LABEL, by=WHO)
        # re-judged: the moved verdict is a new row, and names the gen too
        b3, _ = dash.flag_crop(CROP_A, dash.POS_LABEL, by=WHO)
        hn = lines(dash.HN_LABELS)
        hp = lines(os.path.join(dash.HP_DIR, 'labels.jsonl'))
        check(f'{tag}w1 flag_crop writes gen {want} under a gen-{want} pin',
              b1.get('ok') and b2.get('ok') and b3.get('ok') and hn == []
              and [r.get('gen') for r in hp] == [want, want]
              and all(type(r.get('gen')) is int for r in hp), (b1, hn, hp))
        sb = dash.save_box(CROP_A, 1, [1, 2, 30, 40], by=WHO)
        bx = lines(dash.BOX_LABELS)
        check(f'{tag}w1 save_box writes gen {want}',
              sb.get('ok') and [(r.get('det_idx'), r.get('gen'))
                                for r in bx] == [(1, want)], (sb, bx))
        ms = dash.mark_seen([CROP_C], by=WHO)
        seen = lines(dash.SEEN_FILE)
        check(f'{tag}w1 mark_seen writes gen {want}',
              ms.get('ok') and [r.get('gen') for r in seen] == [want],
              (ms, seen))
        reply, code = post(dash, '/api/review/leash',
                           {'name': CROP_L, 'label': 'leashed'})
        got = leash_rows(w.ls.DB_PATH)
        check(f'{tag}w2 the leash route records gen {want}',
              code == 200 and (reply or {}).get('ok')
              and [(r['crop'], r.get('gen')) for r in got]
              == [(CROP_L, want)], (reply, code, got))
    # no pin: nothing is written, by any of the four
    set_pin(fx, None)
    d = fresh(fx, 'nopin')
    dash_layout(dash, d)
    w.ls.DB_PATH = os.path.join(d, 'leash.db')
    w.ls.CROPS_OUT = os.path.join(d, 'leash_crops')
    w.ls.FULL_OUT = os.path.join(d, 'leash_full')
    b1, _ = dash.flag_crop(CROP_A, dash.FLAG_LABEL, by=WHO)
    sb = dash.save_box(CROP_A, 0, [1, 2, 30, 40], by=WHO)
    ms = dash.mark_seen([CROP_C], by=WHO)
    reply, code = post(dash, '/api/review/leash',
                       {'name': CROP_L, 'label': 'leashed'})
    wrote = (lines(dash.HN_LABELS), lines(dash.BOX_LABELS),
             lines(dash.SEEN_FILE), leash_rows(w.ls.DB_PATH))
    check(f'{tag}w3 no pin: flag, box, seen and leash are refused and write '
          f'nothing',
          not b1.get('ok') and not sb.get('ok') and not ms.get('ok')
          and not (reply or {}).get('ok') and wrote == ([], [], [], []),
          (b1, sb, ms, reply, wrote))
    set_pin(fx, fx.g1)

    # a review set (review_extra_dir) cut from another generation than the
    # pin's: its conf is another detector's, so all four refuse; a set of the
    # pin's own generation is taken, and one with no manifest is gen 1
    for made, ok in (({'gen': 2}, False), ({'gen': 1}, True), (None, True)):
        d = fresh(fx, 'reviewset')
        dash_layout(dash, d)
        extra = os.path.join(d, 'review_set')
        os.makedirs(os.path.join(extra, 'full'))
        for n in (CROP_X, CROP_Y):
            with open(os.path.join(extra, n), 'wb') as fh:
                fh.write(b'\xff\xd8crop')
        if made is not None:
            with open(os.path.join(extra, 'review_set.json'), 'w') as fh:
                json.dump(made, fh)
        dash.review_extra_dir = lambda _e=extra: _e
        w.ls.DB_PATH = os.path.join(d, 'leash.db')
        w.ls.CROPS_OUT = os.path.join(d, 'leash_crops')
        w.ls.FULL_OUT = os.path.join(d, 'leash_full')
        b1, _ = dash.flag_crop(CROP_X, dash.FLAG_LABEL, by=WHO)
        sb = dash.save_box(CROP_X, 0, [1, 2, 30, 40], by=WHO)
        # a page of one live-pool crop and one review-set crop: kept whole
        # or not at all
        ms = dash.mark_seen([CROP_C, CROP_Y], by=WHO)
        reply, code = post(dash, '/api/review/leash',
                           {'name': CROP_Y, 'label': 'leashed'})
        wrote = (lines(dash.HN_LABELS), lines(dash.BOX_LABELS),
                 lines(dash.SEEN_FILE), leash_rows(w.ls.DB_PATH))
        gens = sorted({r.get('gen') for rows in wrote for r in rows})
        replies = [bool(b1.get('ok')), bool(sb.get('ok')),
                   bool(ms.get('ok')), bool((reply or {}).get('ok'))]
        if ok:
            good = replies == [True] * 4 and gens == [1] \
                and [len(x) for x in wrote] == [1, 1, 2, 1]
        else:
            good = replies == [False] * 4 and wrote == ([], [], [], [])
        check(f'{tag}w5 a review set made {made}: '
              f'{"taken as gen 1" if ok else "refused by all four"}',
              good, (replies, wrote))
    dash.review_extra_dir = lambda: None

    # the audit page's box editor
    audit, fa = w.audit, w.fa
    real_paths = fa.paths
    sig = [1.0, 2.0, 30.0, 40.0]           # the stub pool box, as box_sig
    try:
        audit._pool_row = lambda key, ctx: {
            'key': '777#3', 'image_id': '777', 'det_idx': 3,
            'x1': 1.0, 'y1': 2.0, 'x2': 30.0, 'y2': 40.0,
            'cell': 'c', 'drive': 'd'}
        audit._cut_one = lambda *a, **k: True
        audit._roots = lambda ctx=None: {}
        for gen, want in ((1, (1, fa.G1_POOL_ID)), (2, (2, 'P2'))):
            d = fresh(fx, 'audit_w')
            lay = audit_layout(os.path.join(d, 'gate'), gen=gen,
                               detect_root=fx.g2)
            fa.paths = lambda stage='gate', *, gen, _l=lay: _l
            c = fa.ctx('gate', gen)
            # the box was on a page of this generation: a redraw of a key
            # nobody was shown is refused (audit._page_index)
            os.makedirs(c.P['pages'], exist_ok=True)
            with open(os.path.join(c.P['pages'], '00000.json'), 'w') as fh:
                json.dump({'gen': gen, 'pool': c.pool_id,
                           'items': [{'key': '777#3'}]}, fh)
            audit.BOX_FILE = os.path.join(d, 'boxes.jsonl')
            out = audit.save_correction('777#3', [1, 2, 31, 41], c,
                                        pool=c.pool_id, box_sig=sig, by=WHO)
            got = lines(audit.BOX_FILE)
            check(f'{tag}w4 save_correction writes audit:gate with its '
                  f'ctx\'s gen and pool {want}',
                  out.get('ok') and [(r.get('crop'), r.get('gen'),
                                      r.get('pool')) for r in got]
                  == [('audit:gate',) + want], (out, got))
            # a frame of another generation: its pool id, or the box_sig of
            # another box under the same key -- refused, nothing written
            for kw in ({'pool': 'P-other', 'box_sig': sig},
                       {'pool': c.pool_id, 'box_sig': [10, 10, 60, 60]},
                       {'pool': c.pool_id, 'box_sig': None}):
                out = audit.save_correction('777#3', [1, 2, 32, 42], c,
                                            by=WHO, **kw)
                check(f'{tag}w4 gen {gen}: a redraw with {kw} is refused',
                      not out.get('ok') and lines(audit.BOX_FILE) == got,
                      out)
        # the pool.json damaged AFTER the request resolved its ctx: refused
        # at the write, nothing written -- never guessed to be gen 1
        for body in ('{not json', '{"gen": "2"}', '[2]'):
            with open(lay['pool_json'], 'w') as fh:
                fh.write(body)
            audit.BOX_FILE = os.path.join(d, 'boxes_bad.jsonl')
            out = audit.save_correction('777#3', [1, 2, 32, 42], c,
                                        pool=c.pool_id, box_sig=sig, by=WHO)
            got = lines(audit.BOX_FILE)
            check(f'{tag}w4 save_correction refuses a pool.json of {body!r}',
                  not out.get('ok') and got == [], (out, got))
    finally:
        fa.paths = real_paths


# ── leash_store ─────────────────────────────────────────────────────────────

def leash_checks(w, fx, tag):
    ls = w.ls
    d = fresh(fx, 'leash')
    ls.CROPS_OUT = os.path.join(d, 'c')
    ls.FULL_OUT = os.path.join(d, 'f')
    db = os.path.join(d, 'old.db')
    con = sqlite3.connect(db)
    con.execute('PRAGMA journal_mode=WAL')      # as the real store is
    con.executescript(LEASH_SCHEMA_BY)
    con.execute('INSERT INTO leash (crop, image_id, label, conf, ts, '
                'labelled_at, "by") VALUES (?,?,?,?,?,?,?)',
                ('1786000000001_970000000001_090.jpg', '970000000001',
                 'leashed', 0.9, 1, 100, 'old'))
    con.commit()
    con.close()
    before = open(db, 'rb').read()
    try:
        got = ls.read_calls(db)
    except Exception as e:                    # noqa: BLE001
        got = [repr(e)]
    beside = sorted(n for n in os.listdir(d) if n.startswith('old.db-'))
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute('PRAGMA table_info(leash)')]
    con.close()
    missing = os.path.join(d, 'none.db')
    try:
        ls.read_calls(missing)
        made = 'read'
    except sqlite3.Error:
        made = os.path.exists(missing)
    check(f'{tag}l2 read_calls reads an old database without migrating it, '
          f'leaves no -wal/-shm beside it and creates no missing one',
          open(db, 'rb').read() == before and 'gen' not in cols
          and beside == [] and made is False
          and isinstance(got[0], dict) and got[0].get('gen', 'x') is None
          and got[0].get('by') == 'old', (cols, beside, made, got))

    def record(*a, **k):
        # a write that raises is a failed write, said as one
        try:
            return ls.record(*a, **k)
        except Exception as e:                # noqa: BLE001
            return {'ok': False, 'raised': repr(e)}, 500

    refused = [record(CROP_L, 'leashed', path=db, by=WHO, gen=g)
               for g in (None, '2', True, -1)]
    rows0 = leash_rows(db)
    b1, c1 = record(CROP_L, 'leashed', path=db, by=WHO, gen=2)
    b2, c2 = record(CROP_A, 'unleashed', path=db, by=WHO, gen=1)
    con = ls.connect(db)        # a second connect: the migration is idempotent
    con.close()
    rows = {r['crop']: r.get('gen', 'absent') for r in leash_rows(db)}
    cols = [r[1] for r in sqlite3.connect(db).execute(
        'PRAGMA table_info(leash)')]
    check(f'{tag}l1 record refuses a missing or malformed gen',
          all(c == 400 and not b.get('ok') for b, c in refused)
          and len(rows0) == 1, (refused, rows0))
    check(f'{tag}l1 an old database is migrated once, its rows kept NULL',
          c1 == 200 and c2 == 200 and cols.count('gen') == 1
          and rows == {'1786000000001_970000000001_090.jpg': None,
                       CROP_L: 2, CROP_A: 1}, (b1, b2, cols, rows))
    record(CROP_A, 'leashed', path=db, by=WHO + '2', gen=2)
    again = {r['crop']: r for r in leash_rows(db)}
    con = ls.connect(db)
    try:
        legacy = ls.row_dict(con.execute(
            'SELECT * FROM leash WHERE crop = ?',
            ('1786000000001_970000000001_090.jpg',)).fetchone())
    finally:
        con.close()
    check(f'{tag}l1 a re-decision takes the new call\'s gen; NULL reads as 1',
          CROP_A in again and again[CROP_A].get('gen') == 2
          and again[CROP_A]['label'] == 'leashed'
          and legacy.get('gen') == 1, (again.get(CROP_A), legacy))

    # a caller that names no generation at all: the dashboard started before
    # the review page passed one, whose crops all resolve against the pin
    got = {}
    for pin, want in ((fx.g1, 1), (fx.g2, 2), (None, None)):
        set_pin(fx, pin)
        db3 = os.path.join(fresh(fx, 'leash_nogen'), 'l.db')
        b, c = record(CROP_L, 'leashed', path=db3, by=WHO)
        got[want] = (c, [r.get('gen') for r in leash_rows(db3)])
    set_pin(fx, fx.g1)
    check(f'{tag}l3 record with no gen takes the pinned store\'s; no pin '
          f'is refused', got == {1: (200, [1]), 2: (200, [2]),
                                 None: (400, [])}, got)

    # the running dashboard imported fn_audit before it had GEN_FIELD, and
    # leash_store only later: the import must not fail for good
    fa_path = os.path.join(fx.repo, 'tools', 'detect', 'fn_audit.py')
    # importlib.reload finds the module again by name, on sys.path
    sys.path.insert(0, os.path.dirname(fa_path))
    spec = importlib.util.spec_from_file_location('fn_audit', fa_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for attr in ('GEN_FIELD', 'LEGACY_GEN', 'gen_of', 'pool_gen'):
        delattr(old, attr)
    keep = sys.modules.get('fn_audit')
    sys.modules['fn_audit'] = old
    try:
        lspec = importlib.util.spec_from_file_location(
            'gentest_leash_old', os.path.join(fx.repo, 'tools', 'detect',
                                              'leash_store.py'))
        lmod = importlib.util.module_from_spec(lspec)
        lspec.loader.exec_module(lmod)
        loaded = (getattr(lmod, 'GEN_FIELD', None), hasattr(old, 'gen_of'))
    except Exception as e:                    # noqa: BLE001
        loaded = repr(e)
    finally:
        sys.modules['fn_audit'] = keep
    check(f'{tag}l4 leash_store imports under an fn_audit loaded before '
          f'GEN_FIELD existed', loaded == ('gen', True), loaded)


# ── the readers ─────────────────────────────────────────────────────────────

def fake_boxes(confs):
    """hf.boxes_for stand-in: each id one box at the conf given."""
    return lambda ids, root, python_bin=None: {
        i: [(0, 10.0, 10.0, 90.0, 90.0, confs[i])] for i in ids if i in confs}


def reader_checks(w, fx, tag):
    fa, hf, cs = w.fa, w.hf, w.cs
    check(f'{tag}g1 gen_of reads untagged as 1 and names nothing else',
          [fa.gen_of(v) for v in (None, 1, 2, '2', True, -1, 2.0)]
          == [1, 1, 2, None, None, None, None],
          [fa.gen_of(v) for v in (None, 1, 2, '2', True, -1, 2.0)])
    real_paths = fa.paths
    d = fresh(fx, 'readers')
    try:
        lays = {('gate', 1): audit_layout(os.path.join(d, 'gate')),
                ('leash', 1): audit_layout(os.path.join(d, 'leash')),
                ('gate', 2): audit_layout(os.path.join(d, 'gate_gen2'),
                                          gen=2, detect_root=fx.g2),
                ('leash', 2): audit_layout(os.path.join(d, 'leash_gen2'),
                                           gen=2, stage='leash',
                                           detect_root=fx.g2)}
        fa.paths = lambda stage='gate', *, gen: lays[(stage, gen)]
        got = (fa.pool_gen('gate', gen=1), fa.pool_gen('leash', gen=1),
               fa.pool_gen('gate', gen=2))
        check(f'{tag}g1 pool_gen: an untagged gen-1 pool is 1, a ready '
              f'gen-2 pool 2', got == (1, 1, 2), got)
        odd = os.path.join(d, 'odd')
        keep = lays[('gate', 1)]
        lays[('gate', 1)] = audit_layout(odd)
        seen = []
        for body in ('{not json', '[2]', '{"gen": "2"}', '{"gen": true}',
                     '{"gen": 0}', '{"gen": 2}', '{"stage": "x"}'):
            with open(os.path.join(odd, 'pool.json'), 'w') as fh:
                fh.write(body)
            seen.append(fa.pool_gen('gate', gen=1))
        os.remove(os.path.join(odd, 'pool.json'))
        seen.append(fa.pool_gen('gate', gen=1))
        check(f'{tag}g1 pool_gen of gen 1: unreadable, a malformed gen or '
              f'any gen but 1 -> None; no gen or no pool.json -> 1',
              seen == [None, None, None, None, None, None, 1, 1], seen)
        g2j = lays[('gate', 2)]['pool_json']
        os.rename(g2j, g2j + '.away')
        miss = fa.pool_gen('gate', gen=2)
        os.rename(g2j + '.away', g2j)
        check(f'{tag}g1 a gen-2 dir with no pool.json is no pool (not gen 1)',
              miss is None, miss)
        try:
            fa.pool_gen('gate')
            nodefault = False
        except TypeError:
            nodefault = True
        check(f'{tag}g1 pool_gen has no default generation', nodefault)
        # ...and a pool naming none is no pool's: its readers keep nothing,
        # not even an untagged row that would otherwise be gen 1's
        bad = os.path.join(d, 'bad_tags.jsonl')
        write_jsonl(bad, [{'crop': 'audit:gate', 'image_id': '61',
                           'det_idx': 0, 'x1': 1, 'y1': 1, 'x2': 5, 'y2': 5}])
        with open(os.path.join(odd, 'pool.json'), 'w') as fh:
            fh.write('{"gen": "2"}')
        none_read = cs.read_audit_corrections('gate', bad, gen=1)
        check(f'{tag}g1 a pool naming no usable gen: read_audit_corrections '
              f'keeps nothing', none_read == {}, none_read)
        w.audit.BOX_FILE = bad
        try:
            got = w.audit.corrections(fa.ctx('gate', 1))
        except fa.PoolIdentityError:
            got = {}
        check(f'{tag}g1 ...and audit.corrections keeps nothing', got == {},
              got)
        lays[('gate', 1)] = keep

        # the sweep's live preview crops are the review page's queue
        set_pin(fx, fx.g1)
        sw = [w.sweep.preview_refusal(g, repo=fx.repo) for g in (1, 2)]
        set_pin(fx, None)
        sw.append(w.sweep.preview_refusal(1, repo=fx.repo))
        set_pin(fx, fx.g1)
        check(f'{tag}s1 sweep previews only from the pinned store\'s gen',
              sw[0] is None and isinstance(sw[1], str)
              and isinstance(sw[2], str), sw)

        got = [cs.store_gen_of(r) for r in (fx.g1, fx.g2, fx.shard3,
                                            fx.nogen)]
        try:
            cs.join_gen(fx.nogen)
            refused = False
        except SystemExit:
            refused = True
        check(f'{tag}g2 store_gen_of: status, else the only shard gen; '
              f'join_gen refuses none', got == [1, 2, 3, None] and refused,
              (got, refused))

        # the flag ledger: 11 untagged then re-flagged on gen 2, 12 gen 1,
        # 13 gen 2, 14 a string tag, 15 a bool tag
        hn = os.path.join(d, 'labels.jsonl')
        write_jsonl(hn, [
            {'image_id': '11', 'label': 'false_positive', 'conf': 0.5},
            {'image_id': '12', 'label': 'false_positive', 'conf': 0.6,
             'gen': 1},
            {'image_id': '13', 'label': 'false_positive', 'conf': 0.7,
             'gen': 2},
            {'image_id': '14', 'label': 'false_positive', 'conf': 0.8,
             'gen': '1'},
            {'image_id': '15', 'label': 'false_positive', 'conf': 0.9,
             'gen': True},
            {'image_id': '11', 'label': 'false_positive', 'conf': 0.3,
             'gen': 2}])
        corr = os.path.join(d, 'boxes.jsonl')
        write_jsonl(corr, [
            {'crop': 'x.jpg', 'image_id': '11', 'det_idx': 0,
             'x1': 1, 'y1': 1, 'x2': 5, 'y2': 5},
            {'crop': 'y.jpg', 'image_id': '11', 'det_idx': 0,
             'x1': 2, 'y1': 2, 'x2': 6, 'y2': 6, 'gen': 2},
            {'crop': 'audit:gate', 'image_id': '13', 'det_idx': 0,
             'x1': 3, 'y1': 3, 'x2': 7, 'y2': 7, 'gen': 2, 'pool': 'P2'},
            {'crop': 'audit:gate', 'image_id': '14', 'det_idx': 0,
             'x1': 6, 'y1': 6, 'x2': 9, 'y2': 9, 'gen': 2, 'pool': 'OTHER'},
            {'crop': 'audit:gate', 'image_id': '12', 'det_idx': 0,
             'x1': 4, 'y1': 4, 'x2': 8, 'y2': 8},
            {'crop': 'audit:leash', 'image_id': '12', 'det_idx': 0,
             'x1': 5, 'y1': 5, 'x2': 9, 'y2': 9}])
        c1, c2 = {'other_gen': 0}, {'other_gen': 0}
        f1 = hf.read_flags(hn, 'false_positive', gen=1, counts=c1)
        f2 = hf.read_flags(hn, 'false_positive', gen=2, counts=c2)
        check(f'{tag}r1 read_flags keeps only its gen, last write within it',
              sorted(f1) == ['11', '12'] and f1['11']['conf'] == 0.5
              and sorted(f2) == ['11', '13'] and f2['11']['conf'] == 0.3
              and c1['other_gen'] == 4 and c2['other_gen'] == 4,
              (sorted(f1), sorted(f2), c1, c2))
        k1 = {'other_gen': 0}
        r1 = hf.read_corrections(corr, gen=1, counts=k1)
        r2 = hf.read_corrections(corr, gen=2)
        check(f'{tag}r1 read_corrections keeps only its gen (audit rows '
              f'never)', r1 == {('11', 0): (1.0, 1.0, 5.0, 5.0)}
              and r2 == {('11', 0): (2.0, 2.0, 6.0, 6.0)}
              and k1['other_gen'] == 1, (r1, r2, k1))

        hf_boxes = hf.boxes_for
        hf.boxes_for = fake_boxes({'11': 0.5, '12': 0.6, '13': 0.7,
                                   '14': 0.8, '15': 0.9})
        try:
            rows1, n1 = cs.flag_candidates('false_positive', fx.g1,
                                           ledger=hn, corrections=corr)
            hf.boxes_for = fake_boxes({'11': 0.3, '13': 0.7})
            rows2, n2 = cs.flag_candidates('false_positive', fx.g2,
                                           ledger=hn, corrections=corr)
            try:
                cs.flag_candidates('false_positive', fx.nogen, ledger=hn,
                                   corrections=corr)
                refused = False
            except SystemExit:
                refused = True
        finally:
            hf.boxes_for = hf_boxes
        got1 = sorted((r['image_id'], tuple(r['box'])) for r in rows1)
        got2 = sorted((r['image_id'], tuple(r['box'])) for r in rows2)
        check(f'{tag}r2 flag_candidates joins each store with its own '
              f'generation\'s flags and redraws',
              got1 == [('11', (1.0, 1.0, 5.0, 5.0)),
                       ('12', (10.0, 10.0, 90.0, 90.0))]
              and got2 == [('11', (2.0, 2.0, 6.0, 6.0)),
                           ('13', (10.0, 10.0, 90.0, 90.0))]
              and n1.get('other_gen') == 4 and n2.get('other_gen') == 4
              and refused, (got1, got2, dict(n1), dict(n2), refused))

        # leash calls: one legacy (NULL) row, one gen 1, one gen 2
        db = os.path.join(d, 'leash.db')
        con = sqlite3.connect(db)
        con.executescript(LEASH_SCHEMA_BY)
        con.execute('INSERT INTO leash (crop, image_id, label, conf, ts, '
                    'labelled_at, "by") VALUES (?,?,?,?,?,?,?)',
                    ('1786000000001_21_050.jpg', '21', 'leashed', 0.5, 1, 1,
                     None))
        con.commit()
        con.close()
        w.ls.CROPS_OUT = os.path.join(d, 'lc')
        w.ls.FULL_OUT = os.path.join(d, 'lf')
        w.ls.record('1786000000002_22_060.jpg', 'unleashed', path=db,
                    by=WHO, gen=1)
        w.ls.record('1786000000003_23_070.jpg', 'leashed', path=db, by=WHO,
                    gen=2)
        real_db = w.ls.DB_PATH
        w.ls.DB_PATH = db
        hf.boxes_for = fake_boxes({'21': 0.5, '22': 0.6, '23': 0.7})
        try:
            l1, lc1, _ = cs.leash_call_candidates(fx.g1, corrections=corr)
            l2, lc2, _ = cs.leash_call_candidates(fx.g2, corrections=corr)
        finally:
            hf.boxes_for = hf_boxes
            w.ls.DB_PATH = real_db
        check(f'{tag}r3 leash_call_candidates reads only the store\'s gen',
              sorted(r['image_id'] for r in l1) == ['21', '22']
              and [r['image_id'] for r in l2] == ['23']
              and lc1.get('other_gen') == 1 and lc2.get('other_gen') == 2,
              (l1, dict(lc1), l2, dict(lc2)))

        import collections
        c1, c2 = collections.Counter(), collections.Counter()
        a_one = cs.read_audit_corrections('gate', corr, gen=1, counts=c1)
        a_two = cs.read_audit_corrections('gate', corr, gen=2, counts=c2)
        a_leash = cs.read_audit_corrections('leash', corr, gen=1)
        check(f'{tag}r4 read_audit_corrections: its stage, its gen, and for '
              f'gen 2 its pool',
              a_one == {('12', 0): (4.0, 4.0, 8.0, 8.0)}
              and a_two == {('13', 0): (3.0, 3.0, 7.0, 7.0)}
              and a_leash == {('12', 0): (5.0, 5.0, 9.0, 9.0)}
              and c1['other_gen'] == 2 and c2['other_gen'] == 1
              and c2['other_pool'] == 1,
              (a_one, a_two, a_leash, dict(c1), dict(c2)))
        try:
            cs.read_audit_corrections('gate', corr)
            nodefault = False
        except TypeError:
            nodefault = True
        check(f'{tag}r4 read_audit_corrections has no default gen',
              nodefault)

        # the detector builders
        v3, bn = w.v3, w.bn
        v3.HARD_POS = os.path.join(d, 'hp.jsonl')
        write_jsonl(v3.HARD_POS, [
            {'image_id': '41', 'label': 'true_positive'},
            {'image_id': '42', 'label': 'true_positive', 'gen': 2},
            {'image_id': '43', 'label': 'true_positive', 'gen': 1},
            {'image_id': '44', 'label': 'false_positive', 'gen': 1}])
        write_jsonl(lays[('gate', 1)]['verdicts'],
                    [{'key': '51#0', 'verdict': 'dog'}])
        write_jsonl(lays[('gate', 2)]['verdicts'],
                    [{'key': '52#0', 'verdict': 'dog', 'gen': 2,
                      'pool': 'P2'}])
        v3.CORRECTIONS = corr
        out = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for root in (fx.g1, fx.g2):
                v3.DETECT_ROOT = root
                out[root] = sorted(v3.confirmed_positive_ids())
            v3.DETECT_ROOT = fx.g1
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                fix1 = v3.corrections()
            v3.DETECT_ROOT = fx.g2
            fix2 = v3.corrections()
            bn.DETECT_ROOT = fx.g1
            neg1 = sorted(bn.flagged_ids(hn))
            bn.DETECT_ROOT = fx.g2
            neg2 = sorted(bn.flagged_ids(hn))
        check(f'{tag}r5 v3 confirmed positives: the store\'s gen, and the '
              f'gate audit of that gen\'s own dir',
              out == {fx.g1: ['41', '43', '51'],
                      fx.g2: ['42', '52']}, out)
        # ...whose pool must say it was cut from that very store
        g2j = lays[('gate', 2)]['pool_json']
        with open(g2j) as fh:
            real_doc = fh.read()
        doc = json.loads(real_doc)
        doc['detect_root'] = fx.g5
        with open(g2j, 'w') as fh:
            json.dump(doc, fh)
        try:
            cs.audit_ctx_for('gate', fx.g2)
            refused = False
        except SystemExit:
            refused = True
        with open(g2j, 'w') as fh:
            fh.write(real_doc)
        check(f'{tag}r5 a pool cut from another store than the build\'s is '
              f'refused (audit_ctx_for)', refused)
        # '12' det 0: the gate audit's redraw (4..8) stands, the later
        # leash-audit one (5..9, drawn round the leash) does not replace it
        check(f'{tag}r5 v3 corrections: the store\'s gen only, never a '
              f'leash-audit redraw',
              fix1 == {('11', 0): (1.0, 1.0, 5.0, 5.0),
                       ('12', 0): (4.0, 4.0, 8.0, 8.0)}
              and fix2 == {('11', 0): (2.0, 2.0, 6.0, 6.0),
                           ('13', 0): (3.0, 3.0, 7.0, 7.0)}, (fix1, fix2))
        check(f'{tag}r5 v3 corrections count the leash-audit redraws skipped',
              '1 leash-audit box redraw(s) skipped' in said.getvalue(),
              said.getvalue())
        check(f'{tag}r5 build_detector_negatives flags: the store\'s gen',
              neg1 == ['11', '12'] and neg2 == ['11', '13'], (neg1, neg2))

        # the audit page's own reader
        audit = w.audit
        audit.BOX_FILE = corr
        g1 = audit.corrections(fa.ctx('gate', 1))
        g2 = audit.corrections(fa.ctx('gate', 2))
        lv = audit.corrections(fa.ctx('leash', 1))
        check(f'{tag}a1 audit.corrections(ctx): its own stage\'s rows of '
              f'its pool\'s gen (and pool), never the review page\'s',
              {k: v[:4] for k, v in g1.items()}
              == {('12', 0): (4.0, 4.0, 8.0, 8.0)}
              and {k: v[:4] for k, v in g2.items()}
              == {('13', 0): (3.0, 3.0, 7.0, 7.0)}
              and {k: v[:4] for k, v in lv.items()}
              == {('12', 0): (5.0, 5.0, 9.0, 9.0)}, (g1, g2, lv))
        # a row of ANOTHER generation that names the gen-1 pool id (a writer
        # bug, or a hand edit) is still not gen 1's: the generation and the
        # pool are both checked, neither stands in for the other
        odd_gen = os.path.join(d, 'boxes_gen3.jsonl')
        write_jsonl(odd_gen, [{'crop': 'audit:gate', 'image_id': '16',
                               'det_idx': 0, 'x1': 1, 'y1': 1, 'x2': 2,
                               'y2': 2, 'gen': 3, 'pool': fa.G1_POOL_ID}])
        audit.BOX_FILE = odd_gen
        got = audit.corrections(fa.ctx('gate', 1))
        check(f'{tag}a1 a gen-3 row under the gen-1 pool id is not gen 1\'s',
              got == {}, got)
        audit.BOX_FILE = corr
    finally:
        fa.paths = real_paths


def run_all(w, fx, tag=''):
    n0 = len(FAILS)
    os.environ['DETECT_ROOT'] = fx.g5      # the LIVE store: gen 5
    set_pin(fx, fx.g1)
    writer_checks(w, fx, tag)
    leash_checks(w, fx, tag)
    reader_checks(w, fx, tag)
    return len(FAILS) == n0


# ── mutants ─────────────────────────────────────────────────────────────────

MUTANTS = [
    ('flag_crop writes no gen', 'dashboard',
     [("'flagged_at': int(time.time() if now is None else now),\n"
       "                   AUTHOR_FIELD: str(by), GEN_FIELD: gen}",
       "'flagged_at': int(time.time() if now is None else now),\n"
       "                   AUTHOR_FIELD: str(by)}")]),
    ('flag_crop hardcodes gen 1', 'dashboard',
     [('gen = None if undo else _review_gen(name)',
       'gen = None if undo else 1')]),
    ('the review gen read off the live store', 'dashboard',
     [('    gen = store_gen_of(root)\n    if gen is None or name is None:',
       '    gen = _store_gen()\n    if gen is None or name is None:')]),
    ('the review gen without the readers\' shards fallback', 'dashboard',
     [('    gen = store_gen_of(root)\n    if gen is None or name is None:',
       '    gen = _store_gen(root)\n    if gen is None or name is None:')]),
    ('the review gen ignores a review set\'s own gen', 'dashboard',
     [('        if made != gen:\n            return None\n', '')]),
    ('mark_seen takes one crop\'s gen for the page', 'dashboard',
     [('    gens = {_review_gen(str(nm)) for nm in (names or [])}',
       '    gens = {_review_gen(str(nm)) for nm in (names or [])[:1]}')]),
    ('flag_crop writes with no pin', 'dashboard',
     [("    if not undo and gen is None:\n"
       "        return {'ok': False, 'error': NO_GEN}, 200\n", '')]),
    ('save_box writes no gen', 'dashboard',
     [("'saved_at': int(time.time() if now is None else now),\n"
       "           AUTHOR_FIELD: str(by), GEN_FIELD: gen}",
       "'saved_at': int(time.time() if now is None else now),\n"
       "           AUTHOR_FIELD: str(by)}")]),
    ('save_box writes with no pin', 'dashboard',
     [("    gen = _review_gen(name)\n    if gen is None:\n"
       "        return {'ok': False, 'error': NO_GEN}\n",
       "    gen = _review_gen(name)\n")]),
    ('mark_seen writes no gen', 'dashboard',
     [("'seen_at': now,\n"
       "                        AUTHOR_FIELD: str(by), GEN_FIELD: gen})",
       "'seen_at': now,\n"
       "                        AUTHOR_FIELD: str(by)})")]),
    ('mark_seen writes with no pin', 'dashboard',
     [("    if names and gen is None:\n"
       "        return {'ok': False, 'error': NO_GEN, 'added': 0}\n", '')]),
    ('the leash route hardcodes gen 1', 'dashboard',
     [('by=self._annotator(), gen=gen)', 'by=self._annotator(), gen=1)')]),
    ('the leash route leaves the gen to the store', 'dashboard',
     [('by=self._annotator(), gen=gen)', 'by=self._annotator())'),
      ("                    if gen is None:\n"
       "                        body, code = {'ok': False, 'error': NO_GEN}, "
       "400\n                    else:\n",
       "                    if True:\n")]),
    ('save_correction writes no gen', 'audit',
     [("'saved_at': int(time.time()), AUTHOR_FIELD: str(by),\n"
       "           fa.GEN_FIELD: ctx.gen, fa.POOL_FIELD: ctx.pool_id}",
       "'saved_at': int(time.time()), AUTHOR_FIELD: str(by),\n"
       "           fa.POOL_FIELD: ctx.pool_id}")]),
    ('save_correction hardcodes gen 1', 'audit',
     [("int(time.time()), AUTHOR_FIELD: str(by),\n"
       "           fa.GEN_FIELD: ctx.gen, fa.POOL_FIELD",
       "int(time.time()), AUTHOR_FIELD: str(by),\n"
       "           fa.GEN_FIELD: 1, fa.POOL_FIELD")]),
    ('save_correction writes no pool', 'audit',
     [("           fa.GEN_FIELD: ctx.gen, fa.POOL_FIELD: ctx.pool_id}\n"
       "    # what this box was answered",
       "           fa.GEN_FIELD: ctx.gen}\n    # what this box was answered")]),
    ('save_correction writes under a pool.json damaged since', 'audit',
     [("        why = _same_pool(ctx)\n        if why:\n"
       "            return {'ok': False, 'msg': why, 'status': 409}\n"
       "        os.makedirs(os.path.dirname(BOX_FILE)",
       "        os.makedirs(os.path.dirname(BOX_FILE)")]),
    ('save_correction takes another pool\'s page', 'audit',
     [("    if pool != ctx.pool_id:\n"
       "        return {'ok': False, 'msg': STALE_PAGE, 'status': 409}\n"
       "    cand = _pool_row(key, ctx)",
       "    cand = _pool_row(key, ctx)")]),
    ('save_correction ignores box_sig', 'audit',
     [("    if _sig_of(box_sig) != _box_sig(cand):", "    if False:")]),
    ('audit.corrections ignores the pool', 'audit',
     [("                if (gen >= 2 or rp is not None) and rp != pool:\n"
       "                    continue\n", "")]),
    ('audit.corrections reads every page\'s rows (the old bug)', 'audit',
     [("                if d.get('crop') != want or \\\n"
       "                        fa.gen_of(d.get(fa.GEN_FIELD)) != gen:\n",
       "                if fa.gen_of(d.get(fa.GEN_FIELD)) != gen:\n")]),
    ('audit.corrections ignores gen', 'audit',
     [("                if d.get('crop') != want or \\\n"
       "                        fa.gen_of(d.get(fa.GEN_FIELD)) != gen:\n",
       "                if d.get('crop') != want:\n")]),
    ('gen_of reads untagged as unknown', 'fn_audit',
     [('    if tag is None:\n        return LEGACY_GEN\n',
       '    if tag is None:\n        return None\n')]),
    ('gen_of coerces strings and bools', 'fn_audit',
     [('    if isinstance(tag, bool) or not isinstance(tag, int) or tag < 0:'
       '\n        return None\n    return tag\n',
       '    try:\n        return int(tag)\n'
       '    except (TypeError, ValueError):\n        return None\n')]),
    ('pool_gen ignores pool_meta', 'fn_audit',
     [("        return pool_meta(stage, gen)['gen']",
       '        return _want_gen(gen)')]),
    ('pool_meta reads a damaged gen-1 pool.json as gen 1', 'fn_audit',
     [("            doc, why = None, f'{f} is unreadable ({type(e).__name__})'",
       "            doc, why = {}, ''")]),
    ('pool_meta takes any gen in the gen-1 dir', 'fn_audit',
     [("                elif gen_of(doc.get('gen')) != 1:",
       "                elif gen_of(doc.get('gen')) is None:")]),
    ('a gen-2 dir with no pool.json reads as gen 1', 'fn_audit',
     [("        if g == 1:\n            return _gen1_meta(stage, {})\n",
       "        return _gen1_meta(stage, {})\n")]),
    ('paths defaults to gen 1', 'fn_audit',
     [("def paths(stage=DEFAULT_STAGE, *, gen):",
       "def paths(stage=DEFAULT_STAGE, *, gen=1):"),
      ("def pool_gen(stage=DEFAULT_STAGE, *, gen):",
       "def pool_gen(stage=DEFAULT_STAGE, *, gen=1):")]),
    ('read_flags ignores gen', 'harvest_flagged',
     [("                    if gen_of(r.get(GEN_FIELD)) != gen:\n"
       "                        if counts is not None:\n"
       "                            counts['other_gen'] += 1\n"
       "                        continue\n", '')]),
    ('read_corrections ignores gen', 'harvest_flagged',
     [("                if gen_of(r.get(GEN_FIELD)) != gen:\n"
       "                    if counts is not None:\n"
       "                        counts['other_gen'] += 1\n"
       "                    continue\n                try:",
       '                try:')]),
    ('flag_candidates joins gen 1 whatever the store', 'crop_sources',
     [('    gen = join_gen(detect_root) if gen is None else gen\n'
       '    skipped = collections.Counter()\n',
       '    gen = 1\n    skipped = collections.Counter()\n')]),
    ('leash calls of every generation matched', 'crop_sources',
     [('    got = [r for r in calls if fa.gen_of(r.get(fa.GEN_FIELD)) == gen]',
       '    got = list(calls)')]),
    ('read_audit_corrections ignores gen', 'crop_sources',
     [("                if fa.gen_of(r.get(fa.GEN_FIELD)) != gen:\n"
       "                    if counts is not None:\n"
       "                        counts['other_gen'] += 1\n"
       "                    continue\n", '')]),
    ('read_audit_corrections reads under a pool naming no identity',
     'crop_sources',
     [("        pool_id = fa.pool_meta(stage, gen)['pool_id']",
       "        pool_id = fa.G1_POOL_ID")]),
    ('read_audit_corrections ignores the pool', 'crop_sources',
     [("                if (gen >= 2 or rp is not None) and rp != pool_id:",
       "                if False:")]),
    ('read_audit_corrections defaults to gen 1', 'crop_sources',
     [("def read_audit_corrections(stage, path=None, *, gen, counts=None):",
       "def read_audit_corrections(stage, path=None, *, gen=1, counts=None):")]),
    ('join_gen takes 1 for a store naming none', 'crop_sources',
     [('    if gen is None:\n        raise SystemExit(\n'
       "            '%s names no detector generation",
       '    if gen is None:\n        return 1\n        raise SystemExit(\n'
       "            '%s names no detector generation")]),
    ('store_gen_of without the shards fallback', 'crop_sources',
     [('    return gens.pop() if len(gens) == 1 else None',
       '    return None')]),
    ('record with no gen hardcodes 1', 'leash_store',
     [('    if gen is _PINNED:\n        gen = pinned_gen()\n',
       '    if gen is _PINNED:\n        gen = 1\n')]),
    ('record with no gen and no pin writes NULL', 'leash_store',
     [('    if gen is _PINNED:\n        gen = pinned_gen()\n',
       '    if gen is _PINNED:\n        gen = pinned_gen()\n'
       '    if gen is None:\n        gen = 1\n')]),
    ('leash_store fails to import under an older fn_audit', 'leash_store',
     [("if not hasattr(fn_audit, 'GEN_FIELD'):", 'if False:')]),
    ('sweep previews any generation', 'sweep',
     [('    if pinned != gen:', '    if False:')]),
    ('record takes a missing or malformed gen', 'leash_store',
     [("    if gen is None or gen_of(gen) is None:\n"
       "        return {'ok': False, 'error': NO_GEN}, 400\n", ''),
      ('str(by), int(gen)))', 'str(by), gen))')]),
    ('record writes no gen', 'leash_store',
     [('str(by), int(gen)))', 'str(by), None))')]),
    ('the migration forgets gen', 'leash_store',
     [("    for col, decl in ((AUTHOR_FIELD, '\"by\" TEXT'),\n"
       "                      (GEN_FIELD, 'gen INTEGER')):",
       "    for col, decl in ((AUTHOR_FIELD, '\"by\" TEXT'),):")]),
    ('a re-decision keeps the old gen', 'leash_store',
     [("'  \"by\"=excluded.\"by\", gen=excluded.gen'",
       "'  \"by\"=excluded.\"by\"'")]),
    ('read_calls opens read-only (leaves -wal/-shm behind)', 'leash_store',
     [("'file:%s?mode=rw' % urllib", "'file:%s?mode=ro' % urllib")]),
    ('read_calls creates a missing database', 'leash_store',
     [("'file:%s?mode=rw' % urllib", "'file:%s?mode=rwc' % urllib")]),
    ('read_calls migrates the database it reads', 'leash_store',
     [("    con = sqlite3.connect('file:%s?mode=rw' % urllib.parse.quote(p),\n"
       "                          uri=True, timeout=10)\n"
       "    con.row_factory = sqlite3.Row\n",
       '    con = connect(p)\n')]),
    ('v3 hard positives of every generation', 'build_dogdet_v3',
     [("                    if fa.gen_of(d.get(fa.GEN_FIELD)) != gen:\n"
       "                        other += 1\n"
       "                        continue\n"
       "                    out.add(str(d['image_id']))",
       "                    out.add(str(d['image_id']))")]),
    ('v3 reads the gen-2 gate audit whatever the store', 'build_dogdet_v3',
     [("    for v in fa.read_verdicts(stage='gate', gen=ac.gen):",
       "    for v in fa.read_verdicts(stage='gate', gen=2):")]),
    ('v3 skips its own generation\'s gate audit', 'build_dogdet_v3',
     [("    if ac is None:\n        print(f'no gen-{gen} gate audit",
       "    if True:\n        print(f'no gen-{gen} gate audit")]),
    ('v3 takes a gen-2 audit redraw of any pool', 'build_dogdet_v3',
     [("                    if want is None or d.get(fa.POOL_FIELD) != want:",
       "                    if False:")]),
    ('audit_ctx_for skips the detect_root check', 'crop_sources',
     [("    if not have or os.path.realpath(have) != os.path.realpath(detect_root):",
       "    if False:")]),
    ('v3 corrections of every generation', 'build_dogdet_v3',
     [("                if fa.gen_of(d.get(fa.GEN_FIELD)) != gen:\n"
       "                    other += 1\n"
       "                    continue\n"
       "                crop = str(d.get('crop') or '')",
       "                crop = str(d.get('crop') or '')")]),
    ('v3 corrections take leash-audit redraws', 'build_dogdet_v3',
     [("                if d.get('crop') == 'audit:leash':\n"
       "                    leash += 1\n"
       "                    continue\n", '')]),
    ('negatives flags of every generation', 'build_detector_negatives',
     [("                    if fa.gen_of(r.get(fa.GEN_FIELD)) != gen:\n"
       "                        other += 1\n"
       "                        continue\n", '')]),
]


def main():
    fx = fixture()
    env0 = {k: os.environ.get(k) for k in ('DETECT_ROOT', 'TRAINING_ROOT')}
    try:
        w, restore = load_world(fx, {})
        try:
            run_all(w, fx)
        finally:
            restore()
        for name, mod, edits in MUTANTS:
            saved = list(FAILS)
            try:
                w, restore = load_world(fx, {mod: edits})
            except LookupError as e:
                check(f'm  mutant "{name}" applies', False, str(e))
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    try:
                        caught = not run_all(w, fx, 'mut: ')
                    # a reader that now refuses (SystemExit) is caught too
                    except (Exception, SystemExit):   # noqa: BLE001
                        caught = True
            finally:
                restore()
            FAILS[:] = saved
            check(f'm  mutant "{name}" is caught', caught)
    finally:
        for k, v in env0.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(fx.tmp, ignore_errors=True)
    if FAILS:
        print(f'{len(FAILS)} failed')
        return 1
    print('every new ledger row names its detector generation, and every '
          'reader joins a store only with its own generation\'s rows')
    return 0


if __name__ == '__main__':
    sys.exit(main())
