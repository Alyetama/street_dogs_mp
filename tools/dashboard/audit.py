#!/usr/bin/env python3
"""The false-negative audit: sampling, crop cutting, and the verdict ledger.

Serving side of tools/detect/fn_audit.py. The pool is built once by that
script; everything here is per-request.

THE CROP THE PERSON SEES IS THE CROP THE MODEL SAW. PAD_FRAC and PAD_PX are
imported from the runner rather than copied, because a crop framed even
slightly differently is a different picture and the whole exercise would be
measuring a model that never ran.

A PAGE IS A DRAW, AND A DRAW IS KEPT. "Never show the same box twice" and
"let me page back to the one I skipped" are the same requirement seen from two
sides: each draw is written to data/fn_audit/pages/, so paging back re-reads a
page instead of re-rolling it, and every box ever drawn is excluded from every
later draw whether it was judged or not.
"""

import glob
import json
import os
import random
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
import fn_audit as fa                                          # noqa: E402

DEFAULT_STAGE = fa.DEFAULT_STAGE
STAGES = fa.STAGES
# Paths are looked up per request, never held in module state: one process
# serves both audits and two requests can be in flight at once, so a global
# "current stage" would have one page writing into the other's ledger.
#
# A function, not `P = fa.paths`. Binding the function object once looked
# identical and was not: a test that redirects fa.paths at a temp directory
# had no effect here, so the guard that fakes a draw wrote seventeen fixture
# pages into the live audit and the first tile of the real one came back as
# alt text for a box named "z".
def P(stage=DEFAULT_STAGE):
    return fa.paths(stage)
# Two locks per stage, because they guard two different things and one of them
# is slow. A draw holds its lock across the sample and the page write; cutting
# the crops -- seconds, or twenty of them off cold drives -- happens between
# them with nothing held. Sharing one lock meant every verdict recorded while
# the next page was being cut waited for the cutting to finish, which is
# exactly when someone is judging.
_LOCKS = {}


def _locks(stage):
    if stage not in _LOCKS:
        _LOCKS[stage] = (threading.Lock(), threading.Lock())
    return _LOCKS[stage]


# from the runner, never copied -- see the module docstring
try:
    import gate_store
    PAD_FRAC, PAD_PX, MIN_SIDE = (gate_store.PAD_FRAC, gate_store.PAD_PX,
                                  gate_store.MIN_SIDE)
except Exception:                       # pragma: no cover - runner is present
    PAD_FRAC, PAD_PX, MIN_SIDE = 0.12, 4, 8

CROP_PX = 320          # bigger than the model's 224: a person needs to see it


def _roots():
    """Drive label -> grid root, out of the gate's plan.

    The plan carries them because the runner could not look them up itself.
    Same reason applies here in reverse -- this runs inside the dashboard, so
    it could ask directly, but reading the plan means the audit cuts from
    exactly the roots the gate read.
    """
    try:
        with open(os.path.join(REPO, 'data', 'gate', 'plan.json')) as fh:
            got = json.load(fh).get('roots') or {}
    except (OSError, ValueError):
        got = {}
    if got:
        return got
    import dashboard as dash
    return dash._grid_roots()


def pool_ready(stage=DEFAULT_STAGE):
    return os.path.exists(P(stage)['pool'])


def _drawn_keys(stage=DEFAULT_STAGE):
    """Every box ever put in front of anyone, judged or not.

    Two sources, because either alone can be incomplete: the draw log is what
    was shown, and the verdict ledger is what was answered. A box can be
    answered without the draw log surviving -- the pool was rebuilt once and
    the log went with it -- and an answered box is seen by definition, so it
    must never come back round.
    """
    keys, seqs = set(), set()
    pp = P(stage)
    for path in (pp['drawn'], pp['verdicts']):
        try:
            with open(path) as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(d, dict):
                        continue
                    if d.get('key'):
                        keys.add(d['key'])
                    if d.get('seq'):
                        seqs.add(d['seq'])
        except OSError:
            pass
    return keys, seqs


def band_list(band, stage=DEFAULT_STAGE):
    """Whatever the caller asked for, as a list of band indices."""
    n = len(fa.BANDS)
    if band is None or band == 'all':
        return list(range(n))
    if band == 'rejected':
        return [i for i in range(n) if fa.BANDS[i][0] < fa.THRESHOLD]
    if band == 'kept':
        return [i for i in range(n) if fa.BANDS[i][0] >= fa.THRESHOLD]
    try:
        i = int(band)
    except (TypeError, ValueError):
        return list(range(n))
    return [i] if 0 <= i < n else list(range(n))


def sample(n=25, band=None, seed=None, stage=DEFAULT_STAGE):
    """A page of candidates: stratified by band, one box per sequence, none
    ever drawn before.

    One box per sequence is the load-bearing part. Mapillary frames come a
    second apart down one road, so a sequence's boxes are the same handful of
    objects photographed repeatedly -- scoring twenty of them would state a
    confidence twenty independent samples would earn and these do not.
    """
    import duckdb
    keys, seqs = _drawn_keys(stage)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("CREATE TEMP TABLE seen_seq(seq VARCHAR)")
    if seqs:
        con.executemany("INSERT INTO seen_seq VALUES (?)",
                        [(s,) for s in seqs])
    # A band can be one band, or a side of the threshold. False negatives --
    # the whole reason to run this -- only exist where the gate said no, so
    # "rejected" is a first-class choice and the page's default rather than
    # something to be assembled by picking five bands one at a time.
    bands = band_list(band, stage)
    # Evenly over the bands asked for, with the remainder going to the ones
    # NEAREST the threshold -- that is where a wrong answer is most likely to
    # be found, and flooring instead quietly dropped four crops off every
    # page of twenty-four.
    base, extra = divmod(n, max(1, len(bands)))
    quota = {b: base + (1 if i >= len(bands) - extra else 0)
             for i, b in enumerate(bands)}
    per = max(quota.values()) if quota else 1
    salt = str(seed if seed is not None else random.getrandbits(48))
    rows = con.execute(f"""
        WITH fresh AS (
            SELECT p.* FROM read_parquet('{P(stage)['pool']}') p
            ANTI JOIN seen_seq s ON s.seq = p.seq
            WHERE p.band IN ({','.join(str(int(b)) for b in bands)})
        ), one_per_seq AS (
            SELECT * FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY seq ORDER BY hash(image_id || ? )) rn
                FROM fresh) WHERE rn = 1
        )
        SELECT band, image_id, det_idx, p_dog, x1, y1, x2, y2,
               cell, drive, seq, conf
        FROM (SELECT *, row_number() OVER (
                  PARTITION BY band ORDER BY hash(seq || ? )) bn
              FROM one_per_seq)
        WHERE bn <= {int(per)}
        ORDER BY band, bn
    """, [salt, salt]).fetchall()
    con.close()
    cols = ('band', 'image_id', 'det_idx', 'p_dog', 'x1', 'y1', 'x2', 'y2',
            'cell', 'drive', 'seq', 'conf')
    out, taken = [], {}
    for r in rows:
        d = dict(zip(cols, r))
        d['key'] = f"{d['image_id']}#{d['det_idx']}"
        if d['key'] in keys:
            continue
        b = int(d['band'])
        if taken.get(b, 0) >= quota.get(b, 0):
            continue                     # the SQL drew `per`; keep this band's
        taken[b] = taken.get(b, 0) + 1
        d['p_dog'] = float(d['p_dog'])
        d['conf'] = float(d['conf'])
        out.append(d)
    return out


def _cut_one(cand, roots, stage=DEFAULT_STAGE, into=None, force=False):
    """Cut one crop to disk. Returns True if the file is there afterwards.

    `into='edited'` writes only the full-resolution cut, into the corrected
    directory: that path exists so a hand-drawn box can be re-cut without
    disturbing full/, which is the picture the model was actually given.
    """
    from PIL import Image
    pp = P(stage)
    if into == 'edited':
        os.makedirs(os.path.join(pp['out'], 'edited'), exist_ok=True)
    dst = os.path.join(pp['crops'], cand['key'].replace('#', '_') + '.jpg')
    if os.path.exists(dst) and not force:
        return True
    root = roots.get(cand['drive'])
    if not root:
        return False
    src = os.path.join(root, cand['cell'], 'ground_animal_images',
                       f"{cand['image_id']}.jpg")
    try:
        im = Image.open(src)
        im.load()
        if im.mode != 'RGB':
            im = im.convert('RGB')
    except Exception:
        return False
    try:
        w, h = im.size
        x1, y1, x2, y2 = (int(cand['x1']), int(cand['y1']),
                          int(cand['x2']), int(cand['y2']))
        pad = int(PAD_FRAC * max(x2 - x1, y2 - y1)) + PAD_PX
        a, b = max(0, x1 - pad), max(0, y1 - pad)
        c, d = min(w, x2 + pad), min(h, y2 + pad)
        if c - a < MIN_SIDE or d - b < MIN_SIDE:
            return False
        crop = im.crop((a, b, c, d))
        # full resolution first: this is the one a future dataset uses, and
        # thumbnail() resizes in place
        base = os.path.join(pp['out'], 'edited') if into == 'edited' \
            else pp['full']
        os.makedirs(base, exist_ok=True)
        fdst = os.path.join(base, cand['key'].replace('#', '_') + '.jpg')
        ftmp = fdst + '.tmp'
        crop.save(ftmp, 'JPEG', quality=95)
        os.replace(ftmp, fdst)
        if into == 'edited':
            return True          # full/ and the thumbnail stay as the model saw
        crop.thumbnail((CROP_PX, CROP_PX), Image.LANCZOS)
        os.makedirs(pp['crops'], exist_ok=True)
        tmp = dst + '.tmp'
        crop.save(tmp, 'JPEG', quality=88)
        os.replace(tmp, dst)
        return True
    except Exception:
        return False
    finally:
        im.close()


def materialise(cands, workers=8, stage=DEFAULT_STAGE):
    """Cut every crop on a page, in parallel.

    Decoding one 8000x4000 frame is ~116 ms and each candidate is a different
    frame by construction, so a page of 24 is 2.8 seconds serial and a third
    of a second spread over the decoders. Pillow drops the GIL inside decode,
    so threads are enough and there is no process pool to pay for.
    """
    from concurrent.futures import ThreadPoolExecutor
    roots = _roots()
    os.makedirs(P(stage)['crops'], exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = list(ex.map(lambda c: _cut_one(c, roots, stage), cands))
    return [c for c, good in zip(cands, ok) if good]


def _pool_row(key, stage=DEFAULT_STAGE):
    """One box out of the pool, by key, with the geometry to cut it."""
    iid, _, di = str(key).rpartition('_')
    if not iid or not di.isdigit():
        return None
    try:
        import duckdb
        row = duckdb.connect().execute(
            f"""SELECT band, image_id, det_idx, p_dog, x1, y1, x2, y2,
                       cell, drive, seq, conf
                FROM read_parquet('{P(stage)['pool']}')
                WHERE image_id = ? AND det_idx = ?""",
            [iid, int(di)]).fetchone()
    except Exception:
        return None
    if not row:
        return None
    cols = ('band', 'image_id', 'det_idx', 'p_dog', 'x1', 'y1', 'x2', 'y2',
            'cell', 'drive', 'seq', 'conf')
    d = dict(zip(cols, row))
    d['key'] = f"{d['image_id']}#{d['det_idx']}"
    return d


# ── correcting a box ────────────────────────────────────────────────────────
# The project already has one place for hand-drawn boxes -- the review page
# writes them and harvest_flagged.py reads them -- so a correction made here
# lands in the same file, keyed the same way, and every consumer gets it.
BOX_FILE = os.path.join(REPO, 'data', 'box_corrections', 'boxes.jsonl')
_BOX_LOCK = threading.Lock()
# How much of the frame to show around the box while editing. A box is a dog
# in a street; you cannot tell whether it is framed right without seeing what
# is next to it, and you cannot drag an edge that is off-screen.
VIEW_PAD = 2.2
VIEW_MAX = 1100


def corrections():
    """{(image_id, det_idx): (x1, y1, x2, y2)} -- last write wins."""
    out = {}
    try:
        with open(BOX_FILE) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(d, dict) or d.get('image_id') is None:
                    continue
                try:
                    out[(str(d['image_id']), int(d.get('det_idx') or 0))] = (
                        float(d['x1']), float(d['y1']),
                        float(d['x2']), float(d['y2']))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return out


def save_correction(key, box, stage=DEFAULT_STAGE):
    """Record a hand-drawn box and re-cut the crop a future model trains on.

    It does NOT touch what the audit measured. The verdict on this box is a
    verdict about the picture the model was given, and the score it gave is
    the same score whatever anyone redraws afterwards. A correction says only
    "if you train on this crop, train on this framing" -- so the model's cut
    stays in full/ and the corrected one is written beside it.
    """
    cand = _pool_row(str(key).replace('#', '_'), stage)
    if not cand:
        return {'ok': False, 'msg': 'not a box in this pool'}
    try:
        x1, y1, x2, y2 = (float(box[0]), float(box[1]),
                          float(box[2]), float(box[3]))
    except (TypeError, ValueError, IndexError):
        return {'ok': False, 'msg': 'malformed box'}
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return {'ok': False, 'msg': f'a box under {MIN_SIDE}px is not a box'}
    rec = {'crop': f"audit:{stage}", 'image_id': str(cand['image_id']),
           'det_idx': int(cand['det_idx']),
           'x1': round(x1, 2), 'y1': round(y1, 2),
           'x2': round(x2, 2), 'y2': round(y2, 2),
           'saved_at': int(time.time())}
    with _BOX_LOCK:
        os.makedirs(os.path.dirname(BOX_FILE), exist_ok=True)
        with open(BOX_FILE, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    edited = dict(cand, x1=x1, y1=y1, x2=x2, y2=y2)
    ok = _cut_one(edited, _roots(), stage, into='edited', force=True)
    # If this box has already been judged it is already IN the dataset, cut to
    # the old framing. The rebuild would pick the new one up, but until
    # someone ran it the exported crop and the box on record disagreed -- so
    # the crop is re-filed here, the moment the correction is made.
    placed = None
    seen = {v['key']: v.get('verdict') for v in fa.read_verdicts(stage=stage)}
    if cand['key'] in seen:
        placed = place(cand['key'], seen[cand['key']], stage,
                       force=True)
    return {'ok': True, 'recut': bool(ok), 'refiled': placed}


def frame_view(key, stage=DEFAULT_STAGE):
    """A window of the source frame around this box, and the maths to map it.

    The full frame is 8000x4000 and nobody can drag a handle on that, so a
    region around the box is cut and scaled down. Everything the client sends
    back is in VIEW pixels; `scale` and the offsets turn them into the
    original pixels the store speaks.
    """
    from PIL import Image
    cand = _pool_row(str(key).replace('#', '_'), stage)
    if not cand:
        return None, None
    root = _roots().get(cand['drive'])
    if not root:
        return None, None
    src = os.path.join(root, cand['cell'], 'ground_animal_images',
                       f"{cand['image_id']}.jpg")
    cur = corrections().get((str(cand['image_id']), int(cand['det_idx'])))
    x1, y1, x2, y2 = (cur if cur else (cand['x1'], cand['y1'],
                                       cand['x2'], cand['y2']))
    try:
        im = Image.open(src)
        im.load()
        if im.mode != 'RGB':
            im = im.convert('RGB')
    except Exception:
        return None, None
    try:
        w, h = im.size
        pad = int(VIEW_PAD * max(x2 - x1, y2 - y1))
        a, b = max(0, int(x1) - pad), max(0, int(y1) - pad)
        c, d = min(w, int(x2) + pad), min(h, int(y2) + pad)
        view = im.crop((a, b, c, d))
        scale = min(1.0, VIEW_MAX / max(1, max(view.size)))
        if scale < 1.0:
            view = view.resize((max(1, int(view.width * scale)),
                                max(1, int(view.height * scale))),
                               Image.LANCZOS)
        import io
        buf = io.BytesIO()
        view.save(buf, 'JPEG', quality=88)
        return buf.getvalue(), {
            'key': cand['key'], 'stage': stage,
            'off_x': a, 'off_y': b, 'scale': scale,
            'view_w': view.width, 'view_h': view.height,
            'orig_w': w, 'orig_h': h,
            # where the box sits INSIDE the view, in view pixels
            'box': [(x1 - a) * scale, (y1 - b) * scale,
                    (x2 - a) * scale, (y2 - b) * scale],
            'model_box': [(cand['x1'] - a) * scale, (cand['y1'] - b) * scale,
                          (cand['x2'] - a) * scale, (cand['y2'] - b) * scale],
            'corrected': bool(cur)}
    except Exception:
        return None, None
    finally:
        im.close()


def crop_path(key, stage=DEFAULT_STAGE):
    """Absolute path of a cut crop, cutting it again if it has gone.

    A page is kept for ever and its crops are not: the pool can be rebuilt,
    the crop directory cleared, a disk swapped. When that happens the page
    still lists the boxes and every tile comes back as alt text -- which reads
    as a broken page rather than a missing file. The box is still in the pool,
    so it can just be cut again.

    The key is checked against the shape it is generated in, so nothing a
    client sends reaches a path.
    """
    import re
    if not re.fullmatch(r'[0-9]{1,32}_[0-9]{1,6}', str(key or '')):
        return None
    p = os.path.join(P(stage)['crops'], f'{key}.jpg')
    if os.path.exists(p):
        return p
    cand = _pool_row(key, stage)
    if cand and _cut_one(cand, _roots(), stage) and os.path.exists(p):
        return p
    return None


def _page_file(i, stage=DEFAULT_STAGE):
    return os.path.join(P(stage)['pages'], f'{int(i):05d}.json')


def page_count(stage=DEFAULT_STAGE):
    try:
        return len(glob.glob(os.path.join(P(stage)['pages'], '*.json')))
    except OSError:
        return 0


def get_page(i, stage=DEFAULT_STAGE):
    try:
        with open(_page_file(i, stage)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def draw_page(n=25, band=None, stage=DEFAULT_STAGE):
    """Draw, cut, and keep. Returns the page document."""
    draw_lock, _ = _locks(stage)
    pp = P(stage)
    with draw_lock:
        cands = sample(n=n, band=band, stage=stage)
        if not cands:
            return {'index': page_count(stage), 'items': [], 'exhausted': True,
                    'band': band, 'n': n, 'dropped': 0}
        # Reserved before a single frame is opened. A box counts as drawn the
        # moment it is chosen, not when it is judged or even when it is
        # successfully cut: a concurrent draw must not pick it, and a box
        # skipped on screen must not come back three pages later, or "you have
        # not seen this" is untrue and the sample quietly correlates.
        os.makedirs(pp['out'], exist_ok=True)
        with open(pp['drawn'], 'a') as fh:
            for c in cands:
                fh.write(json.dumps({'key': c['key'], 'seq': c['seq']}) + '\n')

    got = materialise(cands, stage=stage)   # slow, and holds nothing

    with draw_lock:
        idx = page_count(stage)
        # Frames that would not open. Usually a jpg pruned off a drive after
        # the sweep read it. Counted rather than hidden, so a short page reads
        # as a short page and not as a page that lost its crops.
        doc = {'index': idx, 'band': band, 'n': n, 'stage': stage,
               'created': time.time(),
               'dropped': len(cands) - len(got),
               'items': [{k: c[k] for k in
                          ('key', 'image_id', 'det_idx', 'p_dog', 'conf',
                           'band', 'seq', 'drive', 'cell')} for c in got]}
        os.makedirs(pp['pages'], exist_ok=True)
        tmp = _page_file(idx, stage) + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(doc, fh)
        os.replace(tmp, _page_file(idx, stage))
        return doc


def place(key, verdict, stage=DEFAULT_STAGE, force=False):
    """Put one judged crop into the dataset, or take it out.

    Hard-linked, not copied: the full-resolution cut already exists and a
    second copy of it would drift from the first the moment either is
    re-cut. 'unsure' is not a class -- it is removed from both, so changing
    your mind to "I cannot tell" does not leave a stale label behind.
    """
    sp, pp = fa.spec(stage), P(stage)
    name = str(key).replace('#', '_') + '.jpg'
    # the hand-drawn framing when there is one, exactly as export() chooses:
    # two paths that pick a different file for the same crop is a dataset that
    # depends on which one ran last
    fixed = os.path.join(pp['out'], 'edited', name)
    src = fixed if os.path.exists(fixed) else os.path.join(pp['full'], name)
    want = fa.verdict_of(verdict, stage)
    classes = (sp['positive'], sp['negative'])
    if want not in classes:
        want = None
    for cls in classes:
        dst = os.path.join(pp['dataset'], cls, name)
        if cls == want:
            continue
        try:
            os.remove(dst)             # a changed mind moves the file
        except OSError:
            pass
    if not want or not os.path.exists(src):
        return False
    dst = os.path.join(pp['dataset'], want, name)
    if os.path.exists(dst):
        # `force` is what a correction needs: the file is already filed under
        # the right class, but it is the OLD framing, and returning early here
        # made re-filing after a redraw a no-op.
        if not force:
            return True
        try:
            os.remove(dst)
        except OSError:
            return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:                    # different filesystem, or no hardlinks
        import shutil
        shutil.copy2(src, dst)
    return True


def record(key, verdict, meta=None, stage=DEFAULT_STAGE):
    """Append one human judgement.

    Append-only and re-readable: a mind changed later is another line, and the
    reader keeps the last one. Nothing rewrites history in place, so a crash
    mid-write costs one line rather than the file.
    """
    # `None` clears: undo is a verdict being withdrawn, not a third opinion,
    # and the ledger is append-only so it is written as one more line.
    if verdict is not None and fa.verdict_of(verdict, stage) is None:
        return {'ok': False, 'msg': f'unknown verdict {verdict!r}'}
    verdict = fa.verdict_of(verdict, stage) if verdict is not None else None
    rec = {'key': str(key), 'verdict': verdict, 'ts': time.time()}
    for k in ('band', 'p_dog', 'seq'):
        if meta and meta.get(k) is not None:
            rec[k] = meta[k]
    _, ledger_lock = _locks(stage)
    pp = P(stage)
    with ledger_lock:
        os.makedirs(pp['out'], exist_ok=True)
        with open(pp['verdicts'], 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        # the ledger is the record; the dataset is a view of it, kept in step
        # as each verdict lands so it is never a rebuild away from usable
        placed = place(key, verdict, stage)
    return {'ok': True, 'placed': placed}


def stats(stage=DEFAULT_STAGE):
    s = fa.summarise(stage=stage)
    s['counts'] = _judged_counts(stage)
    s['pages'] = page_count(stage)
    s['drawn'] = len(_drawn_keys(stage)[0])
    return s


# ── the page ────────────────────────────────────────────────────────────────
# Sibling to /review, so it wears the same clothes: this is one more judging
# surface, not a second product, and a reviewer moving between them should not
# have to learn a second set of controls.
#
# The hero is the MEASUREMENT, not a count. "412 judged" says how much work
# was done; "0.8% of rejected boxes were dogs [0.3%, 1.9%]" is the thing the
# page exists to produce, and it is stated with its interval because at these
# rates a handful of finds moves it a long way.
AUDIT_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit &mdash; __H1__</title><style>
:root{--bg:#13151a;--panel:#1b2027;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;
--red:#ef5350;--gap:20px;
/* Numbers get their own face. Every score, count and interval on this page is
   read by comparison -- 0.048 against 0.462, 6/7 against 4/4 -- and in a
   proportional face the digits move under the eye between rows. This is the
   one typographic choice here and it is a functional one. */
--num:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);-webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  line-height:1.5;padding:0 22px 120px}
.wrap{max-width:1560px;margin:0 auto}
a{color:inherit}
/* ── header ── */
header{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;
  padding:22px 0 16px;border-bottom:1px solid var(--bd);margin-bottom:18px}
h1{font-size:20px;font-weight:660;letter-spacing:-.3px}
.sub{color:var(--dim);font-size:12.5px;margin-top:3px;max-width:56ch}
.tabs{margin-left:auto;display:inline-flex;gap:2px;padding:2px;
  border:1px solid var(--bd);border-radius:10px}
.tab{font-size:12px;color:var(--dim);text-decoration:none;padding:5px 11px;
  border-radius:8px}
.tab:hover{color:var(--tx)}
.tab.on{background:rgba(232,166,69,.15);color:var(--acc);font-weight:640}
.back{font-size:12px;color:var(--mut);text-decoration:none;
  border:1px solid var(--bd);border-radius:8px;padding:6px 11px}
.back:hover{color:var(--tx);border-color:rgba(130,140,150,.3)}
/* ── the measurement ── */
/* THE WORK COMES FIRST. The measurement block, the pool notice and a
   ten-row band table sat above the photographs -- around five hundred pixels
   of statistics before the first crop, on a page whose entire job is showing
   crops and taking answers. The numbers move by a tenth of a percent per
   verdict; they are worth reading once a session, not once a click. So they
   fold into one line that is always true and open when asked. */
.figures{margin-bottom:14px}
.figsum{list-style:none;cursor:pointer;display:flex;gap:12px;
  align-items:baseline;padding:9px 14px;background:var(--panel);
  border:1px solid var(--bd);border-radius:12px;font-size:12.5px;
  color:var(--mut)}
.figsum::-webkit-details-marker{display:none}
.figsum:hover{border-color:rgba(130,140,150,.3)}
.figline b{color:var(--tx);font-weight:640;font-variant-numeric:tabular-nums;
  font-family:var(--num)}
.figmore{margin-left:auto;color:var(--dim);font-size:11.5px}
.figmore::after{content:' \25b8'}
.figures[open] .figmore::after{content:' \25be'}
.figures[open] .figsum{border-radius:12px 12px 0 0;border-bottom:0}
.figures[open] .meas{border-radius:0}
.meas{display:flex;gap:26px;align-items:flex-end;flex-wrap:wrap;
  background:var(--panel);border:1px solid var(--bd);border-top:0;
  padding:16px 20px}
.mbig{font-size:34px;font-weight:680;letter-spacing:-1.2px;line-height:1.1;
  font-variant-numeric:tabular-nums;font-family:var(--num)}
.mlab{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
  color:var(--dim)}
.mci{font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}
.mnote{font-size:12px;color:var(--dim);max-width:44ch}
.mnote b{color:var(--mut);font-weight:600}
/* ── per band ──
   Each row is an ESTIMATE, so it is drawn as one: the 95% interval as a
   segment, the point estimate as a tick inside it. It used to be a plain bar
   whose length was the rate times an invented factor of six, which made a
   25% rate off four crops look exactly as solid as one off four hundred.
   The axis is shared and its top is labelled, because a bar with no scale is
   a shape, not a measurement. */
.poolwarn{background:rgba(232,166,69,.09);border:1px solid rgba(232,166,69,.3);
  border-radius:11px;padding:9px 14px;margin-bottom:12px;font-size:12px;
  color:var(--acc)}
.poolwarn[hidden]{display:none}
.bands{background:var(--panel);border:1px solid var(--bd);border-radius:14px;
  padding:14px 18px 16px;margin-bottom:16px}
.bfoot{display:grid;grid-template-columns:118px 96px 1fr 116px;gap:14px;
  font-size:10px;color:var(--dim);padding-top:7px;margin-top:4px;
  border-top:1px solid var(--bd)}
.bfoot span:nth-child(3){display:flex;justify-content:space-between;
  grid-column:3}
.bfoot span:first-child,.bfoot span:nth-child(2){display:none}
/* the axis runs the width of the track column, so its end labels sit under it */
.bfoot{grid-template-columns:118px 96px 1fr 116px}
.bhead{display:grid;grid-template-columns:118px 96px 1fr 116px;gap:14px;
  align-items:baseline;font-size:10px;text-transform:uppercase;
  letter-spacing:.07em;color:var(--dim);padding-bottom:8px;
  border-bottom:1px solid var(--bd);margin-bottom:4px}
.bhead .ax{text-align:right}
.brow{display:grid;grid-template-columns:118px 96px 1fr 116px;gap:14px;
  align-items:center;font-size:11.5px;color:var(--mut);padding:6px 0;
  font-variant-numeric:tabular-nums}
.brow+.brow{border-top:1px solid rgba(130,140,150,.07)}
.bname{color:var(--tx)}
.brow.nil .bname{color:var(--dim)}
.bwhat{color:var(--dim)}
.btrack{position:relative;height:16px}
/* the axis line the intervals sit on, not a container the bar fills */
.btrack::before{content:'';position:absolute;left:0;right:0;top:50%;
  height:1px;background:rgba(130,140,150,.14)}
/* the gate's threshold, drawn where it actually is on the axis */
.btrack::after{content:'';position:absolute;left:50%;top:-2px;bottom:-2px;
  width:1px;background:rgba(130,140,150,.3)}
.btrack.kept{opacity:.72}
.bci{position:absolute;top:50%;transform:translateY(-50%);height:7px;
  border-radius:4px;background:rgba(232,166,69,.28);min-width:2px}
.bdot{position:absolute;top:50%;transform:translate(-50%,-50%);width:3px;
  height:13px;border-radius:2px;background:var(--acc)}
.bzero .bci{background:rgba(67,181,129,.24)}
.bzero .bdot{background:var(--green)}
.bval{text-align:right}
.bval b{color:var(--tx);font-weight:640}
.bnil{color:var(--dim)}
/* ── toolbar ── */
.bar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
.btn{background:var(--panel2);border:1px solid var(--bd);color:var(--mut);
  border-radius:9px;padding:7px 13px;font-size:12.5px;cursor:pointer;
  font-family:inherit}
.btn:hover:not(:disabled){color:var(--tx);border-color:rgba(130,140,150,.32)}
.btn:disabled{opacity:.4;cursor:default}
.btn.go{color:var(--acc);border-color:rgba(232,166,69,.4)}
.pos{font-size:12px;color:var(--dim);margin-left:2px;
  font-variant-numeric:tabular-nums;font-family:var(--num)}
.spacer{margin-left:auto}
.views{display:inline-flex;gap:2px;padding:2px;border:1px solid var(--bd);
  border-radius:10px;margin-right:4px}
.viewbtn{appearance:none;background:transparent;border:0;color:var(--dim);
  border-radius:8px;padding:6px 11px;font-size:12px;cursor:pointer;
  font-family:inherit}
.viewbtn:hover{color:var(--tx)}
.viewbtn.on{background:rgba(232,166,69,.15);color:var(--acc);font-weight:640}
.viewbtn b{font-family:var(--num);font-weight:640;opacity:.75;margin-left:4px}
.bar.foot{margin:18px 0 0;padding-top:14px;border-top:1px solid var(--bd)}
.pick{display:inline-flex;align-items:center;gap:7px;font-size:11.5px;
  color:var(--dim)}
.pick select{background:var(--panel2);border:1px solid var(--bd);
  color:var(--mut);border-radius:9px;padding:7px 9px;font-size:12.5px;
  font-family:inherit;cursor:pointer}
.pick select:hover{color:var(--tx)}
/* ── the sheet ──
   A CONTACT SHEET, the same conclusion the review page reached and this page
   did not inherit: at rest a tile is a photograph and the model's own verdict,
   nothing else. Three buttons under every crop is seventy-five buttons on a
   page of twenty-five, all saying the same three things, while the pictures
   they are about get what is left. They ride over the foot of the frame now,
   and only when the tile is under the cursor, focused, or the selected one.
   The keyboard never needed them. */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(178px,1fr));
  gap:12px}
.card{background:var(--panel);border:1px solid var(--bd);border-radius:12px;
  overflow:hidden;position:relative}
.card.done{opacity:.5}
/* a flag is the FINDING -- it stays lit rather than greying out like an
   answered question, because the whole point of the page is the pile of them */
.card.miss{border-color:var(--acc);opacity:1;
  box-shadow:0 0 0 1px rgba(232,166,69,.35)}
.card.ok{border-color:rgba(67,181,129,.4)}
.card.cur{box-shadow:0 0 0 2px var(--acc)}
.shot{position:relative;background:#0e1014;aspect-ratio:1;display:flex;
  align-items:center;justify-content:center;cursor:zoom-in}
.shot img{max-width:100%;max-height:100%;display:block}
.ptag{position:absolute;left:6px;top:6px;font-size:10.5px;border-radius:6px;
  padding:2px 7px;letter-spacing:.02em;border:1px solid var(--bd);
  background:rgba(10,12,16,.82);color:var(--mut)}
.ptag.yes{background:rgba(232,166,69,.16);border-color:rgba(232,166,69,.42);
  color:var(--acc);font-weight:620}
/* a box you redrew, so the sheet shows which ones you have been through */
.redrawn{position:absolute;left:6px;bottom:6px;font-size:10px;
  letter-spacing:.04em;text-transform:uppercase;border-radius:5px;
  padding:2px 6px;background:rgba(10,12,16,.86);color:var(--mut);
  border:1px solid var(--bd)}
/* Top-right, opposite the model's verdict -- the two things the MODEL says
   read together along the top, and the bottom belongs to what YOU say. It sat
   bottom-right, in the same corner the buttons appear in, so the score showed
   through "not a dog" whenever the row came up. */
.pchip{position:absolute;right:6px;top:6px;font-size:10.5px;
  font-family:var(--num);
  background:rgba(10,12,16,.86);border:1px solid var(--bd);border-radius:6px;
  padding:2px 6px;color:var(--mut);font-variant-numeric:tabular-nums;
  opacity:0;transition:opacity .12s ease}
.card:hover .pchip,.card:focus-within .pchip,.card.cur .pchip{opacity:1}
/* The row rides over a PHOTOGRAPH, so it cannot be nearly-opaque and nearly
   legible: at 94% a bright cobbled street came through 11.5px type and the
   labels were unreadable on exactly the tiles that most need looking at. The
   buttons are solid, the gaps are dark rather than light, and a short scrim
   above them keeps the row from looking pasted onto the frame. */
.acts{position:absolute;left:0;right:0;bottom:0;display:grid;
  grid-template-columns:1fr 1fr auto;gap:1px;background:#05070a;
  opacity:0;transform:translateY(4px);pointer-events:none;
  transition:opacity .13s ease,transform .13s ease}
.acts::before{content:'';position:absolute;left:0;right:0;bottom:100%;
  height:26px;pointer-events:none;
  background:linear-gradient(to top,rgba(5,7,10,.92),rgba(5,7,10,0))}
.card:hover .acts,.card:focus-within .acts,.card.cur .acts{opacity:1;
  transform:none;pointer-events:auto}
.act{background:#12151b;border:0;color:var(--tx);font-family:inherit;
  font-size:11.5px;font-weight:560;padding:9px 4px;cursor:pointer;
  letter-spacing:.01em}
.act:hover{background:#1a1f27}
.act.m:hover,.act.m.on{background:rgba(232,166,69,.2);color:var(--acc)}
.act.c:hover,.act.c.on{background:rgba(67,181,129,.18);color:var(--green)}
.act.u{padding:9px 10px}
.act.u:hover,.act.u.on{color:var(--acc)}
/* THE MODEL'S LINE. The sheet is ordered by score, so reading it is a walk
   from "certainly not" to "certainly yes" -- and this is drawn where the
   model stops saying one and starts saying the other. Everything above it was
   thrown away; everything below it was kept. It is the whole question of the
   page, made a place on the page. */
.thr{grid-column:1/-1;display:flex;align-items:center;gap:12px;
  color:var(--dim);font-size:11px;text-transform:uppercase;
  letter-spacing:.08em;margin:4px 0 2px}
.thr::before,.thr::after{content:'';height:1px;background:var(--bd);flex:1}
.empty{color:var(--dim);font-size:13px;padding:40px 0;text-align:center}
/* ── lightbox ── */
.lb{position:fixed;inset:0;background:rgba(0,0,0,.9);display:flex;
  align-items:center;justify-content:center;flex-direction:column;gap:12px;
  z-index:50}
.lb[hidden]{display:none}
/* inline-block so the overlay's `inset:0` is the PICTURE's box and not a
   full-width block the image merely sits inside */
.lbstage{position:relative;line-height:0;display:inline-block}
.lb img{max-width:92vw;max-height:80vh;object-fit:contain}
/* The editor draws over the SAME element the picture is in, so a box in view
   pixels lands where the eye says it does. */
.boxwrap{position:absolute;inset:0}
.boxwrap[hidden]{display:none}
.mbox{position:absolute;border:1px dashed rgba(130,140,150,.7);
  pointer-events:none}
.ebox{position:absolute;border:2px solid var(--acc);cursor:move;
  box-shadow:0 0 0 9999px rgba(5,7,10,.45)}
.ebox i{position:absolute;width:14px;height:14px;background:var(--acc);
  border-radius:3px}
.ebox i[data-h=nw]{left:-8px;top:-8px;cursor:nwse-resize}
.ebox i[data-h=ne]{right:-8px;top:-8px;cursor:nesw-resize}
.ebox i[data-h=sw]{left:-8px;bottom:-8px;cursor:nesw-resize}
.ebox i[data-h=se]{right:-8px;bottom:-8px;cursor:nwse-resize}
.lbnote{font-size:12px;color:var(--dim);max-width:60ch;text-align:center}
.lbnote[hidden]{display:none}
.lbcap{font-size:12px;color:var(--mut);display:flex;gap:10px;
  align-items:center}
.lbcap button{background:var(--panel2);border:1px solid var(--bd);
  color:var(--mut);border-radius:7px;padding:4px 9px;font-size:11.5px;
  cursor:pointer;font-family:inherit}
.undotoast{position:fixed;right:24px;bottom:24px;display:flex;gap:12px;
  align-items:center;background:var(--panel2);border:1px solid var(--bd);
  border-radius:12px;padding:10px 12px;z-index:60;
  box-shadow:0 10px 30px rgba(0,0,0,.45)}
.undotoast[hidden]{display:none}
.undotoast img{width:38px;height:38px;object-fit:cover;border-radius:8px;
  background:#000;flex:none}
.undotoast .tt{color:var(--dim);font-size:11px;white-space:nowrap}
.undotoast .tt b{display:block;color:var(--tx);font-size:12.5px;
  font-weight:620}
.toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);
  background:var(--panel2);border:1px solid var(--bd);border-radius:9px;
  padding:8px 14px;font-size:12.5px;color:var(--tx);z-index:60}
.toast[hidden]{display:none}
.keys{font-size:11.5px;color:var(--dim);margin-top:16px}
.keys kbd{background:var(--panel2);border:1px solid var(--bd);border-radius:5px;
  padding:1px 5px;font-family:ui-monospace,monospace;font-size:11px;
  color:var(--mut)}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
@media(prefers-reduced-motion:no-preference){.card{transition:opacity .12s ease,
  border-color .12s ease}}
</style></head><body><div class="wrap">
<header>
  <div><h1>__H1__</h1>
    <div class="sub">__SUB__</div></div>
  <span class="tabs">__TABS__</span>
  <a class="back" href="/">&larr; dashboard</a>
</header>

<details class="figures" id="figures">
  <summary class="figsum">
    <span class="figline" id="figline">nothing judged yet</span>
    <span class="figmore">the numbers</span>
  </summary>
  <div class="meas">
    <div><div class="mlab">__MISSLAB__</div>
      <div class="mbig" id="rate">&mdash;</div>
      <div class="mci" id="ci">nothing judged yet</div></div>
    <div><div class="mlab">you have judged</div>
      <div class="mbig" id="judged">0</div>
      <div class="mci" id="found">&nbsp;</div></div>
    <div class="mnote">Bands are drawn from evenly, so this is not a
      proportion of what you have seen &mdash; it weights each band by how
      many boxes the model really put in it.</div>
  </div>
<div class="poolwarn" id="poolwarn" hidden></div>
<div class="bands" id="bands"></div>
</details>

<div class="bar">
  <span class="views" id="views" role="tablist">
    <button type="button" class="viewbtn on" data-view="sheet">to judge</button>
    <button type="button" class="viewbtn" data-view="flagged">flagged
      <b id="nFlagged">0</b></button>
    <button type="button" class="viewbtn" data-view="all">everything I
      answered <b id="nAll">0</b></button>
  </span>
  <button class="btn" id="prev">&larr; back</button>
  <button class="btn go" id="next">next page &rarr;</button>
  <span class="pos" id="pos">&mdash;</span>
  <span class="spacer"></span>
  <label class="pick">crops per page
    <select id="size">
      <option value="25">25</option><option value="50">50</option>
      <option value="75">75</option><option value="100">100</option>
    </select></label>
  <label class="pick">draw from
    <select id="bandsel">
      <option value="all">every band</option>
      <option value="rejected">below 0.5 &mdash; __BELOWTXT__</option>
      <option value="kept">0.5 and up &mdash; __ABOVETXT__</option>
    </select></label>
  <button class="btn" id="fresh">&#8635; draw a new page</button>
</div>

<div class="grid" id="grid"></div>
<div class="empty" id="empty" hidden></div>
<!-- A page of a hundred crops is a long way from the toolbar, and the way
     out of it should be where the reading finishes, not where it started. -->
<div class="bar foot" id="foot">
  <button class="btn" id="prev2">&larr; back</button>
  <button class="btn go" id="next2">next page &rarr;</button>
  <span class="pos" id="pos2">&mdash;</span>
</div>
<div class="keys">
  <kbd>F</kbd> __YESTXT__ &nbsp; <kbd>2</kbd> __NOTXT__ &nbsp;
  <kbd>3</kbd> unsure &nbsp; <kbd>U</kbd> undo &nbsp;
  <kbd>&larr;</kbd><kbd>&rarr;</kbd> move &nbsp;
  <kbd>Enter</kbd> enlarge &nbsp; <kbd>E</kbd> redraw the box &nbsp;
  <kbd>N</kbd> next page
</div>
</div>

<div class="lb" id="lb" hidden>
  <div class="lbstage" id="lbstage"><img id="lbimg" alt="">
    <div class="boxwrap" id="boxwrap" hidden>
      <div class="mbox" id="mbox"></div>
      <div class="ebox" id="ebox">
        <i data-h="nw"></i><i data-h="ne"></i>
        <i data-h="sw"></i><i data-h="se"></i>
      </div>
    </div></div>
  <div class="lbcap"><span id="lbtxt"></span>
    <button id="lbedit">redraw the box</button>
    <button id="lbsave" hidden>save box</button>
    <button id="lbcancel" hidden>cancel</button>
    <button id="lbcopy">copy image id</button>
    <button id="lbclose">close</button></div>
  <div class="lbnote" id="lbnote" hidden>Drag a corner. This changes the crop
    a future model trains on &mdash; not what this one was judged on.</div>
</div>
<div class="toast" id="toast" hidden></div>
<div class="undotoast" id="undotoast" hidden></div>

<script>
var BANDS=__BANDS__,STAGE=__STAGE__,POS=__POS__,NEG=__NEG__,
    YES=__YES__,NO=__NO__,BELOW=__BELOW__,ABOVE=__ABOVE__,
    DEFAULT_BAND=__DEFBAND__,THRESH=__THRESH__;
var grid=document.getElementById('grid'),empty=document.getElementById('empty'),
    posEl=document.getElementById('pos'),lb=document.getElementById('lb'),
    lbimg=document.getElementById('lbimg'),lbtxt=document.getElementById('lbtxt');
/* -1 is NOT a crop. The page used to open with the first one ringed, which
   reads as a choice already made on your behalf before you have looked. */
/* Where to draw from by default depends on which model this is. A dog the
   gate said no to is gone, so that side is the one worth walking and the
   default is "below 0.5". The leash model's two errors cost the same -- it
   was promoted on balanced accuracy -- so defaulting to one side would
   contradict the sentence at the top of its own page telling you to read
   both. */
var page=null,idx=0,cur=-1,total=0,band=DEFAULT_BAND,busy=false,size=25,
    dirty=false;
/* 'sheet' draws from the pool and is the only view that spends new crops.
   The other two read the ledger back, so you can look at what you answered
   and change it -- the ledger already took withdrawals; the only thing
   missing was a way to find the crop again. */
var view='sheet';

function toast(t){var e=document.getElementById('toast');e.textContent=t;
  e.hidden=false;clearTimeout(e._t);e._t=setTimeout(function(){e.hidden=true},1600)}
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML}
function pctTxt(v){return (v*100).toFixed(1)+'%'}
function bandName(b){
  if(typeof b==='number'&&BANDS[b])
    return BANDS[b][0].toFixed(1)+'–'+BANDS[b][1].toFixed(1);
  return b==='kept'?ABOVE : b==='all'?'every band' : BELOW;
}
function fmtn(n){return (n||0).toLocaleString('en-US')}
/* The ONE place a crop URL is built. There were three, each assembling the
   path by hand, and the stage prefix was added to one of them -- so every
   thumbnail on the leash page asked the gate for a crop it does not have and
   the grid came back as rows of alt text. */
/* What the model called this box. It is not stored on the row -- the score
   is, and the label is the score against the threshold -- so it is derived
   here rather than carried, which means the two can never disagree. */
function predOf(it){return (+it.p_dog>=THRESH)?POS:NEG}
function cropSrc(it){
  return '/audit/crop/'+STAGE+'/'+esc(String(it.key).replace('#','_'))+'.jpg';
}

/* The verdict is sent the moment it is given and the card is marked from the
   local answer, not from a reload: a reviewer working through a page at speed
   must never wait on a round trip, and the ledger is append-only so a lost
   response costs one line, not the page. */
var lastUndo=null,toastT=null;
var VERDICT_TEXT={};VERDICT_TEXT[POS]='Flagged \u2014 '+YES;
VERDICT_TEXT[NEG]=NO;VERDICT_TEXT.unsure='Left as unsure';
function judge(i,verdict){
  if(i<0)return;
  var it=page.items[i]; if(!it)return;
  it.verdict=verdict;
  if(view!=='sheet'){
    /* In a review view the crop is not work to get through -- it is a record
       you came to look at. Answering again changes the record and leaves it
       where it is, so you can see what you changed. */
    paintCard(i);
    send(it.key,verdict,it);
    return;
  }
  /* EVERY answer takes the crop off the grid. The grid is the work left, not
     a record of what was answered -- a page of a hundred where the judged ones
     linger greyed out is a page you have to keep re-reading to find the ones
     you have not done. What was flagged is in the count above and in the
     dataset; what is on screen is what is still to do.
     The cursor does not walk on by itself either: it stays where it was put,
     because the page moving your selection for you is the page choosing. */
  hide(i);
  offerUndo(it,i,verdict);
  send(it.key,verdict,it);
}
function send(key,verdict,it){
  fetch('/api/audit/verdict?stage='+STAGE,{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:key,verdict:verdict,band:it?it.band:null,
                         p_dog:it?it.p_dog:null,seq:it?it.seq:null})})
    .then(function(r){return r.json()})
    .then(function(j){if(!j||!j.ok)toast('not recorded');else loadStats()})
    .catch(function(){toast('not recorded')});
}
function hide(i){
  var el=grid.children[i];
  if(el){el.style.display='none';el.setAttribute('data-gone','1')}
  left();
}
function unhide(i){
  var el=grid.children[i];
  if(el){el.style.display='';el.removeAttribute('data-gone')}
  left();
}
/* Pure: how many crops on this page are still unanswered. Kept separate from
   the redraw because setPos() wants the number too, and the two calling each
   other is a stack overflow rather than a layout. */
function remaining(){
  if(!page||!page.items)return 0;
  var n=0;
  for(var i=0;i<page.items.length;i++)if(!page.items[i].verdict)n++;
  return n;
}
function left(){
  var n=remaining(),e=document.getElementById('empty');
  if(page&&page.items&&page.items.length){
    if(n===0){e.hidden=false;
      e.textContent='Every crop on this page is done \u2014 next page \u2192'}
    else e.hidden=true;
  }
  setPos();
  return n;
}
function offerUndo(it,i,verdict){
  lastUndo={key:it.key,i:i};
  var t=document.getElementById('undotoast');
  t.innerHTML='<img src="'+cropSrc(it)+'" alt="">'+
    '<span class="tt"><b>'+(VERDICT_TEXT[verdict]||'Recorded')+'</b>'+
    esc(it.image_id)+'</span>'+
    '<button class="btn" id="undoB">Undo</button>';
  t.hidden=false;
  document.getElementById('undoB').onclick=undoLast;
  clearTimeout(toastT);
  /* five seconds and it stands -- it is already on disk either way */
  toastT=setTimeout(function(){lastUndo=null;t.hidden=true},5000);
}
function undoLast(){
  var u=lastUndo; if(!u)return;
  lastUndo=null;clearTimeout(toastT);
  document.getElementById('undotoast').hidden=true;
  var it=page.items[u.i];
  if(it){delete it.verdict;unhide(u.i);paintCard(u.i)}
  /* a withdrawal, not a third opinion: the ledger takes a null and the box
     goes back to unjudged */
  send(u.key,null,it);
  toast('put back');
}
function paintCard(i){
  var el=grid.children[i],it=page.items[i]; if(!el||!it)return;
  el.className='card'+(it.verdict&&it.verdict!=='dog'?' done':'')+
    (it.verdict==='dog'?' miss':it.verdict==='not_dog'?' ok':'')+
    (i===cur?' cur':'');
  var b=el.querySelectorAll('.act');
  b[0].classList.toggle('on',it.verdict==='dog');
  b[1].classList.toggle('on',it.verdict==='not_dog');
  b[2].classList.toggle('on',it.verdict==='unsure');
  el.style.boxShadow=i===cur?'0 0 0 2px var(--acc)':'';
}
function render(){
  if(!page||!page.items.length){
    grid.innerHTML='';empty.hidden=false;
    empty.textContent=page&&page.exhausted
      ? 'No sequences left to draw in this band. Every one has been shown.'
      : 'Nothing here yet. Draw a page.';
    return;
  }
  empty.hidden=true;
  /* A frame that would not open is a shorter page, and a shorter page with
     no explanation reads as crops that failed to load. */
  var dr=page.dropped||0;
  posEl.title=dr?dr+' of this page\u2019s frames could not be opened '+
    '(usually a jpg pruned off a drive after the sweep read it)':'';
  /* the sheet runs low score to high, so the crossing happens at most once */
  var crossed=false;
  grid.innerHTML=page.items.map(function(it,i){
    var rule='';
    if(!crossed&&+it.p_dog>=THRESH){
      crossed=true;
      rule='<div class="thr">the model\u2019s line \u2014 '+THRESH+
        ' \u00b7 above this it said '+NO+', below it said '+YES+'</div>';
    }
    var lo=BANDS[it.band][0],hi=BANDS[it.band][1];
    return rule+'<div class="card" data-i="'+i+'">'+
      '<div class="shot" data-zoom="'+i+'">'+
        '<img loading="lazy" src="'+cropSrc(it)+
        '" alt="box '+esc(it.key)+'">'+
        (it.corrected?'<span class="redrawn" title="you redrew this box; '+
          'the training crop uses your framing, the measurement uses the '+
          'model\u2019s">redrawn</span>':'')+
        '<span class="ptag '+(predOf(it)===POS?'yes':'no')+
        '" title="what the model called it: '+predOf(it)+', scoring '+
        it.p_dog.toFixed(3)+' for '+POS+'">'+esc(predOf(it))+'</span>'+
        '<span class="pchip" title="the model scored this '+
        it.p_dog.toFixed(3)+' for '+POS+'; band '+lo.toFixed(1)+'-'+
        hi.toFixed(1)+'">'+
        it.p_dog.toFixed(3)+'</span></div>'+
      '<div class="acts">'+
        '<button class="act m" data-v="'+POS+'" data-i="'+i+'">&#9873; '+YES+'</button>'+
        '<button class="act c" data-v="'+NEG+'" data-i="'+i+'">'+NO+'</button>'+
        '<button class="act u" data-v="unsure" data-i="'+i+'" title="cannot tell">?</button>'+
      '</div></div>';
  }).join('');
  for(var i=0;i<page.items.length;i++)paintCard(i);
}
function move(d){
  if(!page||!page.items.length)return;
  /* the first arrow press selects rather than moving from an assumed first */
  if(cur<0){cur=d>0?0:page.items.length-1}
  else cur=Math.max(0,Math.min(page.items.length-1,cur+d));
  for(var i=0;i<page.items.length;i++)paintCard(i);
  var el=grid.children[cur];
  if(el&&el.scrollIntoView)el.scrollIntoView({block:'nearest'});
}
function setPos(){
  /* `band` is a band index OR a side of the threshold, so it cannot be
     indexed into BANDS without asking which it is -- doing that threw and
     took the whole page down. */
  posEl.textContent=total?('page '+(idx+1)+' of '+total+
    (view!=='sheet'?' \u00b7 '+(view==='flagged'?'flagged':'everything I answered'):'')+
    (page&&page.dropped?' \u00b7 '+page.dropped+' unreadable':'')+
    ' \u00b7 '+bandName(band)+
    (page&&page.items&&page.items.length?' \u00b7 '+remaining()+' left':'')):'—';
  ['prev','prev2'].forEach(function(id){
    document.getElementById(id).disabled=busy||idx<=0});
  ['next','next2'].forEach(function(id){
    document.getElementById(id).disabled=busy||
      (view!=='sheet'&&idx+1>=total)});
  var f=document.getElementById('foot');
  if(f)f.hidden=!(page&&page.items&&page.items.length);
  var p2=document.getElementById('pos2');
  if(p2)p2.textContent=posEl.textContent;
}
function show(doc,at,tot){
  page=doc;idx=at;total=tot;cur=-1;render();setPos();left();
}
function loadJudged(at){
  busy=true;setPos();
  fetch('/api/audit/judged?stage='+STAGE+'&which='+
        (view==='flagged'?'flagged':'all')+'&page='+at+'&n='+size)
    .then(function(r){return r.json()})
    .then(function(j){
      busy=false;
      if(!j){toast('failed');setPos();return}
      counts(j.counts);
      show({index:j.page,items:j.items,dropped:0},j.page,j.pages||1);
    })
    .catch(function(){busy=false;toast('failed');setPos()});
}
function counts(c){
  if(!c)return;
  var a=document.getElementById('nFlagged'),b=document.getElementById('nAll');
  if(a)a.textContent=fmtn(c.flagged||0);
  if(b)b.textContent=fmtn(c.all||0);
}
function load(at){
  if(view!=='sheet')return loadJudged(at);
  busy=true;setPos();
  fetch('/api/audit/page?stage='+STAGE+'&i='+at+'&n='+size+
        (band==null?'':'&band='+encodeURIComponent(band))).then(function(r){return r.json()})
    .then(function(j){busy=false;if(!j||j.error){toast(j&&j.error||'failed');setPos();return}
      show(j.page,j.index,j.total)})
    .catch(function(){busy=false;toast('failed');setPos()});
}
function draw(){
  busy=true;setPos();
  document.getElementById('fresh').textContent='cutting…';
  fetch('/api/audit/draw?stage='+STAGE,{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({band:band,n:size})})
    .then(function(r){return r.json()})
    .then(function(j){busy=false;
      document.getElementById('fresh').textContent='↻ draw a new page';
      if(!j||j.error){toast(j&&j.error||'failed');setPos();return}
      dirty=false;show(j.page,j.index,j.total);loadStats()})
    .catch(function(){busy=false;
      document.getElementById('fresh').textContent='↻ draw a new page';
      toast('failed');setPos()});
}
function loadStats(){
  fetch('/api/audit/stats?stage='+STAGE).then(function(r){return r.json()}).then(paintStats)
    .catch(function(){});
}
function paintStats(s){
  if(!s)return;
  counts(s.counts||{flagged:(s.rejected||{}).wrong,all:s.judged});
  var r=document.getElementById('rate'),ci=document.getElementById('ci'),
      rej=s.rejected||{},kept=s.kept||{};
  document.getElementById('judged').textContent=fmtn(s.judged||0);
  var fl=document.getElementById('figline');
  if(fl)fl.innerHTML=s.judged
    ? '<b>'+fmtn(s.judged)+'</b> judged \u00b7 <b>'+fmtn(s.wrong||0)+
      '</b> the model got wrong \u00b7 <b>'+pctTxt(rej.rate||0)+
      '</b> of what it '+(STAGE==='gate'?'rejected':'called '+NO)
    : 'nothing judged yet \u2014 the numbers fill in as you go';
  document.getElementById('found').textContent=
    (rej.wrong||0)+' the model got wrong below 0.5';
  /* The headline is the one number the page exists to produce: how many dogs
     the gate threw away. It used to read "missed dogs 100.0% -- 3,945,390
     dogs across 3,945,390 rejected boxes", which is what a rate of 1.0 off
     five crops extrapolates to, stated as though it were known. An estimate
     from a handful of crops is not a count of four million things, so the
     extrapolation is only shown once there is enough behind it to mean
     anything, and it is always written as a range. */
  if(!rej.judged){r.textContent='—';
    ci.textContent='judge some crops below 0.5 and this starts counting'}
  else{
    r.textContent=pctTxt(rej.rate||0);
    var lo=0,hi=0,any=false;
    (s.bands||[]).forEach(function(b){
      if(b.kept||!b.judged)return; any=true;
      lo+=b.lo95*b.boxes; hi+=b.hi95*b.boxes;
    });
    var pop=0;(s.bands||[]).forEach(function(b){
      if(!b.kept&&b.judged)pop+=b.boxes});
    ci.textContent = rej.judged<20
      ? 'from '+rej.judged+' crops — too few to put a number on '+
        fmtn(rej.boxes)+' boxes yet'
      : 'somewhere between '+fmtn(Math.round(lo/(pop||1)*rej.boxes))+' and '+
        fmtn(Math.round(hi/(pop||1)*rej.boxes))+' of the '+
        fmtn(rej.boxes)+' it rejected';
  }
  /* One axis, 0 to 100%, with the gate's own threshold drawn on it. The share
     of a band that really is a dog is what the model is trying to predict, so
     against the band's own score it reads as a calibration curve: it should
     climb, and where it crosses 50% is where the threshold belongs. */
  var pi=s.pool_info||{},warn=document.getElementById('poolwarn');
  if(warn){
    warn.textContent=pi.stale
      ? 'This pool was cut from '+pi.shards+' shards and the run has written '+
        pi.shards_now+' since \u2014 the counts below are a snapshot of part '+
        'of the store, not all of it. Rebuild the pool to take them in.'
      : pi.unknown
        ? 'This pool does not record what it was cut from. Rebuild it to find '+
          'out whether it still covers the whole run.'
        : '';
    warn.hidden=!warn.textContent;
  }
  document.getElementById('bands').innerHTML=
    '<div class="bhead"><span>score the gate gave</span>'+
    '<span>in the store</span>'+
    '<span>share that really are dogs, with its 95% interval</span>'+
    '<span class="ax">flagged</span></div>'+
    (s.bands||[]).map(function(b){
      var side=b.kept?' kept':' threw away';
      if(!b.judged)return '<div class="brow nil"><span class="bname">'+
        b.lo.toFixed(1)+'–'+b.hi.toFixed(1)+'</span>'+
        '<span class="bwhat">'+fmtn(b.boxes)+side+'</span>'+
        '<div class="btrack'+(b.kept?' kept':'')+'"></div>'+
        '<span class="bval bnil">none seen</span></div>';
      function at(v){var x=(+v||0)*100;
        return Math.max(0,Math.min(100,x!==x?0:x))}
      var l=at(b.lo95),h=at(b.hi95),m=at(b.rate);
      return '<div class="brow'+(b.dogs?'':' bzero')+'">'+
        '<span class="bname">'+b.lo.toFixed(1)+'–'+b.hi.toFixed(1)+'</span>'+
        '<span class="bwhat">'+fmtn(b.boxes)+side+'</span>'+
        '<div class="btrack'+(b.kept?' kept':'')+'" title="'+b.dogs+' of '+
          b.judged+' were dogs — the bar is the 95% interval, the tick is the '+
          'estimate">'+
          '<div class="bci" style="left:'+l.toFixed(2)+'%;width:'+
            Math.max(0.6,h-l).toFixed(2)+'%"></div>'+
          '<div class="bdot" style="left:'+m.toFixed(2)+'%"></div></div>'+
        '<span class="bval"><b>'+b.dogs+'</b>/'+b.judged+'</span></div>';
    }).join('')+
    '<div class="bfoot"><span>0%</span><span>50% — where the gate '+
    'draws its line</span><span>100%</span></div>';
}
/* clicks */
grid.addEventListener('click',function(e){
  var z=e.target.closest&&e.target.closest('[data-zoom]');
  if(z){zoom(+z.getAttribute('data-zoom'));return}
  var a=e.target.closest&&e.target.closest('.act');
  if(a){cur=+a.getAttribute('data-i');judge(cur,a.getAttribute('data-v'))}
});
var sizeSel=document.getElementById('size'),bandSel=document.getElementById('bandsel');
for(var bi=0;bi<BANDS.length;bi++){
  var o=document.createElement('option');
  o.value=bi;o.textContent='only '+BANDS[bi][0].toFixed(1)+' – '+
    BANDS[bi][1].toFixed(1);
  bandSel.appendChild(o);
}
/* Both choices are remembered. Working through an audit is a long sitting and
   re-picking the page size after every reload is a small tax on the only
   thing this page is for. */
try{
  var sv=localStorage.getItem('sdAuditSize:'+STAGE);
  if(sv&&/^(25|50|75|100)$/.test(sv))sizeSel.value=sv;
  var bv=localStorage.getItem('sdAuditBand:'+STAGE);
  if(bv==='rejected'||bv==='kept'||bv==='all'){band=bv;bandSel.value=bv}
  else if(bv!==null&&bv!==''&&+bv>=0&&+bv<BANDS.length){band=+bv;bandSel.value=bv}
  else bandSel.value=DEFAULT_BAND;
}catch(_){}
size=+sizeSel.value||25;
sizeSel.addEventListener('change',function(){
  size=+sizeSel.value||25;dirty=true;
  try{localStorage.setItem('sdAuditSize:'+STAGE,String(size))}catch(_){}
  toast(size+' crops on the next page');
});
bandSel.addEventListener('change',function(){
  band=/^\d+$/.test(bandSel.value)?+bandSel.value:bandSel.value;dirty=true;
  try{localStorage.setItem('sdAuditBand:'+STAGE,bandSel.value)}catch(_){}
  toast(band==='all'?'drawing from every band'
    :typeof band==='number'
      ?'drawing from '+BANDS[band][0].toFixed(1)+'–'+BANDS[band][1].toFixed(1)+' only'
      :band==='rejected'?'drawing from '+BELOW
      :'drawing from '+ABOVE);
});
/* Changing the score band or the page size is an instruction about what to
   show NEXT. Paging on regardless would hand back a page drawn under the old
   setting -- and one is usually already cut, because reading the last page
   queues the next one -- so you would pick a band, press next, and be judging
   crops from the bands you just excluded. Back still replays exactly what was
   drawn: those pages are the record of what you judged. */
function goNext(){
  if(view!=='sheet'){if(idx+1<total)load(idx+1);return}
  if(dirty||idx+1>=total)draw();else load(idx+1);
}
function goPrev(){if(idx>0)load(idx-1)}
['next','next2'].forEach(function(id){
  document.getElementById(id).addEventListener('click',goNext)});
['prev','prev2'].forEach(function(id){
  document.getElementById(id).addEventListener('click',goPrev)});
document.getElementById('fresh').addEventListener('click',draw);
document.getElementById('views').addEventListener('click',function(e){
  var b=e.target.closest&&e.target.closest('.viewbtn');
  if(!b)return;
  view=b.getAttribute('data-view');
  var all=document.querySelectorAll('.viewbtn');
  for(var i=0;i<all.length;i++)
    all[i].classList.toggle('on',all[i]===b);
  /* drawing new crops is only a thing the sheet does */
  document.getElementById('fresh').hidden=view!=='sheet';
  idx=0;load(0);
});
/* lightbox */
function zoom(i){
  var it=page&&page.items[i]; if(!it)return;
  cur=i;lbimg.src=cropSrc(it);
  lbtxt.textContent=it.image_id+' · box '+it.det_idx+' · scored '+
    it.p_dog.toFixed(3)+' · '+it.drive;
  lb.hidden=false;
}
/* ── redrawing a box ──
   Everything here is in VIEW pixels -- the coordinates of the picture on
   screen. The server sends the offset and scale that turn those back into the
   store's original pixels, and does that conversion itself, because the
   client has no business deciding where in an 8000px frame a box lands. */
var EDIT={on:false,meta:null,box:null,drag:null,url:null};
var boxwrap=document.getElementById('boxwrap'),ebox=document.getElementById('ebox'),
    mbox=document.getElementById('mbox'),lbstage=document.getElementById('lbstage');
function edParts(){return [document.getElementById('lbedit'),
  document.getElementById('lbsave'),document.getElementById('lbcancel'),
  document.getElementById('lbnote')]}
function edShow(on){
  var p=edParts();
  p[0].hidden=on;p[1].hidden=!on;p[2].hidden=!on;p[3].hidden=!on;
  boxwrap.hidden=!on;EDIT.on=on;
}
/* View pixels are not screen pixels. The picture is served at up to 1100px
   and then CSS fits it to the window -- max-width:92vw, max-height:80vh -- so
   on any screen where that shrinks it, a box drawn at its view coordinates
   lands somewhere else entirely, further down and to the right by exactly the
   ratio. Everything is stored in view pixels and drawn through this. */
function edK(){
  var m=EDIT.meta;
  if(!m||!m.view_w||!lbimg.clientWidth)return 1;
  return lbimg.clientWidth/m.view_w;
}
function edPaint(){
  if(!EDIT.box||!EDIT.meta)return;
  var k=edK(),b=EDIT.box,m=EDIT.meta.model_box;
  function put(el,r){
    el.style.left=(r[0]*k)+'px';el.style.top=(r[1]*k)+'px';
    el.style.width=Math.max(2,(r[2]-r[0])*k)+'px';
    el.style.height=Math.max(2,(r[3]-r[1])*k)+'px';
  }
  put(ebox,b);put(mbox,m);
}
function edStart(){
  var it=page&&page.items[cur]; if(!it)return;
  /* One request. The geometry comes back in a header on the picture itself,
     because both are products of the same decode and asking twice meant
     opening an 8000x4000 frame twice for one click. */
  fetch('/audit/frame/'+STAGE+'/'+it.key.replace('#','_')+'.jpg')
    .then(function(r){
      if(!r.ok)throw 0;
      var m=JSON.parse(r.headers.get('X-Audit-Meta')||'null');
      return r.blob().then(function(b){return [m,URL.createObjectURL(b)]});
    })
    .then(function(pair){
      var m=pair[0];
      if(!m||!m.box){toast('cannot open that frame');return}
      EDIT.meta=m;EDIT.box=m.box.slice();
      if(EDIT.url)URL.revokeObjectURL(EDIT.url);
      EDIT.url=pair[1];
      lbimg.onload=function(){edShow(true);edPaint()};
      lbimg.src=EDIT.url;
    }).catch(function(){toast('cannot open that frame')});
}
function edStop(){
  edShow(false);
  if(EDIT.url){URL.revokeObjectURL(EDIT.url);EDIT.url=null}
  var it=page&&page.items[cur];
  if(it)lbimg.src=cropSrc(it);
}
function edSave(){
  var it=page&&page.items[cur],m=EDIT.meta,b=EDIT.box;
  if(!it||!m||!b)return;
  fetch('/api/audit/box?stage='+STAGE,{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:it.key,box:[
      m.off_x+b[0]/m.scale, m.off_y+b[1]/m.scale,
      m.off_x+b[2]/m.scale, m.off_y+b[3]/m.scale]})})
    .then(function(r){return r.json()})
    .then(function(j){
      if(j&&j.ok){
        toast('box saved for training');
        it.corrected=true;paintCard(cur);
        edStop();
      }
      else toast((j&&j.msg)||'not saved');
    }).catch(function(){toast('not saved')});
}
document.getElementById('lbedit').addEventListener('click',edStart);
document.getElementById('lbsave').addEventListener('click',edSave);
document.getElementById('lbcancel').addEventListener('click',edStop);
ebox.addEventListener('mousedown',function(e){
  var h=e.target.getAttribute&&e.target.getAttribute('data-h');
  EDIT.drag={h:h||'move',x:e.clientX,y:e.clientY,box:EDIT.box.slice()};
  e.preventDefault();e.stopPropagation();
});
document.addEventListener('mousemove',function(e){
  var d=EDIT.drag; if(!d||!EDIT.on)return;
  var k=edK()||1,
      dx=(e.clientX-d.x)/k, dy=(e.clientY-d.y)/k, b=d.box.slice(),
      W=EDIT.meta.view_w, H=EDIT.meta.view_h;
  if(d.h==='move'){b[0]+=dx;b[1]+=dy;b[2]+=dx;b[3]+=dy}
  else{
    if(d.h[0]==='n')b[1]+=dy; else b[3]+=dy;
    if(d.h[1]==='w')b[0]+=dx; else b[2]+=dx;
  }
  /* keep it a box, and keep it inside the picture */
  b[0]=Math.max(0,Math.min(b[0],W-4));b[1]=Math.max(0,Math.min(b[1],H-4));
  b[2]=Math.min(W,Math.max(b[2],b[0]+4));b[3]=Math.min(H,Math.max(b[3],b[1]+4));
  EDIT.box=b;edPaint();
});
document.addEventListener('mouseup',function(){EDIT.drag=null});
window.addEventListener('resize',function(){if(EDIT.on)edPaint()});
document.getElementById('lbclose').addEventListener('click',function(){
  if(EDIT.on)edStop();lb.hidden=true});
lb.addEventListener('click',function(e){
  if(e.target===lb){if(EDIT.on)edStop();lb.hidden=true}});
document.getElementById('lbcopy').addEventListener('click',function(){
  var it=page&&page.items[cur]; if(!it)return;
  var t=it.image_id;
  /* navigator.clipboard is absent on a plain-http origin, which this is */
  if(window.isSecureContext&&navigator.clipboard){
    navigator.clipboard.writeText(t).then(function(){toast('copied '+t)},fallback);
  }else fallback();
  function fallback(){
    var ta=document.createElement('textarea');ta.value=t;
    ta.style.position='fixed';ta.style.top='-1000px';document.body.appendChild(ta);
    ta.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}
    document.body.removeChild(ta);toast(ok?'copied '+t:'copy failed');
  }
});
/* keys */
document.addEventListener('keydown',function(e){
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  /* 1/2/3 are verdicts and also the way a keyboard user picks an option in a
     focused <select>. Judging a crop because someone was choosing a page size
     is a wrong answer recorded by the interface itself. */
  var t=e.target&&e.target.tagName;
  if(t==='SELECT'||t==='INPUT'||t==='TEXTAREA')return;
  if(!lb.hidden){
    if(e.key==='Escape'){if(EDIT.on)edStop();else lb.hidden=true;
      e.preventDefault()}
    else if(e.key==='e'||e.key==='E'){edStart();e.preventDefault()}
    else if(EDIT.on&&e.key==='Enter'){edSave();e.preventDefault()}
    return}
  if(e.key==='1'||e.key==='f'||e.key==='F'){judge(cur,POS);e.preventDefault()}
  else if(e.key==='2'){judge(cur,NEG);e.preventDefault()}
  else if(e.key==='3'){judge(cur,'unsure');e.preventDefault()}
  else if(e.key==='u'||e.key==='U'){undoLast();e.preventDefault()}
  else if(e.key==='ArrowRight'){move(1);e.preventDefault()}
  else if(e.key==='ArrowLeft'){move(-1);e.preventDefault()}
  else if(e.key==='Enter'){if(cur>=0)zoom(cur);e.preventDefault()}
  else if(e.key==='n'||e.key==='N'){
    if(idx+1<total)load(idx+1);else draw();e.preventDefault()}
});
/* boot: the last page if there is one, otherwise an invitation */
fetch('/api/audit/page?stage='+STAGE+'&i=-1&n='+size+
      (band==null?'':'&band='+encodeURIComponent(band))).then(function(r){return r.json()})
  .then(function(j){
    if(j&&j.page)show(j.page,j.index,j.total);
    else{total=0;setPos();render()}
    loadStats();
  }).catch(function(){loadStats()});
</script></body></html>
"""
_TEMPLATE = AUDIT_HTML


def page_html(stage=DEFAULT_STAGE):
    """The page, with this stage's words in it.

    One template, because the two audits differ in vocabulary and in nothing
    else -- same grid, same keys, same undo. A second copy would be a second
    place to fix every bug found in the first.
    """
    sp = fa.spec(stage)
    tabs = ''.join(
        f'<a class="tab{" on" if k == stage else ""}" '
        f'href="/audit{"" if k == fa.DEFAULT_STAGE else "/" + k}">'
        f'{v["title"]}</a>' for k, v in fa.STAGES.items())
    if sp['asymmetric']:
        sub = (f'{sp["asks"]} Below a score of 0.5 the model said no, so a '
               f'yes there is a {sp["positive"]} it threw away and nothing '
               f'downstream will ever see. Above 0.5 it said yes already.')
        h1 = sp['miss'].capitalize()
    else:
        sub = (f'{sp["asks"]} Both of this model\'s mistakes cost the same, '
               f'which is why it was promoted on balanced accuracy \u2014 so '
               f'read the two sides of 0.5 together, not one as the error.')
        h1 = f'Where the {sp["title"]} is wrong'
    out = _TEMPLATE
    for k, v in (('__BANDS__', json.dumps(fa.BANDS)),
                 ('__STAGE__', json.dumps(stage)),
                 ('__POS__', json.dumps(sp['positive'])),
                 ('__NEG__', json.dumps(sp['negative'])),
                 ('__YES__', json.dumps(sp['yes'])),
                 ('__NO__', json.dumps(sp['no'])),
                 ('__BELOW__', json.dumps(sp['below'])),
                 ('__ABOVE__', json.dumps(sp['above'])),
                 # asymmetric models get walked from the side where the
                 # unrecoverable error lives; symmetric ones from both
                 ('__DEFBAND__',
                  json.dumps('rejected' if sp['asymmetric'] else 'all')),
                 ('__THRESH__', json.dumps(fa.THRESHOLD)),
                 ('__YESTXT__', sp['yes']), ('__NOTXT__', sp['no']),
                 ('__BELOWTXT__', sp['below']),
                 ('__ABOVETXT__', sp['above']),
                 ('__MISSLAB__', sp['miss']),
                 ('__H1__', h1), ('__SUB__', sub), ('__TABS__', tabs)):
        out = out.replace(k, v)
    return out


# ── prefetch ────────────────────────────────────────────────────────────────
# Cutting a page means opening two dozen 8000x4000 frames off six drives.
# Warm that is under a second; cold it is twenty, and twenty seconds after
# pressing Next is long enough to wonder whether the button worked. So the
# page after the one being read is drawn in the background while it is read.
_PREFETCH = {}


def prefetch(band=None, n=25, stage=DEFAULT_STAGE):
    """Draw the next page in the background, unless one is already coming."""
    t = _PREFETCH.get(stage)
    if t is not None and t.is_alive():
        return
    def go():
        try:
            draw_page(n=n, band=band, stage=stage)
        except Exception:
            pass
    t = threading.Thread(target=go, daemon=True)
    _PREFETCH[stage] = t
    t.start()


def with_verdicts(doc, stage=DEFAULT_STAGE):
    """Stamp each item with the answer already on record for it.

    A page document is the draw, not the judging, so paging back to one
    re-read it as untouched and every verdict on it looked lost. The ledger is
    the record; the page is just which boxes were on screen.
    """
    if not doc or not doc.get('items'):
        return doc
    seen = {v['key']: v.get('verdict') for v in fa.read_verdicts(stage=stage)}
    fixed = corrections()
    for it in doc['items']:
        v = seen.get(it['key'])
        if v:
            it['verdict'] = v
        iid, _, di = str(it['key']).partition('#')
        if (iid, int(di or 0)) in fixed:
            it['corrected'] = True
    return doc


# The sizes the page offers. Anything else is somebody's URL, not a choice
# the interface made, and a page of 40,000 crops is a way to take the server
# down by typing. Each divides by the band count, so every band gets the same
# quota and the strata stay even.
PAGE_SIZES = (25, 50, 75, 100)


BAND_GROUPS = ('rejected', 'kept', 'all')


def band_arg(v):
    """A band selection off the wire: a group name, an index, or nothing."""
    if v in BAND_GROUPS:
        return v
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if 0 <= i < len(fa.BANDS) else None


def page_size(v, default=25):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return v if v in PAGE_SIZES else default


# ── what you have already answered ──────────────────────────────────────────
# The sheet only ever shows boxes nobody has seen; that is the whole point of
# it. But an answer given at speed is an answer worth being able to look at
# again -- and the ledger already supports changing your mind, so the only
# thing missing was a way to find the crop.
JUDGED_VIEWS = ('flagged', 'wrong', 'all')


def judged(stage=DEFAULT_STAGE, which='flagged', page=0, n=25):
    """Crops already answered, newest first.

    `flagged` is the positive class -- the dogs found, the leashes seen --
    because that is what someone means by "what I flagged". `wrong` is every
    answer that disagrees with the model, which is the same set plus the
    false positives above the threshold. `all` is everything, unsure included.
    """
    sp = fa.spec(stage)
    rows = [v for v in fa.read_verdicts(stage=stage)
            if fa.verdict_of(v.get('verdict'), stage)]
    for v in rows:
        v['verdict'] = fa.verdict_of(v['verdict'], stage)
    # p_dog is what decides the tag, the score chip and the threshold rule.
    # Early rows predate it being sent, so the missing ones are looked up in
    # one query rather than one per crop.
    need = [v['key'] for v in rows if v.get('p_dog') is None]
    if need:
        got = {}
        try:
            import duckdb
            keys = [tuple(k.split('#')) for k in need]
            con = duckdb.connect()
            con.execute('CREATE TEMP TABLE want(i VARCHAR, d INTEGER)')
            con.executemany('INSERT INTO want VALUES (?, ?)',
                            [(a, int(b or 0)) for a, b in keys])
            for iid, di, p, band in con.execute(
                    f"""SELECT p.image_id, p.det_idx, p.p_dog, p.band
                        FROM read_parquet('{P(stage)['pool']}') p
                        JOIN want w ON w.i = p.image_id AND w.d = p.det_idx"""
            ).fetchall():
                got[f'{iid}#{di}'] = (float(p), int(band))
            con.close()
        except Exception:
            got = {}
        for v in rows:
            if v['key'] in got:
                v['p_dog'], v['band'] = got[v['key']]
    if which == 'flagged':
        rows = [v for v in rows if v['verdict'] == sp['positive']]
    elif which == 'wrong':
        rows = [v for v in rows
                if v.get('p_dog') is not None
                and ((float(v['p_dog']) >= fa.THRESHOLD)
                     != (v['verdict'] == sp['positive']))]
    rows.sort(key=lambda v: -(v.get('ts') or 0))
    total = len(rows)
    pages = max(1, -(-total // max(1, n)))
    page = max(0, min(pages - 1, int(page)))
    fixed = corrections()
    out = []
    for v in rows[page * n:(page + 1) * n]:
        iid, _, di = str(v['key']).partition('#')
        it = {'key': v['key'], 'image_id': iid,
              'det_idx': int(di or 0), 'verdict': v['verdict'],
              'p_dog': float(v['p_dog']) if v.get('p_dog') is not None else 0.0,
              'band': v.get('band'), 'seq': v.get('seq'),
              'judged_at': v.get('ts'),
              'unknown_score': v.get('p_dog') is None}
        if (iid, int(di or 0)) in fixed:
            it['corrected'] = True
        out.append(it)
    return {'items': out, 'total': total, 'page': page, 'pages': pages,
            'which': which, 'stage': stage,
            'counts': _judged_counts(stage)}


def _judged_counts(stage=DEFAULT_STAGE):
    """How many sit behind each view, so the switch can say so."""
    sp = fa.spec(stage)
    rows = [fa.verdict_of(v.get('verdict'), stage)
            for v in fa.read_verdicts(stage=stage)]
    rows = [v for v in rows if v]
    return {'flagged': sum(1 for v in rows if v == sp['positive']),
            'all': len(rows)}


def api_page(i, n=25, band=None, stage=DEFAULT_STAGE):
    """One page by index; -1 means the most recent. Draws if there are none."""
    total = page_count(stage)
    if total == 0:
        doc = draw_page(n=n, band=band, stage=stage)
        return {'page': with_verdicts(doc, stage), 'index': doc.get('index', 0),
                'total': max(1, page_count(stage))}
    if i is None or int(i) < 0:
        i = total - 1
    i = max(0, min(total - 1, int(i)))
    doc = get_page(i, stage)
    if doc is None:
        return {'error': f'page {i} is missing'}
    if i >= total - 1:
        prefetch(band=band, n=n, stage=stage)   # line up the next one
    return {'page': with_verdicts(doc, stage), 'index': i, 'total': total}


def api_draw(n=25, band=None, stage=DEFAULT_STAGE):
    doc = draw_page(n=n, band=band, stage=stage)
    total = max(1, page_count(stage))
    if doc.get('items'):
        prefetch(band=band, n=n, stage=stage)
    return {'page': with_verdicts(doc, stage),
            'index': doc.get('index', total - 1), 'total': total}
