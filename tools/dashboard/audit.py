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

OUT_DIR = fa.OUT_DIR
CROPS = os.path.join(OUT_DIR, 'crops')
# The full-resolution cut, kept beside the thumbnail. harvest_flagged.py
# learned this the hard way: the ~160px thumbnail the review UI copied was
# "too small and too lossy for training", so it re-opens the original jpg to
# re-cut every flagged box. The audit already has the frame open to make the
# thumbnail, so it writes both from that one decode and a verdict costs no
# second pass over an 8000x4000 panorama.
FULL = os.path.join(OUT_DIR, 'full')
DATASET = os.path.join(REPO, 'data', 'audit_finds')
PAGES = os.path.join(OUT_DIR, 'pages')
DRAWN = os.path.join(OUT_DIR, 'drawn.jsonl')
# Two locks, because they guard two different things and one of them is slow.
# A draw holds its lock across the sample and the page write; cutting the
# crops -- seconds, or twenty of them off cold drives -- happens between them
# with nothing held. Sharing one lock meant every verdict recorded while the
# next page was being cut waited for the cutting to finish, which is exactly
# when someone is judging: the page they are on loads, the next one starts
# cutting behind it, and the first keystroke hangs.
_DRAW_LOCK = threading.Lock()
_LEDGER_LOCK = threading.Lock()

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


def pool_ready():
    return os.path.exists(fa.POOL)


def _drawn_keys():
    """Every box ever put in front of anyone, judged or not.

    Two sources, because either alone can be incomplete: the draw log is what
    was shown, and the verdict ledger is what was answered. A box can be
    answered without the draw log surviving -- the pool was rebuilt once and
    the log went with it -- and an answered box is seen by definition, so it
    must never come back round.
    """
    keys, seqs = set(), set()
    for path, get in ((DRAWN, None), (fa.VERDICTS, None)):
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


def sample(n=25, band=None, seed=None):
    """A page of candidates: stratified by band, one box per sequence, none
    ever drawn before.

    One box per sequence is the load-bearing part. Mapillary frames come a
    second apart down one road, so a sequence's boxes are the same handful of
    objects photographed repeatedly -- scoring twenty of them would state a
    confidence twenty independent samples would earn and these do not.
    """
    import duckdb
    keys, seqs = _drawn_keys()
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("CREATE TEMP TABLE seen_seq(seq VARCHAR)")
    if seqs:
        con.executemany("INSERT INTO seen_seq VALUES (?)",
                        [(s,) for s in seqs])
    bands = [band] if band is not None else list(range(len(fa.BANDS)))
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
            SELECT p.* FROM read_parquet('{fa.POOL}') p
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


def _cut_one(cand, roots):
    """Cut one crop to disk. Returns True if the file is there afterwards."""
    from PIL import Image
    dst = os.path.join(CROPS, cand['key'].replace('#', '_') + '.jpg')
    if os.path.exists(dst):
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
        os.makedirs(FULL, exist_ok=True)
        fdst = os.path.join(FULL, cand['key'].replace('#', '_') + '.jpg')
        ftmp = fdst + '.tmp'
        crop.save(ftmp, 'JPEG', quality=95)
        os.replace(ftmp, fdst)
        crop.thumbnail((CROP_PX, CROP_PX), Image.LANCZOS)
        os.makedirs(CROPS, exist_ok=True)
        tmp = dst + '.tmp'
        crop.save(tmp, 'JPEG', quality=88)
        os.replace(tmp, dst)
        return True
    except Exception:
        return False
    finally:
        im.close()


def materialise(cands, workers=8):
    """Cut every crop on a page, in parallel.

    Decoding one 8000x4000 frame is ~116 ms and each candidate is a different
    frame by construction, so a page of 24 is 2.8 seconds serial and a third
    of a second spread over the decoders. Pillow drops the GIL inside decode,
    so threads are enough and there is no process pool to pay for.
    """
    from concurrent.futures import ThreadPoolExecutor
    roots = _roots()
    os.makedirs(CROPS, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = list(ex.map(lambda c: _cut_one(c, roots), cands))
    return [c for c, good in zip(cands, ok) if good]


def crop_path(key):
    """Absolute path of a cut crop, or None. The key is checked against the
    shape it is generated in, so nothing a client sends reaches a path."""
    import re
    if not re.fullmatch(r'[0-9]{1,32}_[0-9]{1,6}', str(key or '')):
        return None
    p = os.path.join(CROPS, f'{key}.jpg')
    return p if os.path.exists(p) else None


def _page_file(i):
    return os.path.join(PAGES, f'{int(i):05d}.json')


def page_count():
    try:
        return len(glob.glob(os.path.join(PAGES, '*.json')))
    except OSError:
        return 0


def get_page(i):
    try:
        with open(_page_file(i)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def draw_page(n=25, band=None):
    """Draw, cut, and keep. Returns the page document."""
    with _DRAW_LOCK:
        cands = sample(n=n, band=band)
        if not cands:
            return {'index': page_count(), 'items': [], 'exhausted': True,
                    'band': band, 'n': n, 'dropped': 0}
        # Reserved before a single frame is opened. A box counts as drawn the
        # moment it is chosen, not when it is judged or even when it is
        # successfully cut: a concurrent draw must not pick it, and a box
        # skipped on screen must not come back three pages later, or "you have
        # not seen this" is untrue and the sample quietly correlates.
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(DRAWN, 'a') as fh:
            for c in cands:
                fh.write(json.dumps({'key': c['key'], 'seq': c['seq']}) + '\n')

    got = materialise(cands)          # slow, and holds nothing

    with _DRAW_LOCK:
        idx = page_count()
        # Frames that would not open. Usually a jpg pruned off a drive after
        # the sweep read it. Counted rather than hidden, so a short page reads
        # as a short page and not as a page that lost its crops.
        doc = {'index': idx, 'band': band, 'n': n, 'created': time.time(),
               'dropped': len(cands) - len(got),
               'items': [{k: c[k] for k in
                          ('key', 'image_id', 'det_idx', 'p_dog', 'conf',
                           'band', 'seq', 'drive', 'cell')} for c in got]}
        os.makedirs(PAGES, exist_ok=True)
        tmp = _page_file(idx) + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(doc, fh)
        os.replace(tmp, _page_file(idx))
        return doc


VERDICTS = fa.ANSWERS           # 'dog' | 'not_dog' | 'unsure'
CLASS_OF = fa.CLASS_OF


def place(key, verdict):
    """Put one judged crop into the dataset, or take it out.

    Hard-linked, not copied: the full-resolution cut already exists and a
    second copy of it would drift from the first the moment either is
    re-cut. 'unsure' is not a class -- it is removed from both, so changing
    your mind to "I cannot tell" does not leave a stale label behind.
    """
    name = str(key).replace('#', '_') + '.jpg'
    src = os.path.join(FULL, name)
    want = CLASS_OF.get(fa.verdict_of(verdict))
    for cls in ('dog', 'not_dog'):
        dst = os.path.join(DATASET, cls, name)
        if cls == want:
            continue
        try:
            os.remove(dst)             # a changed mind moves the file
        except OSError:
            pass
    if not want or not os.path.exists(src):
        return False
    dst = os.path.join(DATASET, want, name)
    if os.path.exists(dst):
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:                    # different filesystem, or no hardlinks
        import shutil
        shutil.copy2(src, dst)
    return True


def record(key, verdict, meta=None):
    """Append one human judgement.

    Append-only and re-readable: a mind changed later is another line, and the
    reader keeps the last one. Nothing rewrites history in place, so a crash
    mid-write costs one line rather than the file.
    """
    # `None` clears: undo is a verdict being withdrawn, not a third opinion,
    # and the ledger is append-only so it is written as one more line.
    if verdict is not None and fa.verdict_of(verdict) is None:
        return {'ok': False, 'msg': f'unknown verdict {verdict!r}'}
    verdict = fa.verdict_of(verdict) if verdict is not None else None
    rec = {'key': str(key), 'verdict': verdict, 'ts': time.time()}
    for k in ('band', 'p_dog', 'seq'):
        if meta and meta.get(k) is not None:
            rec[k] = meta[k]
    with _LEDGER_LOCK:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(fa.VERDICTS, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        # the ledger is the record; the dataset is a view of it, kept in step
        # as each verdict lands so it is never a rebuild away from usable
        placed = place(key, verdict)
    return {'ok': True, 'placed': placed}


def stats():
    s = fa.summarise()
    s['pages'] = page_count()
    s['drawn'] = len(_drawn_keys()[0])
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
<title>What the gate threw away</title><style>
:root{--bg:#13151a;--panel:#1b2027;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;
--red:#ef5350;--gap:20px}
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
.back{margin-left:auto;font-size:12px;color:var(--mut);text-decoration:none;
  border:1px solid var(--bd);border-radius:8px;padding:6px 11px}
.back:hover{color:var(--tx);border-color:rgba(130,140,150,.3)}
/* ── the measurement ── */
.meas{display:flex;gap:26px;align-items:flex-end;flex-wrap:wrap;
  background:var(--panel);border:1px solid var(--bd);border-radius:14px;
  padding:16px 20px;margin-bottom:14px}
.mbig{font-size:34px;font-weight:680;letter-spacing:-1.2px;line-height:1.1;
  font-variant-numeric:tabular-nums}
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
  font-variant-numeric:tabular-nums}
.spacer{margin-left:auto}
.pick{display:inline-flex;align-items:center;gap:7px;font-size:11.5px;
  color:var(--dim)}
.pick select{background:var(--panel2);border:1px solid var(--bd);
  color:var(--mut);border-radius:9px;padding:7px 9px;font-size:12.5px;
  font-family:inherit;cursor:pointer}
.pick select:hover{color:var(--tx)}
/* ── the grid ── */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(212px,1fr));
  gap:14px}
.card{background:var(--panel);border:1px solid var(--bd);border-radius:13px;
  overflow:hidden;display:flex;flex-direction:column}
.card.done{opacity:.5}
/* a flag is the FINDING -- it stays lit rather than greying out like an
   answered question, because the whole point of the page is the pile of them */
.card.miss{border-color:var(--acc);opacity:1;
  box-shadow:0 0 0 1px rgba(232,166,69,.35)}
.card.ok{border-color:rgba(67,181,129,.4)}
.shot{position:relative;background:#0e1014;aspect-ratio:1;display:flex;
  align-items:center;justify-content:center;cursor:zoom-in}
.shot img{max-width:100%;max-height:100%;display:block}
.pchip{position:absolute;right:6px;bottom:6px;font-size:10.5px;
  background:rgba(10,12,16,.82);border:1px solid var(--bd);border-radius:6px;
  padding:2px 6px;color:var(--mut);font-variant-numeric:tabular-nums}
.acts{display:grid;grid-template-columns:1fr 1fr auto;gap:1px;
  background:var(--bd)}
.act{background:var(--panel2);border:0;color:var(--mut);font-family:inherit;
  font-size:11.5px;padding:9px 4px;cursor:pointer}
.act:hover{color:var(--tx)}
.act.m:hover,.act.m.on{background:rgba(239,83,80,.16);color:var(--red)}
.act.c:hover,.act.c.on{background:rgba(67,181,129,.14);color:var(--green)}
.act.u{padding:9px 10px}
.act.u:hover,.act.u.on{color:var(--acc)}
.empty{color:var(--dim);font-size:13px;padding:40px 0;text-align:center}
/* ── lightbox ── */
.lb{position:fixed;inset:0;background:rgba(0,0,0,.9);display:flex;
  align-items:center;justify-content:center;flex-direction:column;gap:12px;
  z-index:50}
.lb[hidden]{display:none}
.lb img{max-width:92vw;max-height:80vh;object-fit:contain}
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
  <div><h1>Dogs the gate threw away</h1>
    <div class="sub">Flag anything that is a dog. Below a score of 0.5 the
      gate rejected it, so a flag there is a dog that is gone from everything
      downstream. Above 0.5 it kept it, and a flag just says it was right.</div></div>
  <a class="back" href="/">&larr; dashboard</a>
</header>

<div class="meas">
  <div><div class="mlab">dogs it threw away</div>
    <div class="mbig" id="rate">&mdash;</div>
    <div class="mci" id="ci">nothing flagged yet</div></div>
  <div><div class="mlab">you have flagged</div>
    <div class="mbig" id="judged">0</div>
    <div class="mci" id="found">&nbsp;</div></div>
  <div class="mnote" id="note">Bands are drawn from evenly, so this is not a
    proportion of what you have seen &mdash; it weights each band by how many
    boxes the gate really put in it.</div>
</div>

<div class="bands" id="bands"></div>

<div class="bar">
  <button class="btn" id="prev">&larr; back</button>
  <button class="btn go" id="next">next page &rarr;</button>
  <span class="pos" id="pos">&mdash;</span>
  <span class="spacer"></span>
  <label class="pick">crops per page
    <select id="size">
      <option value="25">25</option><option value="50">50</option>
      <option value="75">75</option><option value="100">100</option>
    </select></label>
  <label class="pick">score
    <select id="bandsel"><option value="">every band</option></select></label>
  <button class="btn" id="fresh">&#8635; draw a new page</button>
</div>

<div class="grid" id="grid"></div>
<div class="empty" id="empty" hidden></div>
<div class="keys">
  <kbd>F</kbd> it&rsquo;s a dog &nbsp; <kbd>2</kbd> not a dog &nbsp;
  <kbd>3</kbd> unsure &nbsp; <kbd>U</kbd> undo &nbsp;
  <kbd>&larr;</kbd><kbd>&rarr;</kbd> move &nbsp;
  <kbd>Enter</kbd> enlarge &nbsp; <kbd>N</kbd> next page
</div>
</div>

<div class="lb" id="lb" hidden>
  <img id="lbimg" alt="">
  <div class="lbcap"><span id="lbtxt"></span>
    <button id="lbcopy">copy image id</button>
    <button id="lbclose">close</button></div>
</div>
<div class="toast" id="toast" hidden></div>
<div class="undotoast" id="undotoast" hidden></div>

<script>
var BANDS=__BANDS__;
var grid=document.getElementById('grid'),empty=document.getElementById('empty'),
    posEl=document.getElementById('pos'),lb=document.getElementById('lb'),
    lbimg=document.getElementById('lbimg'),lbtxt=document.getElementById('lbtxt');
var page=null,idx=0,cur=0,total=0,band=null,busy=false,size=25,dirty=false;

function toast(t){var e=document.getElementById('toast');e.textContent=t;
  e.hidden=false;clearTimeout(e._t);e._t=setTimeout(function(){e.hidden=true},1600)}
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML}
function pctTxt(v){return (v*100).toFixed(1)+'%'}
function fmtn(n){return (n||0).toLocaleString('en-US')}

/* The verdict is sent the moment it is given and the card is marked from the
   local answer, not from a reload: a reviewer working through a page at speed
   must never wait on a round trip, and the ledger is append-only so a lost
   response costs one line, not the page. */
var lastUndo=null,toastT=null;
function judge(i,verdict){
  var it=page.items[i]; if(!it)return;
  it.verdict=verdict;
  /* The cursor stays where it was put. It used to step to the next crop after
     every answer, which reads as the page choosing for you -- and on a grid,
     where the eye is already on the crop it means to judge, moving the ring
     somewhere else is just wrong. Arrows move it; nothing else does. */
  if(verdict==='not_dog'){
    /* Not a dog is a DISMISSAL: it is the answer for almost every crop here,
       and leaving three hundred of them on screen greyed out buries the few
       that matter. It leaves the grid, and the toast is the way back. */
    hide(i);
    offerUndo(it,i);
  }else{
    paintCard(i);
  }
  send(it.key,verdict,it);
}
function send(key,verdict,it){
  fetch('/api/audit/verdict',{method:'POST',
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
}
function unhide(i){
  var el=grid.children[i];
  if(el){el.style.display='';el.removeAttribute('data-gone')}
}
function offerUndo(it,i){
  lastUndo={key:it.key,i:i};
  var t=document.getElementById('undotoast');
  t.innerHTML='<img src="/audit/crop/'+esc(it.key.replace("#","_"))+
    '.jpg" alt="">'+
    '<span class="tt"><b>Not a dog</b>'+it.image_id+'</span>'+
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
  grid.innerHTML=page.items.map(function(it,i){
    var lo=BANDS[it.band][0],hi=BANDS[it.band][1];
    return '<div class="card" data-i="'+i+'">'+
      '<div class="shot" data-zoom="'+i+'">'+
        '<img loading="lazy" src="/audit/crop/'+esc(it.key.replace("#","_"))+
        '.jpg" alt="rejected box '+esc(it.key)+'">'+
        '<span class="pchip" title="the gate scored this '+it.p_dog.toFixed(3)+
        ' for dog; band '+lo.toFixed(1)+'-'+hi.toFixed(1)+'">'+
        it.p_dog.toFixed(3)+'</span></div>'+
      '<div class="acts">'+
        '<button class="act m" data-v="dog" data-i="'+i+'">&#9873; it\u2019s a dog</button>'+
        '<button class="act c" data-v="not_dog" data-i="'+i+'">not a dog</button>'+
        '<button class="act u" data-v="unsure" data-i="'+i+'" title="cannot tell">?</button>'+
      '</div></div>';
  }).join('');
  for(var i=0;i<page.items.length;i++)paintCard(i);
}
function move(d){
  if(!page||!page.items.length)return;
  cur=Math.max(0,Math.min(page.items.length-1,cur+d));
  for(var i=0;i<page.items.length;i++)paintCard(i);
  var el=grid.children[cur];
  if(el&&el.scrollIntoView)el.scrollIntoView({block:'nearest'});
}
function setPos(){
  posEl.textContent=total?('page '+(idx+1)+' of '+total+
    (page&&page.dropped?' · '+page.dropped+' unreadable':'')+
    (band!=null?' · band '+BANDS[band][0].toFixed(1)+'-'+BANDS[band][1].toFixed(1):'')):'—';
  document.getElementById('prev').disabled=busy||idx<=0;
  document.getElementById('next').disabled=busy;
}
function show(doc,at,tot){
  page=doc;idx=at;total=tot;cur=0;render();setPos();
}
function load(at){
  busy=true;setPos();
  fetch('/api/audit/page?i='+at+'&n='+size+
        (band==null?'':'&band='+band)).then(function(r){return r.json()})
    .then(function(j){busy=false;if(!j||j.error){toast(j&&j.error||'failed');setPos();return}
      show(j.page,j.index,j.total)})
    .catch(function(){busy=false;toast('failed');setPos()});
}
function draw(){
  busy=true;setPos();
  document.getElementById('fresh').textContent='cutting…';
  fetch('/api/audit/draw',{method:'POST',
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
  fetch('/api/audit/stats').then(function(r){return r.json()}).then(paintStats)
    .catch(function(){});
}
function paintStats(s){
  if(!s)return;
  var r=document.getElementById('rate'),ci=document.getElementById('ci'),
      rej=s.rejected||{},kept=s.kept||{};
  document.getElementById('judged').textContent=fmtn(s.judged||0);
  document.getElementById('found').textContent=
    (rej.wrong||0)+' flagged below 0.5';
  /* The headline is the one number the page exists to produce: how many dogs
     the gate threw away. It used to read "missed dogs 100.0% -- 3,945,390
     dogs across 3,945,390 rejected boxes", which is what a rate of 1.0 off
     five crops extrapolates to, stated as though it were known. An estimate
     from a handful of crops is not a count of four million things, so the
     extrapolation is only shown once there is enough behind it to mean
     anything, and it is always written as a range. */
  if(!rej.judged){r.textContent='—';
    ci.textContent='flag a dog below 0.5 and this starts counting'}
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
  o.value=bi;o.textContent=BANDS[bi][0].toFixed(1)+' – '+BANDS[bi][1].toFixed(1);
  bandSel.appendChild(o);
}
/* Both choices are remembered. Working through an audit is a long sitting and
   re-picking the page size after every reload is a small tax on the only
   thing this page is for. */
try{
  var sv=localStorage.getItem('sdAuditSize');
  if(sv&&/^(25|50|75|100)$/.test(sv))sizeSel.value=sv;
  var bv=localStorage.getItem('sdAuditBand');
  if(bv!==null&&bv!==''&&+bv>=0&&+bv<BANDS.length){band=+bv;bandSel.value=bv}
}catch(_){}
size=+sizeSel.value||25;
sizeSel.addEventListener('change',function(){
  size=+sizeSel.value||25;dirty=true;
  try{localStorage.setItem('sdAuditSize',String(size))}catch(_){}
  toast(size+' crops on the next page');
});
bandSel.addEventListener('change',function(){
  band=bandSel.value===''?null:+bandSel.value;dirty=true;
  try{localStorage.setItem('sdAuditBand',bandSel.value)}catch(_){}
  toast(band==null?'drawing from every band'
    :'drawing from '+BANDS[band][0].toFixed(1)+'–'+BANDS[band][1].toFixed(1)+' only');
});
/* Changing the score band or the page size is an instruction about what to
   show NEXT. Paging on regardless would hand back a page drawn under the old
   setting -- and one is usually already cut, because reading the last page
   queues the next one -- so you would pick a band, press next, and be judging
   crops from the bands you just excluded. Back still replays exactly what was
   drawn: those pages are the record of what you judged. */
document.getElementById('next').addEventListener('click',function(){
  if(dirty||idx+1>=total)draw();else load(idx+1);
});
document.getElementById('prev').addEventListener('click',function(){
  if(idx>0)load(idx-1)});
document.getElementById('fresh').addEventListener('click',draw);
/* lightbox */
function zoom(i){
  var it=page&&page.items[i]; if(!it)return;
  cur=i;lbimg.src='/audit/crop/'+it.key.replace('#','_')+'.jpg';
  lbtxt.textContent=it.image_id+' · box '+it.det_idx+' · scored '+
    it.p_dog.toFixed(3)+' · '+it.drive;
  lb.hidden=false;
}
document.getElementById('lbclose').addEventListener('click',function(){lb.hidden=true});
lb.addEventListener('click',function(e){if(e.target===lb)lb.hidden=true});
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
  if(!lb.hidden){if(e.key==='Escape'){lb.hidden=true;e.preventDefault()}return}
  if(e.key==='1'||e.key==='f'||e.key==='F'){judge(cur,'dog');e.preventDefault()}
  else if(e.key==='2'){judge(cur,'not_dog');e.preventDefault()}
  else if(e.key==='3'){judge(cur,'unsure');e.preventDefault()}
  else if(e.key==='u'||e.key==='U'){undoLast();e.preventDefault()}
  else if(e.key==='ArrowRight'){move(1);e.preventDefault()}
  else if(e.key==='ArrowLeft'){move(-1);e.preventDefault()}
  else if(e.key==='Enter'){zoom(cur);e.preventDefault()}
  else if(e.key==='n'||e.key==='N'){
    if(idx+1<total)load(idx+1);else draw();e.preventDefault()}
});
/* boot: the last page if there is one, otherwise an invitation */
fetch('/api/audit/page?i=-1&n='+size+
      (band==null?'':'&band='+band)).then(function(r){return r.json()})
  .then(function(j){
    if(j&&j.page)show(j.page,j.index,j.total);
    else{total=0;setPos();render()}
    loadStats();
  }).catch(function(){loadStats()});
</script></body></html>
"""
AUDIT_HTML = AUDIT_HTML.replace('__BANDS__', json.dumps(fa.BANDS))


# ── prefetch ────────────────────────────────────────────────────────────────
# Cutting a page means opening two dozen 8000x4000 frames off six drives.
# Warm that is under a second; cold it is twenty, and twenty seconds after
# pressing Next is long enough to wonder whether the button worked. So the
# page after the one being read is drawn in the background while it is read.
_PREFETCH = {'thread': None}


def prefetch(band=None, n=25):
    """Draw the next page in the background, unless one is already coming."""
    t = _PREFETCH.get('thread')
    if t is not None and t.is_alive():
        return
    def go():
        try:
            draw_page(n=n, band=band)
        except Exception:
            pass
    t = threading.Thread(target=go, daemon=True)
    _PREFETCH['thread'] = t
    t.start()


def with_verdicts(doc):
    """Stamp each item with the answer already on record for it.

    A page document is the draw, not the judging, so paging back to one
    re-read it as untouched and every verdict on it looked lost. The ledger is
    the record; the page is just which boxes were on screen.
    """
    if not doc or not doc.get('items'):
        return doc
    seen = {v['key']: v.get('verdict') for v in fa.read_verdicts()}
    for it in doc['items']:
        v = seen.get(it['key'])
        if v:
            it['verdict'] = v
    return doc


# The sizes the page offers. Anything else is somebody's URL, not a choice
# the interface made, and a page of 40,000 crops is a way to take the server
# down by typing. Each divides by the band count, so every band gets the same
# quota and the strata stay even.
PAGE_SIZES = (25, 50, 75, 100)


def page_size(v, default=25):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return v if v in PAGE_SIZES else default


def api_page(i, n=25, band=None):
    """One page by index; -1 means the most recent. Draws if there are none."""
    total = page_count()
    if total == 0:
        doc = draw_page(n=n, band=band)
        return {'page': with_verdicts(doc), 'index': doc.get('index', 0),
                'total': max(1, page_count())}
    if i is None or int(i) < 0:
        i = total - 1
    i = max(0, min(total - 1, int(i)))
    doc = get_page(i)
    if doc is None:
        return {'error': f'page {i} is missing'}
    if i >= total - 1:
        prefetch(band=band, n=n)      # reading the last one: line up another
    return {'page': with_verdicts(doc), 'index': i, 'total': total}


def api_draw(n=25, band=None):
    doc = draw_page(n=n, band=band)
    total = max(1, page_count())
    if doc.get('items'):
        prefetch(band=band, n=n)
    return {'page': with_verdicts(doc), 'index': doc.get('index', total - 1),
            'total': total}
