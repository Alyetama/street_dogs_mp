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
    """Every box ever put in front of anyone, judged or not."""
    keys, seqs = set(), set()
    try:
        with open(DRAWN) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get('key'):
                    keys.add(d['key'])
                if d.get('seq'):
                    seqs.add(d['seq'])
    except OSError:
        pass
    return keys, seqs


def sample(n=24, band=None, seed=None):
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


def draw_page(n=24, band=None):
    """Draw, cut, and keep. Returns the page document."""
    with _DRAW_LOCK:
        cands = sample(n=n, band=band)
        if not cands:
            return {'index': page_count(), 'items': [], 'exhausted': True,
                    'band': band, 'dropped': 0}
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
        doc = {'index': idx, 'band': band, 'created': time.time(),
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


VERDICTS = ('missed', 'correct', 'unsure')


def record(key, verdict, meta=None):
    """Append one human judgement.

    Append-only and re-readable: a mind changed later is another line, and the
    reader keeps the last one. Nothing rewrites history in place, so a crash
    mid-write costs one line rather than the file.
    """
    if verdict not in VERDICTS:
        return {'ok': False, 'msg': f'unknown verdict {verdict!r}'}
    rec = {'key': str(key), 'verdict': verdict, 'ts': time.time()}
    for k in ('band', 'p_dog', 'seq'):
        if meta and meta.get(k) is not None:
            rec[k] = meta[k]
    with _LEDGER_LOCK:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(fa.VERDICTS, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    return {'ok': True}


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
/* ── per band ── */
.bands{display:grid;gap:7px;margin-bottom:16px}
.brow{display:grid;grid-template-columns:104px 1fr 128px;gap:12px;
  align-items:center;font-size:11.5px;color:var(--mut);
  font-variant-numeric:tabular-nums}
.btrack{height:7px;border-radius:4px;background:rgba(130,140,150,.12);
  position:relative;overflow:hidden}
.bfill{height:100%;border-radius:4px;background:var(--acc)}
.bnil{color:var(--dim)}
.bsel{cursor:pointer;background:none;border:0;color:inherit;font:inherit;
  text-align:left;padding:0}
.bsel:hover{color:var(--tx)}
.brow.on{color:var(--tx)}
.brow.on .bnil{color:var(--mut)}
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
/* ── the grid ── */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(212px,1fr));
  gap:14px}
.card{background:var(--panel);border:1px solid var(--bd);border-radius:13px;
  overflow:hidden;display:flex;flex-direction:column}
.card.done{opacity:.42}
.card.miss{border-color:rgba(239,83,80,.55)}
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
  <div><h1>What the gate threw away</h1>
    <div class="sub">dogbin_008 rejected 3.9M of 4.7M boxes. This samples them
      so the ones it got wrong can be counted &mdash; a dog dropped here is
      gone from everything downstream.</div></div>
  <a class="back" href="/">&larr; dashboard</a>
</header>

<div class="meas">
  <div><div class="mlab">missed dogs</div>
    <div class="mbig" id="rate">&mdash;</div>
    <div class="mci" id="ci">nothing judged yet</div></div>
  <div><div class="mlab">boxes judged</div>
    <div class="mbig" id="judged">0</div>
    <div class="mci" id="found">&nbsp;</div></div>
  <div class="mnote" id="note">Each band is drawn from evenly, so these are
    not proportions of what you have seen &mdash; the headline weights each
    band by how many boxes the gate really put in it.</div>
</div>

<div class="bands" id="bands"></div>

<div class="bar">
  <button class="btn" id="prev">&larr; back</button>
  <button class="btn go" id="next">next page &rarr;</button>
  <span class="pos" id="pos">&mdash;</span>
  <span class="spacer"></span>
  <button class="btn" id="fresh">&#8635; draw a new page</button>
</div>

<div class="grid" id="grid"></div>
<div class="empty" id="empty" hidden></div>
<div class="keys">
  <kbd>1</kbd> missed a dog &nbsp; <kbd>2</kbd> not a dog &nbsp;
  <kbd>3</kbd> unsure &nbsp; <kbd>&larr;</kbd><kbd>&rarr;</kbd> move &nbsp;
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

<script>
var BANDS=__BANDS__;
var grid=document.getElementById('grid'),empty=document.getElementById('empty'),
    posEl=document.getElementById('pos'),lb=document.getElementById('lb'),
    lbimg=document.getElementById('lbimg'),lbtxt=document.getElementById('lbtxt');
var page=null,idx=0,cur=0,total=0,band=null,busy=false;

function toast(t){var e=document.getElementById('toast');e.textContent=t;
  e.hidden=false;clearTimeout(e._t);e._t=setTimeout(function(){e.hidden=true},1600)}
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML}
function pctTxt(v){return (v*100).toFixed(1)+'%'}

/* The verdict is sent the moment it is given and the card is marked from the
   local answer, not from a reload: a reviewer working through a page at speed
   must never wait on a round trip, and the ledger is append-only so a lost
   response costs one line, not the page. */
function judge(i,verdict){
  var it=page.items[i]; if(!it)return;
  it.verdict=verdict; paintCard(i);
  fetch('/api/audit/verdict',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:it.key,verdict:verdict,band:it.band,
                         p_dog:it.p_dog,seq:it.seq})})
    .then(function(r){return r.json()})
    .then(function(j){if(!j||!j.ok)toast('not recorded');else loadStats()})
    .catch(function(){toast('not recorded')});
  if(i===cur&&cur<page.items.length-1)move(1);
}
function paintCard(i){
  var el=grid.children[i],it=page.items[i]; if(!el||!it)return;
  el.className='card'+(it.verdict?' done':'')+
    (it.verdict==='missed'?' miss':it.verdict==='correct'?' ok':'')+
    (i===cur?' cur':'');
  var b=el.querySelectorAll('.act');
  b[0].classList.toggle('on',it.verdict==='missed');
  b[1].classList.toggle('on',it.verdict==='correct');
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
        '<button class="act m" data-v="missed" data-i="'+i+'">missed a dog</button>'+
        '<button class="act c" data-v="correct" data-i="'+i+'">not a dog</button>'+
        '<button class="act u" data-v="unsure" data-i="'+i+'" title="unsure">?</button>'+
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
  fetch('/api/audit/page?i='+at).then(function(r){return r.json()})
    .then(function(j){busy=false;if(!j||j.error){toast(j&&j.error||'failed');setPos();return}
      show(j.page,j.index,j.total)})
    .catch(function(){busy=false;toast('failed');setPos()});
}
function draw(){
  busy=true;setPos();
  document.getElementById('fresh').textContent='cutting…';
  fetch('/api/audit/draw',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({band:band})})
    .then(function(r){return r.json()})
    .then(function(j){busy=false;
      document.getElementById('fresh').textContent='↻ draw a new page';
      if(!j||j.error){toast(j&&j.error||'failed');setPos();return}
      show(j.page,j.index,j.total);loadStats()})
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
  var r=document.getElementById('rate'),ci=document.getElementById('ci');
  document.getElementById('judged').textContent=(s.judged||0).toLocaleString();
  document.getElementById('found').textContent=
    (s.missed||0)+' were dogs';
  if(!s.judged){r.textContent='—';ci.textContent='nothing judged yet'}
  else{
    r.textContent=pctTxt(s.weighted_rate||0);
    ci.textContent='≈'+Math.round((s.weighted_rate||0)*(s.pool||0)).toLocaleString()+
      ' dogs across the '+(s.pool||0).toLocaleString()+' rejected boxes';
  }
  document.getElementById('bands').innerHTML=(s.bands||[]).map(function(b,i){
    var w=b.judged?Math.max(1.2,Math.min(100,b.rate*100*6)):0;
    return '<div class="brow'+(band===i?' on':'')+'">'+
      '<button class="bsel" data-band="'+i+'">p_dog '+b.lo.toFixed(1)+'–'+
        b.hi.toFixed(1)+'</button>'+
      '<div class="btrack">'+(b.judged?'<div class="bfill" style="width:'+
        w.toFixed(1)+'%"></div>':'')+'</div>'+
      '<span class="'+(b.judged?'':'bnil')+'">'+
        (b.judged?b.missed+'/'+b.judged+'  '+pctTxt(b.rate):
          b.boxes.toLocaleString()+' unsampled')+'</span></div>';
  }).join('');
}
/* clicks */
grid.addEventListener('click',function(e){
  var z=e.target.closest&&e.target.closest('[data-zoom]');
  if(z){zoom(+z.getAttribute('data-zoom'));return}
  var a=e.target.closest&&e.target.closest('.act');
  if(a){cur=+a.getAttribute('data-i');judge(cur,a.getAttribute('data-v'))}
});
document.getElementById('bands').addEventListener('click',function(e){
  var b=e.target.closest&&e.target.closest('[data-band]');
  if(!b)return;
  var v=+b.getAttribute('data-band');
  band=(band===v)?null:v;
  loadStats();toast(band==null?'drawing from every band':'drawing from that band only');
});
document.getElementById('next').addEventListener('click',function(){
  if(idx+1<total)load(idx+1);else draw();
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
  if(!lb.hidden){if(e.key==='Escape'){lb.hidden=true;e.preventDefault()}return}
  if(e.key==='1'){judge(cur,'missed');e.preventDefault()}
  else if(e.key==='2'){judge(cur,'correct');e.preventDefault()}
  else if(e.key==='3'){judge(cur,'unsure');e.preventDefault()}
  else if(e.key==='ArrowRight'){move(1);e.preventDefault()}
  else if(e.key==='ArrowLeft'){move(-1);e.preventDefault()}
  else if(e.key==='Enter'){zoom(cur);e.preventDefault()}
  else if(e.key==='n'||e.key==='N'){
    if(idx+1<total)load(idx+1);else draw();e.preventDefault()}
});
/* boot: the last page if there is one, otherwise an invitation */
fetch('/api/audit/page?i=-1').then(function(r){return r.json()})
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


def prefetch(band=None, n=24):
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


def api_page(i, n=24, band=None):
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


def api_draw(n=24, band=None):
    doc = draw_page(n=n, band=band)
    total = max(1, page_count())
    if doc.get('items'):
        prefetch(band=band, n=n)
    return {'page': with_verdicts(doc), 'index': doc.get('index', total - 1),
            'total': total}
