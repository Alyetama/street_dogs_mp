"""Duel: a blinded look at the dogs two detectors disagree about.

A detector comparison ends in a list of dogs only one of the two models counts.
data/detector_diagnosis/<run>/review.py draws a sample of them, half from each
side, shuffles the two halves together and writes a review set: a close-up
and a whole-frame thumbnail per item, and a key.json that says which model
drew each box and why the other did not. A person then answers one question
per item -- is the thing in the pink box a dog? -- and review.py scores the
answers against the key.

That used to be a standalone HTML page opened off the disk, with the answers
kept in the browser until somebody remembered to download them. This is the
same page behind the dashboard's sign-in: the answers are written on the
server as they are given, signed with the annotator's name, and survive a
closed tab, a cleared browser and a restart.

BLINDING IS THE POINT. The person judging must not know which model drew the
box, or the judgement measures what they expect of the models rather than
what is in the picture. So NOTHING from key.json reaches the browser beyond:

  the set's name, its item numbers and their count, and which items show a
  cyan box (the page has to offer the fourth answer on those, and the cyan
  box is already in the picture)

-- never the side, the reason, the confidences, the detection index, the
image id, a coordinate, the per-side totals or anything else. key.json is
read on the server, cut down to that projection the moment it is parsed, and
only the projection is cached. Pictures are asked for by set and item number,
never by image id, and key.json and the set's old index.html are never served.

THE CYAN BOX. A model can count one dog with two boxes that NMS does not
merge. Asked only "is this a dog?" a person would score the second box as a
real dog, so review.py draws the same model's earlier, surer, overlapping
dogs in cyan, and those items take a fourth answer: "same dog as the cyan
box" -- an extra count, not a new dog. score() refuses that answer on an item
that showed no cyan box, so save() refuses it too, before it is written.

WHAT IS WRITTEN. data/duel_answers/answers.jsonl, append-only, one line per
answer, the same shape of ledger as the Annotate page's: the set, the item,
the answer, who gave it and when. The LATEST line for (set, annotator, item)
is the answer, so changing one's mind is a second line rather than an edit.
The author comes from the signed-in session and never from the request. The
review sets themselves are read-only inputs, and nothing is ever written in
them.

A REBUILT SET STARTS FRESH. review.py can be run again over the same
directory with another seed or another comparison, and item 17 is then a
different box. Every line therefore carries a fingerprint of the key's items
(server-side only -- it is a hash and leaves this process as nothing), and
answers are read back only against the build they were given for. Answers
for a build that no longer exists stay in the ledger, unread.

WHERE THE SETS ARE is config: duel_dir (or $DUEL_DIR), resolved by
dashboard.py and passed in, so this module never imports the dashboard. Each
set is a subdirectory review_<name>/ holding key.json and img/NNNc.jpg (the
close-up) and img/NNNf.jpg (the whole frame), NNN the item number padded to
three digits. Unset, the page says which key to set.
"""

import hashlib
import json
import os
import re
import threading
import time

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER_DIR = os.path.join(REPO, 'data', 'duel_answers')
LEDGER = os.path.join(LEDGER_DIR, 'answers.jsonl')
SOURCE = 'duel_page'

# \Z, not $: `$` also matches before a trailing newline, which would let
# 'review_name\n' through as a second, different-looking set
SET_DIR_RE = re.compile(r'^review_([a-z0-9][a-z0-9_]{0,60})\Z')
SET_RE = re.compile(r'^[a-z0-9][a-z0-9_]{0,60}\Z')
ANSWERS = ('dog', 'not_dog', 'unsure', 'dup')
VIEWS = ('c', 'f')
# review.py numbers items from 1 and pads to three digits; a set holds 150.
# A number past this is not an item, it is somebody counting upward.
MAX_N = 99999

KEY = 'duel_dir'
ENV = 'DUEL_DIR'
CFG_FILE = 'tools/dashboard/dashboard.config.json'
# Never the configured path: that is a fact about this machine, and the page
# is served to volunteers on the public internet.
UNSET = ('duel_dir is not set — put the directory that holds the '
         'review_<name>/ sets in "duel_dir" in %s, or in $%s, and reload.' %
         (CFG_FILE, ENV))
MISSING = ('duel_dir is set but is not a directory this server can read '
           '— check "duel_dir" in %s or $%s.' % (CFG_FILE, ENV))
NO_AUTHOR = 'no annotator -- this answer was not given by a signed-in account'
NO_SET = 'no such review set'

_lock = threading.Lock()
_cache = {}


def _stamp(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _cached(key, path, load):
    """load() once per version of `path` on disk."""
    stamp = _stamp(path)
    hit = _cache.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    val = load() if stamp else None
    _cache[key] = (stamp, val)
    return val


def _root(root):
    """The real path of duel_dir, or None when it is unset or unreadable.

    realpath because duel_dir may itself sit behind a symlink, and every
    containment test below compares real paths on both sides.
    """
    if not root or not isinstance(root, str):
        return None
    r = os.path.realpath(root)
    return r if os.path.isdir(r) else None


def state(root):
    """None when duel_dir can be used, else the sentence that says why not."""
    if not root:
        return UNSET
    return None if _root(root) else MISSING


def _set_dirs(root):
    """{name: real directory} for every review set under duel_dir.

    THE CLIENT'S STRING IS LOOKED UP HERE, NEVER JOINED ONTO A PATH. A set is
    an entry of duel_dir named review_<name> that is a real directory -- not
    a link, even one pointing at another set, because a link is a second name
    for something and the whole point is that a name means one thing -- with
    a key.json and a real img/ directory in it.
    """
    r = _root(root)
    if not r:
        return {}
    out = {}
    try:
        names = sorted(os.listdir(r))
    except OSError:
        return {}
    for entry in names:
        m = SET_DIR_RE.match(entry)
        if not m:
            continue
        d = os.path.join(r, entry)
        if os.path.realpath(d) != d or not os.path.isdir(d):
            continue
        k = os.path.join(d, 'key.json')
        if os.path.realpath(k) != k or not os.path.isfile(k):
            continue
        # a set whose img/ is a link has no picture this page will serve, so
        # it is not offered as a set at all
        img = os.path.join(d, 'img')
        if os.path.realpath(img) != img or not os.path.isdir(img):
            continue
        out[m.group(1)] = d
    return out


def _n(v):
    """An item number as the client may send it -- a whole number, or the
    digits of one -- else None. int() alone would take 1.5 or True as 1."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if 0 < v <= MAX_N else None
    if isinstance(v, str) and v.isascii() and v.isdigit() and len(v) < 6:
        n = int(v)
        return n if 0 < n <= MAX_N else None
    return None


def _key(d):
    """The ONE projection of a set's key.json this module keeps.

    {'items': [n, ...] ascending, 'cyan': frozenset of n, 'build': hash}.
    Everything else in the file -- side, reason, confidences, image ids,
    boxes, totals -- is dropped right here, so no later code path can hand it
    to the browser by accident. None when the key cannot be read.
    """
    path = os.path.join(d, 'key.json')

    def load():
        with open(path) as fh:
            raw = json.load(fh)
        items = raw.get('items') if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return None
        ns, cyan = [], set()
        for it in items:
            if not isinstance(it, dict):
                continue
            n = it.get('n')
            if isinstance(n, bool) or not isinstance(n, int) \
                    or not 0 < n <= MAX_N:
                continue
            ns.append(n)
            if it.get('earlier_dogs'):
                cyan.add(n)
        if len(set(ns)) != len(ns):
            return None  # two items, one number: answers could not be told
        build = hashlib.sha256(
            json.dumps(items, sort_keys=True,
                       default=str).encode()).hexdigest()[:16]
        return {'items': sorted(ns), 'cyan': frozenset(cyan), 'build': build}

    try:
        return _cached(('key', path), path, load)
    except (OSError, ValueError, TypeError):
        return None


def _set(root, name):
    """(directory, key projection) for a set the client named, or None."""
    if not isinstance(name, str) or not SET_RE.match(name):
        return None
    d = _set_dirs(root).get(name)
    if not d:
        return None
    k = _key(d)
    return (d, k) if k and k['items'] else None


def _ledger():
    """{(set, build, by, n): answer}, the latest line for each winning."""

    def load():
        out = {}
        with open(LEDGER) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a torn last line is not an answer
                if not isinstance(r, dict):
                    continue
                try:
                    k = (str(r['set']), str(r['build']), str(r['by']),
                         int(r['n']))
                except (KeyError, TypeError, ValueError):
                    continue
                if r.get('answer') in ANSWERS:
                    out[k] = r['answer']
        return out

    return _cached(('ledger', ), LEDGER, load) or {}


def _mine(name, k, who):
    """{n: answer} for one annotator on one build of one set -- only items
    the key still has, so a count is of distinct items, never of lines."""
    if not who:
        return {}
    led = _ledger()
    out = {}
    for n in k['items']:
        a = led.get((name, k['build'], who, n))
        if a is not None:
            out[n] = a
    return out


def sets(root, who):
    """Every set there is, with how far THIS annotator has got in each."""
    why = state(root)
    if why:
        return {'configured': False, 'key': KEY, 'error': why, 'sets': []}
    out = []
    for name, d in sorted(_set_dirs(root).items()):
        k = _key(d)
        if not k or not k['items']:
            continue
        out.append({
            'set': name,
            'total': len(k['items']),
            'answered': len(_mine(name, k, who)),
        })
    return {'configured': True, 'sets': out}


def set_view(root, name, who):
    """Everything the page needs for one set, and nothing else."""
    why = state(root)
    if why:
        return {'configured': False, 'key': KEY, 'error': why}
    got = _set(root, name)
    if not got:
        return {'configured': True, 'error': NO_SET}
    _d, k = got
    mine = _mine(name, k, who)
    return {
        'configured': True,
        'set': name,
        'items': list(k['items']),
        'cyan': sorted(k['cyan']),
        'total': len(k['items']),
        'answered': len(mine),
        'answers': {
            str(n): a
            for n, a in sorted(mine.items())
        },
    }


def image_path(root, name, n, v):
    """The jpg for one item's close-up ('c') or whole frame ('f'), or None.

    Nothing the client sends reaches a path: the set is looked up among the
    directories that exist, n must be one of that key's item numbers, v one
    of two letters, and the file name is built here. The file it ENDS at is
    then held to the set's own img/ directory, so a link in img/ pointing at
    another set or anywhere else is refused rather than followed.
    """
    if v not in VIEWS:
        return None
    got = _set(root, name)
    if not got:
        return None
    d, k = got
    n = _n(n)
    if n is None or n not in set(k['items']):
        return None
    img = os.path.realpath(os.path.join(d, 'img'))
    if img != os.path.join(d, 'img'):
        return None
    p = os.path.realpath(os.path.join(img, '%03d%s.jpg' % (n, v)))
    if not p.startswith(img + os.sep) or not os.path.isfile(p):
        return None
    return p


def save(root, name, n, answer, by, now=None):
    """Record one annotator's answer to one item. {'ok': ...}."""
    if not by:
        return {'ok': False, 'error': NO_AUTHOR}
    why = state(root)
    if why:
        return {'ok': False, 'configured': False, 'key': KEY, 'error': why}
    got = _set(root, name)
    if not got:
        return {'ok': False, 'error': NO_SET}
    _d, k = got
    n = _n(n)
    if n is None or n not in set(k['items']):
        return {'ok': False, 'error': 'no such item in that set'}
    if answer not in ANSWERS:
        return {
            'ok': False,
            'error': 'the answer must be one of ' + ', '.join(ANSWERS)
        }
    if answer == 'dup' and n not in k['cyan']:
        return {
            'ok':
            False,
            'error':
            '"same dog as cyan" is only an answer where the '
            'picture shows a cyan box'
        }
    rec = {
        'set': name,
        'build': k['build'],
        'n': n,
        'answer': answer,
        'by': str(by),
        'at': int(time.time() if now is None else now),
        'source': SOURCE,
    }
    try:
        with _lock:
            os.makedirs(LEDGER_DIR, exist_ok=True)
            with open(LEDGER, 'a+b') as fh:
                # a line torn by a crash would swallow this one into it --
                # the two would read back as one line that is not JSON
                lead = b''
                if fh.tell() > 0:
                    fh.seek(-1, os.SEEK_END)
                    if fh.read(1) != b'\n':
                        lead = b'\n'
                fh.write(lead + (json.dumps(rec) + '\n').encode())
                fh.flush()
                os.fsync(fh.fileno())
    except OSError as e:
        # the type, not the text: an OSError's text names the file
        return {
            'ok': False,
            'error': 'not saved (%s) -- try again' % (type(e).__name__, )
        }
    mine = _mine(name, k, str(by))
    # what the page shows back is the item and the answer, never the line:
    # the build fingerprint stays on this side
    return {
        'ok': True,
        'n': n,
        'answer': answer,
        'answered': len(mine),
        'total': len(k['items'])
    }


def export(root, name, who):
    """The annotator's answers in the shape review.py score --answers reads:
    {"store": "<set>_<annotator>", "answers": {"<n>": "<answer>"}}, keys the
    plain digits of n with no padding, as score() looks them up."""
    why = state(root)
    if why:
        return {'configured': False, 'key': KEY, 'error': why}
    got = _set(root, name)
    if not got:
        return {'configured': True, 'error': NO_SET}
    _d, k = got
    mine = _mine(name, k, who)
    return {
        'store': '%s_%s' % (name, who),
        'answers': {
            str(n): a
            for n, a in sorted(mine.items())
        }
    }


def page_html(account=('', '')):
    """The page, with the identity strip spliced in like every other."""
    out = DUEL_HTML
    for key, value in (('__ACCTCSS__', account[0]), ('__ACCOUNT__',
                                                     account[1])):
        out = out.replace(key, value)
    return out


# The wording, keys and colours are review.py's PAGE, kept on purpose: answers
# given here and answers given on the old page must mean the same thing, or
# the two cannot be scored together.
DUEL_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Duel &mdash; Street Dogs</title><style>
:root{--bg:#13151a;--panel:#1b2027;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;
--red:#ef5350;--num:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
--dog:#3fa66b;--not:#c8553d;--unsure:#b48a2c;--mark:#ff00c8;--cyan:#00c8ff;--dup:#1f7f9e}
*{box-sizing:border-box;margin:0;padding:0}
[hidden]{display:none!important}
body{background:var(--bg);color:var(--tx);-webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  line-height:1.5;padding:0 18px 18px}
a{color:inherit}
button,select{font:inherit}
header{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;
  padding:16px 0 12px;border-bottom:1px solid var(--bd)}
h1{font-size:19px;font-weight:660;letter-spacing:-.3px}
.sub{color:var(--dim);font-size:12.5px;margin-top:2px;max-width:78ch}
.hdrend{display:flex;align-items:center;gap:12px;margin-left:auto}
.back{font-size:12px;color:var(--mut);text-decoration:none;
  border:1px solid var(--bd);border-radius:8px;padding:6px 11px}
.back:hover{color:var(--tx);border-color:rgba(130,140,150,.3)}
__ACCTCSS__

main{max-width:1120px;margin:0 auto}
.bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 0}
.bsel{background:var(--panel2);color:var(--tx);border:1px solid var(--bd);
  border-radius:8px;padding:5px 9px;font-size:12.5px}
.bsel:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.prog{display:flex;align-items:center;gap:9px;font:400 12px var(--num);
  color:var(--mut);font-variant-numeric:tabular-nums}
.pbar{width:160px;height:5px;border-radius:3px;background:var(--panel2);
  overflow:hidden}
.pbar i{display:block;height:100%;background:var(--green);width:0}
.done{color:var(--green)}
.q1{font-size:16px;font-weight:600;margin:4px 0 10px}
.view{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}
@media (max-width:900px){.view{grid-template-columns:1fr}}
figure{background:var(--panel);border:1px solid var(--bd);border-radius:12px;
  padding:8px}
figure img{display:block;width:100%;height:auto;border-radius:6px;
  background:#0b0d10;min-height:120px}
figcaption{color:var(--mut);font-size:13px;margin-top:6px}
.q{margin:16px 0 8px;font-size:16px}
.cy{color:var(--cyan);font-weight:600}
.dupq{margin:4px 0 8px;font-size:14px;color:var(--mut);max-width:78ch}
.btns,.nav{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.nav{margin-top:16px}
.btn{appearance:none;background:var(--panel2);color:var(--tx);
  border:1px solid var(--bd);border-radius:8px;padding:9px 15px;
  font-size:14px;cursor:pointer;line-height:1.3;white-space:nowrap}
.btns .btn{min-width:132px}
.btn:hover{border-color:rgba(130,140,150,.4)}
.btn:focus-visible{outline:2px solid var(--mark);outline-offset:2px}
.btn:disabled{opacity:.4;cursor:default}
.btn kbd{font:500 11px var(--num);color:var(--dim);margin-left:6px}
.btn.on[data-v=dog]{background:var(--dog);border-color:var(--dog)}
.btn.on[data-v=not_dog]{background:var(--not);border-color:var(--not)}
.btn.on[data-v=unsure]{background:var(--unsure);border-color:var(--unsure)}
.btn.on[data-v=dup]{background:var(--dup);border-color:var(--dup)}
.btn.on kbd{color:var(--tx)}
.note{color:var(--mut);font-size:13px;margin-top:18px;max-width:78ch}
.status{min-height:20px;font-size:12.5px;color:var(--mut);padding-top:10px}
.status.bad{color:var(--red)}
.status.good{color:var(--green)}
.empty{border:1px solid var(--bd);border-radius:12px;background:var(--panel);
  padding:18px 20px;margin-top:16px;max-width:78ch;color:var(--mut);font-size:14px}
.empty b{color:var(--tx)}
.empty code{font:400 12.5px var(--num);color:var(--tx)}
</style></head><body>
<header>
  <div>
    <h1>Duel</h1>
    <p class="sub">Dogs only one of two detectors counts, shuffled together.
    Nothing here says which detector drew the box &mdash; judge only the
    picture.</p>
  </div>
  <div class="hdrend"><a class="back" href="/">&larr; dashboard</a>__ACCOUNT__</div>
</header>
<main>
<div class="empty" id="empty" hidden></div>
<div id="work" hidden>
<div class="bar">
  <select class="bsel" id="bsel" aria-label="review set" hidden></select>
  <div class="prog"><span class="pbar"><i id="pfill"></i></span>
    <span id="count">&mdash;</span></div>
</div>
<p class="q1">Is the boxed thing a dog?</p>
<div class="view">
<figure><img id="crop" alt="close-up of the boxed object"><figcaption>Close-up. The pink box is the one to judge.</figcaption></figure>
<figure><img id="frame" alt="the whole frame, box marked"><figcaption>Where it sits in the frame.</figcaption></figure>
</div>
<p class="q">Is the thing inside the pink box a dog?</p>
<p class="dupq" id="dupq" hidden>This one has a <span class="cy">cyan</span> box nearby: the same source
already counted a dog there with a surer box. If the pink box is that same dog again, choose
<b>Same dog</b>. If it is a different dog, choose <b>Dog</b>.</p>
<div class="btns" id="btns">
<button type="button" class="btn" data-v="dog">Dog<kbd>D</kbd></button>
<button type="button" class="btn" data-v="not_dog">Not a dog<kbd>N</kbd></button>
<button type="button" class="btn" data-v="dup" id="dupbtn" hidden>Same dog as cyan<kbd>S</kbd></button>
<button type="button" class="btn" data-v="unsure">Can't tell<kbd>U</kbd></button>
</div>
<div class="nav">
<button type="button" class="btn" id="prev">&larr; Previous</button>
<button type="button" class="btn" id="next">Next &rarr;</button>
<button type="button" class="btn" id="skip">Next unanswered</button>
<button type="button" class="btn" id="dl">Download answers</button>
</div>
<div class="status" id="status" role="status" aria-live="polite"></div>
<p class="note">Judge only what is in the pink box: a dog, part of a dog, or a dog
seen from far away counts as a dog; a cat, a person, a bag or a shadow does not.
If the box holds a dog together with its leash or owner, it is a dog, and so is
someone's pet on a lead. A <span class="cy">cyan</span> box, when there is one, is
a dog the same source already counted close by. Each answer is saved on the
server under your name as you give it, and you can change it by answering
again; &ldquo;Download answers&rdquo; gives you the file the scoring script reads.</p>
</div>
</main>
<script>
(function(){
var $=function(id){return document.getElementById(id)};
var SET='',ITEMS=[],DUPS=new Set(),ans={},i=0,TOTAL=0;

function say(t,kind){var s=$('status');s.textContent=t||'';
  s.className='status'+(kind?' '+kind:'')}
function api(url,opts){
  opts=opts||{};opts.credentials='same-origin';
  return fetch(url,opts).then(function(r){
    return r.json().catch(function(){return {}}).then(function(j){
      if(!r.ok&&!j.error)j.error='the server answered '+r.status;return j})})}
function img(n,v){return '/duel/img?set='+encodeURIComponent(SET)+'&n='+n+'&v='+v}
function empty(parts){var e=$('empty');e.textContent='';
  parts.forEach(function(p){var n;
    if(typeof p==='string')n=document.createTextNode(p);
    else{n=document.createElement(p[0]);n.textContent=p[1]}
    e.appendChild(n)});
  e.hidden=false;$('work').hidden=true}
function unset(j){
  empty([['b','The Duel page has nothing to show yet. '],(j&&j.error)||'',
    ' The key is ',['code','duel_dir'],', or the environment variable ',
    ['code','$DUEL_DIR'],', in ',['code','tools/dashboard/dashboard.config.json'],'.'])}

function show(){
  if(!ITEMS.length)return;
  var n=ITEMS[i];
  $('crop').src=img(n,'c');$('frame').src=img(n,'f');
  var done=ITEMS.filter(function(k){return ans[k]}).length;
  var c=$('count');c.textContent='Item '+(i+1)+' of '+ITEMS.length+' · '+done+' answered';
  if(done===ITEMS.length){var s=document.createElement('span');s.className='done';
    s.textContent=' · all answered';c.appendChild(s)}
  $('pfill').style.width=(100*done/ITEMS.length).toFixed(1)+'%';
  $('dupbtn').hidden=$('dupq').hidden=!DUPS.has(n);
  document.querySelectorAll('#btns .btn').forEach(function(b){
    var on=ans[n]===b.dataset.v;b.classList.toggle('on',on);
    b.setAttribute('aria-pressed',on?'true':'false')});
  $('prev').disabled=i===0;$('next').disabled=i>=ITEMS.length-1;
  var o=$('bsel').querySelector('option[value="'+SET+'"]');
  if(o)o.textContent=SET+' — '+done+'/'+ITEMS.length;
  var nx=ITEMS[i+1];if(nx){new Image().src=img(nx,'c')}}

/* the answer shows at once and the page moves on; if the server refuses it,
   the item goes back to what it was and the status line says which one */
function pick(v){
  if(!ITEMS.length)return;
  var n=ITEMS[i],was=ans[n],set=SET;
  if(v==='dup'&&!DUPS.has(n))return;
  ans[n]=v;if(i<ITEMS.length-1)i++;show();say('saving item '+n+'…');
  api('/api/duel/answer',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({set:set,n:n,answer:v})})
    .then(function(j){if(set!==SET)return;
      if(!j.ok){if(was)ans[n]=was;else delete ans[n];show();
        say('Item '+n+' was not saved: '+(j.error||'the server refused it'),'bad');return}
      say('Item '+n+' saved.','good')})
    .catch(function(){if(set!==SET)return;if(was)ans[n]=was;else delete ans[n];show();
      say('Item '+n+' was not saved — the server did not answer.','bad')})}
document.querySelectorAll('#btns .btn').forEach(function(b){
  b.addEventListener('click',function(){pick(b.dataset.v)})});
$('prev').addEventListener('click',function(){if(i>0)i--;show()});
$('next').addEventListener('click',function(){if(i<ITEMS.length-1)i++;show()});
$('skip').addEventListener('click',function(){
  var j=ITEMS.findIndex(function(k){return !ans[k]});
  if(j>=0){i=j;show()}else say('Every item is answered.','good')});
$('dl').addEventListener('click',function(){
  if(!SET)return;
  api('/api/duel/export?set='+encodeURIComponent(SET)).then(function(j){
    if(j.error||!j.answers){say(j.error||'nothing to download','bad');return}
    var blob=new Blob([JSON.stringify({store:j.store,answers:j.answers},null,1)],
      {type:'application/json'});
    var a=document.createElement('a');a.href=URL.createObjectURL(blob);
    a.download=j.store+'_answers.json';document.body.appendChild(a);a.click();
    setTimeout(function(){URL.revokeObjectURL(a.href);a.remove()},1000)})
  .catch(function(){say('The server did not answer.','bad')})});
document.addEventListener('keydown',function(e){
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  var tag=(e.target&&e.target.tagName)||'';
  if(tag==='SELECT'||tag==='INPUT'||tag==='TEXTAREA')return;
  var k=(e.key||'').toLowerCase();
  if(k==='d')pick('dog');else if(k==='n')pick('not_dog');else if(k==='u')pick('unsure');
  else if(k==='s'){if(DUPS.has(ITEMS[i]))pick('dup')}
  else if(e.key==='ArrowRight')$('next').click();else if(e.key==='ArrowLeft')$('prev').click()});

function open(name){
  say('loading…');
  api('/api/duel/set?set='+encodeURIComponent(name)).then(function(j){
    if(j.configured===false){unset(j);return}
    if(j.error||!j.items){say(j.error||'that set could not be opened','bad');return}
    SET=j.set;ITEMS=j.items;DUPS=new Set(j.cyan||[]);ans={};
    Object.keys(j.answers||{}).forEach(function(k){ans[+k]=j.answers[k]});
    var first=ITEMS.findIndex(function(k){return !ans[k]});i=first>=0?first:0;
    try{localStorage.setItem('duel.set',SET)}catch(e){}
    $('empty').hidden=true;$('work').hidden=false;say('');show()})
  .catch(function(){say('The server did not answer.','bad')})}
function load(){
  api('/api/duel/sets').then(function(j){
    if(j.configured===false){unset(j);return}
    var list=(j&&j.sets)||[];
    if(!list.length){empty([['b','There is no review set to judge yet. '],
      'A set is a review_<name>/ directory, with key.json and img/, inside ',
      ['code','duel_dir'],'; review.py build makes one.']);return}
    var sel=$('bsel');sel.textContent='';
    list.forEach(function(s){var o=document.createElement('option');o.value=s.set;
      o.textContent=s.set+' — '+s.answered+'/'+s.total;sel.appendChild(o)});
    sel.hidden=list.length<2;
    var want=new URLSearchParams(location.search).get('set');
    try{want=want||localStorage.getItem('duel.set')}catch(e){}
    var pk=list.filter(function(s){return s.set===want})[0]||
      list.filter(function(s){return s.answered<s.total})[0]||list[0];
    sel.value=pk.set;open(pk.set)})
  .catch(function(){empty([['b','The server did not answer.']])})}
$('bsel').addEventListener('change',function(){var v=this.value;this.blur();open(v)});
load();
})();
</script>
</body></html>
"""
