"""Draw every dog in a whole frame -- the labels the detector cannot ask for.

The review page and the audits judge boxes the detector drew: one crop, one
question, and at most a nudge to its geometry. A dog the detector never boxed
has no crop, so it never reaches either of them, and those are the dogs the
detector is worst at -- small, far off, and missing from its training set.
This page is where they get labels. It serves whole frames out of a label
batch (tools/detect/pick_label_frames.py chooses them), and a person draws a
box round every dog in the frame, however small, or says there is none.

A LEASHED DOG'S BOX TAKES IN ITS LEASH. That is the Label Studio convention,
kept on purpose: the same boxes are cut into the leash classifier's crops, and
a crop without the leash cannot show one. The detector trains on Label
Studio's boxes and these together, so the page asks for the same thing.

WHAT IS WRITTEN. data/frame_labels/labels.jsonl, append-only, one line per
save: the image, its batch, every box in the frame's own pixels, whether the
frame was declared empty, the frame's size, who drew it and when. The LATEST
line for an image is its label, so a correction is a second save rather than
an edit. The author comes from the signed-in session and never from the
request -- a client that could name its own annotator could sign somebody
else's name to a label, and the ledger is append-only.

WHAT IS NOT. Anything a model said. The seed dog a batch was chosen for is
shown as a hint -- where the gate saw a dog a second earlier, in the next
frame along, or where a second model thinks a dog nobody labelled is -- and
is never recorded as a label or copied into the ledger: the one thing in this
file is what a person drew.

WHAT THE PAGE SAYS DEPENDS ON THE BATCH. Two kinds exist, and what is true of
a frame differs between them, so the sentence over the frame and the hint's
tooltip come from the frame (see COPY), never from one fixed header:

  neighbours   pick_label_frames.py: the detector drew nothing here, and the
               hint is where the gate saw a dog a second earlier
  hidden_dogs  hidden_dogs.py: a person already confirmed one dog here (or
               rejected the detector's box), and the hint is where a second
               model thinks there is a dog nobody labelled -- not always right

A batch says which it is in summary.json `kind`; an older batch without one is
read off its manifest (hidden-dog rows carry a `role`). A single fixed header
told annotators of the first hidden-dog batch that the detector drew nothing
and that the hint was the gate's dog, and several boxed only the hinted dog
and left the confirmed one out.

HOW THE WORK IS SHARED. Every annotator walks a batch in their own order -- a
hash of their name and the image id -- so two people on one batch rarely meet
on the same frame, without anything having to be claimed, held or released.
"""

import hashlib
import json
import math
import os
import re
import threading
import time

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCHES = os.path.join(REPO, 'data', 'label_batches')
LEDGER_DIR = os.path.join(REPO, 'data', 'frame_labels')
LEDGER = os.path.join(LEDGER_DIR, 'labels.jsonl')
SOURCE = 'annotate_page'

BATCH_RE = re.compile(r'^[a-z0-9][a-z0-9_]{2,60}$')
ID_RE = re.compile(r'^[0-9]{6,24}$')
# More than this in one street frame is a slip of the mouse, not a pack.
MAX_BOXES = 60
# A box narrower than this, in the frame's own pixels, is a click, not a dog.
MIN_SIDE = 2.0

NO_AUTHOR = 'no annotator -- this save was not made by a signed-in account'

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


def _batch_dir(batch):
    if not isinstance(batch, str) or not BATCH_RE.match(batch):
        return None
    d = os.path.join(BATCHES, batch)
    return d if os.path.isfile(os.path.join(d, 'manifest.jsonl')) else None


def manifest(batch):
    """The frames of one batch, in the order the tool wrote them."""
    d = _batch_dir(batch)
    if not d:
        return None
    path = os.path.join(d, 'manifest.jsonl')

    def load():
        rows = []
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if ID_RE.match(str(r.get('image_id', ''))):
                        rows.append(r)
        return rows

    return _cached(('manifest', batch), path, load) or []


def seeds(batch):
    """Where each frame's seed dog was: the box in the SEED frame's pixels and
    that frame's size. Optional -- a batch without it just draws no hint."""
    d = _batch_dir(batch)
    if not d:
        return {}
    path = os.path.join(d, 'seeds.json')

    def load():
        with open(path) as fh:
            return json.load(fh)

    try:
        return _cached(('seeds', batch), path, load) or {}
    except (OSError, ValueError):
        return {}


def labels():
    """{image_id: the latest record for it}. A later save replaces an earlier
    one, which is how a label is corrected in an append-only ledger."""

    def load():
        out = {}
        with open(LEDGER) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a torn last line is not a label
                if isinstance(r, dict) and r.get('image_id'):
                    out[str(r['image_id'])] = r
        return out

    return _cached(('labels', ), LEDGER, load) or {}


# What the page says over a frame, by kind of batch and the frame's role.
# Plain text: the page sets it with textContent. Every one ends by asking for
# every dog and for the leash, which are true of every batch.
_EVERY = ('Box every dog you can see, however small — zoom in to look. '
          'For a dog on a leash, take the leash into the box, as in Label '
          'Studio.')
COPY = {
    'neighbours': {
        'intro': ('The detector drew nothing on this frame, but a dog was '
                  'seen a second earlier on the same road. ' + _EVERY +
                  ' If there is none, mark the frame “no dog”.'),
        'hint':
        'the gate saw a dog about here, a second earlier',
    },
    'hidden_dogs/addition': {
        'intro': ('A reviewer already confirmed one dog in this frame, and a '
                  'second model thinks there is another — the dashed '
                  'square marks where. Box the confirmed dog too, not only '
                  'the one in the square. ' + _EVERY),
        'hint': ('a second model thinks a dog nobody labelled is about here '
                 '— check it: it is not always a dog'),
    },
    'hidden_dogs/background': {
        'intro': ('A reviewer said the detector’s box in this frame was '
                  'not a dog, but a second model thinks there is a dog '
                  '— the dashed square marks where. It is often a cat, '
                  'a goat or part of the car: if there is no dog anywhere, '
                  'mark the frame “no dog”. ' + _EVERY),
        'hint': ('a second model thinks a dog is about here — often it '
                 'is a cat, a goat or part of the car'),
    },
}


def batch_kind(batch):
    """'neighbours' or 'hidden_dogs', from summary.json or the manifest."""
    d = _batch_dir(batch)
    if not d:
        return None
    try:
        with open(os.path.join(d, 'summary.json')) as fh:
            kind = json.load(fh).get('kind')
        if kind in ('neighbours', 'hidden_dogs'):
            return kind
    except (OSError, ValueError, AttributeError):
        pass
    rows = manifest(batch) or []
    return 'hidden_dogs' if any(r.get('role') for r in rows) else 'neighbours'


def copy_for(batch, row):
    """{'intro', 'hint'}: what the page says over this frame."""
    if batch_kind(batch) == 'hidden_dogs':
        role = 'background' if row.get('role') == 'background' else 'addition'
        return COPY['hidden_dogs/' + role]
    return COPY['neighbours']


def batches():
    """Every batch there is, with how far along it is."""
    lab = labels()
    out = []
    try:
        names = sorted(os.listdir(BATCHES))
    except OSError:
        names = []
    for name in names:
        rows = manifest(name)
        if not rows:
            continue
        done = [lab[r['image_id']] for r in rows if r['image_id'] in lab]
        out.append({
            'batch': name,
            'total': len(rows),
            'done': len(done),
            'empty': sum(1 for x in done if x.get('empty')),
            'dogs': sum(len(x.get('boxes') or []) for x in done),
        })
    return out


def _order(user, image_id):
    return hashlib.sha1(('%s|%s' % (user, image_id)).encode()).hexdigest()


def _index(i):
    """A manifest index as the client may send it -- a whole number, or the
    digits of one -- else None. int() alone would take 1.5 or True as 1."""
    if isinstance(i, bool):
        return None
    if isinstance(i, int):
        return i
    if isinstance(i, str) and i.isascii() and i.isdigit() and len(i) < 9:
        return int(i)
    return None


def frame_path(batch, i):
    """The jpg behind one frame of a batch, or None.

    Resolved from the batch's own manifest BY INDEX: nothing the client sends
    is joined onto a path. The file name must still be the image id the
    manifest says it is, so an edited manifest cannot point here at anything
    that is not a harvested frame.
    """
    rows = manifest(batch)
    i = _index(i)
    if i is None or not rows or i < 0 or i >= len(rows):
        return None
    r = rows[i]
    # the name is checked on the file the path ENDS at, so neither a line
    # naming another frame nor a link dressed as this one gets through
    p = os.path.realpath(str(r.get('jpg') or ''))
    if os.path.basename(p) != r['image_id'] + '.jpg' or not os.path.isfile(p):
        return None
    return p


def frame_size(path):
    """The frame's real size, read off the file. The manifest's is the
    sweep's reading of it, and a label is only right against the file."""

    def load():
        from PIL import Image
        with Image.open(path) as im:
            # A frame stored on its side is shown upright by the browser and
            # read raw by the trainer, so a box drawn on it would land in the
            # wrong place. None of the harvest is, as of writing; if one ever
            # is, it is refused here rather than labelled wrong.
            if im.getexif().get(0x0112, 1) != 1:
                return None
            return im.size

    try:
        return _cached(('size', path), path, load)
    except Exception:  # noqa: BLE001 - an unreadable frame has no size
        return None


def _hint(batch, row, w, h):
    """Where the gate saw a dog a second earlier, in THIS frame's pixels.

    The seed box is scaled across by the two frames' sizes -- the same camera
    down the same road -- which puts it near the dog rather than on it. It is
    somewhere to look first, never a label.
    """
    s = seeds(batch).get(row['image_id'])
    if not s:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in s['box'])
        sw, sh = float(s['w']), float(s['h'])
        if sw <= 0 or sh <= 0:
            return None
        return [
            round(x1 / sw * w, 1),
            round(y1 / sh * h, 1),
            round(x2 / sw * w, 1),
            round(y2 / sh * h, 1)
        ]
    except (KeyError, TypeError, ValueError):
        return None


def frame(batch, i, user=''):
    """Everything the page needs to show one frame, or None."""
    rows = manifest(batch)
    path = frame_path(batch, i)
    if not path:
        return None
    size = frame_size(path)
    if not size:
        return None
    w, h = size
    i = _index(i)
    row = rows[i]
    lab = labels()
    got = lab.get(row['image_id'])
    mine = sum(
        1 for r in rows
        if r['image_id'] in lab and lab[r['image_id']].get('by') == user)
    done = sum(1 for r in rows if r['image_id'] in lab)
    return {
        'batch':
        batch,
        'i':
        i,
        'image_id':
        row['image_id'],
        'w':
        w,
        'h':
        h,
        'hint':
        _hint(batch, row, w, h),
        'intro':
        copy_for(batch, row)['intro'],
        'hint_title':
        copy_for(batch, row)['hint'],
        'label': ({
            'boxes': got.get('boxes') or [],
            'empty': bool(got.get('empty')),
            'by': got.get('by'),
            'at': got.get('at')
        } if got else None),
        'progress': {
            'done': done,
            'total': len(rows),
            'mine': mine
        },
    }


def next_frame(batch, user, after=None):
    """The next frame this annotator has not labelled, in THEIR order.

    `after` is the image id they just left: the walk carries on past it,
    and wraps, so skipping never serves the same frame straight back. A frame
    that cannot be read (its drive unmounted) is stepped over rather than
    stopping the batch.
    """
    rows = manifest(batch)
    if not rows:
        return None
    lab = labels()
    todo = sorted((_order(user, r['image_id']), k) for k, r in enumerate(rows)
                  if r['image_id'] not in lab)
    if not todo:
        return {
            'batch': batch,
            'done': True,
            'progress': {
                'done': len(rows),
                'total': len(rows)
            }
        }
    start = 0
    if after is not None and ID_RE.match(str(after)):
        key = _order(user, str(after))
        start = next((n for n, t in enumerate(todo) if t[0] > key), 0)
    for _, k in todo[start:] + todo[:start]:
        got = frame(batch, k, user)
        if got:
            return got
    return {
        'batch':
        batch,
        'error':
        'none of the %d frames left in this '
        'batch can be read -- is a drive unmounted?' % (len(todo), )
    }


def _clean_boxes(boxes, w, h):
    """The boxes as [[x1, y1, x2, y2]] in the frame's pixels, or an error."""
    if not isinstance(boxes, list):
        return None, 'boxes must be a list'
    if len(boxes) > MAX_BOXES:
        return None, 'more than %d boxes in one frame' % (MAX_BOXES, )
    out = []
    for b in boxes:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            return None, 'a box is four numbers'
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            return None, 'a box is four numbers'
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return None, 'a box is four numbers'
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        # a box dragged a few pixels past the edge is clamped, not refused;
        # one that is mostly outside the frame is a mistake
        if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
            return None, 'a box lies outside the frame'
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
            return None, 'a box is too small to be a dog -- delete it'
        out.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
    return out, None


def save(batch, i, boxes, empty, by, now=None):
    """Record what a person drew on one frame. {'ok': ...}."""
    if not by:
        return {'ok': False, 'error': NO_AUTHOR}
    path = frame_path(batch, i)
    if not path:
        return {'ok': False, 'error': 'no such frame in that batch'}
    size = frame_size(path)
    if not size:
        return {'ok': False, 'error': 'the frame could not be read'}
    w, h = size
    clean, err = _clean_boxes(boxes if boxes is not None else [], w, h)
    if err:
        return {'ok': False, 'error': err}
    empty = bool(empty)
    # the two answers must agree: "no dog" with a box drawn, or a save with
    # nothing drawn that was not declared empty, is a slip either way
    if empty and clean:
        return {'ok': False, 'error': 'marked "no dog" but boxes are drawn'}
    if not empty and not clean:
        return {
            'ok': False,
            'error': 'draw a box round each dog, or mark '
            'the frame "no dog"'
        }
    row = manifest(batch)[_index(i)]
    rec = {
        'image_id': row['image_id'],
        'batch': batch,
        'boxes': clean,
        'empty': empty,
        'w': w,
        'h': h,
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
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'saved': rec}


def page_html(account=('', '')):
    """The page, with the identity strip spliced in like every other."""
    out = ANNOTATE_HTML
    for key, value in (('__ACCTCSS__', account[0]), ('__ACCOUNT__',
                                                     account[1])):
        out = out.replace(key, value)
    return out


ANNOTATE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotate &mdash; Street Dogs</title><style>
:root{--bg:#13151a;--panel:#1b2027;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;
--red:#ef5350;--num:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
[hidden]{display:none!important}
html,body{height:100%}
body{background:var(--bg);color:var(--tx);-webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  line-height:1.5;display:flex;flex-direction:column;padding:0 18px 14px}
a{color:inherit}
button,select{font:inherit}
header{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;
  padding:16px 0 12px;border-bottom:1px solid var(--bd)}
h1{font-size:19px;font-weight:660;letter-spacing:-.3px}
.sub{color:var(--dim);font-size:12.5px;margin-top:2px;max-width:78ch}
.sub b{color:var(--mut);font-weight:600}
.hdrend{display:flex;align-items:center;gap:12px;margin-left:auto}
.back{font-size:12px;color:var(--mut);text-decoration:none;
  border:1px solid var(--bd);border-radius:8px;padding:6px 11px}
.back:hover{color:var(--tx);border-color:rgba(130,140,150,.3)}
__ACCTCSS__

/* ── the bar: which batch, how far, and the four answers ── */
.bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 0}
.bsel{background:var(--panel2);color:var(--tx);border:1px solid var(--bd);
  border-radius:8px;padding:5px 9px;font-size:12.5px}
.bsel:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.prog{display:flex;align-items:center;gap:9px;font:400 12px var(--num);
  color:var(--mut);font-variant-numeric:tabular-nums}
.pbar{width:130px;height:5px;border-radius:3px;background:var(--panel2);
  overflow:hidden}
.pbar i{display:block;height:100%;background:var(--green);width:0}
.acts{display:flex;align-items:center;gap:7px;margin-left:auto;flex-wrap:wrap}
.btn{appearance:none;background:var(--panel2);color:var(--tx);
  border:1px solid var(--bd);border-radius:8px;padding:6px 12px;
  font-size:12.5px;cursor:pointer;line-height:1.3;white-space:nowrap}
.btn:hover{border-color:rgba(130,140,150,.4)}
.btn:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.btn:disabled{opacity:.4;cursor:default}
.btn kbd{font:500 10.5px var(--num);color:var(--dim);margin-left:6px}
.btn.go{background:rgba(67,181,129,.16);border-color:rgba(67,181,129,.5);
  font-weight:600}
.btn.go:hover{background:rgba(67,181,129,.24)}
.btn.none{border-color:rgba(232,166,69,.35)}
.btn.tog[aria-pressed="true"]{color:var(--acc);border-color:rgba(232,166,69,.45)}

/* ── the stage and the rail beside it ── */
.work{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 250px;
  gap:14px}
@media(max-width:900px){html,body{height:auto}
  .work{display:flex;flex-direction:column}
  .stage{height:62vh;min-height:300px;flex:none}
  .rail{overflow:visible}}
.stage{position:relative;min-height:0;border:1px solid var(--bd);
  border-radius:12px;background:#0b0d10;overflow:hidden;touch-action:none}
.stage svg{position:absolute;inset:0;width:100%;height:100%;display:block;
  cursor:crosshair;user-select:none}
.stage.pan svg,.stage.panning svg{cursor:grab}
.stage.panning svg{cursor:grabbing}
.stmsg{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:center;flex-direction:column;gap:8px;color:var(--mut);
  font-size:13.5px;text-align:center;padding:24px;pointer-events:none}
.stmsg b{color:var(--tx);font-size:15px}
.zoom{position:absolute;right:10px;bottom:10px;font:400 11px var(--num);
  color:var(--mut);background:rgba(11,13,16,.8);border:1px solid var(--bd);
  border-radius:6px;padding:2px 7px;pointer-events:none}
.bx{fill:rgba(232,166,69,.06);stroke:var(--acc);stroke-width:2;
  vector-effect:non-scaling-stroke;cursor:move}
.bx.on{fill:rgba(232,166,69,.12);stroke:#ffd27a;stroke-width:2.5}
.halo{fill:none;stroke:#000;stroke-opacity:.55;stroke-width:5;
  vector-effect:non-scaling-stroke;pointer-events:none}
.hd{fill:#ffd27a;stroke:#000;stroke-width:1;vector-effect:non-scaling-stroke}
.hd.nw,.hd.se{cursor:nwse-resize}
.hd.ne,.hd.sw{cursor:nesw-resize}
.hint{fill:none;stroke:#dfe3e8;stroke-opacity:.75;stroke-width:1.5;
  stroke-dasharray:6 5;vector-effect:non-scaling-stroke;pointer-events:none}

.rail{display:flex;flex-direction:column;gap:12px;min-height:0;overflow:auto}
.card{border:1px solid var(--bd);border-radius:12px;background:var(--panel);
  padding:11px 12px}
.card h2{font-size:11px;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;color:var(--dim);margin-bottom:7px}
.kv{font:400 11.5px/1.7 var(--num);color:var(--mut);word-break:break-all}
.kv b{color:var(--tx);font-weight:500}
.saved{margin-top:6px;font-size:12px;color:var(--green)}
.blist{list-style:none;display:flex;flex-direction:column;gap:4px}
.blist li{display:flex;align-items:center;gap:8px;font:400 12px var(--num);
  color:var(--mut);padding:3px 6px;border-radius:6px;cursor:pointer}
.blist li:hover,.blist li.on{background:var(--panel2);color:var(--tx)}
.blist li em{font-style:normal;color:var(--acc);min-width:24px}
.blist li button{margin-left:auto;appearance:none;background:none;border:0;
  color:var(--dim);cursor:pointer;font-size:14px;padding:0 4px}
.blist li button:hover{color:var(--red)}
.blist li button:focus-visible{outline:2px solid var(--acc);border-radius:4px}
.empty{font-size:12px;color:var(--dim)}
.keys{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-size:11.5px;
  color:var(--mut)}
.keys kbd{font:500 10.5px var(--num);color:var(--tx);background:var(--panel2);
  border:1px solid var(--bd);border-radius:4px;padding:0 5px;justify-self:start}
.status{min-height:20px;font-size:12.5px;color:var(--mut);padding-top:8px}
.status.bad{color:var(--red)}
.status.good{color:var(--green)}
</style></head><body>
<header>
  <div>
    <h1>Annotate</h1>
    <p class="sub" id="intro"><b>Box every dog you can see, however
    small</b> &mdash; zoom in to look &mdash; or mark the frame &ldquo;no
    dog&rdquo;. For a dog on a leash, <b>take the leash into the box</b>, as
    in Label Studio.</p>
  </div>
  <div class="hdrend"><a class="back" href="/">&larr; dashboard</a>__ACCOUNT__</div>
</header>

<div class="bar">
  <select class="bsel" id="bsel" aria-label="label batch"></select>
  <div class="prog"><span class="pbar"><i id="pfill"></i></span>
    <span id="ptext">&mdash;</span></div>
  <div class="acts">
    <button type="button" class="btn tog" id="bhint" aria-pressed="true"
      title="Show or hide the hint: where somebody thinks a dog is (H)">hint<kbd>H</kbd></button>
    <button type="button" class="btn" id="bundo" title="Undo (Ctrl+Z)">undo<kbd>Ctrl Z</kbd></button>
    <button type="button" class="btn" id="bback" title="Reopen the frame you last saved (B)">back<kbd>B</kbd></button>
    <button type="button" class="btn" id="bskip" title="Leave this one for later (S)">skip<kbd>S</kbd></button>
    <button type="button" class="btn none" id="bnone" title="There is no dog in this frame (N)">no dog<kbd>N</kbd></button>
    <button type="button" class="btn go" id="bsave" title="Save the boxes and go on (Enter)">save &amp; next<kbd>&#8629;</kbd></button>
  </div>
</div>

<div class="work">
  <div class="stage" id="stage">
    <svg id="st" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet"
      role="img" aria-label="the frame to annotate">
      <image id="stimg" preserveAspectRatio="none"></image>
      <g id="sthint"></g><g id="stbox"></g>
    </svg>
    <div class="stmsg" id="stmsg">loading&hellip;</div>
    <div class="zoom" id="zoom" hidden></div>
  </div>
  <aside class="rail">
    <div class="card"><h2>This frame</h2><div class="kv" id="finfo">&mdash;</div>
      <div class="saved" id="fsaved" hidden></div></div>
    <div class="card"><h2>Dogs boxed</h2><ul class="blist" id="blist"></ul>
      <p class="empty" id="bnone2">None yet &mdash; drag on the frame to draw one.</p></div>
    <div class="card"><h2>How</h2><div class="keys">
      <kbd>drag</kbd><span>draw a box</span>
      <kbd>wheel</kbd><span>zoom where you point</span>
      <kbd>space</kbd><span>hold and drag to move around</span>
      <kbd>F</kbd><span>fit the whole frame</span>
      <kbd>Del</kbd><span>delete the selected box</span>
      <kbd>&larr;&uarr;&rarr;&darr;</kbd><span>nudge it (Shift: 10px)</span>
      <kbd>Esc</kbd><span>deselect</span>
    </div></div>
  </aside>
</div>
<div class="status" id="status" role="status" aria-live="polite"></div>

<script>
(function(){
var $=function(id){return document.getElementById(id)};
var NS='http://www.w3.org/2000/svg';
var stage=$('stage'),svg=$('st'),img=$('stimg'),gh=$('sthint'),gb=$('stbox');
var F=null,boxes=[],sel=-1,hist=[],view=null,showHint=true,batch='',busy=false;
var prev=[],drag=null,spaceDown=false,armed='',dirty=false;

function say(t,kind){var s=$('status');s.textContent=t||'';
  s.className='status'+(kind?' '+kind:'')}
function api(url,opts){
  return fetch(url,opts).then(function(r){
    return r.json().catch(function(){return {}}).then(function(j){
      if(!r.ok&&!j.error)j.error='the server answered '+r.status;return j})})}
function snap(){hist.push(JSON.stringify(boxes));if(hist.length>200)hist.shift()}
function setDirty(v){dirty=v;window.onbeforeunload=v?function(){return true}:null}

/* ── where a pointer is, in the frame's own pixels ── */
function pt(e){var p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;
  var m=svg.getScreenCTM();return m?p.matrixTransform(m.inverse()):{x:0,y:0}}
function perPx(){return view.w/Math.max(1,svg.clientWidth)}

/* ── the view: a window onto the frame, the stage's own shape ── */
function stageAR(){return Math.max(1,svg.clientWidth)/Math.max(1,svg.clientHeight)}
function fit(){if(!F)return;var ar=stageAR(),w=F.w,h=F.h;
  if(w/h>ar)h=w/ar;else w=h*ar;
  view={x:(F.w-w)/2,y:(F.h-h)/2,w:w,h:h};apply()}
function clampView(){var mw=view.w*0.15,mh=view.h*0.15;
  if(view.w>=F.w)view.x=(F.w-view.w)/2;
  else view.x=Math.max(-mw,Math.min(F.w-view.w+mw,view.x));
  if(view.h>=F.h)view.y=(F.h-view.h)/2;
  else view.y=Math.max(-mh,Math.min(F.h-view.h+mh,view.y))}
function zoomAt(p,f){if(!F)return;var ar=view.w/view.h;
  var full=Math.max(F.w,F.h*ar);
  var nw=Math.max(16,Math.min(full*1.05,view.w*f)),nh=nw/ar;
  view.x=p.x-(p.x-view.x)*nw/view.w;view.y=p.y-(p.y-view.y)*nh/view.h;
  view.w=nw;view.h=nh;clampView();apply()}
function apply(){if(!view)return;
  svg.setAttribute('viewBox',[view.x,view.y,view.w,view.h].map(function(v){
    return +v.toFixed(2)}).join(' '));
  var ar=stageAR(),fitW=Math.max(F.w,F.h*ar);
  var z=$('zoom');z.hidden=false;z.textContent=(fitW/view.w).toFixed(1)+'×';
  draw()}

/* ── drawing what there is ── */
function el(tag,attrs){var n=document.createElementNS(NS,tag);
  for(var k in attrs)n.setAttribute(k,attrs[k]);return n}
function draw(){
  while(gb.firstChild)gb.removeChild(gb.firstChild);
  while(gh.firstChild)gh.removeChild(gh.firstChild);
  if(!F)return;
  if(showHint&&F.hint){var b=F.hint,cx=(b[0]+b[2])/2,cy=(b[1]+b[3])/2,
      r=Math.max(b[2]-b[0],b[3]-b[1])*1.6+12;
    var c=el('rect',{x:cx-r,y:cy-r,width:2*r,height:2*r,rx:r*0.25,'class':'hint'});
    var t=el('title',{});t.textContent=F.hint_title||'somebody thinks a dog is about here';
    c.appendChild(t);gh.appendChild(c)}
  var hs=7*perPx();
  boxes.forEach(function(b,k){
    var w=b[2]-b[0],h=b[3]-b[1];
    gb.appendChild(el('rect',{x:b[0],y:b[1],width:w,height:h,'class':'halo'}));
    var r=el('rect',{x:b[0],y:b[1],width:w,height:h,
      'class':'bx'+(k===sel?' on':''),'data-k':k});
    gb.appendChild(r);
    if(k===sel)[['nw',b[0],b[1]],['ne',b[2],b[1]],['sw',b[0],b[3]],['se',b[2],b[3]]]
      .forEach(function(c){gb.appendChild(el('rect',{x:c[1]-hs/2,y:c[2]-hs/2,
        width:hs,height:hs,'class':'hd '+c[0],'data-h':c[0]}))})});
  rail()}
function rail(){
  var ul=$('blist');ul.textContent='';
  boxes.forEach(function(b,k){
    var li=document.createElement('li');if(k===sel)li.className='on';
    var em=document.createElement('em');em.textContent='#'+(k+1);li.appendChild(em);
    li.appendChild(document.createTextNode(Math.round(b[2]-b[0])+'×'+
      Math.round(b[3]-b[1])+' px'));
    var x=document.createElement('button');x.type='button';x.textContent='×';
    x.setAttribute('aria-label','delete box '+(k+1));
    x.addEventListener('click',function(e){e.stopPropagation();snap();
      boxes.splice(k,1);sel=-1;setDirty(true);draw()});
    li.appendChild(x);
    li.addEventListener('click',function(){sel=k;draw()});
    ul.appendChild(li)});
  $('bnone2').hidden=boxes.length>0;
  $('bundo').disabled=!hist.length}

/* ── pointer: draw, select, move, resize, pan ── */
svg.addEventListener('pointerdown',function(e){
  if(!F||busy)return;
  var p=pt(e),t=e.target;
  armed='';
  if(e.button===1||e.button===2||spaceDown){
    drag={mode:'pan',cx:e.clientX,cy:e.clientY,v:{x:view.x,y:view.y}};
    stage.classList.add('panning')}
  else if(e.button===0&&t.getAttribute('data-h')){
    snap();drag={mode:'resize',k:sel,h:t.getAttribute('data-h')}}
  else if(e.button===0&&t.getAttribute('data-k')!==null&&t.getAttribute('data-k')!==undefined
          &&t.classList.contains('bx')){
    sel=+t.getAttribute('data-k');snap();
    drag={mode:'move',k:sel,p:p,b:boxes[sel].slice()};draw()}
  else if(e.button===0){
    snap();boxes.push([p.x,p.y,p.x,p.y]);sel=boxes.length-1;
    drag={mode:'draw',k:sel,p:p};draw()}
  else return;
  svg.setPointerCapture(e.pointerId);e.preventDefault()});
svg.addEventListener('pointermove',function(e){
  if(!drag)return;
  if(drag.mode==='pan'){var s=perPx();
    view.x=drag.v.x-(e.clientX-drag.cx)*s;view.y=drag.v.y-(e.clientY-drag.cy)*s;
    clampView();apply();return}
  var p=pt(e),b=boxes[drag.k];if(!b)return;
  var X=Math.max(0,Math.min(F.w,p.x)),Y=Math.max(0,Math.min(F.h,p.y));
  if(drag.mode==='draw'){b[0]=Math.min(drag.p.x,X);b[2]=Math.max(drag.p.x,X);
    b[1]=Math.min(drag.p.y,Y);b[3]=Math.max(drag.p.y,Y)}
  else if(drag.mode==='move'){var dx=p.x-drag.p.x,dy=p.y-drag.p.y,o=drag.b;
    dx=Math.max(-o[0],Math.min(F.w-o[2],dx));dy=Math.max(-o[1],Math.min(F.h-o[3],dy));
    b[0]=o[0]+dx;b[2]=o[2]+dx;b[1]=o[1]+dy;b[3]=o[3]+dy}
  else if(drag.mode==='resize'){
    if(drag.h.indexOf('w')>=0)b[0]=X;else b[2]=X;
    if(drag.h.indexOf('n')>=0)b[1]=Y;else b[3]=Y;
    if(b[0]>b[2]){var a=b[0];b[0]=b[2];b[2]=a;drag.h=drag.h.replace(/[we]/,function(c){return c==='w'?'e':'w'})}
    if(b[1]>b[3]){var q=b[1];b[1]=b[3];b[3]=q;drag.h=drag.h.replace(/[ns]/,function(c){return c==='n'?'s':'n'})}}
  setDirty(true);draw()});
function endDrag(){
  if(!drag)return;
  if(drag.mode==='draw'){var b=boxes[drag.k];
    /* a click is not a box: under 3 screen px either way, it goes */
    if(!b||(b[2]-b[0])/perPx()<3||(b[3]-b[1])/perPx()<3){
      boxes.splice(drag.k,1);hist.pop();sel=-1}}
  if(drag.mode==='pan')stage.classList.remove('panning');
  drag=null;draw()}
svg.addEventListener('pointerup',endDrag);
svg.addEventListener('pointercancel',endDrag);
svg.addEventListener('contextmenu',function(e){e.preventDefault()});
svg.addEventListener('wheel',function(e){if(!F)return;e.preventDefault();
  zoomAt(pt(e),Math.pow(1.0018,e.deltaY))},{passive:false});
addEventListener('resize',function(){if(!F||!view)return;
  /* a whole frame stays whole; a zoomed view keeps its zoom and centre */
  if(view.w>=F.w-1&&view.h>=F.h-1){fit();return}
  {var c={x:view.x+view.w/2,y:view.y+view.h/2};
  var ar=stageAR();view.h=view.w/ar;view.x=c.x-view.w/2;view.y=c.y-view.h/2;clampView();apply()}});

/* ── keys ── */
addEventListener('keydown',function(e){
  var tag=(e.target&&e.target.tagName)||'';
  if(tag==='SELECT'||tag==='INPUT'||tag==='TEXTAREA')return;
  var k=e.key;
  if(k===' '){if(!spaceDown){spaceDown=true;stage.classList.add('pan')}e.preventDefault();return}
  if((e.ctrlKey||e.metaKey)&&(k==='z'||k==='Z')){e.preventDefault();undo();return}
  if(e.ctrlKey||e.metaKey||e.altKey)return;
  if(k==='Enter'){e.preventDefault();saveIt(false)}
  else if(k==='n'||k==='N')saveIt(true);
  else if(k==='s'||k==='S')skip();
  else if(k==='b'||k==='B')back();
  else if(k==='h'||k==='H')toggleHint();
  else if(k==='f'||k==='F'||k==='0')fit();
  else if(k==='+'||k==='=')zoomAt({x:view.x+view.w/2,y:view.y+view.h/2},0.8);
  else if(k==='-'||k==='_')zoomAt({x:view.x+view.w/2,y:view.y+view.h/2},1.25);
  else if((k==='Delete'||k==='Backspace')&&sel>=0){e.preventDefault();snap();
    boxes.splice(sel,1);sel=-1;setDirty(true);draw()}
  else if(k==='Escape'){sel=-1;draw()}
  else if(sel>=0&&k.indexOf('Arrow')===0){e.preventDefault();
    var d=e.shiftKey?10:1,b=boxes[sel],dx=0,dy=0;
    if(k==='ArrowLeft')dx=-d;else if(k==='ArrowRight')dx=d;
    else if(k==='ArrowUp')dy=-d;else dy=d;
    dx=Math.max(-b[0],Math.min(F.w-b[2],dx));dy=Math.max(-b[1],Math.min(F.h-b[3],dy));
    snap();b[0]+=dx;b[2]+=dx;b[1]+=dy;b[3]+=dy;setDirty(true);draw()}});
addEventListener('keyup',function(e){if(e.key===' '){spaceDown=false;
  stage.classList.remove('pan')}});

function undo(){if(!hist.length)return;boxes=JSON.parse(hist.pop());sel=-1;
  setDirty(true);draw()}
function toggleHint(){showHint=!showHint;$('bhint').setAttribute('aria-pressed',
  showHint?'true':'false');draw()}

/* ── frames ── */
function msg(html){var m=$('stmsg');if(html){m.innerHTML=html;m.hidden=false}else m.hidden=true}
function info(){
  if(!F){$('finfo').textContent='—';$('fsaved').hidden=true;return}
  $('finfo').innerHTML='';
  [['image',F.image_id],['size',F.w+' × '+F.h],['batch',F.batch]].forEach(function(r){
    var d=document.createElement('div');d.appendChild(document.createTextNode(r[0]+' '));
    var b=document.createElement('b');b.textContent=r[1];d.appendChild(b);$('finfo').appendChild(d)});
  var s=$('fsaved');
  if(F.label){s.hidden=false;s.textContent='labelled by '+(F.label.by||'?')+
    (F.label.empty?' as no dog':' — '+F.label.boxes.length+' dog'+
    (F.label.boxes.length===1?'':'s'))+'. Saving again replaces it.'}
  else s.hidden=true}
function progress(p){if(!p)return;var pct=p.total?100*p.done/p.total:0;
  $('pfill').style.width=pct.toFixed(1)+'%';
  $('ptext').textContent=p.done+' of '+p.total+' labelled'+
    (p.mine!=null?' · '+p.mine+' by you':'');
  var o=$('bsel').querySelector('option[value="'+batch+'"]');
  if(o)o.textContent=batch+' — '+p.done+'/'+p.total}
function show(d,note){
  if(!d){F=null;msg('<b>That frame could not be opened.</b>');return}
  if(d.error){F=null;msg('<b>'+esc(d.error)+'</b>');return}
  if(d.done){F=null;boxes=[];draw();info();progress(d.progress);
    msg('<b>Every frame in this batch is labelled.</b>Pick another batch above, or ask for a new one.');
    return}
  F=d;boxes=d.label&&!d.label.empty?d.label.boxes.map(function(b){return b.slice()}):[];
  sel=-1;hist=[];setDirty(false);armed='';
  msg('loading the frame…');
  img.onload=function(){msg('')};img.onerror=function(){msg('<b>The picture would not load.</b>')};
  img.setAttribute('width',d.w);img.setAttribute('height',d.h);
  img.setAttribute('href','/annotate/frame.jpg?batch='+encodeURIComponent(d.batch)+'&i='+d.i);
  if(d.intro)$('intro').textContent=d.intro;
  fit();info();progress(d.progress);say(note||'',note?'good':'')}
function esc(s){return String(s).replace(/[&<>"]/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
function next(after,note){if(!batch)return;busy=true;
  api('/api/annotate/next?batch='+encodeURIComponent(batch)+
      (after?'&after='+encodeURIComponent(after):''))
    .then(function(d){busy=false;show(d,note)}).catch(function(){busy=false;
      msg('<b>The server did not answer.</b>')})}
function open(b,i){busy=true;api('/api/annotate/frame?batch='+encodeURIComponent(b)+'&i='+i)
  .then(function(d){busy=false;show(d)}).catch(function(){busy=false})}
function saveIt(empty){
  if(!F||busy)return;
  if(empty&&boxes.length){say('There are boxes drawn — delete them first, or save them.','bad');return}
  if(!empty&&!boxes.length){say('Nothing is drawn — box each dog, or press N if there is none.','bad');return}
  busy=true;say('saving…');
  api('/api/annotate/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({batch:F.batch,i:F.i,empty:!!empty,
      boxes:empty?[]:boxes.map(function(b){return b.map(function(v){return +v.toFixed(1)})})})})
    .then(function(j){busy=false;
      if(!j.ok){say(j.error||'not saved','bad');return}
      prev.push({batch:F.batch,i:F.i});setDirty(false);
      next(F.image_id,empty?'Saved: no dog.':'Saved: '+boxes.length+' dog'+
        (boxes.length===1?'':'s')+'.')})
    .catch(function(){busy=false;say('not saved — the server did not answer','bad')})}
/* leaving a frame with boxes drawn and not saved takes the key twice */
function leaving(key){
  if(dirty&&boxes.length&&armed!==key){armed=key;
    say('You have unsaved boxes — press '+key+' again to leave them.','bad');
    return false}
  armed='';setDirty(false);return true}
function skip(){if(!F||busy||!leaving('S'))return;next(F.image_id)}
function back(){if(busy||!prev.length||!leaving('B'))return;
  var p=prev.pop();open(p.batch,p.i)}
$('bsave').addEventListener('click',function(){saveIt(false)});
$('bnone').addEventListener('click',function(){saveIt(true)});
$('bskip').addEventListener('click',skip);
$('bback').addEventListener('click',back);
$('bundo').addEventListener('click',undo);
$('bhint').addEventListener('click',toggleHint);

/* ── batches ── */
function loadBatches(){
  api('/api/annotate/batches').then(function(j){
    var sel=$('bsel'),list=(j&&j.batches)||[];sel.textContent='';
    if(!list.length){msg('<b>There is nothing to label yet.</b>A batch is made with tools/detect/pick_label_frames.py plan.');return}
    var want=new URLSearchParams(location.search).get('batch');
    try{want=want||localStorage.getItem('annotate.batch')}catch(e){}
    list.forEach(function(b){var o=document.createElement('option');o.value=b.batch;
      o.textContent=b.batch+' — '+b.done+'/'+b.total;sel.appendChild(o)});
    var pick=list.filter(function(b){return b.batch===want})[0]||
      list.filter(function(b){return b.done<b.total})[0]||list[0];
    sel.value=pick.batch;batch=pick.batch;next()})
    .catch(function(){msg('<b>The server did not answer.</b>')})}
$('bsel').addEventListener('change',function(){batch=this.value;this.blur();
  try{localStorage.setItem('annotate.batch',batch)}catch(e){}prev=[];next()});
loadBatches();
})();
</script>
</body></html>
"""
