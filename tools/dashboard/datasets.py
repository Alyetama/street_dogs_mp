#!/usr/bin/env python3
"""The datasets page: what has been built, what trained on it, what is inside.

Serving side of tools/detect/dataset_index.py. That module answers "which
datasets are on this machine and what is in them"; everything here turns those
answers into a page and cuts the thumbnails a grid of a hundred photographs
needs.

WHY THIS IS NOT A FILE MANAGER. A dataset directory can be opened in a
terminal. What cannot is the line between a directory and the training history:
thirty-eight runs trained on a Label Studio export that has since been deleted,
and the only place that fact exists is the runs' own `data` strings read
against the disk. So every row carries the runs that trained on it, and a
dataset whose directory is gone is a row that says so rather than a row that
is missing.

NOTHING HERE WRITES INTO A DATASET. The only file this module creates is a
thumbnail, under data/dashboard/dataset_thumbs/, and deleting that directory
costs one slow page. Datasets are training sets that took hours to build and
hold human annotations; this page opens them and never edits them.

EVERY PATH ON THIS PAGE CAME OFF A QUERY STRING. A key and a relative path
arrive from the client and turn into a file read, which makes traversal the
whole security surface. There is exactly one door -- _source() -- and it goes
through dataset_index.resolve(), which realpaths both sides and refuses
anything that lands outside a dataset root -- this dataset's own, or another
indexed one's, because symlink-assembled exports point their files into the
set they were cut from. A request for a .yaml, a .cache or a label file
through an image route is a 404 rather than a download.
"""

import atexit
import hashlib
import json
import os
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DETECT = os.path.join(REPO, 'tools', 'detect')
if _DETECT not in sys.path:
    sys.path.insert(0, _DETECT)
import dataset_index as index                                   # noqa: E402

# What the image routes will serve, and what each is called on the wire. The
# map IS the allow-list: an extension that is not in it never reaches
# resolve(), so a dataset's dataset.yaml, its ultralytics *.cache and its label
# .txt files cannot be fetched through /datasets/image however they are named.
MIME = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
        '.webp': 'image/webp', '.bmp': 'image/bmp', '.gif': 'image/gif',
        '.tif': 'image/tiff', '.tiff': 'image/tiff'}

THUMB_DIR = os.path.join(REPO, 'data', 'dashboard', 'dataset_thumbs')
# Long side. The tiles are ~170 px, so this is a shade over 2x for a retina
# screen and still ~9 KB a piece; a page of 120 is about a megabyte.
THUMB_PX = 240
# The cache is bounded by COUNT rather than by anything about the datasets,
# because a dataset can be walked page by page and there is no pool to prune
# against the way the review page's crops have one. Twenty thousand cuts is
# roughly 180 MB. Pruning takes it back to KEEP so it does not run every time.
THUMB_MAX = 20000
THUMB_KEEP = 16000
# How many thumbnails one warm pass will cut: the largest page the interface
# offers, and never more. A pass reads the same disk a live training run is
# writing to, so it is bounded by what the reader is about to look at rather
# than by how much of a folder could be cut.
WARM_MAX = 240
# A whole original is read into memory before it is written to the socket, so
# there has to be a ceiling somewhere. Nothing in either repo's datasets comes
# close (the largest full-resolution frames here are ~2 MB); this exists so a
# 400 MB tiff dropped in a folder cannot take the dashboard's memory with it.
FULL_MAX = 64 << 20

# The sizes the page offers. Anything else is somebody's URL rather than a
# choice the interface made -- dataset_index clamps at 1000, but a page of a
# thousand full tiles is a megabyte of thumbnails nobody asked for.
PAGE_SIZES = (60, 120, 240)
PAGE_DEFAULT = 120


def page_size(v, default=PAGE_DEFAULT):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return v if v in PAGE_SIZES else default


# ── the one door to the filesystem ──────────────────────────────────────────

def _source(key, rel):
    """An image inside a dataset, as (absolute path, content type).

    Answers (None, None) for anything it will not serve. Both image routes go
    through here and nothing else opens a file. The name the client sent is
    checked first, so a request for a dataset's descriptor or its label files
    is refused without so much as a stat -- the routes serve pictures, and
    anything else being downloadable from a directory the client can name
    would be a file server nobody asked for. resolve() then does the traversal
    work: it realpaths both sides, refuses '..', an absolute rel and a symlink
    that leaves every indexed dataset root, and answers None rather than
    raising.

    THE ALLOW-LIST DECIDES ON THE FILE THAT WILL BE OPENED, not on the name it
    was asked for by. resolve() hands back a realpath, and a dataset assembled
    with symlinks -- which is how exports and dedup'd builds avoid copying --
    can hold an images/train/x.jpg pointing at the dataset.yaml beside it.
    Checking only the request string served that yaml as image/jpeg while the
    same file asked for by its own name was correctly refused, so the extension
    that counts is the resolved one and the request's is a cheap first no.
    """
    if not MIME.get(os.path.splitext(str(rel or ''))[1].lower()):
        return None, None
    path = index.resolve(key, rel)
    if not path or not os.path.isfile(path):
        return None, None
    ctype = MIME.get(os.path.splitext(path)[1].lower())
    if not ctype:
        return None, None
    return path, ctype


# ── thumbnails ──────────────────────────────────────────────────────────────

def _cache_name(path, st):
    """The cache file name for one source image.

    The source's size and modification time are in the hash, so a file
    rewritten under the same name is a different thumbnail rather than a stale
    one served for ever. Nothing in either repo rewrites a dataset image in
    place -- a rebuild writes a new directory -- but the cache outlives the
    dataset that filled it, and a name that only carried the path would hand
    the old picture to whatever landed at that path next.
    """
    seed = f'{path}\0{st.st_mtime_ns}\0{st.st_size}'
    return hashlib.sha1(seed.encode('utf-8', 'replace')).hexdigest() + '.jpg'


def _cut(src, out):
    """Cut one thumbnail. Returns the path, or None if it would not decode."""
    # One temp name per writer. A shared '<name>.part' is the bug the review
    # page's hq_crop() carried: the browser fetches six tiles at a time over
    # the same names the warmer is cutting, and the second os.replace hit a
    # file the first had already moved -- so the cut came back None, the
    # handler 404'd, and the tile stayed alt text for the life of the page.
    tmp = '%s.%d.%d.part' % (out, os.getpid(), threading.get_ident())
    try:
        from PIL import Image
        im = Image.open(src)
        # draft() lets the JPEG decoder scale while it decodes instead of
        # after, which is the difference between 22 ms and 7.5 ms per tile
        # here -- a page of 120 in under a second rather than three.
        im.draft('RGB', (THUMB_PX, THUMB_PX))
        im = im.convert('RGB')
        im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
        os.makedirs(THUMB_DIR, exist_ok=True)
        im.save(tmp, 'JPEG', quality=80, optimize=True)
        os.replace(tmp, out)
        return out
    except Exception as e:
        # the prune below leaves '.part' names alone, so a half-written temp
        # would sit in the cache directory for ever if nobody dropped it here
        try:
            os.remove(tmp)
        except OSError:
            pass
        sys.stderr.write('dataset thumb(%s): %s\n' % (src, e))
        return None


def thumb_path(key, rel):
    """Path to the cached thumbnail for one image, cutting it if it is new.

    None covers every way this can fail -- the path did not resolve, the file
    has gone since the listing was taken, it is not an image, it will not
    decode -- and the caller answers 404 to all of them. The client draws a
    tile that says the picture would not open, which is a fact about the
    dataset and belongs on the page.
    """
    src, _ = _source(key, rel)
    if not src:
        return None
    try:
        st = os.stat(src)
    except OSError:
        return None
    out = os.path.join(THUMB_DIR, _cache_name(src, st))
    if os.path.exists(out):
        return out
    return _cut(src, out)


def thumb(key, rel):
    """(jpeg bytes, content type) for a thumbnail, or (None, None)."""
    p = thumb_path(key, rel)
    if not p:
        return None, None
    try:
        with open(p, 'rb') as fh:
            return fh.read(), 'image/jpeg'
    except OSError:
        return None, None


def full(key, rel):
    """(the original file's bytes, its content type), or (None, None).

    The original is served exactly as it sits on disk. Nothing is re-encoded:
    the point of clicking a tile is to see the picture the model was trained
    on, and a re-compressed copy of it is a different picture.
    """
    src, ctype = _source(key, rel)
    if not src:
        return None, None
    try:
        if os.path.getsize(src) > FULL_MAX:
            return None, None
        with open(src, 'rb') as fh:
            return fh.read(), ctype
    except OSError:
        return None, None      # unlinked between the listing and the click


# ── warming a page ahead of the reader ──────────────────────────────────────
# Cold, a tile costs ~7.5 ms; a page of 120 is a second of decoding spread over
# however many connections the browser opens. That is fine once. Cutting the
# NEXT page while this one is being read is what makes every page after the
# first instant.

_warm_lock = threading.Lock()
_warm_busy = False
# The warmer outlives the request that started it, which is the point -- but it
# must not outlive the interpreter. This is the second bug the review page's
# warmer carried: a daemon thread still holding files when the process exits
# takes duckdb's static teardown down with it, so a script that imports the
# dashboard, renders one page and returns dies of SIGABRT reporting failure
# about work that succeeded.
_warm_stop = threading.Event()
_warm_thread = None


def _warm_shutdown():
    """Ask the warmer to stop, and give it a moment to notice."""
    _warm_stop.set()
    t = _warm_thread
    if t is not None and t.is_alive():
        t.join(timeout=2.0)


atexit.register(_warm_shutdown)


def warm(key, rels):
    """Cut a set of thumbnails in the background, one worker at a time.

    One worker because this reads the same disks a live training run is
    writing to, and because a page turn is cheap enough that two of them
    racing would buy nothing. A second call while one is running is dropped
    rather than queued: the reader has moved on, and the pass in flight is
    already cutting the folder they are looking at.
    """
    global _warm_busy, _warm_thread
    with _warm_lock:
        if _warm_busy:
            return
        todo = list(rels)[:WARM_MAX]
        if not todo:
            return
        _warm_busy = True

    def work():
        global _warm_busy
        try:
            for r in todo:
                if _warm_stop.is_set():
                    return
                try:
                    thumb_path(key, r)
                except Exception:
                    pass
            if not _warm_stop.is_set():
                _prune()
        finally:
            _warm_busy = False

    _warm_thread = threading.Thread(target=work, daemon=True)
    _warm_thread.start()


def _prune():
    """Hold the thumbnail cache to a bounded number of files.

    Oldest first, and only when the cache has grown past THUMB_MAX, so paging
    through one big folder does not spend its time deleting the thumbnails of
    the folder before it. Best effort throughout -- every one of these files
    can be cut again.
    """
    try:
        names = os.listdir(THUMB_DIR)
    except OSError:
        return
    if len(names) <= THUMB_MAX:
        return
    rows = []
    for n in names:
        # an in-flight '<name>.<pid>.<tid>.part' belongs to a writer; removing
        # one is the shared-temp bug again, one directory along
        if n.endswith('.part'):
            continue
        p = os.path.join(THUMB_DIR, n)
        try:
            rows.append((os.stat(p).st_mtime, p))
        except OSError:
            continue
    rows.sort()
    for _, p in rows[:max(0, len(rows) - THUMB_KEEP)]:
        try:
            os.remove(p)
        except OSError:
            pass


# ── what the page asks for ──────────────────────────────────────────────────

def api_list(refresh=False):
    """Every dataset, newest first, with the runs that trained on each.

    `error` is the index's own: when the dashboard module will not import,
    discovery falls back to the disk scan alone and every dataset comes back
    with no runs attached. That is a page missing the one column it exists
    for, so it is said out loud rather than shown as "0 runs" everywhere.
    """
    rows = index.datasets(refresh=bool(refresh))
    return {'datasets': rows, 'error': index._INDEX['error'],
            'roots': index.scan_roots()}


def api_tree(key):
    """The folder structure of one dataset: directories, counts and bytes.

    Passed through from the index, including its two failure answers -- an
    unknown key and a root that has gone since the list was taken. The second
    is the ordinary one: the list is up to 20 seconds old and a rebuild moves
    a directory in less than that.
    """
    return index.tree(key)


def api_files(key, rel='', page=0, n=PAGE_DEFAULT):
    """One page of the files in one folder, and a warm pass for the next.

    The page just served is not warmed. The browser is already fetching those
    hundred-odd tiles the moment this answer lands, so a warmer racing it
    would decode every one of them twice; the page AFTER this one is the one
    that would otherwise be cold, and cutting it now is what makes the next
    click instant.
    """
    n = page_size(n)
    got = index.listing(key, rel, page, n)
    if got.get('ok'):
        ahead = index.listing(key, got['rel'], got['page'] + 1, n)
        if ahead.get('ok'):
            warm(key, [f['rel'] for f in ahead['files'] if f['image']])
    return got


_WORDS = {}


def _status_words():
    """The training tracker's own wording for a run's state.

    Read off the dashboard rather than copied, so a run reads the same here as
    it does in the training panel -- 'early-stopped', not 'early_stopped'. A
    second copy of these five words would drift the first time either changed.

    Imported lazily and by path, the way audit.py reaches _grid_roots(): the
    dashboard imports THIS module, so a top-level import would be a cycle. Held
    afterwards because they cannot change inside one process, and because the
    path insert must not run once per page render.
    """
    if not _WORDS:
        try:
            here = os.path.join(REPO, 'tools', 'dashboard')
            if here not in sys.path:
                sys.path.insert(0, here)
            import dashboard
            _WORDS.update({k: v[0] for k, v in dashboard.TRK_STATUS.items()})
        except Exception:
            # the page then prints the raw status, which is still true
            pass
    return dict(_WORDS)


# ── the page ────────────────────────────────────────────────────────────────
# A sibling of /review and /audit, wearing the same clothes: same palette, same
# controls, same type scale. Three panes, and it should never be in doubt which
# is which -- what has been built, what is inside the one you opened, and the
# pictures in the folder you picked.
#
# The dataset list carries its runs because that is the whole reason this page
# is not a file manager. "dogbin_v5, 3,174 images" is something ls can say;
# "and three runs trained on it, the last of which early-stopped" is not.
DATASETS_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Datasets</title><style>
:root{--bg:#13151a;--panel:#1b2027;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;
--red:#ef5350;
/* Numbers get their own face, for the same reason the audit page's do: every
   count here is read by comparison -- 1,333 against 1,199, 2,416 against 480 --
   and in a proportional face the digits move under the eye between rows. */
--num:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
/* The training tracker's two validated marks on these panels, plus its green.
   A class bar needs one colour per class and inventing a second palette for
   this page is exactly the thing the rest of the dashboard does not do. */
--c1:#c2872e;--c2:#5b93cf;--c3:#43b581;--c4:#8b7fd4}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);-webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  line-height:1.5;padding:0 22px 60px}
.wrap{max-width:1720px;margin:0 auto}
a{color:inherit}
header{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;
  padding:22px 0 16px;border-bottom:1px solid var(--bd);margin-bottom:16px}
h1{font-size:20px;font-weight:660;letter-spacing:-.3px}
.sub{color:var(--dim);font-size:12.5px;margin-top:3px;max-width:62ch}
.back{font-size:12px;color:var(--mut);text-decoration:none;margin-left:auto;
  border:1px solid var(--bd);border-radius:8px;padding:6px 11px}
.back:hover{color:var(--tx);border-color:rgba(130,140,150,.3)}
.banner{background:rgba(232,166,69,.09);border:1px solid rgba(232,166,69,.3);
  border-radius:11px;padding:9px 14px;margin-bottom:14px;font-size:12px;
  color:var(--acc)}
.banner[hidden]{display:none}
/* ── the three panes ── */
.cols{display:grid;grid-template-columns:322px minmax(0,1fr);gap:16px;
  align-items:start}
/* Stacked, not a sidebar. The structure is read once and then consulted;
   the pictures are the work, and they run for pages. A 290px column spent a
   screen-tall strip of every row on an element about 400px tall -- the grid
   paid for it on every scroll. So the structure lies as a band across the
   top: balance on the left, the folder tree flowing into columns beside it. */
.panes{display:grid;grid-template-columns:minmax(0,1fr);gap:16px;
  align-items:start;margin-top:14px}
/* 300px, one number: the balance is a fixed label over a fixed-shape bar, and
   a track that answers the viewport (the minmax this used to be) re-seated
   the train/val rows with the window size. The slack all goes to the tree,
   which absorbs it in whole columns. */
.struct{display:grid;grid-template-columns:300px minmax(0,1fr);
  grid-template-rows:auto minmax(0,1fr)}
.struct .chead{grid-column:1/-1}
.struct .balance{border-bottom:0;border-right:1px solid var(--bd)}
/* `display:grid` beats the hidden attribute's `display:none`, so without this
   the two empty panes sat under "pick a dataset" before anything was opened. */
.panes[hidden]{display:none}
.card{background:var(--panel);border:1px solid var(--bd);border-radius:14px}
.chead{display:flex;gap:10px;align-items:baseline;padding:11px 14px;
  border-bottom:1px solid var(--bd);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.07em;color:var(--dim)}
.chead .n{margin-left:auto;font-family:var(--num);letter-spacing:0;
  text-transform:none;font-size:11px;color:var(--mut)}
/* ── the dataset list ── */
/* Its own scroller, so opening a dataset with two dozen siblings does not push
   the pictures off the bottom of the window. */
.dslist{max-height:calc(100vh - 190px);overflow:auto;padding:6px}
.ds{display:block;width:100%;text-align:left;background:transparent;border:0;
  border-radius:10px;padding:9px 10px;cursor:pointer;font-family:inherit;
  color:inherit}
.ds+.ds{margin-top:2px}
.ds:hover{background:var(--panel2)}
.ds.on{background:rgba(232,166,69,.12);box-shadow:inset 0 0 0 1px
  rgba(232,166,69,.32)}
.dstop{display:flex;gap:7px;align-items:baseline}
.dsname{font-size:13px;font-weight:600;letter-spacing:-.1px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
/* Only drawn when two rows share a name, so it is never noise on a page where
   every name is already its own. */
.dsname .dsin{color:var(--dim);font-weight:400;font-size:11px}
.ds.on .dsname{color:var(--acc)}
.chip{font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;
  border:1px solid var(--bd);border-radius:5px;padding:1px 5px;
  color:var(--dim);flex:none}
.chip.detect{border-color:rgba(91,147,207,.4);color:#7fb0e0}
.chip.classify{border-color:rgba(194,135,46,.42);color:#d09b47}
/* Blocks, not inline spans. A button's children are inline by default, so the
   counts and the run line ran together into one wrapped paragraph and every
   row read "... 11.1 h ago1 run dogdet_v3_002". */
.dsnums{display:block;font-size:11.5px;color:var(--mut);font-family:var(--num);
  font-variant-numeric:tabular-nums;margin-top:2px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.dsruns{display:block;font-size:11.5px;color:var(--dim);margin-top:2px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dsruns b{color:var(--mut);font-weight:600}
.live{color:var(--green)}
/* A dataset whose directory has gone is not hidden and not greyed into
   nothing: thirty-eight runs trained on one of them, and that is the row that
   explains what those runs were. */
.ds.gone .dsname{color:var(--mut)}
.ds.gone .dsnums{color:var(--red);font-family:inherit}
/* ── folding the list away ─────────────────────────────────────────────────
   The list is how a dataset gets PICKED; once one is open it is 322px of
   names the reader already chose between, on every screenful of pictures.
   Folded, the card narrows to a rail wide enough to say what it is holding
   and to bring it back. Class-driven display, never the hidden attribute:
   .chead carries display:flex, which beats [hidden] -- the exact bug this
   codebase has shipped twice. Open by default; the choice is remembered the
   way the page already remembers the dataset and the page size. */
.cols.folded{grid-template-columns:44px minmax(0,1fr)}
.side.folded .chead,.side.folded .dslist{display:none}
.rail{display:none}
.side.folded .rail{display:block;padding:6px}
.rail button{width:100%;background:transparent;border:0;border-radius:8px;
  cursor:pointer;font-family:inherit;color:var(--dim);font-size:11.5px;
  padding:10px 4px;writing-mode:vertical-rl;letter-spacing:.06em;
  display:flex;align-items:center;gap:8px}
.rail button:hover{background:var(--panel2);color:var(--tx)}
.rail .n{font-family:var(--num);color:var(--mut)}
.fold{background:transparent;border:0;color:var(--dim);cursor:pointer;
  font-size:13px;padding:2px 6px;border-radius:6px;line-height:1}
.fold:hover{background:var(--panel2);color:var(--tx)}
.rescan{background:var(--panel2);border:1px solid var(--bd);color:var(--dim);
  border-radius:8px;padding:4px 9px;font-size:11px;cursor:pointer;
  font-family:inherit;margin-left:10px;text-transform:none;letter-spacing:0}
.rescan:hover:not(:disabled){color:var(--tx);
  border-color:rgba(130,140,150,.32)}
.rescan:disabled{opacity:.45;cursor:default}
/* ── the opened dataset ── */
.dshead{padding:15px 18px}
.dstitle{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.dstitle h2{font-size:17px;font-weight:640;letter-spacing:-.3px}
.dspath{font-size:11.5px;color:var(--dim);font-family:var(--num);
  word-break:break-all}
.facts{display:flex;gap:20px;flex-wrap:wrap;margin-top:10px;font-size:12px;
  color:var(--mut)}
.facts b{color:var(--tx);font-weight:640;font-family:var(--num);
  font-variant-numeric:tabular-nums}
.gonenote{margin-top:10px;font-size:12.5px;color:var(--red)}
.runs{margin-top:13px;border-top:1px solid var(--bd);padding-top:11px}
.runs .rlab{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
  color:var(--dim);margin-bottom:6px}
/* Thirty-eight runs trained on the one dataset that is gone, and printing all
   of them pushed the folders and the pictures off the bottom of the window on
   the row where the run list matters most. */
.runlist{max-height:214px;overflow:auto}
.run{display:grid;grid-template-columns:minmax(0,1fr) 78px 130px 108px;
  gap:12px;font-size:12px;color:var(--mut);padding:4px 0;align-items:baseline;
  font-variant-numeric:tabular-nums}
.run+.run{border-top:1px solid rgba(130,140,150,.07)}
.run .rname{color:var(--tx);overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
/* nowrap and wide enough for the longest of them: "61.7 days ago" broke over
   two lines and every run row in the list was a different height */
.run .rwhen{text-align:right;color:var(--dim);font-family:var(--num);
  white-space:nowrap}
.run .rst.live{color:var(--green)}
.built{margin-top:12px;border-top:1px solid var(--bd);padding-top:10px}
.built summary{list-style:none;cursor:pointer;font-size:11.5px;
  color:var(--dim)}
.built summary::-webkit-details-marker{display:none}
.built summary::after{content:' \25b8'}
.built[open] summary::after{content:' \25be'}
.built summary:hover{color:var(--tx)}
.kv{display:grid;grid-template-columns:auto minmax(0,1fr);gap:3px 14px;
  margin-top:5px;font-size:11.5px}
.kv dt{color:var(--dim)}
.kv dd{color:var(--mut);font-family:var(--num);word-break:break-all}
/* The descriptor and the manifest are two files and they use some of the same
   words: dogdet_v3's yaml says train: images/train and its manifest says
   train: 2416, and in one list those read as a contradiction. */
.src{margin-top:11px;font-size:10.5px;text-transform:uppercase;
  letter-spacing:.07em;color:var(--dim);font-family:var(--num)}
.note{margin-top:4px;font-size:11.5px;color:var(--dim)}
/* ── the structure ── */
.balance{padding:12px 14px;border-bottom:1px solid var(--bd)}
.balrow+.balrow{margin-top:11px}
.ballab{display:flex;gap:8px;align-items:baseline;font-size:12px;
  color:var(--tx);margin-bottom:5px}
.ballab .bn{margin-left:auto;color:var(--mut);font-family:var(--num);
  font-size:11.5px;font-variant-numeric:tabular-nums}
.balbar{display:flex;height:9px;border-radius:5px;overflow:hidden;
  background:var(--panel2)}
.balseg{height:100%}
.ballegend{display:flex;gap:12px;flex-wrap:wrap;margin-top:5px;font-size:11px;
  color:var(--dim);font-variant-numeric:tabular-nums}
.ballegend i{display:inline-block;width:7px;height:7px;border-radius:2px;
  margin-right:5px;vertical-align:baseline}
.ballegend b{color:var(--mut);font-weight:600;font-family:var(--num)}
/* An image with no label file beside it is a background frame; a SPLIT with
   fewer label files than images is usually a build that stopped half way, and
   that is worth saying rather than leaving as two numbers to subtract. */
.mismatch{color:var(--acc)}
/* Wrapped flex, not multicol: columns:15em stretched every column to fill
   the pane and re-balanced the nodes across them, so a window resize marched
   images/val from under images to the top of its own column and back. Here a
   column is 240px whatever the viewport does, a node's seat depends on the
   pane's height alone, and extra width is an empty stripe on the right -- or,
   short of it, a scroll inside this pane -- never a re-seating. */
.tree{padding:8px 10px 12px;max-height:236px;overflow:auto;
  display:flex;flex-flow:column wrap;align-content:flex-start;gap:0 4px}
.tnode{display:flex;gap:8px;align-items:baseline;width:240px;flex:none;
  background:transparent;border:0;border-radius:8px;padding:5px 8px;
  cursor:pointer;font-family:inherit;color:var(--mut);font-size:12px;
  text-align:left}
.tnode:hover{background:var(--panel2);color:var(--tx)}
.tnode.on{background:rgba(232,166,69,.12);color:var(--acc)}
.tname{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tnum{margin-left:auto;font-family:var(--num);font-size:11px;color:var(--dim);
  font-variant-numeric:tabular-nums;flex:none}
.tnode.on .tnum{color:var(--acc)}
/* ── the pictures ── */
.ftool{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:10px 14px;
  border-bottom:1px solid var(--bd)}
.where{font-size:12px;color:var(--mut);font-family:var(--num);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.btn{background:var(--panel2);border:1px solid var(--bd);color:var(--mut);
  border-radius:9px;padding:6px 12px;font-size:12.5px;cursor:pointer;
  font-family:inherit}
.btn:hover:not(:disabled){color:var(--tx);border-color:rgba(130,140,150,.32)}
.btn:disabled{opacity:.4;cursor:default}
.pos{font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums;
  font-family:var(--num)}
.spacer{margin-left:auto}
.pick{display:inline-flex;align-items:center;gap:7px;font-size:11.5px;
  color:var(--dim)}
.pick select{background:var(--panel2);border:1px solid var(--bd);
  color:var(--mut);border-radius:9px;padding:6px 9px;font-size:12.5px;
  font-family:inherit;cursor:pointer}
.pick select:hover{color:var(--tx)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));
  gap:10px;padding:12px 14px}
.tile{background:var(--panel2);border:1px solid var(--bd);border-radius:10px;
  overflow:hidden}
.pic{position:relative;background:#0e1014;aspect-ratio:1;display:flex;
  align-items:center;justify-content:center;cursor:zoom-in}
.pic img{max-width:100%;max-height:100%;display:block}
.pic.flat{cursor:default}
.ext{font-family:var(--num);font-size:11px;color:var(--dim);
  letter-spacing:.08em}
/* The box count rides on the picture because it is a fact ABOUT the picture.
   Zero is not a missing number -- it is a background frame, put in the set on
   purpose with nothing to find -- so it gets its own word and its own colour
   rather than reading as a label file that failed to load. */
.boxes{position:absolute;right:5px;bottom:5px;font-size:10px;
  font-family:var(--num);background:rgba(10,12,16,.86);border-radius:5px;
  padding:1px 5px;border:1px solid var(--bd);color:var(--mut)}
.boxes.bg{color:var(--acc);border-color:rgba(232,166,69,.42)}
.bad{font-size:10.5px;color:var(--dim);text-align:center;padding:0 8px;
  line-height:1.35}
/* A folder with no pictures in it -- labels/val, or a root holding a manifest
   -- gets a list rather than a grid. Four hundred and eighty square tiles
   reading "TXT" is a page of empty boxes where four hundred and eighty lines
   of filename and size is the whole content. */
.flist{padding:6px 14px 12px}
.frow{display:grid;grid-template-columns:minmax(0,1fr) 90px;gap:12px;
  font-size:12px;color:var(--mut);padding:5px 2px;
  font-variant-numeric:tabular-nums}
.frow+.frow{border-top:1px solid rgba(130,140,150,.07)}
.frow .fn{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  font-family:var(--num)}
.frow .fs{text-align:right;color:var(--dim);font-family:var(--num)}
.cap{padding:6px 7px 7px}
.fname{display:block;font-size:11px;color:var(--tx);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.fmeta{display:block;font-size:10.5px;color:var(--dim);font-family:var(--num);
  font-variant-numeric:tabular-nums;margin-top:1px}
.pager{display:flex;gap:9px;align-items:center;padding:12px 14px;
  border-top:1px solid var(--bd)}
.pager[hidden]{display:none}
.empty{color:var(--dim);font-size:12.5px;padding:34px 18px;text-align:center;
  line-height:1.6;max-width:62ch;margin:0 auto}
.empty b{color:var(--mut);font-weight:600}
/* ── the full picture ── */
.lb{position:fixed;inset:0;background:rgba(0,0,0,.9);display:flex;
  align-items:center;justify-content:center;flex-direction:column;gap:12px;
  z-index:50}
.lb[hidden]{display:none}
.lb img{max-width:92vw;max-height:80vh;object-fit:contain}
.lbcap{font-size:12px;color:var(--mut);display:flex;gap:10px;
  align-items:center;flex-wrap:wrap;justify-content:center;max-width:92vw}
.lbcap button{background:var(--panel2);border:1px solid var(--bd);
  color:var(--mut);border-radius:7px;padding:4px 9px;font-size:11.5px;
  cursor:pointer;font-family:inherit}
.lbcap button:hover{color:var(--tx)}
.toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);
  background:var(--panel2);border:1px solid var(--bd);border-radius:9px;
  padding:8px 14px;font-size:12.5px;color:var(--tx);z-index:60}
.toast[hidden]{display:none}
.keys{font-size:11.5px;color:var(--dim);margin-top:14px}
.keys kbd{background:var(--panel2);border:1px solid var(--bd);border-radius:5px;
  padding:1px 5px;font-family:var(--num);font-size:11px;color:var(--mut)}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
@media(max-width:1180px){
  /* minmax(0,1fr), not 1fr: a bare 1fr track refuses to shrink below its
     content's min-content, and the sidebar rows' nowrap lines (412px) plus
     the run table's fixed columns were holding both panes wider than a
     phone screen -- the page scrolled sideways to 468px at 390px wide. */
  .cols{grid-template-columns:minmax(0,1fr)}
  .cols.folded{grid-template-columns:minmax(0,1fr)}
  .rail button{writing-mode:horizontal-tb;justify-content:center}
  .panes{grid-template-columns:minmax(0,1fr)}
  .struct{grid-template-columns:minmax(0,1fr)}
  .struct .balance{border-right:0;border-bottom:1px solid var(--bd)}
  /* stacked, the tree is a plain list again: display beats the desktop flex
     the way columns:auto used to beat the multicol */
  .struct .tree{display:block;max-height:52vh}
  .tnode{width:100%}
  .dslist{max-height:340px}
}
@media(max-width:560px){
  /* the run rows' three fixed columns (78+130+108px) cannot share 390px
     with a name; two lines per run beats an ellipsis three glyphs in */
  .run{grid-template-columns:minmax(0,1fr) auto;row-gap:2px}
}
</style></head><body><div class="wrap">
<header>
  <div><h1>Datasets</h1>
    <div class="sub">Every training set on this machine, found by walking the
      disk rather than from a list &mdash; so one built five minutes ago is
      here. Each carries the runs that trained on it.</div></div>
  <a class="back" href="/">&larr; dashboard</a>
</header>

<div class="banner" id="banner" hidden></div>

<div class="cols">
  <aside class="card side" id="side">
    <div class="chead"><span>built</span>
      <span class="n" id="dscount">&mdash;</span>
      <button class="rescan" id="rescan" title="walk the disk again — a dataset
        built in the last few seconds may not be in the cached answer">&#8635;
        rescan</button>
      <button class="fold" id="fold" title="fold the list away — the pictures
        get the width">&#9666;</button></div>
    <div class="dslist" id="dslist"></div>
    <div class="rail"><button id="unfold" title="bring the dataset list back">
      &#9656; <span id="railn" class="n"></span> datasets</button></div>
  </aside>
  <main>
    <div class="card dshead" id="dshead" hidden></div>
    <div class="empty" id="pickone">Pick a dataset on the left to see what is
      inside it.</div>
    <div class="panes" id="panes" hidden>
      <section class="card struct">
        <div class="chead"><span>structure</span></div>
        <div class="balance" id="balance"></div>
        <div class="tree" id="tree"></div>
      </section>
      <section class="card">
        <div class="ftool">
          <span class="where" id="where">&mdash;</span>
          <span class="spacer"></span>
          <label class="pick">per page
            <select id="size">__SIZEOPTS__</select></label>
        </div>
        <div class="grid" id="grid"></div>
        <div class="flist" id="flist" hidden></div>
        <div class="empty" id="fempty" hidden></div>
        <div class="pager" id="pager" hidden>
          <button class="btn" id="prev">&larr; back</button>
          <button class="btn" id="next">next &rarr;</button>
          <span class="pos" id="pos">&mdash;</span>
        </div>
      </section>
    </div>
    <div class="keys" id="keys" hidden><kbd>&larr;</kbd><kbd>&rarr;</kbd> move
      through the pictures &nbsp; <kbd>Esc</kbd> close</div>
  </main>
</div>
</div>

<div class="lb" id="lb" hidden>
  <img id="lbimg" alt="">
  <div class="lbcap"><span id="lbtxt"></span>
    <button id="lbcopy">copy filename</button>
    <button id="lbclose">close</button></div>
</div>
<div class="toast" id="toast" hidden></div>

<script>
var STATUS=__STATUS__,DEFSIZE=__DEFSIZE__;
var rows=[],cur=null,tree=null,rel='',pg=0,size=DEFSIZE,files=null,shot=-1;
/* Set when the index could not read the run list. Every row is then missing
   its runs, and "no run has trained on it" would be this page stating as a
   fact the one thing it was unable to look up. */
var noRuns=false;
var $=function(id){return document.getElementById(id)};

function esc(s){var d=document.createElement('div');d.textContent=s==null?'':s;
  return d.innerHTML}
/* esc() is a text-node round trip, and a text node has no quotes to escape --
   so it is safe between tags and unsafe inside title="" or alt="". Every name
   on this page is a directory entry read off a disk anyone can unpack an
   export onto, and a quote in one of them closes the attribute and turns the
   rest of the file name into attributes. The review page carries the same
   helper for the same reason. */
function att(s){return esc(s).replace(/"/g,'&quot;').replace(/'/g,'&#39;')}
function fmtn(n){return (+n||0).toLocaleString('en-US')}
function human(b){b=+b||0;var u=['B','KB','MB','GB','TB'],i=0;
  while(b>=1024&&i<u.length-1){b/=1024;i++}
  return (i?b.toFixed(1):Math.round(b))+' '+u[i]}
/* the wording the dashboard's own panels use, so a run reads the same on both */
function agoTxt(t){
  if(!t)return '—';
  var s=Math.max(0,Math.round(Date.now()/1000-t));
  if(s<90)return s+' s ago';
  if(s<5400)return Math.round(s/60)+' min ago';
  if(s<129600)return (s/3600).toFixed(1)+' h ago';
  return (s/86400).toFixed(1)+' days ago';
}
function toast(t){var e=$('toast');e.textContent=t;e.hidden=false;
  clearTimeout(e._t);e._t=setTimeout(function(){e.hidden=true},1800)}
function q(o){var p=[],k;
  for(k in o)p.push(k+'='+encodeURIComponent(o[k]));return p.join('&')}
/* Every picture URL is built here and nowhere else. The version is the source
   file's own mtime: thumbnails are served with a long cache, which is right
   for a file that never changes and wrong the moment a rebuild puts a
   different picture at the same path. */
function thumbSrc(f){
  return '/datasets/thumb?'+q({key:cur.key,rel:f.rel,v:Math.round(f.mtime||0)})}
function fullSrc(f){return '/datasets/image?'+q({key:cur.key,rel:f.rel})}
var COLORS=['var(--c1)','var(--c2)','var(--c3)','var(--c4)'];
function colr(i){return COLORS[i%COLORS.length]}

/* ── the list ── */
function loadList(refresh){
  $('rescan').disabled=true;
  fetch('/api/datasets'+(refresh?'?refresh=1':''))
    .then(function(r){return r.json()})
    .then(function(j){
      $('rescan').disabled=false;
      rows=(j&&j.datasets)||[];
      noRuns=!!(j&&j.error);
      var b=$('banner');
      /* Without the run list this page is a directory listing: every dataset
         comes back with nothing attached, and "0 runs" everywhere reads as a
         fact about the runs rather than about the lookup. */
      b.textContent=j&&j.error
        ? 'The run list is unavailable ('+j.error+') — these are the datasets '+
          'found on disk, with no runs attached to any of them.':'';
      b.hidden=!b.textContent;
      paintList();
      if(!rows.length){cur=null;$('dshead').hidden=true;$('panes').hidden=true;
        $('keys').hidden=true;return}
      var want=cur&&cur.key;
      if(!want){try{want=localStorage.getItem('sdDataset')}catch(_){}}
      var hit=null,i;
      for(i=0;i<rows.length;i++)if(rows[i].key===want)hit=rows[i];
      /* A rescan must not throw away where you were. Re-opening the dataset
         already on screen would reset the folder back to the default one, so
         the counts are refreshed in place and the pictures are left alone. */
      if(cur&&hit){cur=hit;paintHead();return}
      /* Otherwise open the newest, which is what "newest first" is for: a page
         that lands on an empty right-hand side has made the reader click
         before it has told them anything. */
      openDs((hit||rows[0]).key);
    })
    .catch(function(){$('rescan').disabled=false;toast('could not read the '+
      'dataset list')});
}
/* Two rows can carry the same basename and mean two different directories: an
   export moved into archived_datasets sits on the same page as the path the
   runs recorded, which is not there any more. key_for() disambiguates the URL
   key already; without the same on the label the page reads as double-listing
   one dataset, and the greyed row asserts that nothing can be opened while a
   directory of that name is three rows above it. */
function twinned(r){
  var i,n=0;
  for(i=0;i<rows.length;i++)if(rows[i].name===r.name)n++;
  return n>1;
}
function parentOf(p){
  var s=String(p||'').replace(/\/+$/,'').split('/');
  s.pop();
  return s.pop()||'/';
}
function nameOf(r){
  return esc(r.name)+(twinned(r)
    ? ' <span class="dsin">in '+esc(parentOf(r.root))+'</span>':'');
}
function paintList(){
  var el=$('dslist');
  $('dscount').textContent=rows.length?fmtn(rows.length)+
    (rows.length===1?' dataset':' datasets'):'none';
  $('railn').textContent=rows.length?fmtn(rows.length):'0';
  if(!rows.length){
    el.innerHTML='<div class="empty">Nothing on this machine looks like a '+
      'dataset yet. Build one &mdash; a detector set under the training root, '+
      'or a crop export from this repo &mdash; and it shows up here on the '+
      'next rescan.</div>';
    $('pickone').hidden=true;
    return;
  }
  el.innerHTML=rows.map(function(r){
    var runs=r.runs||[],live=false,i;
    for(i=0;i<runs.length;i++)if(runs[i].live)live=true;
    var who=runs.length
      ? '<b>'+runs.length+(runs.length===1?' run</b> ':' runs</b> ')+
        esc(runs[0].name)+(runs.length>1?' +'+(runs.length-1)+' more':'')+
        (live?' <span class="live">&#9679; running</span>':'')
      : noRuns?'which runs used it is not known':'no run has trained on it';
    var nums=r.exists
      ? fmtn(r.images)+' images · '+human(r.bytes)+' · '+agoTxt(r.mtime)
      : 'the directory is gone';
    return '<button class="ds'+(r.exists?'':' gone')+
      (cur&&cur.key===r.key?' on':'')+'" data-key="'+att(r.key)+'">'+
      '<span class="dstop"><span class="dsname">'+nameOf(r)+'</span>'+
      '<span class="chip '+att(r.kind)+'">'+esc(r.kind)+'</span></span>'+
      '<span class="dsnums">'+nums+'</span>'+
      '<span class="dsruns">'+who+'</span></button>';
  }).join('');
}
$('dslist').addEventListener('click',function(e){
  var b=e.target.closest&&e.target.closest('.ds');
  if(b)openDs(b.getAttribute('data-key'));
});
$('rescan').addEventListener('click',function(){loadList(true)});

/* ── opening one ── */
function rowFor(key){
  for(var i=0;i<rows.length;i++)if(rows[i].key===key)return rows[i];
  return null;
}
function openDs(key){
  var r=rowFor(key); if(!r)return;
  cur=r;tree=null;files=null;rel='';pg=0;
  try{localStorage.setItem('sdDataset',key)}catch(_){}
  paintList();
  paintHead();
  $('pickone').hidden=true;
  $('panes').hidden=false;
  $('grid').innerHTML='';
  $('pager').hidden=true;
  if(!r.exists){
    /* The row stays on screen with its runs -- that is the whole point of it --
       and the two panes say plainly that there is nothing left to open. */
    $('panes').hidden=true;
    $('keys').hidden=true;
    return;
  }
  fetch('/api/datasets/tree?'+q({key:key}))
    .then(function(res){return res.json()})
    .then(function(j){
      if(cur.key!==key)return;         /* another dataset was clicked meanwhile */
      if(!j||!j.ok){
        noTree((j&&j.error)||'could not read this dataset');
        return;
      }
      tree=j.tree;
      paintBalance();
      paintTree();
      pick(firstWithImages(tree));
    })
    /* A fetch that never produced JSON has to clear the two panes as well.
       The header has already been repainted for the dataset just clicked, and
       leaving the previous one's tree and class balance under it puts two
       datasets on one screen with nothing saying so -- and a click on one of
       those stale nodes asks the new dataset for the old one's folder. */
    .catch(function(){
      if(cur.key!==key)return;
      noTree('could not read this dataset');
      toast('could not read this dataset');
    });
}
function noTree(why){
  tree=null;
  $('balance').innerHTML='';
  $('tree').innerHTML='<div class="empty">'+esc(why)+'.</div>';
  showEmpty('This dataset could not be opened. It may have been moved '+
    'or rebuilt since the list was taken — rescan to find out.');
}
function paintHead(){
  var r=cur,el=$('dshead');
  el.hidden=false;
  var runs=r.runs||[];
  var cls=(r.classes&&r.classes.length)
    ? '<span>'+r.classes.length+' '+
      (r.classes.length===1?'class':'classes')+': '+
      esc(r.classes.join(', '))+'</span>':'';
  /* A dataset that is gone keeps its class list -- the descriptor outlived the
     directory -- but not its counts and not a "last written", which would be
     the date of the last run that used it dressed up as a fact about a
     directory that is not there. */
  var facts=r.exists
    ? '<span><b>'+fmtn(r.images)+'</b> images</span>'+
      '<span><b>'+human(r.bytes)+'</b> on disk</span>'+
      '<span><b>'+fmtn(r.folders)+'</b> folders</span>'+cls+
      '<span>last written '+agoTxt(r.mtime)+'</span>'
    : cls;
  var gone=r.exists?'':
    '<div class="gonenote">This directory is gone. '+
    (runs.length?'The '+runs.length+' run'+(runs.length===1?'':'s')+
      ' below trained on it and nothing here can be opened again':
      'Nothing is left to open')+'.'+
    (cls?' Its descriptor still reads, which is where the class list comes '+
      'from.':'')+
    /* "nothing can be opened again" is a strong sentence to print while a
       directory of the same name is sitting in the list, full of images. It
       is still true of THIS path -- but an export that was archived rather
       than deleted looks exactly like this, and the reader should be told
       where to look before they take the sentence at face value. */
    (twinned(r)?' A dataset of the same name is on the left: if this export '+
      'was moved rather than deleted, that is where it went.':'')+'</div>';
  var manifest=r.manifest,spec=r.descriptor&&r.descriptor.spec,built='';
  function kvOf(o){
    var kv='',k;
    for(k in o)kv+='<dt>'+esc(k)+'</dt><dd>'+esc(o[k])+'</dd>';
    return kv?'<dl class="kv">'+kv+'</dl>':'';
  }
  if(manifest||spec){
    var body='';
    if(r.descriptor)
      body+='<div class="src">'+esc(r.descriptor.name)+'</div>'+
        (kvOf(spec)||'<div class="note">nothing readable in it.</div>');
    if(manifest){
      body+='<div class="src">'+esc(manifest.name)+'</div>'+
        /* A per-crop manifest is one line per file and can be tens of
           thousands of them, so the index describes it rather than parsing
           it. Saying that beats an expander that opens on nothing. */
        (kvOf(manifest.summary)||
          '<div class="note">one row per crop, '+human(manifest.size)+
          ' &mdash; described rather than read.</div>');
    }
    built='<details class="built"><summary>how it was built</summary>'+
      body+'</details>';
  }
  el.innerHTML='<div class="dstitle"><h2>'+esc(r.name)+'</h2>'+
    '<span class="chip '+att(r.kind)+'">'+esc(r.kind)+'</span></div>'+
    '<div class="dspath">'+esc(r.root)+'</div>'+
    (facts?'<div class="facts">'+facts+'</div>':'')+gone+
    (runs.length
      ? '<div class="runs"><div class="rlab">trained on this — '+
        runs.length+' run'+(runs.length===1?'':'s')+'</div>'+
        '<div class="runlist">'+runs.map(function(x){
          return '<div class="run"><span class="rname">'+esc(x.key)+'</span>'+
            '<span>'+esc(x.task||'—')+'</span>'+
            '<span class="rst'+(x.live?' live':'')+'">'+
            (x.live?'&#9679; ':'')+
            esc(STATUS[x.status]||x.status||'—')+'</span>'+
            '<span class="rwhen">'+agoTxt(x.mtime)+'</span></div>';
        }).join('')+'</div></div>'
      : '<div class="runs"><div class="rlab">trained on this</div>'+
        '<div class="run"><span class="rname" style="color:var(--dim)">'+
        (noRuns
          ? 'The run list could not be read, so nothing here can say what '+
            'trained on this.'
          : 'No logged run has trained on this one yet.')+
        '</span></div></div>')+
    built;
}

/* ── the structure ── */
function kids(n){return (n&&n.dirs)||[]}
function byName(n,name){
  var d=kids(n),i;for(i=0;i<d.length;i++)if(d[i].name===name)return d[i];
  return null;
}
/* Depth first, the first folder that actually holds pictures. On a detect set
   that lands on images/train, on a classify set on train/<first class>, and on
   a flat crop export on its first class folder -- without any of those names
   being written down here, which is the rule the whole feature is built on. */
function firstWithImages(n){
  if(!n)return '';
  if(n.own_images)return n.rel;
  var d=kids(n),i,got;
  for(i=0;i<d.length;i++){got=firstWithImages(d[i]);if(got)return got}
  return '';
}
function bars(groups){
  return groups.map(function(g){
    var tot=0,i;for(i=0;i<g.parts.length;i++)tot+=g.parts[i].n;
    var segs=g.parts.map(function(p,j){
      return '<div class="balseg" style="width:'+
        (tot?(p.n/tot*100):0).toFixed(2)+'%;background:'+colr(j)+
        '" title="'+att(p.name)+': '+fmtn(p.n)+'"></div>';
    }).join('');
    var leg=g.parts.map(function(p,j){
      return '<span><i style="background:'+colr(j)+'"></i>'+esc(p.name)+
        ' <b>'+fmtn(p.n)+'</b>'+(tot?' ('+(p.n/tot*100).toFixed(1)+'%)':'')+
        '</span>';
    }).join('');
    return '<div class="balrow"><div class="ballab">'+esc(g.name)+
      /* What the parts COUNT, said on the bar. Folders hold image files and a
         crop manifest holds rows, and the two are not the same number for the
         same crops -- printing either as "images" is how a count of rows ends
         up read as a count of pictures. */
      '<span class="bn">'+fmtn(tot)+' '+(g.unit||'images')+'</span></div>'+
      /* A bar of nothing is a grey box that looks like a bar that failed to
         draw. An export that has been created but not filled is a real state
         here -- audit_finds_leash is one -- so it gets a sentence. */
      (tot?'<div class="balbar">'+segs+'</div>'
         :'<div class="note">nothing has been written into it yet.</div>')+
      '<div class="ballegend">'+leg+'</div>'+
      (g.note?'<div class="note">'+g.note+'</div>':'')+'</div>';
  }).join('');
}
function paintBalance(){
  var el=$('balance');
  if(!tree){el.innerHTML='';return}
  if(cur.kind==='detect'&&byName(tree,'images')){
    /* A detector's two halves face each other: images/<split> and
       labels/<split>. The number worth showing is not a class share -- there
       is one class -- it is whether every picture has a label file beside it,
       because a split with fewer is a build that stopped half way. */
    var im=byName(tree,'images'),lb=byName(tree,'labels'),tot=0;
    kids(im).forEach(function(s){tot+=s.images});
    el.innerHTML=kids(im).map(function(s){
      var l=lb?byName(lb,s.name):null,
          nl=l?l.files:null,
          off=nl!==null&&nl!==s.images,
          /* the split's share of the set. A bar drawn full width for every
             split is a shape rather than a measurement -- what anyone opening
             a detector's data wants off this block is how much of it is
             training and how much is held out. */
          share=tot?s.images/tot*100:0;
      return '<div class="balrow"><div class="ballab">'+esc(s.name)+
        '<span class="bn">'+fmtn(s.images)+' images · '+
        share.toFixed(1)+'%</span></div>'+
        '<div class="balbar"><div class="balseg" style="width:'+
        share.toFixed(2)+'%;background:'+colr(0)+'"></div></div>'+
        '<div class="ballegend"><span>'+
        (nl===null
          ? 'no labels/'+esc(s.name)+' beside it'
          : '<b>'+fmtn(nl)+'</b> label file'+(nl===1?'':'s')+
            (off?' <span class="mismatch">— '+
              fmtn(Math.abs(nl-s.images))+' '+(nl<s.images?'short':'spare')+
              '</span>':''))+
        '</span><span>'+human(s.bytes)+'</span></div></div>';
    }).join('')||'<div class="ballegend">no splits under images/</div>';
    return;
  }
  /* Classify. The groups are the folders that hold class folders -- train and
     val on a split set, the root itself on a flat crop export, which is the
     shape this repo's own exporters write. */
  var groups=kids(tree).filter(function(d){return kids(d).length}).map(
    function(d){return {name:d.name,parts:kids(d).map(function(c){
      return {name:c.name,n:c.images}})}});
  if(!groups.length&&kids(tree).length)
    groups=[{name:cur.name,parts:kids(tree).map(function(c){
      return {name:c.name,n:c.images}})}];
  /* THE FOLDERS ARE NOT ALWAYS THE CLASSES. data/hard_negatives holds crops/
     and full/ -- a crop and the frame it was cut from, the same name in both
     -- and every row of its labels.jsonl reads the one word. Drawn off the
     folders that is a 50/50 split between two classes that do not exist, over
     an image count that has every crop in it twice. The index reads the
     labels off the manifest; where they are not the folder names, they are
     what this draws, and the folders keep their counts in the tree below. */
  var byLabel=labelBar();
  el.innerHTML=byLabel?bars([byLabel])
    :groups.length?bars(groups)
    :'<div class="ballegend">no folders inside this one</div>';
}
function labelBar(){
  var lab=cur.labels,names=[],k;
  for(k in (lab||{}))names.push(k);
  if(!names.length)return null;
  var folders=kids(tree).map(function(d){return d.name}),same=true,i;
  if(folders.length!==names.length)same=false;
  for(i=0;i<folders.length;i++)if(names.indexOf(folders[i])<0)same=false;
  if(same)return null;               /* the folders ARE the classes */
  names.sort();
  return {name:(cur.manifest&&cur.manifest.name)||'the manifest',
          unit:'rows',
          parts:names.map(function(n){return {name:n,n:lab[n]}}),
          note:'The folders in this one are not classes &mdash; the labels '+
            'are recorded per crop in this file. The image count in the '+
            'header is every file in every folder, so a crop kept in more '+
            'than one of them is counted more than once.'};
}
function paintTree(){
  var out=[];
  (function walk(n,depth){
    /* Images where there are images, files where there are none. A detect
       dataset's labels/val held 480 label files and the tree said "0", which
       reads as an empty folder rather than as a folder of text. */
    var num=n.images?fmtn(n.images)
      :n.files?fmtn(n.files)+' files':'—';
    out.push('<button class="tnode'+(n.rel===rel?' on':'')+'" data-rel="'+
      att(n.rel)+'" style="padding-left:'+(8+depth*14)+'px" title="'+
      att(n.rel||cur.name)+' — '+fmtn(n.images)+' images, '+human(n.bytes)+
      '">'+
      '<span class="tname">'+esc(depth?n.name:n.name+'/')+'</span>'+
      '<span class="tnum">'+num+'</span></button>');
    kids(n).forEach(function(k){walk(k,depth+1)});
  })(tree,0);
  $('tree').innerHTML=out.join('');
}
$('tree').addEventListener('click',function(e){
  var b=e.target.closest&&e.target.closest('.tnode');
  if(b)pick(b.getAttribute('data-rel'));
});

/* ── the pictures ── */
function pick(r){rel=r||'';pg=0;paintTree();loadFiles()}
function showEmpty(msg){
  $('grid').innerHTML='';
  $('flist').innerHTML='';$('flist').hidden=true;
  $('fempty').hidden=false;
  $('fempty').innerHTML=msg;
  $('pager').hidden=true;
  $('keys').hidden=true;
}
/* `after` runs once the new page is on screen, and exists for one caller:
   arrowing out of the last picture on a page turns to the next one and opens
   its first, so the reader keeps going instead of hitting a wall the folder
   does not have. */
function loadFiles(after){
  var key=cur.key,at=rel;
  $('where').textContent=cur.name+'/'+(rel||'');
  fetch('/api/datasets/files?'+q({key:key,rel:rel,page:pg,n:size}))
    .then(function(r){return r.json()})
    .then(function(j){
      if(cur.key!==key||rel!==at)return;
      if(!j||!j.ok){
        showEmpty(j&&j.error==='no such folder'
          ? 'That folder is not there any more. A rebuild moves a dataset '+
            'while you are looking at it &mdash; rescan, or pick another '+
            'folder.'
          : 'That folder could not be read.');
        return;
      }
      files=j;
      /* The canonical rel comes back from the server; linking on to the string
         we sent would let './train' and 'train//dog' keep growing. */
      rel=j.rel;
      paintTree();
      $('where').textContent=cur.name+'/'+(rel||'');
      paintFiles();
      if(after)after();
    })
    .catch(function(){showEmpty('That folder could not be read.')});
}
function paintFiles(){
  var j=files;
  if(!j.files.length){
    showEmpty(j.dirs.length
      ? 'No files here &mdash; only the '+j.dirs.length+' folder'+
        (j.dirs.length===1?'':'s')+' listed on the left.'
      : 'This folder is empty.');
    return;
  }
  $('fempty').hidden=true;
  var any=false,i;
  for(i=0;i<j.files.length;i++)if(j.files[i].image)any=true;
  if(!any){
    $('grid').innerHTML='';
    $('flist').hidden=false;
    $('flist').innerHTML=j.files.map(function(f){
      return '<div class="frow"><span class="fn" title="'+att(f.name)+'">'+
        esc(f.name)+'</span><span class="fs">'+human(f.size)+'</span></div>';
    }).join('');
    paintPager();
    /* nothing to arrow through, so the hint would be describing a control
       that does nothing here */
    $('keys').hidden=true;
    return;
  }
  $('flist').innerHTML='';$('flist').hidden=true;
  $('grid').innerHTML=j.files.map(function(f,i){
    /* 0 boxes is not a missing label -- it is a background frame, put in the
       set on purpose with nothing to find -- and how many of those a split
       holds is one of the few things you cannot get by looking at the
       pictures. */
    var box=(f.labels===null||f.labels===undefined)?''
      :'<span class="boxes'+(f.labels?'':' bg')+'" title="'+
        (f.labels?f.labels+' box'+(f.labels===1?'':'es')+' in the matching '+
          'label file':'no boxes — a background frame')+'">'+
        (f.labels?f.labels+(f.labels===1?' box':' boxes'):'background')+
        '</span>';
    var pic=f.image
      ? '<div class="pic" data-i="'+i+'"><img loading="lazy" src="'+
        thumbSrc(f)+'" alt="'+att(f.name)+'">'+box+'</div>'
      : '<div class="pic flat"><span class="ext">'+
        esc((f.name.split('.').pop()||'file').toUpperCase().slice(0,5))+
        '</span>'+box+'</div>';
    return '<figure class="tile">'+pic+
      '<figcaption class="cap"><span class="fname" title="'+att(f.name)+'">'+
      esc(f.name)+'</span><span class="fmeta">'+human(f.size)+
      '</span></figcaption></figure>';
  }).join('');
  /* A thumbnail that will not cut comes back 404, and a broken-image icon
     reads as a broken page rather than as a picture this dataset cannot
     open. */
  var imgs=$('grid').querySelectorAll('img');
  for(i=0;i<imgs.length;i++)imgs[i].onerror=function(){
    var p=this.parentNode;
    p.innerHTML='<span class="bad">would not open</span>';
  };
  paintPager();
  $('keys').hidden=false;
}
function paintPager(){
  var j=files;
  $('pager').hidden=false;
  $('pos').textContent='page '+(j.page+1)+' of '+Math.max(1,j.pages)+
    ' · '+fmtn(j.total)+' file'+(j.total===1?'':'s');
  $('prev').disabled=j.page<=0;
  $('next').disabled=j.page+1>=j.pages;
}
$('grid').addEventListener('click',function(e){
  var p=e.target.closest&&e.target.closest('.pic[data-i]');
  if(p)zoom(+p.getAttribute('data-i'));
});
$('prev').addEventListener('click',function(){if(pg>0){pg--;loadFiles()}});
$('next').addEventListener('click',function(){
  if(files&&pg+1<files.pages){pg++;loadFiles()}});
/* The list folds to a rail and the pictures take the width. Open by
   default; the choice survives a reload the way the picked dataset and the
   page size already do. */
function foldList(want){
  document.querySelector('.cols').classList.toggle('folded',want);
  $('side').classList.toggle('folded',want);
  try{localStorage.setItem('sdDatasetList',want?'folded':'')}catch(_){}
}
$('fold').addEventListener('click',function(){foldList(true)});
$('unfold').addEventListener('click',function(){foldList(false)});
try{if(localStorage.getItem('sdDatasetList')==='folded')foldList(true)}catch(_){}
var sizeSel=$('size');
try{
  var sv=localStorage.getItem('sdDatasetSize');
  if(sv&&/^(60|120|240)$/.test(sv))sizeSel.value=sv;
}catch(_){}
size=+sizeSel.value||DEFSIZE;
sizeSel.addEventListener('change',function(){
  size=+sizeSel.value||DEFSIZE;
  try{localStorage.setItem('sdDatasetSize',String(size))}catch(_){}
  /* Back to the first page: the page NUMBER means something different at a
     different size, and keeping it steps over a block of files with nothing
     saying so. */
  pg=0;
  if(cur)loadFiles();
});

/* ── the full picture ── */
function zoom(i){
  var f=files&&files.files[i]; if(!f||!f.image)return;
  shot=i;
  $('lbimg').src=fullSrc(f);
  $('lbtxt').textContent=f.name+' · '+human(f.size)+
    (f.labels===null||f.labels===undefined?''
      :' · '+(f.labels?f.labels+(f.labels===1?' box':' boxes')
                      :'background frame, no boxes'));
  $('lb').hidden=false;
}
function step(d){
  if(!files)return;
  var i=shot+d;
  while(i>=0&&i<files.files.length&&!files.files[i].image)i+=d;
  if(i>=0&&i<files.files.length){zoom(i);return}
  /* Off the edge of the PAGE, which is an artefact of how many tiles fit on a
     screen and not a fact about the folder. dogdet_v3's images/train is 21
     pages at the default, and stopping dead at the 120th picture with no
     message meant Esc, click "next", click a thumbnail -- twenty times over
     to read one split. So the key turns the page and carries on. */
  var want=pg+d;
  if(want<0||want>=files.pages){
    toast(d>0?'that is the last picture in this folder'
             :'that is the first picture in this folder');
    return;
  }
  pg=want;
  loadFiles(function(){
    var k=d>0?0:files.files.length-1;
    while(k>=0&&k<files.files.length&&!files.files[k].image)k+=d;
    /* A page of a mixed folder can hold no pictures at all. The grid behind
       has already turned, so the lightbox closes rather than hanging on a
       picture from the page before it. */
    if(k>=0&&k<files.files.length)zoom(k);
    else {shot=-1;$('lb').hidden=true}
  });
}
$('lbclose').addEventListener('click',function(){$('lb').hidden=true});
$('lb').addEventListener('click',function(e){
  if(e.target===$('lb'))$('lb').hidden=true});
$('lbimg').addEventListener('error',function(){
  $('lbtxt').textContent='that image would not open'});
$('lbcopy').addEventListener('click',function(){
  var f=files&&files.files[shot]; if(!f)return;
  /* navigator.clipboard is absent on a plain-http origin, which this is */
  if(window.isSecureContext&&navigator.clipboard)
    navigator.clipboard.writeText(f.name).then(
      function(){toast('copied '+f.name)},fallback);
  else fallback();
  function fallback(){
    var ta=document.createElement('textarea');ta.value=f.name;
    ta.style.position='fixed';ta.style.top='-1000px';document.body.appendChild(ta);
    ta.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}
    document.body.removeChild(ta);toast(ok?'copied '+f.name:'copy failed');
  }
});
document.addEventListener('keydown',function(e){
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  var t=e.target&&e.target.tagName;
  if(t==='SELECT'||t==='INPUT'||t==='TEXTAREA')return;
  if(!$('lb').hidden){
    if(e.key==='Escape'){$('lb').hidden=true;e.preventDefault()}
    else if(e.key==='ArrowRight'){step(1);e.preventDefault()}
    else if(e.key==='ArrowLeft'){step(-1);e.preventDefault()}
    return;
  }
  if(e.key==='ArrowRight'&&files&&pg+1<files.pages){pg++;loadFiles();
    e.preventDefault()}
  else if(e.key==='ArrowLeft'&&pg>0){pg--;loadFiles();e.preventDefault()}
});
loadList(false);
</script></body></html>
"""


def page_html():
    """The whole /datasets page.

    Nothing about the datasets is baked in: the page arrives empty and asks
    /api/datasets for what is here. That is not a style preference -- the whole
    promise of this page is that a dataset built after it was written turns up
    on it, and a template with rows rendered into it would be a snapshot taken
    when the server started.
    """
    opts = ''.join(
        f'<option value="{n}"{" selected" if n == PAGE_DEFAULT else ""}>'
        f'{n}</option>' for n in PAGE_SIZES)
    out = DATASETS_HTML
    for k, v in (('__STATUS__', json.dumps(_status_words())),
                 ('__DEFSIZE__', json.dumps(PAGE_DEFAULT)),
                 ('__SIZEOPTS__', opts)):
        out = out.replace(k, v)
    return out
