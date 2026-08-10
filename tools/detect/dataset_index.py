#!/usr/bin/env python3
"""Every dataset on this machine, found by shape rather than from a list.

    python tools/detect/dataset_index.py                 # what is here
    python tools/detect/dataset_index.py <key>           # its folders
    python tools/detect/dataset_index.py <key> <folder>  # the files in one

WHY THIS IS A SCAN. A training run records the string it was launched with --
`dataset.yaml`, or an absolute path to a directory -- and nothing else. Any
list of datasets written down here would be correct until the next build and
silently wrong afterwards, which is the whole failure this exists to avoid: a
dataset built this afternoon has to show up without anyone editing a file.

TWO PASSES, MERGED ON THE RESOLVED ROOT.

  the runs   every logged run's `data`, resolved. 39 of the 52 runs on this
             box recorded a BARE name, so a resolver that only accepted
             absolute paths would drop three quarters of the history.
  the disk   the places datasets get BUILT -- the training root, and this
             repo's own data/ -- walked shallowly and tested for a shape.
             This is the half that finds a dataset no run has trained on yet.

Both passes end at a realpath and the rows are keyed on it, so one dataset is
one row however it was found.

A DATASET THAT IS GONE IS STILL A ROW. `dataset.yaml` at the training root
points at a Label Studio export directory that has since been deleted, and
thirty-nine runs trained on it. Reporting it as exists=False is the answer;
dropping it would show those runs as having trained on nothing at all.

WHAT A DATASET LOOKS LIKE. Three layouts, recognised by structure and never by
name:

    detect     images/{train,val} beside labels/{train,val}
    classify   {train,val}/<class>/*.jpg
    classify   <class>/*.jpg beside a per-crop manifest -- what this repo's
               own export tools write into data/

WHAT IT COSTS. Counting is cached per directory against that directory's own
mtime, so a tree nobody has touched is re-measured with one stat per DIRECTORY
rather than one per file; see _own(). Manifests and sorted listings are cached
the same way. Nothing here parses a yaml with PyYAML -- the dashboard's
interpreter does not have it.

READ-ONLY. Nothing opens a file for writing inside a dataset. The one thing
this writes is its own measurement cache, under data/dashboard/.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')
SPLITS = ('train', 'val', 'test')
# The descriptor a detect dataset carries. Preferred in this order; a single
# stray *.yaml in the root is taken only when neither of these is there.
DESCRIPTORS = ('dataset.yaml', 'data.yaml')
# Whole-dataset manifests, in the order they are preferred when a root holds
# more than one. Every builder in both repos writes one of these.
MANIFESTS = ('manifest.json', 'rebuild_manifest.json', 'build_manifest.json',
             'negatives_manifest.json')
# Per-crop manifests: one line per file. These are what tell a flat directory
# of class folders apart from a working directory that merely holds folders of
# crops -- data/fn_audit has crops/ and full/ and is scratch, not a dataset.
ROW_MANIFESTS = ('manifest.jsonl', 'labels.jsonl')
# How far below a scan root a dataset is looked for. Two covers every case
# here (dogbin_v5 sits directly under the training root, leash_src one deeper
# under data/harvest) and keeps the walk off the run directories.
SCAN_DEPTH = 2
SKIP_DIRS = ('__pycache__', 'node_modules')
# A dataset is three levels deep. Past this it is a link loop or it is not a
# dataset, and either way the page does not want it.
MAX_DEPTH = 12


def _dash():
    """The dashboard module, for the run list and the training root.

    Imported lazily and by path, the way fn_audit reaches _grid_roots(): the
    serving layer imports THIS module out of the dashboard, so a top-level
    import would be a cycle. Run discovery is never reimplemented here -- a
    second walk of the run directories is a second answer, and the two would
    drift the first time either changed.

    The insert is guarded. Both of these run inside the discovery loop, once
    per logged run, so an unguarded one prepended fifty-odd copies of the same
    directory to sys.path per refresh -- in a process that stays up for days,
    with every cold import anywhere in the dashboard paying a linear scan of
    the result.
    """
    here = os.path.join(REPO, 'tools', 'dashboard')
    if here not in sys.path:
        sys.path.insert(0, here)
    import dashboard
    return dashboard


def _mistakes():
    """run_mistakes, for its dataset-path resolver and its yaml reader.

    Guarded like _dash() above, and for the same reason: resolve_data() calls
    this once per run.
    """
    here = os.path.join(REPO, 'tools', 'detect')
    if here not in sys.path:
        sys.path.insert(0, here)
    import run_mistakes
    return run_mistakes


def paths():
    """Every file this module owns.

    One file, under data/dashboard/, and it is a cache: deleting it costs one
    slow render and nothing else. Computed rather than module state for the
    same reason fn_audit.paths() is -- the config it hangs off can change
    under a running server.
    """
    out = os.path.join(REPO, 'data', 'dashboard')
    return {'out': out, 'sizes': os.path.join(out, 'dataset_sizes.json')}


# ── the filesystem, defensively ─────────────────────────────────────────────
# A detector is training into the training root while this runs, and an export
# writes into data/ while someone is looking at it. Everything below tolerates
# a directory or a file vanishing between one call and the next; a scan that
# raises because a file was unlinked mid-walk would take the page down at
# exactly the moment there is something new to look at.

def _entries(path):
    """(file names, directory names) in one directory, or ([], []).

    Symlinked directories are not returned as directories: a symlinked subtree
    can loop, and it can point clean out of the dataset, where resolve() would
    refuse to serve anything inside it anyway. A symlinked FILE still counts as
    a file -- the crop datasets are built with os.link and a hard link is not
    distinguishable from the original here.
    """
    files, dirs = [], []
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        dirs.append(e.name)
                    elif e.is_file():
                        files.append(e.name)
                except OSError:
                    continue
    except OSError:
        return [], []
    return files, dirs


def _any_image(path):
    """True as soon as one image is seen.

    Short-circuits deliberately: the shape test must not pay for a 200,000
    file directory just to learn that the directory has pictures in it.
    """
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.name.lower().endswith(IMAGE_EXT):
                    try:
                        if e.is_file():
                            return True
                    except OSError:
                        continue
    except OSError:
        pass
    return False


# ── what a dataset is ───────────────────────────────────────────────────────

def shape(root):
    """'detect' | 'classify' | 'unknown' -- read off the layout.

    By layout and never by name. The point of the page is that a dataset built
    an hour ago appears on it, and nothing in this file can know its name in
    advance.
    """
    files, dirs = _entries(root)
    have = set(dirs)
    # detect: the two halves face each other, images/<split> and labels/<split>
    if 'images' in have and 'labels' in have:
        return 'detect'
    # classify: <split>/<class>/*.jpg. One split is enough -- a build writes
    # train before it writes val, and half a dataset is still a dataset.
    for s in SPLITS:
        if s in have:
            for c in _entries(os.path.join(root, s))[1]:
                if _any_image(os.path.join(root, s, c)):
                    return 'classify'
    # the flat class layout this repo's own exporters write. The per-crop
    # manifest is load-bearing: without it this matches every working
    # directory under data/ that happens to hold folders of crops. The class
    # directories must also be leaves, which is what keeps data/harvest -- fp/
    # and tp/, each holding another folder -- out of the list.
    if any(f in files for f in ROW_MANIFESTS) and dirs and all(
            not _entries(os.path.join(root, d))[1] for d in dirs):
        return 'classify'
    return 'unknown'


def key_for(root):
    """A handle for a dataset root that is safe to put in a URL.

    Name plus a hash of the path. The name alone will not do -- dogdet_v2
    exists under the training root and again under archived_datasets -- and
    the path alone must never travel in a query string, both because it is
    ugly and because it is one bug away from being joined onto something. No
    slashes and no dots, so nothing in it can climb.
    """
    base = os.path.basename(root.rstrip(os.sep))
    slug = re.sub(r'[^A-Za-z0-9_-]+', '-', base)[:40].strip('-')
    return (slug or 'dataset') + '-' + \
        hashlib.sha1(root.encode('utf-8', 'replace')).hexdigest()[:8]


# ── the descriptor ──────────────────────────────────────────────────────────

_SPECS = {}


def descriptor(path):
    """{path, spec, names, root} for a dataset yaml, or None.

    The flat keys come from run_mistakes._spec(), which is the reader of
    record: the scorer decides whether a detector can be scored by reading the
    same file, and a second parser here would eventually disagree with it
    about which directory a run trained on. It stops at the first indented
    line, so the class list is read separately below.

    Cached by (path, mtime) -- the page re-renders on a timer and this is
    otherwise a file open per dataset per render.
    """
    if not path:
        return None
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    hit = _SPECS.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        spec = _mistakes()._spec(path) or {}
    except Exception:
        spec = {}
    if not isinstance(spec, dict):
        spec = {}
    names = spec.get('names')
    if not isinstance(names, (dict, list)):
        names = _names(path)
    doc = {'path': path, 'name': os.path.basename(path),
           'spec': {k: v for k, v in spec.items() if k != 'names'},
           'names': _name_list(names),
           'root': _root_of(path, spec)}
    _SPECS[path] = (mt, doc)
    return doc


def _names(path):
    """The class names out of a dataset yaml's `names:` block.

    Hand-read for the same reason run_mistakes reads the flat keys by hand:
    there is no PyYAML in the dashboard's interpreter, and with the parser
    behind a try/except a missing library is indistinguishable from a dataset
    with no classes. Understands the two forms these files are written in --
    an indented `0: target` block, and an inline `[dog, not_dog]`.
    """
    out, block = {}, False
    try:
        with open(path) as fh:
            for ln in fh:
                if ln.lstrip().startswith('#'):
                    continue
                if ln[:1].strip():
                    k, sep, v = ln.partition(':')
                    if not sep:
                        block = False
                        continue
                    block = k.strip() == 'names'
                    v = v.split('#')[0].strip()
                    if block and v:
                        return [x.strip().strip('"\'')
                                for x in v.strip('[]').split(',') if x.strip()]
                elif block:
                    k, sep, v = ln.partition(':')
                    if sep and k.strip():
                        out[k.strip()] = v.split('#')[0].strip().strip('"\'')
    except OSError:
        return []
    return out


def _name_list(names):
    """Class names in index order, however the yaml wrote them."""
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        def order(k):
            try:
                return (0, int(k))
            except (TypeError, ValueError):
                return (1, 0)
        return [str(names[k]) for k in sorted(names, key=order)]
    return []


def _root_of(yaml_path, spec):
    """The directory a dataset yaml describes.

    Usually the yaml's own directory, and `path:` is what makes it not: the
    descriptor at the training root points at a Label Studio export somewhere
    else entirely, and following it is the difference between naming the
    dataset thirty-nine runs used and naming the directory the file sits in.
    """
    base = str((spec or {}).get('path') or '').strip()
    if not base:
        return os.path.dirname(os.path.abspath(yaml_path))
    if not os.path.isabs(base):
        base = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), base)
    return os.path.abspath(base)


def _descriptor_in(root):
    """The dataset yaml sitting in a dataset root, if there is one."""
    files = _entries(root)[0]
    for want in DESCRIPTORS:
        if want in files:
            return os.path.join(root, want)
    loose = [f for f in files if f.lower().endswith(('.yaml', '.yml'))]
    # only when it is unambiguous: a root with three yamls in it is a root
    # where picking one at random would attribute the wrong classes
    return os.path.join(root, loose[0]) if len(loose) == 1 else None


# ── manifests ───────────────────────────────────────────────────────────────

_MANS = {}
# dogdet_v3's manifest is 145 KB, nearly all of it a holdout list of image ids.
# Reading it is fine; shipping it inside every row of the dataset list is not,
# so the list carries a summary and manifest() hands over the whole document
# when a dataset is actually opened.
MAN_MAX = 4 << 20


def manifest_path(root):
    """The build manifest in a dataset root, or None."""
    files = _entries(root)[0]
    for want in MANIFESTS + ROW_MANIFESTS:
        if want in files:
            return os.path.join(root, want)
    return None


def manifest(key_or_path):
    """The whole build manifest for a dataset, or None.

    Takes a dataset key or a path so the serving layer can ask by key without
    resolving anything itself. A .jsonl is never parsed -- it is one row per
    crop and can be tens of thousands of lines -- only described.
    """
    path = key_or_path
    if path and os.path.sep not in str(path):
        row = dataset(path)
        path = (row or {}).get('manifest', {}).get('path')
    return _manifest_doc(path)


def _manifest_doc(path):
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    hit = _MANS.get(path)
    if hit and hit[0] == st.st_mtime:
        return hit[1]
    doc = None
    if path.endswith('.json') and st.st_size <= MAN_MAX:
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            doc = None
    got = {'path': path, 'name': os.path.basename(path),
           'size': st.st_size, 'mtime': st.st_mtime, 'doc': doc}
    _MANS[path] = (st.st_mtime, got)
    return got


def _summary(man):
    """A manifest small enough to sit in a list row.

    Scalars survive; a list or an object becomes its length. The holdout list
    in dogdet_v3's manifest is a thousand image ids and the fact worth showing
    is that there are a thousand of them.
    """
    if not man:
        return None
    out = {k: v for k, v in man.items() if k != 'doc'}
    doc, small = man.get('doc'), {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                small[k] = v
            elif isinstance(v, (list, tuple, dict)):
                small[k] = f'{len(v)} items'
    out['summary'] = small
    return out


_ROWS = {}
# A per-crop manifest is one line per file and nothing bounds how many lines
# that is. Enough of one to learn which labels a build wrote, and no more: a
# label appearing for the first time in row 200,001 is not a label this page
# has to name, and reading 40 MB of json per render to find it would be.
ROW_MAX = 4 << 20
# One entry per flat crop export, so a handful; four of them exist here.
ROW_KEEP = 32
# What a per-crop row calls the thing it recorded. 'label' is what both this
# repo's exporters write; the other two are what the audit ledgers use.
LABEL_KEYS = ('label', 'class', 'verdict')


def row_labels(path):
    """{label: rows} out of a per-crop manifest, or {}.

    WHY THIS EXISTS. data/hard_negatives holds crops/ and full/ -- one crop cut
    out, and the frame it came from, under the same name in both -- beside a
    labels.jsonl in which every row reads false_positive. Read the class list
    off the directories there and the page prints two VIEWS as two classes and
    draws them at 50/50, which is a sentence about the layout standing where a
    sentence about the data belongs. The labels are in the manifest, so this
    reads them from there.

    Cached against the file's mtime, and bounded by ROW_MAX: a manifest bigger
    than that is described by its first few megabytes rather than not at all.
    """
    if not path:
        return {}
    try:
        mt = os.stat(path).st_mtime
    except OSError:
        return {}
    hit = _ROWS.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    out, read = {}, 0
    try:
        with open(path) as fh:
            for ln in fh:
                read += len(ln)
                if read > ROW_MAX:
                    break
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except ValueError:
                    continue       # a half-written last line during an export
                if not isinstance(row, dict):
                    continue
                for k in LABEL_KEYS:
                    v = row.get(k)
                    if isinstance(v, str) and v:
                        out[v] = out.get(v, 0) + 1
                        break
    except OSError:
        return {}
    if len(_ROWS) >= ROW_KEEP:
        _ROWS.clear()
    _ROWS[path] = (mt, out)
    return out


# ── counting, cached against the directory's mtime ──────────────────────────

_LOCK = threading.Lock()
_SIZES = None
_DIRTY = False
_FLUSHED = 0.0
# One row per directory, so a few dozen for everything on this box. A tree big
# enough to blow past this is not a dataset tree, and a cache that outgrows
# its bound is cheaper to rebuild than to curate.
CACHE_MAX = 20000
FLUSH_EVERY = 60


def _cache():
    global _SIZES
    if _SIZES is None:
        with _LOCK:
            if _SIZES is None:
                try:
                    with open(paths()['sizes']) as fh:
                        got = json.load(fh)
                except (OSError, ValueError):
                    got = {}
                _SIZES = got if isinstance(got, dict) else {}
    return _SIZES


def _own(path):
    """[mtime, images, bytes, files, subdirs] for one directory.

    CACHED AGAINST THE DIRECTORY'S OWN MTIME, which is the entire cost story
    of this module. Adding or removing a file moves the mtime of the directory
    holding it, so a tree nobody has touched is re-measured with one stat per
    DIRECTORY -- seven of them for a classify dataset -- instead of one stat
    per file. Cold, or after a rebuild, it is a full walk; that is paid once
    and then persisted, so a dashboard restart does not pay it again.

    Only the directory's OWN files are cached. The totals are summed back up
    from the children on every call, because a directory's mtime does not move
    when something changes two levels below it -- a cached total would be a
    number that is quietly wrong, which is worse than a slow one.

    What it cannot see is a file rewritten in place under the same name.
    Nothing in either repo does that; a rebuild writes a new directory.
    """
    global _DIRTY
    try:
        mt = os.stat(path).st_mtime
    except OSError:
        return None
    cache = _cache()
    row = cache.get(path)
    if isinstance(row, list) and len(row) == 5 and row[0] == mt:
        return row
    files, dirs = _entries(path)
    images = size = 0
    for f in files:
        if f.lower().endswith(IMAGE_EXT):
            images += 1
        try:
            size += os.stat(os.path.join(path, f)).st_size
        except OSError:
            continue    # a live run can unlink between the listing and here
    row = [mt, images, size, len(files), sorted(dirs)]
    with _LOCK:
        if len(cache) >= CACHE_MAX:
            cache.clear()
        cache[path] = row
        _DIRTY = True
    return row


def _flush(force=False):
    """Persist the size cache. Best effort -- it is only ever a cache."""
    global _DIRTY, _FLUSHED
    now = time.time()
    with _LOCK:
        if not _DIRTY or (not force and now - _FLUSHED < FLUSH_EVERY):
            return
        doc, _DIRTY, _FLUSHED = dict(_cache()), False, now
    P = paths()
    try:
        os.makedirs(P['out'], exist_ok=True)
        tmp = P['sizes'] + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(doc, fh)
        # swapped in, never written over: the dashboard reads this file from
        # another thread while it is being replaced
        os.replace(tmp, P['sizes'])
    except OSError:
        pass


def measure(root, rel='', depth=0):
    """One directory and everything under it, as a node the page can render.

    ``images`` and ``bytes`` are totals including descendants; ``own`` is what
    sits directly in this directory. Directories only -- listing three
    thousand filenames is what listing() is for.
    """
    here = os.path.join(root, rel) if rel else root
    row = _own(here)
    if row is None:
        return {'name': os.path.basename(here) or here, 'rel': rel,
                'images': 0, 'bytes': 0, 'files': 0, 'own_images': 0,
                'mtime': 0, 'dirs': []}
    mt, images, size, files, subs = row
    node = {'name': os.path.basename(here) or here, 'rel': rel,
            'images': images, 'bytes': size, 'files': files,
            'own_images': images, 'mtime': mt, 'dirs': []}
    if depth < MAX_DEPTH:
        for d in subs:
            kid = measure(root, os.path.join(rel, d) if rel else d, depth + 1)
            node['images'] += kid['images']
            node['bytes'] += kid['bytes']
            node['dirs'].append(kid)
    return node


# ── discovery ───────────────────────────────────────────────────────────────

def resolve_data(raw, run_dir, root):
    """Where a run's `data` string points, whether or not it is still there.

    run_mistakes.resolve_data() is the resolver of record -- reusing it is how
    this page and the scorer are kept from disagreeing about which file a run
    trained on -- but it answers '' for anything that no longer exists, and a
    dataset that is GONE is precisely what this page has to be able to say. So
    its answer is taken when it has one, and otherwise the same string is
    resolved the same way and handed back unverified.
    """
    raw = str(raw or '').strip()
    if not raw:
        return ''
    try:
        got = _mistakes().resolve_data(raw, run_dir, root)
    except Exception:
        got = ''
    if got:
        return got
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(root or run_dir or '', raw))


def _from_data(path):
    """(dataset root, descriptor path) for what a run's `data` resolved to.

    A path ending .yaml is the DESCRIPTOR, not the dataset; the dataset is
    whatever its `path:` names, which is usually but not always the directory
    the file sits in.
    """
    if path.lower().endswith(('.yaml', '.yml')):
        doc = descriptor(path)
        if doc:
            return doc['root'], path
        # The descriptor has gone as well, so which directory it named cannot
        # be known -- and the directory it SAT in is not a safe guess: a bare
        # `data: dataset.yaml` resolves against the training root, so guessing
        # would put the training root itself on the page as a dataset and then
        # measure it, walking every run directory underneath. The row is the
        # descriptor the runs recorded, and it says it is gone.
        return os.path.abspath(path), path
    return os.path.abspath(path), _descriptor_in(path)


def scan_roots():
    """Where datasets get built on this machine.

    The training root, because that is where the training repo assembles them,
    and this repo's data/, because its own export tools write crop datasets
    there. Never a grid root: those hold the harvest, they are spread over five
    drives and they are millions of files.
    """
    out = []
    try:
        troot = _dash().training_root()
    except Exception:
        troot = ''
    for d in (troot, os.path.join(REPO, 'data')):
        if not d:
            continue
        real = os.path.realpath(d)
        if os.path.isdir(real) and real not in out:
            out.append(real)
    return out


def _scan(root, depth=SCAN_DEPTH):
    """[(dataset root, kind)] under one scan root, recognised by shape."""
    found, stack = [], [(root, 0)]
    while stack:
        path, d = stack.pop()
        for name in _entries(path)[1]:
            if name.startswith('.') or name in SKIP_DIRS:
                continue
            sub = os.path.join(path, name)
            kind = shape(sub)
            if kind != 'unknown':
                found.append((sub, kind))
                continue      # a dataset's own images/ is not another dataset
            if d + 1 < depth:
                stack.append((sub, d + 1))
    return found


def _blank(root, kind):
    return {'key': key_for(root), 'name': os.path.basename(root) or root,
            'root': root, 'kind': kind, 'exists': os.path.isdir(root),
            'images': 0, 'bytes': 0, 'folders': 0, 'mtime': 0.0,
            'classes': [], 'labels': {}, 'splits': [], 'descriptor': None,
            'manifest': None, 'runs': [], 'found': []}


def _fill(row):
    """Everything about a dataset that costs a look at the disk."""
    root = row['root']
    row['exists'] = os.path.isdir(root)
    if row['exists']:
        if row['kind'] == 'unknown':
            row['kind'] = shape(root)
        node = measure(root)
        row['images'], row['bytes'] = node['images'], node['bytes']
        row['mtime'] = node['mtime']
        row['folders'] = _count_dirs(node)
        row['splits'] = [d['name'] for d in node['dirs']]
        row['classes'] = _classes(root, row['kind'])
        # The label counts, and only where they are what the class list was
        # read from. A split set's classes are its directories, and a bar
        # drawn off manifest ROWS beside a folder that holds FILES would put
        # two denominators in one panel -- which is the trap that has already
        # cost this project a phantom count once.
        if row['kind'] == 'classify' and not any(
                os.path.isdir(os.path.join(root, s)) for s in SPLITS):
            row['labels'] = _flat_labels(root)
        if not row['descriptor']:
            row['descriptor'] = _descriptor_in(root)
        row['manifest'] = _summary(_manifest_doc(manifest_path(root)))
    if row['descriptor'] and not isinstance(row['descriptor'], dict):
        row['descriptor'] = descriptor(row['descriptor'])
    if isinstance(row['descriptor'], dict):
        if row['kind'] == 'unknown':
            # The directory is gone, so its layout cannot be read -- but the
            # fact that a run pointed at a yaml at all says what it was:
            # ultralytics takes a descriptor for a detector and a bare
            # directory for a classifier. "38 runs trained on a detect
            # dataset that is gone" is the sentence the page needs.
            row['kind'] = 'detect'
        # a detector's classes only exist in the yaml -- there are no class
        # directories to read them off
        row['classes'] = row['classes'] or row['descriptor']['names']
    return row


def _count_dirs(node):
    return 1 + sum(_count_dirs(k) for k in node['dirs'])


def _classes(root, kind):
    """The class names a classify dataset is split into.

    For a SPLIT set, off the disk rather than out of the manifest, because the
    manifest records what a build INTENDED and the directories are what it left
    behind.

    For the FLAT layout the manifest wins, and data/hard_negatives is why: its
    two directories are a crop and the frame it was cut from, not two labels,
    and every row of its labels.jsonl carries the same one word. Directory
    names there answer a question nobody asked. The manifest is only believed
    when it has labels in it -- an export created and not yet filled falls back
    to the directories, which is all it has.
    """
    if kind != 'classify':
        return []
    for s in SPLITS:
        d = os.path.join(root, s)
        if os.path.isdir(d):
            return sorted(_entries(d)[1])
    got = _flat_labels(root)
    return sorted(got) if got else sorted(_entries(root)[1])


def _flat_labels(root):
    """{label: rows} for a flat crop export, or {} -- see row_labels()."""
    man = manifest_path(root)
    return row_labels(man) if man and man.endswith('.jsonl') else {}


_INDEX = {'at': 0.0, 'rows': [], 'by_key': {}, 'error': None}
# Matches the tracker's own TTL. Datasets do not appear faster than runs do,
# and resolve() -- which every image request goes through -- must not pay for
# a fresh scan per tile.
INDEX_TTL = 20
# One walk at a time, and everyone else reads its answer. The server gives
# every request its own thread, so a page of a hundred tiles landing on an
# index that has just gone stale used to start a hundred independent walks of
# both roots -- across the same disks the live training run is reading from,
# each one re-stat'ing every directory whose mtime had moved.
_SCAN_LOCK = threading.Lock()


def datasets(refresh=False):
    """Every dataset discovered right now, newest first.

    Newest by the root's mtime; a dataset that is gone borrows the mtime of
    the most recent run that trained on it, which is the only date it still
    has and keeps it from sinking to the bottom of the page.
    """
    now, at = time.time(), _INDEX['at']
    if not refresh and now - at < INDEX_TTL:
        return _INDEX['rows']
    with _SCAN_LOCK:
        # Whoever was ahead in the queue has walked while this call waited, so
        # their answer is this one's -- it was taken after the request that is
        # asking. The test is on the index having MOVED and not on its age:
        # age alone would answer a rescan click with the cached rows it was
        # pressed to get past.
        if _INDEX['at'] != at:
            return _INDEX['rows']
        return _discover(now)


def _discover(now):
    """The walk itself, run with _SCAN_LOCK held. See datasets()."""
    rows, err = {}, None

    def row_for(root, kind, how):
        root = os.path.realpath(root)
        got = rows.get(root)
        if got is None:
            got = rows[root] = _blank(root, kind)
        if kind != 'unknown':
            got['kind'] = kind
        if how not in got['found']:
            got['found'].append(how)
        return got

    # pass one: what the runs say they trained on
    troot = ''
    try:
        dash = _dash()
        troot = dash.training_root()
        runs = dash.training_runs()
    except Exception as e:
        # a dashboard that will not import must not empty the page -- the
        # disk pass below still finds every dataset that is actually there
        runs, err = [], f'{type(e).__name__}: {e}'
    for r in runs:
        got = resolve_data(r.get('data'), r.get('dir') or '', troot)
        if not got:
            continue
        root, desc = _from_data(got)
        row = row_for(root, 'unknown', 'run')
        if desc and not row['descriptor']:
            row['descriptor'] = desc
        row['runs'].append({
            'key': f"{r.get('project')}/{r.get('name')}",
            'project': r.get('project'), 'name': r.get('name'),
            'task': r.get('task'), 'status': r.get('status'),
            'live': bool(r.get('live')), 'mtime': r.get('mtime') or 0,
            'dir': r.get('dir')})

    # pass two: what is on the disk, whether or not anything trained on it
    for base in scan_roots():
        for root, kind in _scan(base):
            row_for(root, kind, 'scan')

    out = []
    for root in sorted(rows):
        row = _fill(rows[root])
        row['runs'].sort(key=lambda x: x['mtime'] or 0, reverse=True)
        if not row['mtime'] and row['runs']:
            row['mtime'] = row['runs'][0]['mtime'] or 0
        out.append(row)
    # Newest first, but everything still on disk before anything that is gone:
    # a deleted dataset keeps the date of the last run that used it, and that
    # date can be recent enough to put it at the top of a page it is not the
    # subject of.
    out.sort(key=lambda x: (x['exists'], x['mtime'] or 0), reverse=True)
    _flush()
    _INDEX.update(at=now, rows=out, error=err,
                  by_key={r['key']: r for r in out})
    return out


def dataset(key):
    """One dataset by key, or None.

    A key that is not in the index is answered None rather than rescanning to
    look for it: everything the page serves goes through resolve(), which goes
    through here, and a made-up key must not be able to buy a walk of every
    dataset root per request. The key is therefore looked up FIRST, and an age
    check on the index is not allowed to run a walk on its behalf -- that was
    the shape of it before, and it handed a stranger the whole scan for the
    cost of a query string.

    The one walk this does buy is the first in the process. A dashboard that
    has just restarted has discovered nothing yet, and an open page's next
    thumbnail would otherwise 404 until somebody reloaded it. Nothing newer
    than the last walk is reachable this way and nothing needs to be: a key is
    only ever learned from the list, and asking for the list is what refreshes
    it.
    """
    if not key:
        return None
    if not _INDEX['at']:
        datasets()
    return _INDEX['by_key'].get(key)


# ── the security primitive ──────────────────────────────────────────────────

def resolve(key, rel=''):
    """The absolute path for `rel` inside dataset `key`, or None.

    This is the one function the serving layer's safety rests on: everything
    the page reads -- a thumbnail, a listing, a label file -- is a string that
    came off a query string, and it is only ever opened after passing through
    here.

    Both sides are realpath'd before they are compared, which is what makes a
    symlink inside a dataset pointing at /etc fail rather than succeed; the
    separator is appended to the root before the prefix test, because
    '/data/foo_old' starts with '/data/foo' and is a different dataset.
    Absolute rel, any '..' segment and any NUL are refused outright rather
    than normalised away, since none of them is ever a thing the page asks
    for. Returns None instead of raising: a bad path is an answer, and the
    caller is rendering a page.
    """
    row = dataset(key)
    if not row:
        return None
    try:
        root = os.path.realpath(row['root'])
        rel = str(rel or '')
        if '\0' in rel or os.path.isabs(rel) or rel.startswith(('/', '\\')):
            return None
        parts = [p for p in rel.replace('\\', '/').split('/')
                 if p not in ('', '.')]
        if any(p == '..' for p in parts):
            return None
        full = os.path.realpath(os.path.join(root, *parts)) if parts else root
    except (ValueError, OSError):
        return None
    if full != root and not full.startswith(root + os.sep):
        return None
    return full


# ── opening one ─────────────────────────────────────────────────────────────

def tree(key):
    """The folder structure under a dataset: directories, counts and bytes.

    Directories only. A detect dataset has two folders of images and two of
    labels; naming their three thousand files here is what listing() is for.
    """
    row = dataset(key)
    if not row:
        return {'ok': False, 'error': 'no such dataset'}
    if not row['exists']:
        return {'ok': False, 'error': 'the directory is gone', 'row': row}
    node = measure(row['root'])
    _flush()
    return {'ok': True, 'key': key, 'root': row['root'], 'kind': row['kind'],
            'tree': node}


_PAGES = {}
# Only ever the directory being paged through, plus whatever was open before
# it. Kept so that walking a 200,000-image folder page by page enumerates and
# sorts it once instead of once per page.
PAGE_KEEP = 8


def _sorted(path):
    """(file names, directory names), sorted, cached against the mtime."""
    try:
        mt = os.stat(path).st_mtime
    except OSError:
        return [], []
    hit = _PAGES.get(path)
    if hit and hit[0] == mt:
        return hit[1], hit[2]
    files, dirs = _entries(path)
    files.sort()
    dirs.sort()
    if len(_PAGES) >= PAGE_KEEP:
        _PAGES.clear()
    _PAGES[path] = (mt, files, dirs)
    return files, dirs


def _label_lines(root, rel):
    """How many boxes the label file facing this image holds, or None.

    A detect dataset mirrors images/<split>/<stem>.jpg at
    labels/<split>/<stem>.txt. A missing file is not missing information --
    it is a background image, deliberately put there with nothing to find --
    so it counts 0 once the labels/ directory itself exists. Asked only of the
    files on the page being served, which is one small read each.
    """
    parts = [p for p in rel.split('/') if p]
    if len(parts) < 2 or parts[0] != 'images':
        return None
    if not os.path.isdir(os.path.join(root, 'labels')):
        return None
    lab = os.path.join(root, 'labels', *parts[1:])
    try:
        lab = os.path.splitext(lab)[0] + '.txt'
        with open(lab) as fh:
            return sum(1 for ln in fh if ln.strip())
    except OSError:
        return 0


def listing(key, rel='', page=0, n=120):
    """One page of the files in one directory of a dataset.

    Sorted by name, because a page number has to mean the same thing twice.
    """
    row = dataset(key)
    if not row:
        return {'ok': False, 'error': 'no such dataset'}
    full = resolve(key, rel)
    if not full or not os.path.isdir(full):
        return {'ok': False, 'error': 'no such folder'}
    # Re-derived from the resolved path rather than echoed back, so the rels
    # this hands out are canonical: './train' and 'train//dog' both arrive,
    # both resolve, and a page that links on to the string it was given would
    # otherwise keep growing it.
    rel = os.path.relpath(full, row['root'])
    rel = '' if rel == os.curdir else rel
    try:
        n = max(1, min(int(n), 1000))
        page = max(0, int(page))
    except (TypeError, ValueError):
        n, page = 120, 0
    files, dirs = _sorted(full)
    pages = (len(files) + n - 1) // n
    out = []
    for name in files[page * n:page * n + n]:
        sub = (rel.rstrip('/') + '/' + name) if rel else name
        try:
            st = os.stat(os.path.join(full, name))
            size, mt = st.st_size, st.st_mtime
        except OSError:
            size, mt = 0, 0     # unlinked since the listing; still show it
        image = name.lower().endswith(IMAGE_EXT)
        out.append({'name': name, 'rel': sub, 'size': size, 'mtime': mt,
                    'image': image,
                    'labels': _label_lines(row['root'], sub)
                    if image and row['kind'] == 'detect' else None})
    return {'ok': True, 'key': key, 'rel': rel, 'kind': row['kind'],
            'total': len(files), 'page': page, 'pages': pages, 'n': n,
            'dirs': [{'name': d, 'rel': (rel.rstrip('/') + '/' + d)
                      if rel else d} for d in dirs],
            'files': out}


# ── a quick look from the command line ──────────────────────────────────────

def _human(b):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if b < 1024 or unit == 'TB':
            return f'{b:.0f} {unit}' if unit == 'B' else f'{b:.1f} {unit}'
        b /= 1024.0


def _print_tree(node, indent='  '):
    print(f"{indent}{node['name']}/  {node['images']:,} images  "
          f"{_human(node['bytes'])}")
    for kid in node['dirs']:
        _print_tree(kid, indent + '  ')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('key', nargs='?', help='a dataset key, or part of a name')
    ap.add_argument('rel', nargs='?', default='',
                    help='a folder inside it, to list its files')
    ap.add_argument('--page', type=int, default=0)
    ap.add_argument('-n', type=int, default=20)
    a = ap.parse_args(argv)
    try:
        return _look(a)
    finally:
        # what this run measured is worth keeping whichever way it went out;
        # the next look, and the dashboard's first render, are both faster for
        # it
        _flush(force=True)


def _look(a):
    rows = datasets(refresh=True)
    if _INDEX['error']:
        print(f"NOTE: the run list is unavailable ({_INDEX['error']}) -- what "
              f"follows is the disk scan only\n")
    if not a.key:
        print(f'{len(rows)} datasets\n')
        for r in rows:
            gone = '' if r['exists'] else '  GONE'
            print(f"  {r['key']:<28} {r['kind']:<9} {r['images']:>8,} images  "
                  f"{_human(r['bytes']):>9}  {len(r['runs']):>2} runs  "
                  f"{','.join(r['found'])}{gone}")
            print(f"      {r['name']}"
                  + (f"  [{', '.join(r['classes'])}]" if r['classes'] else ''))
        return 0

    hit = dataset(a.key) or next(
        (r for r in rows if a.key in r['key'] or a.key in r['name']), None)
    if not hit:
        print(f'no dataset matching {a.key!r}')
        return 1
    if a.rel:
        got = listing(hit['key'], a.rel, a.page, a.n)
        if not got['ok']:
            print(got['error'])
            return 1
        print(f"{hit['name']}/{a.rel}  {got['total']:,} files, "
              f"page {got['page'] + 1} of {got['pages']}")
        for f in got['files']:
            lab = '' if f['labels'] is None else f"  {f['labels']} boxes"
            print(f"  {f['name']:<44} {_human(f['size']):>9}{lab}")
        return 0
    print(f"{hit['name']}  ({hit['kind']})\n  {hit['root']}")
    if not hit['exists']:
        print('  the directory is gone')
    for r in hit['runs']:
        print(f"  trained: {r['key']:<30} {r['task'] or '?':<9} {r['status']}")
    got = tree(hit['key'])
    if got['ok']:
        _print_tree(got['tree'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
