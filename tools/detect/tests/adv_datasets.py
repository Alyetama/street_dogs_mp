#!/usr/bin/env python3
"""The datasets page turns query strings into file reads. This is the guard.

    python tools/detect/tests/adv_datasets.py

Everything here defends one of the claims the feature makes:

  * a path that came off a query string is opened only after it has been
    proved to sit inside one dataset root -- and the image routes serve the
    file they actually open, not the one they were asked for by name;
  * a dataset built five minutes ago is on the page without anyone editing a
    file, and a dataset that has been deleted is a row that says so rather
    than a row that is missing;
  * the counts are the filesystem's, and a page number means the same thing
    twice;
  * a file name is a string off a disk anybody can unpack an export onto, so
    it lands on the page as text and never as markup.

The fixtures are a temp directory and the index is pointed at it -- nothing
here reads, writes or measures a real dataset except the one live pass in
live_checks(), which only counts. Nothing anywhere writes into one.

Every check is written to fail if the defect it names comes back; a check that
cannot be made to fail is a certificate of nothing, and this suite has shipped
three of those before. Where a check cannot run at all -- no node, no PIL --
it SAYS it skipped, because a guard that prints nothing looks exactly like a
guard that passed.
"""

import html.parser
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
# Guarded, because one of the checks below counts sys.path: a test that
# duplicates an entry itself would report its own doing as the defect.
MINE = (os.path.join(REPO, 'tools', 'dashboard'),
        os.path.join(REPO, 'tools', 'detect'))
for _p in MINE:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    Image, HAVE_PIL = None, False

# A name that is a valid file name and an attribute break-out at the same
# time. Everything on this page that prints a name prints one of these, and
# what it must never do is turn into markup.
EVIL_FILE = 'a" onload="BOOM(1)" x.jpg'
EVIL_DIR = 'q" onx="1'
# More than one page at the smallest size the interface offers, so the last
# page is short and the boundary is a real boundary.
N_TRAIN = 130


# ── the fixture ─────────────────────────────────────────────────────────────

def _jpeg(path):
    """A picture. Really decodable where PIL is here, bytes where it is not.

    The thumbnail cutter decodes what it is given, so the one cut this suite
    asks for has to be of a real picture. Everything else in the fixture
    exists to be counted, listed and paged through, and PIL is not needed for
    any of that -- so where it is missing the counts are still checked and the
    cut says it was skipped.
    """
    if HAVE_PIL:
        Image.new('RGB', (16, 12), (90, 110, 130)).save(path, 'JPEG')
        return
    with open(path, 'wb') as fh:
        fh.write(b'\xff\xd8\xff\xdb' + b'0' * 64)


def build_fixture(base):
    """Three datasets in the two shapes this machine builds, plus a ghost.

    Laid out to match what is really here: a detect set with its descriptor
    and its label files, a split classify set, a flat crop export whose
    directories are VIEWS of one set of crops rather than classes, and a
    descriptor whose `path:` names a directory that does not exist -- which is
    the shape of the export thirty-eight real runs trained on.
    """
    det = os.path.join(base, 'det_v1')
    for d in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(det, d))
    for i in range(N_TRAIN):
        _jpeg(os.path.join(det, 'images/train', 'f%03d.jpg' % i))
        with open(os.path.join(det, 'labels/train', 'f%03d.txt' % i), 'w') as f:
            f.write('0 0.5 0.5 0.2 0.2\n' * (i % 3))
    # the hostile name lives in the split the page opens on
    _jpeg(os.path.join(det, 'images/train', EVIL_FILE))
    for i in range(4):
        _jpeg(os.path.join(det, 'images/val', 'v%03d.jpg' % i))
    with open(os.path.join(det, 'dataset.yaml'), 'w') as f:
        f.write('path: .\ntrain: images/train\nval: images/val\n'
                'nc: 1\nnames: [target]\n')
    with open(os.path.join(det, 'manifest.json'), 'w') as f:
        json.dump({'built': 'fixture', 'train': N_TRAIN, 'holdout': [1, 2, 3]},
                  f)

    cls = os.path.join(base, 'cls_v1')
    for split, name, n in (('train', 'dog', 5), ('train', EVIL_DIR, 3),
                           ('val', 'dog', 2), ('val', EVIL_DIR, 1)):
        d = os.path.join(cls, split, name)
        os.makedirs(d)
        for i in range(n):
            _jpeg(os.path.join(d, 'c%03d.jpg' % i))

    # The flat crop export. crops/ and full/ hold the SAME names -- the cut
    # out and the frame it came from -- and the label is a column in the row
    # manifest, identical on every row. Calling those two directories two
    # classes is the defect this shape exists here to catch.
    flat = os.path.join(base, 'flat_v1')
    for d in ('crops', 'full'):
        os.makedirs(os.path.join(flat, d))
    with open(os.path.join(flat, 'labels.jsonl'), 'w') as f:
        for i in range(6):
            _jpeg(os.path.join(flat, 'crops', 'k%03d.jpg' % i))
            _jpeg(os.path.join(flat, 'full', 'k%03d.jpg' % i))
            f.write(json.dumps({'crop': 'k%03d.jpg' % i,
                                'label': 'false_positive'}) + '\n')

    # A descriptor at the training root pointing somewhere that is not there.
    with open(os.path.join(base, 'dataset.yaml'), 'w') as f:
        f.write('path: %s\ntrain: images/train\nval: images/val\n'
                'names:\n  0: target\n' % os.path.join(base, 'ghost_export'))

    # ...and a directory of that same basename, archived one level down. This
    # is the real pair on this machine: the runs recorded a path that is gone
    # and the export was moved rather than deleted, so the page carries two
    # rows reading the same name, one of them saying nothing can be opened.
    arch = os.path.join(base, 'archived', 'ghost_export')
    for d in ('images/train', 'labels/train'):
        os.makedirs(os.path.join(arch, d))
    for i in range(3):
        _jpeg(os.path.join(arch, 'images/train', 'a%03d.jpg' % i))

    # Something to escape TO, and a link that tries to.
    with open(os.path.join(base, 'outside.txt'), 'w') as f:
        f.write('not yours\n')
    try:
        os.symlink(os.path.join(base, 'outside.txt'),
                   os.path.join(det, 'images/train/escape.jpg'))
        os.symlink('../../dataset.yaml',
                   os.path.join(det, 'images/train/sneaky.jpg'))
    except OSError:
        pass       # a filesystem without links; the traversal checks say so

    # A symlink-ASSEMBLED dataset: every image is a link into det_v1, and only
    # the links are its own. This is how exports and dedup'd builds avoid
    # copying -- the real pairs on this machine are holdout180 -> dogdet_v3
    # and leash_src -> leash_3class_v2 -- and refusing the off-root realpath
    # put "would not open" on every tile of a dataset whose files were fine.
    linked = os.path.join(base, 'linked_v1')
    for d in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(linked, d))
    with open(os.path.join(linked, 'dataset.yaml'), 'w') as f:
        f.write('path: .\ntrain: images/train\nval: images/val\n'
                'nc: 1\nnames: [target]\n')
    try:
        for i in range(3):
            os.symlink(os.path.join(det, 'images/train', 'f%03d.jpg' % i),
                       os.path.join(linked, 'images/train', 'l%03d.jpg' % i))
        os.symlink(os.path.join(det, 'images/val', 'v000.jpg'),
                   os.path.join(linked, 'images/val', 'l000.jpg'))
        # the sharpened yaml hazard: a CROSS-dataset link called x.jpg
        # pointing at the other dataset's descriptor. Containment now lets
        # the realpath through, so the resolved-extension allow-list is the
        # only thing standing between this name and a yaml served as JPEG.
        os.symlink(os.path.join(det, 'dataset.yaml'),
                   os.path.join(linked, 'images/train', 'sneak2.jpg'))
    except OSError:
        pass       # a filesystem without links; the checks say so
    return {'base': base, 'det': det, 'cls': cls, 'flat': flat,
            'linked': linked}


class _FakeDash:
    """training_root() and training_runs(), and nothing else is asked of it.

    The runs are invented so the gone-dataset check does not depend on this
    machine still having the deleted Label Studio export on it. Their `data`
    strings are the two shapes that really occur -- a bare name resolved
    against the training root, and an absolute directory -- and they go
    through run_mistakes' own resolver, which is the point.
    """

    def __init__(self, root, runs):
        self.root, self.runs = root, runs

    def training_root(self):
        return self.root

    def training_runs(self):
        return list(self.runs)


def _runs(fx):
    now = time.time()
    return [
        {'project': 'p', 'name': 'det_001', 'task': 'detect', 'status': 'done',
         'data': fx['det'], 'dir': os.path.join(fx['base'], 'runs', 'det_001'),
         'mtime': now - 100, 'live': False},
        {'project': 'p', 'name': 'ghost_001', 'task': 'detect',
         'status': 'done', 'data': 'dataset.yaml',
         'dir': os.path.join(fx['base'], 'runs', 'ghost_001'),
         'mtime': now - 200, 'live': False},
        {'project': 'p', 'name': 'ghost_002', 'task': 'detect',
         'status': 'stopped', 'data': 'dataset.yaml',
         'dir': os.path.join(fx['base'], 'runs', 'ghost_002'),
         'mtime': now - 300, 'live': False},
        # a bare name whose descriptor is gone as well: the resolver of record
        # answers nothing for it, and the row exists only because the same
        # string is resolved again and handed back unverified
        {'project': 'p', 'name': 'lost_001', 'task': 'detect',
         'status': 'done', 'data': 'gone_forever.yaml',
         'dir': os.path.join(fx['base'], 'runs', 'lost_001'),
         'mtime': now - 400, 'live': False},
    ]


class Fixture:
    """The index, pointed at a temp directory and put back afterwards.

    The size cache is redirected too. It is only a cache, but it lives under
    data/dashboard/ and a guard has no business writing fixture paths into a
    file the dashboard reads.
    """

    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix='adv_datasets_')
        self.paths = build_fixture(os.path.join(self.tmp, 'root'))
        self.saved = {}

    def __enter__(self):
        import dataset_index as ix
        import datasets as ds
        self.ix, self.ds = ix, ds
        out = os.path.join(self.tmp, 'cache')
        self.saved = {'dash': ix._dash, 'scan': ix.scan_roots,
                      'paths': ix.paths, 'sizes': ix._SIZES,
                      'index': dict(ix._INDEX), 'thumbs': ds.THUMB_DIR}
        fake = _FakeDash(self.paths['base'], _runs(self.paths))
        ix._dash = lambda: fake
        ix.scan_roots = lambda: [self.paths['base']]
        ix.paths = lambda: {'out': out,
                            'sizes': os.path.join(out, 'sizes.json')}
        ix._SIZES = {}
        ds.THUMB_DIR = os.path.join(self.tmp, 'thumbs')
        self.reset()
        return self

    def __exit__(self, *a):
        ix, ds = self.ix, self.ds
        ix._dash = self.saved['dash']
        ix.scan_roots = self.saved['scan']
        ix.paths = self.saved['paths']
        ix._SIZES = self.saved['sizes']
        ix._INDEX.clear()
        ix._INDEX.update(self.saved['index'])
        ix._PAGES.clear()
        ix._ROWS.clear()
        ds.THUMB_DIR = self.saved['thumbs']
        shutil.rmtree(self.tmp, ignore_errors=True)
        return False

    def reset(self):
        """Forget everything measured, so the next call really walks."""
        self.ix._INDEX.update(at=0.0, rows=[], by_key={}, error=None)
        self.ix._PAGES.clear()
        self.ix._ROWS.clear()

    def rows(self, refresh=True):
        return self.ix.datasets(refresh=refresh)

    def key(self, name):
        for r in self.rows(refresh=False) or self.rows():
            if r['name'] == name:
                return r['key']
        return None


def _row(fx, name):
    for r in fx.ix.datasets(refresh=False):
        if r['name'] == name:
            return r
    return None


# ── the security surface ────────────────────────────────────────────────────

# Every one of these is a string a browser can put in a query string, and each
# of them lands OUTSIDE the dataset if it is followed: climbing, absolute,
# separator-swapped, and a symlink that leaves the root.
ESCAPES = (
    '../outside.txt',
    '../../etc/passwd',
    'images/../../outside.txt',
    'images/train/../../../outside.txt',
    '..',
    './../outside.txt',
    '/etc/passwd',
    '/etc/passwd\0.jpg',
    '\\..\\..\\outside.txt',
    'images\\..\\..\\outside.txt',
    'images/train/escape.jpg',        # a symlink out of the root
    'images/./../../outside.txt',
)
# And these are refused even though every one of them lands back INSIDE the
# root. That is the point of them: the page never builds a path in any of
# these shapes, so accepting one means the string is being rewritten before it
# is checked rather than turned down. Refusal is a property the next reader
# can confirm off one line; "the rewrite is total" is a proof nobody has.
REFUSALS = (
    'images/../images/train/f000.jpg',
    'images/train/../../images/train/f000.jpg',
    '/images/train/f000.jpg',
    '/',
    'images/train/f000.jpg\0',
)


def traversal_checks(bad, fx):
    """No string from a query string opens a file outside its dataset.

    Checked at resolve(), which is the one door, AND at every route that goes
    through it -- the listing, the tree, the thumbnail and the original --
    because a route that grew its own path join would pass the first and fail
    the second.
    """
    ix, ds = fx.ix, fx.ds
    key = fx.key('det_v1')
    if not key:
        bad.append('the fixture detect dataset was not discovered at all — '
                   'nothing below tested anything')
        return
    root = os.path.realpath(_row(fx, 'det_v1')['root'])
    for rel in ESCAPES + REFUSALS:
        got = ix.resolve(key, rel)
        if got is not None and (got != root
                                and not got.startswith(root + os.sep)):
            bad.append(f'resolve({rel!r}) escaped the dataset root: {got}')
        elif got is not None:
            bad.append(f'resolve({rel!r}) answered {got!r} instead of '
                       f'refusing it — a climbing, absolute or NUL-bearing '
                       f'rel is turned down, not normalised into something '
                       f'that happens to land inside the root')
        # the routes, not just the primitive
        if ds.thumb(key, rel)[0] is not None:
            bad.append(f'/datasets/thumb served {rel!r}')
        if ds.full(key, rel)[0] is not None:
            bad.append(f'/datasets/image served {rel!r}')
        got = ds.api_files(key, rel)
        if got.get('ok'):
            bad.append(f'/api/datasets/files listed {rel!r} as '
                       f'{got.get("rel")!r}')
    # the escape link is the one that proves realpath is being used rather
    # than a string prefix test on the path as written
    if os.path.islink(os.path.join(root, 'images/train/escape.jpg')):
        if ix.resolve(key, 'images/train/escape.jpg') is not None:
            bad.append('a symlink out of the dataset resolved — the root is '
                       'being compared before the link is followed')
    else:
        print('SKIP: no symlinks on this filesystem — the link-escape case '
              'was not exercised')
    # a key nobody issued reaches nothing at all
    for k in ('', 'no-such-key', '../..', 'det_v1'):
        if ix.resolve(k, 'images/train/f000.jpg') is not None:
            bad.append(f'resolve() accepted the made-up key {k!r}')
        if ds.thumb(k, 'images/train/f000.jpg')[0] is not None:
            bad.append(f'/datasets/thumb accepted the made-up key {k!r}')
        if ix.tree(k).get('ok'):
            bad.append(f'/api/datasets/tree accepted the made-up key {k!r}')
        if ix.listing(k, '').get('ok'):
            bad.append(f'/api/datasets/files accepted the made-up key {k!r}')
    # and the legitimate path still works, or everything above passes because
    # nothing is served at all
    if ix.resolve(key, 'images/train/f000.jpg') is None:
        bad.append('resolve() refuses a file that is really in the dataset — '
                   'the checks above prove nothing')
    if ix.resolve(key, '') != root:
        bad.append('resolve() does not answer the root for an empty rel')


def symlink_dataset_checks(bad, fx):
    """A dataset assembled from links into ANOTHER dataset serves its images.

    holdout180 (180/180 images) and leash_src (every sampled crop) are built
    exactly this way, and resolve() used to realpath the target and refuse
    anything outside the dataset's OWN root -- so /api/datasets/files listed
    every name while /datasets/thumb and /datasets/image answered 404 for all
    of them, and the default landing grid rendered sixty tiles of "would not
    open" over a dataset whose files were fine. The line the refusal exists
    for has not moved: a link that leaves EVERY dataset root is still None,
    and a cross-dataset link to a .yaml is still not an image.
    """
    ix, ds = fx.ix, fx.ds
    if not os.path.islink(os.path.join(fx.paths['linked'],
                                       'images/train/l000.jpg')):
        print('SKIP: no symlinks on this filesystem — the symlink-assembled '
              'dataset was not exercised')
        return
    key = fx.key('linked_v1')
    if not key:
        bad.append('the symlink-assembled fixture dataset was not discovered '
                   'at all — nothing below tested anything')
        return
    det_root = os.path.realpath(_row(fx, 'det_v1')['root'])
    got = ix.resolve(key, 'images/train/l000.jpg')
    if got is None:
        bad.append('resolve() refuses a symlink into a sibling indexed '
                   'dataset — every tile of a symlink-assembled export '
                   'renders "would not open" while the file listing says '
                   'the images are there')
    elif not got.startswith(det_root + os.sep):
        bad.append(f'the linked image resolved to {got!r}, which is not '
                   f'inside det_v1 — the realpath went somewhere else')
    # the listing and the pictures must AGREE: a name the files API hands
    # out is a name the image route serves
    listing = ix.listing(key, 'images/train', 0, 10)
    names = [f['rel'] for f in (listing.get('files') or [])]
    linked_names = [n for n in names if n.endswith('l000.jpg')]
    if listing.get('ok') and linked_names:
        if ds.full(key, linked_names[0])[0] is None:
            bad.append('/api/datasets/files lists a linked image that '
                       '/datasets/image then 404s — the listing asserts the '
                       'dataset is readable and the pictures deny it')
    # a cross-dataset link to a descriptor is refused on the RESOLVED
    # extension, exactly like the in-dataset sneaky.jpg
    if ds.full(key, 'images/train/sneak2.jpg')[0] is not None:
        bad.append('a linked dataset.yaml was served as an image through a '
                   'cross-dataset symlink')
    # and the outside line has not moved: det_v1's escape link points at a
    # file under the scan root that no dataset owns
    if ix.resolve(fx.key('det_v1'), 'images/train/escape.jpg') is not None:
        bad.append('a symlink pointing outside every dataset root resolved — '
                   'widening containment to sibling datasets opened the '
                   'whole disk')


def allowlist_checks(bad, fx):
    """The image routes serve images, decided on the file that gets opened.

    The promise in the module docstring is that a .yaml, a .cache or a label
    file is a 404 through an image route. Enforced against the name the client
    sent, a symlink called x.jpg pointing at the descriptor beside it served
    the descriptor as image/jpeg -- containment intact, guarantee broken.
    """
    ds, ix = fx.ds, fx.ix
    key = fx.key('det_v1')
    for rel in ('dataset.yaml', 'labels/train/f000.txt', 'manifest.json'):
        if ds.full(key, rel)[0] is not None:
            bad.append(f'/datasets/image served {rel!r}, which is not an image')
        if ds.thumb(key, rel)[0] is not None:
            bad.append(f'/datasets/thumb served {rel!r}')
    link = os.path.join(_row(fx, 'det_v1')['root'], 'images/train/sneaky.jpg')
    if not os.path.islink(link):
        print('SKIP: no symlinks on this filesystem — the .jpg-that-is-a-yaml '
              'case was not exercised')
        return
    body, ctype = ds.full(key, 'images/train/sneaky.jpg')
    if body is not None:
        bad.append('/datasets/image served a .jpg that is a symlink to the '
                   f'descriptor, as {ctype}: {body[:40]!r} — the allow-list '
                   f'is being read off the request string, not off the file')
    if ds.thumb(key, 'images/train/sneaky.jpg')[0] is not None:
        bad.append('/datasets/thumb cut a thumbnail of the descriptor')
    # a real image still comes back, or the check above is vacuous
    body, ctype = ds.full(key, 'images/train/f000.jpg')
    if not body or ctype != 'image/jpeg':
        bad.append('/datasets/image will not serve a real image in the '
                   'dataset — the refusals above prove nothing')
    if not HAVE_PIL:
        print('SKIP: no PIL in this interpreter — no thumbnail was cut, so '
              'only the refusals were checked on /datasets/thumb')
        return
    body, ctype = ds.thumb(key, 'images/train/f000.jpg')
    if not body or ctype != 'image/jpeg':
        bad.append('/datasets/thumb cut nothing for a real picture in the '
                   'dataset')
    elif not body.startswith(b'\xff\xd8'):
        bad.append('/datasets/thumb answered something that is not a JPEG')
    # and the cut landed in the cache this module owns, nowhere near the
    # dataset it was cut from
    for here, _, files in os.walk(_row(fx, 'det_v1')['root']):
        if any(f.endswith(('.part', '.thumb')) for f in files):
            bad.append(f'the thumbnail cutter wrote into the dataset itself: '
                       f'{here}')


# ── discovery, counts, and what is gone ─────────────────────────────────────

def discovery_checks(bad, fx):
    """A dataset built after the page was written turns up on it.

    The whole feature rests on this: discovery is a walk, so a directory that
    did not exist when the server started is found on the next pass with
    nobody editing a list.
    """
    ix = fx.ix
    names = {r['name'] for r in fx.rows()}
    for want in ('det_v1', 'cls_v1', 'flat_v1'):
        if want not in names:
            bad.append(f'{want} was not discovered — a dataset in a shape '
                       f'this machine really builds is invisible')
    # built now, while the index is warm and the page is open
    fresh = os.path.join(fx.paths['base'], 'built_just_now')
    os.makedirs(os.path.join(fresh, 'train', 'dog'))
    _jpeg(os.path.join(fresh, 'train', 'dog', 'a.jpg'))
    if 'built_just_now' in {r['name'] for r in ix.datasets(refresh=False)}:
        bad.append('a directory created after the last walk is in the CACHED '
                   'answer — the index is not answering from its cache')
    rows = fx.rows()
    hit = [r for r in rows if r['name'] == 'built_just_now']
    if not hit:
        bad.append('a dataset built while the page was open never appeared, '
                   'which is the one thing this page exists to do')
        return
    if hit[0]['kind'] != 'classify':
        bad.append(f'the new dataset came back as {hit[0]["kind"]!r}, not '
                   f'classify — its shape was not read')
    # the key is a handle, not a path, and it survives a rescan
    before = hit[0]['key']
    if '/' in before or '..' in before:
        bad.append(f'the dataset key carries a path: {before!r}')
    if fx.rows() and _row(fx, 'built_just_now')['key'] != before:
        bad.append('the key for one dataset changed between two walks — every '
                   'open picture URL on the page died with it')
    # two roots with the same basename are two rows, not one
    twin = os.path.join(fx.paths['base'], 'archived', 'det_v1')
    os.makedirs(os.path.join(twin, 'images', 'train'))
    os.makedirs(os.path.join(twin, 'labels', 'train'))
    _jpeg(os.path.join(twin, 'images', 'train', 'a.jpg'))
    same = [r for r in fx.rows() if r['name'] == 'det_v1']
    if len(same) != 2:
        bad.append(f'{len(same)} rows named det_v1 after a second one was '
                   f'archived; two directories are two datasets')
    elif len({r['key'] for r in same}) != 2:
        bad.append('two datasets with the same basename share one key — one '
                   'of them is unreachable and the other serves its files')
    shutil.rmtree(twin, ignore_errors=True)
    shutil.rmtree(fresh, ignore_errors=True)
    fx.reset()
    fx.rows()


def count_checks(bad, fx):
    """The numbers on the page are the filesystem's.

    Counted here by a plain walk, which is deliberately not how the index
    counts: it caches per directory against that directory's mtime, and a
    cache that reports a stale total is the failure mode worth a check.
    """
    ix = fx.ix
    for name in ('det_v1', 'cls_v1', 'flat_v1'):
        row = _row(fx, name)
        if not row:
            bad.append(f'{name} is missing from the index')
            continue
        want = _count_images(row['root'])
        if row['images'] != want:
            bad.append(f'{name} reports {row["images"]} images, the disk has '
                       f'{want}')
        node = ix.tree(row['key'])
        if not node.get('ok'):
            bad.append(f'{name} would not open: {node.get("error")}')
            continue
        if node['tree']['images'] != want:
            bad.append(f'{name} tree totals {node["tree"]["images"]}, the '
                       f'disk has {want}')
    # a file added under a directory nobody has touched still moves the total
    row = _row(fx, 'cls_v1')
    before = ix.tree(row['key'])['tree']['images']
    _jpeg(os.path.join(row['root'], 'val', 'dog', 'late.jpg'))
    after = ix.tree(row['key'])['tree']['images']
    if after != before + 1:
        bad.append(f'a picture added to a split moved the count from {before} '
                   f'to {after} — the per-directory cache is being trusted '
                   f'past its own mtime')
    os.remove(os.path.join(row['root'], 'val', 'dog', 'late.jpg'))
    # the listing counts what the directory holds, not what the page shows
    got = ix.listing(_row(fx, 'det_v1')['key'], 'images/train', 0, 60)
    disk = len(os.listdir(os.path.join(_row(fx, 'det_v1')['root'],
                                       'images/train')))
    if got['total'] != disk:
        bad.append(f'images/train lists {got["total"]} files, the directory '
                   f'holds {disk}')


def _count_images(root):
    """Every image file under a root, by a plain walk.

    Deliberately not how the index counts -- it caches per directory against
    that directory's own mtime -- so the two agreeing means something.
    """
    n = 0
    for here, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp',
                                   '.tif', '.tiff')):
                if os.path.isfile(os.path.join(here, f)):
                    n += 1
    return n


def gone_checks(bad, fx):
    """A dataset that is not there is a row that says so.

    `dataset.yaml` at the training root points at an export that has been
    deleted and thirty-eight runs trained on it. Dropping that row would show
    those runs as having trained on nothing at all, so it is reported with
    exists=False and it keeps its runs, its class list and a date.
    """
    ix = fx.ix
    rows = fx.rows()
    ghost = [r for r in rows if not r['exists']]
    if not ghost:
        bad.append('the deleted export is not on the list at all — the runs '
                   'that trained on it now point at nothing')
        return
    g = ghost[0]
    if len(g['runs']) != 2:
        bad.append(f'the deleted export carries {len(g["runs"])} runs, not '
                   f'the 2 that trained on it')
    if g['kind'] != 'detect':
        bad.append(f'the deleted export came back as {g["kind"]!r}: a run '
                   f'that trained on a descriptor trained on a detector')
    if g['classes'] != ['target']:
        bad.append(f'the deleted export lost its class list ({g["classes"]}) '
                   f'— its descriptor still reads')
    if g['images'] or g['bytes']:
        bad.append('a directory that is gone reports images and bytes')
    if not g['mtime']:
        bad.append('a directory that is gone has no date at all, so it sorts '
                   'to the bottom under everything')
    if rows[-1]['exists']:
        bad.append('a deleted dataset sorted above one that is still there')
    if ix.tree(g['key']).get('ok'):
        bad.append('the tree of a directory that is gone came back ok')
    if ix.listing(g['key'], '').get('ok'):
        bad.append('a listing of a directory that is gone came back ok')
    # the archived copy of the same name is its own row, with its own key and
    # its own images, and it is not the one reported as gone
    twins = [r for r in rows if r['name'] == g['name']]
    if len(twins) != 2 or len({r['key'] for r in twins}) != 2:
        bad.append(f'the archived directory of the same basename is not a '
                   f'separate row: {[(r["key"], r["exists"]) for r in twins]}')
    elif not any(r['exists'] and r['images'] for r in twins):
        bad.append('the archived copy came back empty, so the page cannot '
                   'point at it')
    # and the bare-name resolution that finds it in the first place
    if not any('run' in r['found'] for r in rows):
        bad.append('no dataset was found via the run list — the bare `data` '
                   'names 39 of the 52 real runs carry resolved to nothing')
    # A run whose descriptor has gone as well keeps a row, and that row is
    # NOT the directory the descriptor sat in: a bare name resolves against
    # the training root, so guessing the parent would put the training root
    # on the page as a dataset and then measure every run underneath it.
    if not [r for r in rows if r['name'] == 'gone_forever.yaml']:
        bad.append('a run whose descriptor is gone has no row at all, so it '
                   'is a run that trained on nothing')
    base = os.path.realpath(fx.paths['base'])
    if any(os.path.realpath(r['root']) == base for r in rows):
        bad.append('the training root itself is on the list as a dataset')


def paging_checks(bad, fx):
    """A page number means the same thing twice, and the last page is short.

    Sorted by name for exactly that reason. The boundaries are where this goes
    wrong: a page past the end must be empty rather than wrapping, and the
    pages must partition the directory with nothing seen twice and nothing
    missed.
    """
    ix = fx.ix
    key = fx.key('det_v1')
    total = ix.listing(key, 'images/train', 0, 60)['total']
    seen, pages = [], (total + 59) // 60
    for p in range(pages):
        got = ix.listing(key, 'images/train', p, 60)
        if got['pages'] != pages:
            bad.append(f'page {p} says there are {got["pages"]} pages, not '
                       f'{pages}')
        if got['page'] != p:
            bad.append(f'asked for page {p}, got page {got["page"]}')
        seen += [f['name'] for f in got['files']]
    if len(seen) != total:
        bad.append(f'the pages hold {len(seen)} of {total} files — a page '
                   f'boundary drops or repeats')
    if len(set(seen)) != len(seen):
        bad.append('a file appears on two pages')
    if seen != sorted(seen):
        bad.append('the pages are not in name order, so a page number means '
                   'something different every time it is asked for')
    last = ix.listing(key, 'images/train', pages - 1, 60)
    if len(last['files']) != total - 60 * (pages - 1):
        bad.append(f'the last page holds {len(last["files"])} files, not the '
                   f'{total - 60 * (pages - 1)} left over')
    past = ix.listing(key, 'images/train', pages + 3, 60)
    if not past['ok'] or past['files']:
        bad.append(f'a page past the end returned {len(past["files"])} files '
                   f'— it must be empty, not a wrap')
    # the page size the interface offers is the only one it honours
    for n, want in ((60, 60), (120, 120), (240, 240), (99, 120),
                    (0, 120), (-5, 120), ('x', 120), (100000, 120)):
        if fx.ds.page_size(n) != want:
            bad.append(f'page_size({n!r}) = {fx.ds.page_size(n)}, not {want}')
    # a rel that is written oddly comes back canonical, or the page keeps
    # growing the string it links on
    for rel in ('./images/train', 'images//train', 'images/train/'):
        got = ix.listing(key, rel, 0, 60)
        if not got['ok'] or got['rel'] != 'images/train':
            bad.append(f'{rel!r} listed as {got.get("rel")!r} rather than '
                       f'the canonical images/train')


def label_checks(bad, fx):
    """Two directories that hold one set of crops are not two classes.

    data/hard_negatives holds crops/ and full/ -- the cut-out and the frame it
    came from -- beside a labels.jsonl in which every row reads the same word.
    Read off the directories that is a two-class dataset with a 50/50 balance,
    which is a fact about the layout printed where a fact about the data goes.
    """
    ix = fx.ix
    flat = _row(fx, 'flat_v1')
    if not flat:
        bad.append('the flat crop export was not discovered')
        return
    if flat['classes'] != ['false_positive']:
        bad.append(f'the flat crop export reports classes '
                   f'{flat["classes"]} — crops/ and full/ are two views of '
                   f'one crop, and every row of its manifest says '
                   f'false_positive')
    if flat['labels'] != {'false_positive': 6}:
        bad.append(f'the per-crop labels came back {flat["labels"]}, not the '
                   f'6 rows the manifest holds')
    # a split set's classes are still its directories: the manifest records
    # what a build intended and the directories are what it left behind
    cls = _row(fx, 'cls_v1')
    if cls['classes'] != sorted(['dog', EVIL_DIR]):
        bad.append(f'the split classify set reports classes '
                   f'{cls["classes"]}, not its class directories')
    if cls['labels']:
        bad.append(f'a split set carries per-row labels ({cls["labels"]}) — '
                   f'a bar drawn off manifest rows beside folders of files '
                   f'puts two denominators in one panel')


# ── what a stranger with a URL can buy ──────────────────────────────────────

def cost_checks(bad, fx):
    """A made-up key must not buy a walk, and one walk serves everybody.

    The dashboard has no authentication and every thumbnail, listing and tree
    goes through dataset(). Rescanning on the age of the index BEFORE looking
    the key up handed anyone who could reach the port a full walk of every
    dataset root per request, and without a lock every thread that arrived
    inside the window ran its own.
    """
    ix = fx.ix
    fx.rows()
    key = fx.key('det_v1')
    walks = [0]
    real = ix._discover

    def counted(now):
        walks[0] += 1
        return real(now)

    # An index that was walked long ago, which is not the same thing as one
    # that was never walked: a zero there means a process that has just
    # started, and that one first walk is the one this is allowed to buy.
    stale = time.time() - 10 * ix.INDEX_TTL
    ix._discover = counted
    try:
        ix._INDEX['at'] = stale
        for i in range(8):
            if ix.dataset('made-up-key-%d' % i) is not None:
                bad.append('a made-up key resolved to a dataset')
        if walks[0]:
            bad.append(f'{walks[0]} full walks of every dataset root were '
                       f'bought by 8 made-up keys — the key is being looked '
                       f'up after the rescan, not before it')
        # and a known key still answers from the stale index rather than
        # forcing a walk of its own
        walks[0] = 0
        ix._INDEX['at'] = stale
        ix.dataset(key)
        ix.resolve(key, 'images/train/f000.jpg')
        if walks[0]:
            bad.append(f'serving one picture off a stale index cost '
                       f'{walks[0]} walks of every root')

        # a page of tiles landing together on a stale index is ONE walk
        walks[0] = 0
        ix._INDEX['at'] = stale
        threads = [threading.Thread(target=ix.datasets) for _ in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if walks[0] != 1:
            bad.append(f'24 requests arriving together on a stale index ran '
                       f'{walks[0]} independent walks — there is no single '
                       f'flight, and they read the disks a live training run '
                       f'is writing to')
        # the rescan button is not answered with the cache it was pressed to
        # get past, however recently the last walk finished
        walks[0] = 0
        ix.datasets(refresh=True)
        if walks[0] != 1:
            bad.append(f'a rescan straight after a walk ran {walks[0]} walks '
                       f'— the button reports what it was pressed to replace')
    finally:
        ix._discover = real
    # an empty index still answers, or a restarted dashboard serves 404s to
    # every open page until somebody reloads it
    fx.reset()
    if ix.dataset(fx.key('det_v1')) is None:
        bad.append('the first lookup in a fresh process found nothing — a '
                   'dashboard that has just restarted serves nothing')


def syspath_checks(bad, fx):
    """Refreshing the index must not lengthen sys.path.

    Both lazy imports run inside the discovery loop, once per logged run. An
    unguarded insert added about fifty entries per refresh to a process that
    stays up for days, and every cold import anywhere in the dashboard pays a
    linear scan of the result.
    """
    ix = fx.ix
    fx.rows()
    before = [len(sys.path)] + [sys.path.count(p) for p in MINE]
    for _ in range(5):
        ix.datasets(refresh=True)
    after = [len(sys.path)] + [sys.path.count(p) for p in MINE]
    grew = after[0] - before[0]
    if grew:
        bad.append(f'sys.path grew by {grew} entries over 5 index refreshes '
                   f'({grew / 5:.0f} per refresh), and nothing ever removes '
                   f'them')
    # Which directory, because that names the function that did it. Measured
    # as growth rather than as a count: something else in the dashboard may
    # already have put one of these on the path, and that is not this check.
    for i, p in enumerate(MINE, start=1):
        if after[i] > before[i]:
            bad.append(f'refreshing the index added {after[i] - before[i]} '
                       f'more copies of {os.path.basename(p)} to sys.path')


# ── the page ────────────────────────────────────────────────────────────────

def attribute_checks(bad, fx):
    """No name off the disk is printed into an attribute by esc() alone.

    esc() is the textContent -> innerHTML round trip. A text node has no
    quotes to escape, so the serialiser leaves them alone: esc() is right
    between tags and wrong inside title="" or alt="". Every name this page
    prints came from a directory entry, and the page exists so that an export
    somebody unpacked into the training root shows up without a code change.

    Read off the source rather than off a render, because the render only
    reaches the attributes the fixture happens to exercise and there are six
    of them.
    """
    script = _script(fx.ds.page_html())
    flat = re.sub(r'\s+', ' ', script)
    # an attribute's opening quote, then the end of the JS string, then esc()
    hits = re.findall(r'=\s*"[^"\']{0,60}\'\s*\+\s*esc\(([^)]*)\)', flat)
    if hits:
        bad.append(f'esc() is used inside a double-quoted attribute at '
                   f'{len(hits)} place(s) ({", ".join(hits[:4])}) — it does '
                   f'not escape the quote, so a name holding one closes the '
                   f'attribute and the rest becomes markup. att() is the one '
                   f'that escapes it.')
    if 'function att(' not in script:
        bad.append('the page has no att() helper at all')


class _Tags(html.parser.HTMLParser):
    """Every tag and its attributes, as a real parser sees them."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, attrs))


def _attrs_of(markup, tag, cls=None):
    p = _Tags()
    p.feed(markup)
    out = []
    for t, a in p.tags:
        if t != tag:
            continue
        d = dict(a)
        if cls and cls not in (d.get('class') or '').split():
            continue
        out.append(d)
    return out


def _script(html_text):
    return html_text[html_text.rindex('<script>') + 8:
                     html_text.rindex('</script>')]


HARNESS = r"""
// ── a DOM small enough to run this page and honest about what it escapes ───
var THROWS = [], FETCHED = [], listeners = {}, els = {};
// Printed as it happens rather than collected and printed at the end: a
// throw further down the drive would take the whole list with it, and the
// run would be reported as one crash with nothing said about the six things
// that had already been found wrong before it.
function chk(c, m){ if(!c) console.log('FAIL ' + m) }

function mk(id){
  var el = {id:id, className:'', title:'', hidden:false, disabled:false,
    style:{}, children:[], value:'', src:'', dataset:{}, _attrs:{},
    appendChild:function(c){ this.children.push(c) },
    setAttribute:function(k,v){ this._attrs[k] = v },
    getAttribute:function(k){ return this._attrs[k] },
    removeAttribute:function(k){ delete this._attrs[k] },
    addEventListener:function(t,f){ (listeners[id] = listeners[id]||{})[t]=f },
    querySelectorAll:function(){ return [] },
    querySelector:function(){ return null },
    closest:function(){ return null },
    select:function(){}, scrollIntoView:function(){},
    classList:{add:function(){},remove:function(){},toggle:function(){}}};
  // The escaping the page depends on, modelled exactly: a text node
  // serialises &, < and > and LEAVES THE QUOTE ALONE. Escape the quote here
  // and esc() would look safe in attribute position, which is the bug.
  var _html = '', _text = '';
  Object.defineProperty(el, 'textContent', {
    get:function(){ return _text },
    set:function(v){ _text = String(v);
      _html = _text.replace(/&/g,'&amp;').replace(/</g,'&lt;')
                   .replace(/>/g,'&gt;'); }});
  Object.defineProperty(el, 'innerHTML', {
    get:function(){ return _html },
    set:function(v){ _html = String(v); _text = ''; }});
  return el;
}
function E(id){ return els[id] || (els[id] = mk(id)) }
global.document = {getElementById:E,
  createElement:function(t){ return mk('new:' + t) },
  addEventListener:function(t,f){ (listeners.doc = listeners.doc||{})[t]=f },
  body:{appendChild:function(){}, removeChild:function(){}},
  execCommand:function(){ return true }};
global.window = {isSecureContext:false, addEventListener:function(){},
  removeEventListener:function(){}};
global.localStorage = {_d:{}, getItem:function(k){ return this._d[k]||null },
  setItem:function(k,v){ this._d[k] = String(v) }};
global.setTimeout = function(){ return 1 };
global.clearTimeout = function(){};

// ── the server, answering with what the real one answered ─────────────────
// A thenable rather than a promise: the assertions run after a synchronous
// boot, so nothing has to be awaited -- but a THROW inside a .then is
// invisible from outside (the page swallows it and the chain resolves), so
// every one is recorded and the tail fails on it.
var FAIL_TREE = false;
function settled(v){
  return {then:function(f){
            var out; try { out = f(v) } catch(e){ THROWS.push(''+e);
                                                  return rejected(e) }
            return settled(out) },
          catch:function(){ return settled(v) }};
}
function rejected(e){
  return {then:function(){ return rejected(e) },
          catch:function(c){ try { c(e) } catch(x){ THROWS.push(''+x) }
                             return settled(null) }};
}
global.fetch = function(u){
  FETCHED.push(u);
  if (FAIL_TREE && /\/api\/datasets\/tree/.test(u))
    return rejected(new Error('socket closed'));
  var body = RESP(u);
  return settled({ok:true, json:function(){ return body }});
};
function qval(u, k){
  var m = new RegExp('[?&]' + k + '=([^&]*)').exec(u);
  return m ? decodeURIComponent(m[1]) : '';
}
function RESP(u){
  if (/^\/api\/datasets($|\?)/.test(u)) return LIST;
  if (/^\/api\/datasets\/tree/.test(u)) return TREE[qval(u,'key')] ||
    {ok:false, error:'no such dataset'};
  if (/^\/api\/datasets\/files/.test(u)) {
    var f = (FILES[qval(u,'key')] || {})[qval(u,'rel')];
    if (!f) return {ok:false, error:'no such folder'};
    return f[+qval(u,'page') || 0] || {ok:false, error:'no such folder'};
  }
  return {};
}
"""

TAIL = r"""
// ── what the page drew ────────────────────────────────────────────────────
chk(THROWS.length === 0, 'the page threw inside a handler: ' + THROWS[0]);
chk(rows.length === LIST.datasets.length,
  'the list painted ' + rows.length + ' of ' + LIST.datasets.length + ' rows');
chk(/det_v1/.test(els.dslist.innerHTML), 'the dataset list is empty');
chk(cur && cur.name === 'det_v1',
  'the page did not open a dataset by itself');
chk(files && files.files.length > 0, 'no files were listed');
console.log('MARK grid ' + JSON.stringify(els.grid.innerHTML));
console.log('MARK tree ' + JSON.stringify(els.tree.innerHTML));
console.log('MARK list ' + JSON.stringify(els.dslist.innerHTML));

// the label files beside a detect split are a list, not a grid of TXT tiles
pick('labels/train');
console.log('MARK flist ' + JSON.stringify(els.flist.innerHTML));
chk(els.flist.hidden === false, 'a folder of label files drew as a grid');
pick('images/train');

// ── the page boundary in the lightbox ─────────────────────────────────────
// step() used to bound itself by the page it was on, so holding the arrow key
// stopped dead at the 120th picture of a 2,416-file split with no message.
chk(files.pages > 1, 'the fixture has only one page — the boundary below is '
  + 'not being tested');
zoom(0);
var opened = els.lbtxt.textContent;
step(-1);
chk(els.lbtxt.textContent === opened,
  'stepping back off the FIRST page moved somewhere: ' + els.lbtxt.textContent);
chk(pg === 0, 'stepping back off the first page turned the page to ' + pg);
var last = files.files.length - 1, wasPage = files.page;
zoom(last);
step(1);
chk(pg === wasPage + 1,
  'arrowing off the end of a page left the reader on page ' + pg
  + ' with nothing happening');
chk(files.page === wasPage + 1,
  'the page turned but the files did not: ' + files.page);
chk(els.lbtxt.textContent !== opened && /\S/.test(els.lbtxt.textContent),
  'the lightbox shows nothing after the page turned: '
  + els.lbtxt.textContent);
chk(els.lb.hidden === false, 'the lightbox closed itself at a page boundary');
els.lb.hidden = true;

// ── a tree fetch that never produced JSON ─────────────────────────────────
// The header is repainted before the fetch. If the rejection leaves the old
// dataset's structure under the new one's name, two datasets are on one
// screen with nothing saying so.
var other = null, i;
for (i = 0; i < rows.length; i++)
  if (rows[i].name === 'cls_v1') other = rows[i];
chk(other, 'the fixture classify set is not in the list');
var beforeTree = els.tree.innerHTML;
FAIL_TREE = true;
openDs(other.key);
chk(/cls_v1/.test(els.dshead.innerHTML), 'the header did not change');
chk(els.tree.innerHTML !== beforeTree,
  'a rejected tree fetch left the previous dataset\'s folders on screen '
  + 'under the new one\'s name');
chk(!/f000/.test(els.balance.innerHTML) && els.balance.innerHTML === '',
  'a rejected tree fetch left the previous dataset\'s class balance up');
chk(tree === null,
  'a rejected tree fetch left the previous dataset\'s tree live — clicking a '
  + 'node in it asks the new dataset for the old one\'s folder');
FAIL_TREE = false;

// ── the classify page ─────────────────────────────────────────────────────
openDs(other.key);
console.log('MARK balance ' + JSON.stringify(els.balance.innerHTML));
// this one's folders carry the hostile name, which the detect fixture's do not
console.log('MARK tree2 ' + JSON.stringify(els.tree.innerHTML));

// ── a flat crop export, whose folders are not its classes ─────────────────
// crops/ and full/ are one crop and the frame it came from. Drawn as a class
// balance they are two classes at 50/50 over an image count that holds every
// crop twice, and the labels -- one word, on every row of the manifest -- are
// nowhere on the screen.
var flat = null;
for (i = 0; i < rows.length; i++)
  if (rows[i].name === 'flat_v1') flat = rows[i];
chk(flat, 'the flat crop export is not in the list');
if (flat) {
  openDs(flat.key);
  chk(/false_positive/.test(els.balance.innerHTML),
    'the flat crop export does not name the label every row of its manifest '
    + 'carries: ' + els.balance.innerHTML.slice(0, 200));
  chk(/>6 rows</.test(els.balance.innerHTML),
    'the flat crop export counts something other than the 6 manifest rows, '
    + 'or calls them images: ' + els.balance.innerHTML.slice(0, 200));
  chk(/not classes/.test(els.balance.innerHTML),
    'nothing says the folders are not the classes, so the reader has no way '
    + 'to know the image count holds every crop twice');
}

// ── a dataset that is gone, beside one of the same name that is not ───────
var ghost = null, lost = null;
for (i = 0; i < rows.length; i++) {
  if (!rows[i].exists && rows[i].name === 'ghost_export') ghost = rows[i];
  if (!rows[i].exists && rows[i].name === 'gone_forever.yaml') lost = rows[i];
}
chk(ghost, 'no deleted dataset in the payload — the row below is untested');
if (ghost) {
  openDs(ghost.key);
  chk(els.panes.hidden === true,
    'a directory that is gone still shows a structure pane');
  chk(/gonenote/.test(els.dshead.innerHTML),
    'nothing on the page says the directory is gone');
  chk(/\d+ runs?/.test(els.dshead.innerHTML),
    'the deleted dataset does not say how many runs trained on it, which is '
    + 'the only reason its row exists');
  chk(/same name is on the left/.test(els.dshead.innerHTML),
    '"nothing here can be opened again" is stated as a fact while a '
    + 'directory of the same name sits in the list full of images, and the '
    + 'note says nothing about it');
  console.log('MARK gone ' + JSON.stringify(els.dshead.innerHTML));
}
// ...and the same sentence must not appear where there is no twin, or it is
// a sentence the page always says rather than one it means
chk(lost, 'the run whose descriptor is gone has no row');
if (lost) {
  openDs(lost.key);
  chk(/gonenote/.test(els.dshead.innerHTML),
    'a descriptor that is gone does not say so');
  chk(!/same name is on the left/.test(els.dshead.innerHTML),
    'the page points at a dataset of the same name when there is not one');
}
chk(THROWS.length === 0, 'the page threw inside a handler: ' + THROWS[0]);
"""


def page_checks(bad, fx):
    """The page runs, and a hostile name lands on it as text.

    node --check only proves the script parses. This executes it against a
    stub DOM, feeding it exactly what the real routes answered for the
    fixture, and then parses the markup it produced with a real HTML parser --
    because "does the browser see one attribute or three" is a question only a
    parser can answer.
    """
    if not shutil.which('node'):
        print('SKIP: node is not on PATH — the datasets page was not run, '
              'and nothing below about the rendered markup was checked')
        return
    ds = fx.ds
    payload = _payload(fx)
    script = _script(ds.page_html())
    body = (HARNESS
            + '\nvar LIST=' + json.dumps(payload['list']) + ';'
            + '\nvar TREE=' + json.dumps(payload['tree']) + ';'
            + '\nvar FILES=' + json.dumps(payload['files']) + ';'
            + "\nlocalStorage.setItem('sdDatasetSize','60');"
            + "\nlocalStorage.setItem('sdDataset','" + payload['key'] + "');"
            # strict mode, so a name the page forgot to declare is a
            # ReferenceError here rather than a global nobody notices
            + '\n(function(){"use strict";\n' + script + '\n' + TAIL + '\n})();')
    path = os.path.join(fx.tmp, 'page.js')
    with open(path, 'w') as fh:
        fh.write(body)
    r = subprocess.run(['node', path], capture_output=True, text=True)
    marks = {}
    # Read whatever it managed to say before it died, then report the death:
    # a crash half way through is still six findings and a crash.
    for line in r.stdout.splitlines():
        if line.startswith('FAIL '):
            bad.append('datasets page: ' + line[5:])
        elif line.startswith('MARK '):
            what, _, raw = line[5:].partition(' ')
            marks[what] = json.loads(raw)
    if r.returncode != 0:
        err = [x.strip() for x in (r.stderr or '').splitlines() if x.strip()]
        why = next((x for x in err if re.match(r'^\w*Error\b', x)),
                   err[0] if err else '?')
        bad.append(f'the datasets page threw and never finished: {why[:200]}')
        return
    _markup_checks(bad, marks)


def _markup_checks(bad, marks):
    """A real parser on the markup the page built, name by name.

    Every one of these is the same defect at a different site: a file or
    folder name holding a double quote closes the attribute it was printed
    into, and everything after it in the name is parsed as further attributes
    -- on this origin, which has no authentication and POST routes that start
    GPU runs and rewrite human annotations.
    """
    if not marks:
        bad.append('the page printed nothing to check')
        return
    imgs = _attrs_of(marks.get('grid', ''), 'img')
    if not imgs:
        bad.append('the grid drew no pictures at all')
    if EVIL_FILE.replace('"', '&quot;') not in marks.get('grid', ''):
        bad.append(f'the file named {EVIL_FILE!r} is not in the grid with its '
                   f'quotes escaped: either it was never listed, and the '
                   f'checks below tested nothing, or it was written out raw')
    for i in imgs:
        extra = set(i) - {'loading', 'src', 'alt'}
        if extra:
            bad.append(f'a file name became {sorted(extra)} on the <img> '
                       f'tag: alt={i.get("alt")!r} — the name broke out of '
                       f'alt="" and the browser is parsing it as markup')
        if i.get('alt') == 'a':
            bad.append('the picture\'s alt text stops at the first quote of '
                       'its file name')
    for cap in _attrs_of(marks.get('grid', ''), 'span', 'fname'):
        extra = set(cap) - {'class', 'title'}
        if extra:
            bad.append(f'a file name became {sorted(extra)} on its caption')
    for fn in _attrs_of(marks.get('flist', ''), 'span', 'fn'):
        extra = set(fn) - {'class', 'title'}
        if extra:
            bad.append(f'a file name became {sorted(extra)} in the file list')
    nodes = _attrs_of(marks.get('tree', ''), 'button', 'tnode')
    nodes += _attrs_of(marks.get('tree2', ''), 'button', 'tnode')
    if not nodes:
        bad.append('the structure pane drew no folders')
    rels = [str(n.get('data-rel')) for n in nodes]
    if not any(EVIL_DIR in r for r in rels):
        bad.append(f'no tree node carries the folder named {EVIL_DIR!r}: '
                   f'either the classify fixture was not opened, or the name '
                   f'was cut short at its quote and that folder can never be '
                   f'opened. Got {rels}')
    for n in nodes:
        extra = set(n) - {'class', 'data-rel', 'style', 'title'}
        if extra:
            bad.append(f'a folder name became {sorted(extra)} on its tree '
                       f'node: data-rel={n.get("data-rel")!r} — the folder '
                       f'can never be opened again, and the name is markup')
    bars = _attrs_of(marks.get('balance', ''), 'div', 'balseg')
    if not bars:
        bad.append('the class balance drew no bars for the classify set')
    hostile_bar = [b for b in bars if str(b.get('title', '')).startswith('q"')]
    if not hostile_bar:
        bad.append(f'the class folder named {EVIL_DIR!r} is not in the class '
                   f'balance, so its attribute was not tested')
    for b in bars:
        extra = set(b) - {'class', 'style', 'title'}
        if extra:
            bad.append(f'a class folder name became {sorted(extra)} on its '
                       f'bar: title={b.get("title")!r}')
    # The two rows sharing a basename are told apart in the list. Without it
    # the page reads as double-listing one dataset, and the reader picks the
    # wrong one or believes the greyed row and stops looking.
    if marks.get('list', '').count('class="dsin"') != 2:
        bad.append('two rows on the list carry the same basename and nothing '
                   'in the label tells them apart')


def _payload(fx):
    """What the real routes answer for the fixture, as the page's fetch sees.

    Generated rather than written out, so the page is driven against the
    server's actual shape -- a payload invented here would keep passing after
    the server stopped sending one of its fields.
    """
    ix = fx.ix
    rows = fx.rows()
    key = None
    for r in rows:
        if r['name'] == 'det_v1':
            key = r['key']
    out = {'list': {'datasets': rows, 'error': None, 'roots': []},
           'tree': {}, 'files': {}, 'key': key}
    for r in rows:
        if not r['exists']:
            continue
        got = ix.tree(r['key'])
        out['tree'][r['key']] = got
        if not got.get('ok'):
            continue
        out['files'][r['key']] = {}

        def walk(node):
            pages = {}
            p = 0
            while True:
                one = ix.listing(r['key'], node['rel'], p, 60)
                pages[p] = one
                p += 1
                if not one.get('ok') or p >= max(1, one.get('pages', 1)):
                    break
            out['files'][r['key']][node['rel']] = pages
            for k in node['dirs']:
                walk(k)

        walk(got['tree'])
    return out


# ── the structure band stands still ─────────────────────────────────────────

def _css_rules(html_text):
    """The page's stylesheet, split at its first @media."""
    css = html_text[html_text.index('<style>') + 7:html_text.index('</style>')]
    base, _, narrow = css.partition('@media')
    return base, narrow


def _decls(css, selector):
    """Every declaration a selector carries in this sheet, later one winning.

    The brace must follow the selector immediately, so '.struct' does not
    read '.struct .balance' and report the wrong rule's properties.
    """
    out = {}
    for m in re.finditer(re.escape(selector) + r'\{([^}]*)\}', css):
        for d in m.group(1).split(';'):
            k, _, v = d.partition(':')
            if k.strip():
                out[k.strip().lower()] = re.sub(r'\s+', ' ', v.strip())
    return out


def structure_css_checks(bad, fx):
    """The structure band's geometry is written down, not negotiated.

    The user watched train and val re-seat themselves with every window size:
    the balance column was a minmax the viewport answered, and the tree was
    multicol, which stretches every column to fill the pane and re-balances
    the nodes across them at every width. The sheet now pins both -- one
    fixed balance track, fixed-width flex columns packed left -- and this
    reads the sheet, because the node harness has no layout engine.
    structure_layout_checks measures the real thing where chromium is here.
    """
    base, narrow = _css_rules(fx.ds.page_html())
    first = (_decls(base, '.struct').get('grid-template-columns')
             or '').split(' ')[0]
    if not re.fullmatch(r'\d+(\.\d+)?px', first):
        bad.append(f'the balance column is {first!r}, not one fixed px width '
                   f'— a track that answers the viewport (minmax, fr, %) '
                   f're-seats the train/val bars with every window size')
    tree = _decls(base, '.tree')
    multicol = [k for k in ('columns', 'column-width', 'column-count')
                if k in tree]
    if multicol:
        bad.append(f'the tree is multicol again ({", ".join(multicol)}) — '
                   f'multicol stretches every column to fill the pane and '
                   f're-balances the nodes across them at every width')
    if 'flex' not in (tree.get('display') or '') or \
            (tree.get('flex-flow') or '') != 'column wrap':
        bad.append('the tree is not a wrapped column flex — nothing else '
                   'here keeps a node\'s seat independent of the viewport')
    if (tree.get('align-content') or '') != 'flex-start':
        bad.append('the tree\'s columns are not packed to the left — the '
                   'browser then spreads the slack between them, and every '
                   'column moves as the window does')
    width = _decls(base, '.tnode').get('width') or 'auto'
    if not re.fullmatch(r'\d+(\.\d+)?px', width):
        bad.append(f'a tree node is {width!r} wide, not a fixed px — the '
                   f'flex columns then size to their content and shift as '
                   f'folders come and go')
    # under the breakpoint the stacked layout is still the stacked layout
    if _decls(narrow, '.struct').get('grid-template-columns') != 'minmax(0,1fr)':
        bad.append('below the breakpoint the structure band no longer '
                   'stacks — the phone layout was the one part meant to stay')
    if _decls(narrow, '.struct .tree').get('display') != 'block':
        bad.append('below the breakpoint the tree keeps its desktop '
                   'columns — stacked, it is a plain full-width list')
    if _decls(narrow, '.tnode').get('width') != '100%':
        bad.append('below the breakpoint a tree node keeps its fixed '
                   'desktop width instead of taking the row')


def _widened(treejson):
    """A tree tall enough that the desktop columns must wrap.

    The fixture's real trees fit one column, which would let a broken wrap
    pass as a stable single column. The extra directories are clones of a
    real node -- the shape stays the server's own -- renamed so each seat is
    identifiable.
    """
    import copy
    t = copy.deepcopy(treejson)
    root = t['tree']

    def re_rel(n, rel):
        n['rel'] = rel
        for k in n.get('dirs', []):
            re_rel(k, rel + '/' + k['name'])

    for i in range(8):
        c = copy.deepcopy(root['dirs'][0])
        c['name'] = 'z%02d' % i
        re_rel(c, c['name'])
        root['dirs'].append(c)
    return t


_GEOMETRY = r"""
() => {
  const r = el => { const b = el.getBoundingClientRect();
    return {x: +b.x.toFixed(1), y: +b.y.toFixed(1),
            w: +b.width.toFixed(1), h: +b.height.toFixed(1)}; };
  const tr = r(document.getElementById('tree'));
  return {balance: r(document.getElementById('balance')),
          display: getComputedStyle(document.getElementById('tree')).display,
          labs: [...document.querySelectorAll('#balance .ballab')].map(e => ({
            t: e.textContent.trim().slice(0, 12),
            y: +e.getBoundingClientRect().y.toFixed(1)})),
          nodes: [...document.querySelectorAll('#tree .tnode')].map(e => {
            const b = e.getBoundingClientRect();
            return [e.getAttribute('data-rel'), +(b.x - tr.x).toFixed(1),
                    +(b.y - tr.y).toFixed(1), +b.width.toFixed(1)]; })};
}
"""


def structure_layout_checks(bad, fx):
    """The band measured in a real browser: nothing moves with the width.

    The sheet can hold every promised declaration while some later rule
    undoes it -- an author display: rule fighting [hidden] shipped twice on
    this dashboard exactly that way -- so the page is also measured.
    page_html() and the fixture's own route answers are served through
    playwright's request interception: no server, no network, no live
    anything. Above the breakpoint the balance box and every tree node must
    sit at the same place at 1600, 1440 and 1280 CSS px; below it the tree
    must still be the plain stacked list.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print('SKIP: playwright is not in this interpreter — the structure '
              'band was read off the sheet but never measured in a browser')
        return
    from urllib.parse import parse_qs, urlparse
    payload = _payload(fx)
    det_key, cls_key = fx.key('det_v1'), fx.key('cls_v1')
    html = fx.ds.page_html()
    trees = dict(payload['tree'])
    trees[cls_key] = _widened(trees[cls_key])

    def serve(route, request):
        u = urlparse(request.url)
        q = parse_qs(u.query)

        def one(k):
            return (q.get(k) or [''])[0]

        if u.path == '/datasets':
            route.fulfill(content_type='text/html', body=html)
        elif u.path == '/api/datasets':
            route.fulfill(json=payload['list'])
        elif u.path == '/api/datasets/tree':
            route.fulfill(json=trees.get(one('key'),
                                         {'ok': False, 'error': 'no such'}))
        elif u.path == '/api/datasets/files':
            pages = (payload['files'].get(one('key')) or {}).get(one('rel'))
            route.fulfill(json=(pages or {}).get(int(one('page') or 0),
                                                 {'ok': False,
                                                  'error': 'no such folder'}))
        else:
            route.fulfill(status=404, body='')

    got = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for w in (1600, 1440, 1280, 1100):
                ctx = browser.new_context(viewport={'width': w, 'height': 950})
                ctx.add_init_script("localStorage.setItem('sdDataset', %s);"
                                    % json.dumps(det_key))
                pg = ctx.new_page()
                pg.route('**/*', serve)
                pg.goto('http://datasets.fixture/datasets')
                pg.wait_for_function("document.querySelectorAll("
                                     "'#tree .tnode').length > 0",
                                     timeout=15000)
                det = pg.evaluate(_GEOMETRY)
                pg.evaluate('openDs(%s)' % json.dumps(cls_key))
                pg.wait_for_function("document.getElementById('tree')"
                                     ".innerHTML.indexOf('z07') >= 0",
                                     timeout=15000)
                got[w] = {'det': det, 'cls': pg.evaluate(_GEOMETRY)}
                ctx.close()
            browser.close()
    except Exception as e:
        print(f'SKIP: playwright would not run a browser here '
              f'({type(e).__name__}: {str(e).splitlines()[0][:120]}) — the '
              f'structure band was read off the sheet but never measured')
        return

    for w, d in got.items():
        labs = d['det']['labs']
        # the split name runs straight into its count ('val4 images'), so
        # startswith, not equality -- and 'train' first keeps 'val' honest
        if len(labs) < 2 or not labs[0]['t'].startswith('train') \
                or not labs[1]['t'].startswith('val'):
            bad.append(f'at {w}px the balance rows read '
                       f'{[l["t"] for l in labs]} — train has lost its place '
                       f'above val')
        elif labs[0]['y'] >= labs[1]['y']:
            bad.append(f'at {w}px the train bar sits below the val bar')
    ref = got[1600]
    for w in (1440, 1280):
        a, b = ref['det']['balance'], got[w]['det']['balance']
        if abs(a['x'] - b['x']) > 0.5 or abs(a['w'] - b['w']) > 0.5:
            bad.append(f'the balance box moved with the window: '
                       f'x={a["x"]} w={a["w"]} at 1600px but x={b["x"]} '
                       f'w={b["w"]} at {w}px — its column answers the '
                       f'viewport again')
        for n0, n1 in zip(ref['cls']['nodes'], got[w]['cls']['nodes']):
            if n0[0] != n1[0] or abs(n0[1] - n1[1]) > 0.5 \
                    or abs(n0[2] - n1[2]) > 0.5 or abs(n0[3] - n1[3]) > 0.5:
                bad.append(f'the tree re-seated {n0[0]!r} between 1600px '
                           f'(x={n0[1]}, y={n0[2]}, w={n0[3]}) and {w}px '
                           f'(x={n1[1]}, y={n1[2]}, w={n1[3]}) — node '
                           f'placement follows the window size again')
                break
    if len({n[1] for n in ref['cls']['nodes']}) < 2:
        bad.append('a tree too tall for the pane never wrapped into a '
                   'second column at 1600px — every big dataset is one '
                   'endless first column')
    if got[1100]['det']['display'] != 'block':
        bad.append(f'at 1100px the tree is {got[1100]["det"]["display"]}, '
                   f'not the plain stacked list the narrow layout keeps')


# ── the routes hand the strings straight through ────────────────────────────

def route_checks(bad, fx):
    """The dashboard's own handler forms no second opinion about a path.

    Two places deciding whether a path is safe is how a traversal gets
    through: one of them is eventually relaxed. The handler's job is to hand
    `key` and `rel` over verbatim and to answer 404 to (None, None).
    """
    src = open(os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')).read()
    try:
        block = src[src.index('def _datasets_get('):]
        block = block[:block.index('\n    def ')]
    except ValueError:
        bad.append('the dashboard has no _datasets_get — /datasets is not '
                   'served by the module this file guards')
        return
    for token in ('open(', 'os.path.join', 'os.path.realpath', 'listdir'):
        if token in block:
            bad.append(f'the /datasets handler calls {token} itself; the path '
                       f'work belongs behind resolve(), and two opinions '
                       f'about one path is how a traversal gets through')
    # The whole comparison, not the substring: '/datasets/image' is inside
    # '/datasets/image-anything', so looking for the path alone passes on a
    # route that has been renamed out from under the page.
    for route in ('/datasets/thumb', '/datasets/image', '/api/datasets/files',
                  '/api/datasets/tree', '/api/datasets'):
        if f"path == '{route}'" not in block:
            bad.append(f'{route} is not served')
    if 'send_error(404)' not in block:
        bad.append('a refused path does not answer 404')
    # and the page is claimed before the static file handler, which is what
    # turned /audit/leash into a 404 the day it was added
    if "_p.startswith('/datasets')" not in src:
        bad.append('/datasets is not claimed before the static handler')


# ── one pass over what is really on this machine ────────────────────────────

def live_checks(bad):
    """The real index, on the real disks, counting and nothing else.

    Everything above runs against a fixture, which proves the code and not the
    wiring. This one proves the wiring: the training root the dashboard
    reports, the repo's own data/, and a count that matches a plain walk.
    """
    import dataset_index as ix
    roots = ix.scan_roots()
    if not roots:
        print('SKIP: no scan roots on this machine — live discovery was not '
              'checked')
        return
    rows = ix.datasets(refresh=True)
    if not rows:
        print('SKIP: no datasets on this machine — live discovery found '
              'nothing to check')
        return
    live = [r for r in rows if r['exists'] and r['images']]
    if not live:
        bad.append('every dataset on this machine reports zero images')
        return
    small = min(live, key=lambda r: r['images'])
    want = _count_images(small['root'])
    if small['images'] != want:
        bad.append(f'{small["name"]} reports {small["images"]} images, a walk '
                   f'of {small["root"]} counts {want}')
    for r in rows:
        if r['key'].count('/') or os.sep in r['key']:
            bad.append(f'a real dataset key carries a separator: {r["key"]!r}')
    # the roots are the two places datasets are built, and never a grid root:
    # those are the harvest, spread over five drives and millions of files
    if not any(os.path.join(REPO, 'data') == r or
               os.path.realpath(os.path.join(REPO, 'data')) == r
               for r in roots):
        bad.append("this repo's own data/ is not scanned, so a crop export "
                   "built by this repo's tools would never show up")


def main():
    bad = []
    try:
        live_checks(bad)
    except Exception as e:
        bad.append(f'live_checks threw {type(e).__name__}: {e}')
    try:
        with Fixture() as fx:
            fx.rows()
            for fn in (traversal_checks, symlink_dataset_checks,
                       allowlist_checks, discovery_checks,
                       count_checks, gone_checks, paging_checks, label_checks,
                       cost_checks, syspath_checks, attribute_checks,
                       structure_css_checks, structure_layout_checks,
                       route_checks, page_checks):
                try:
                    fn(bad, fx)
                except Exception as e:      # noqa: BLE001 - report, not die
                    bad.append(f'{fn.__name__} threw {type(e).__name__}: {e}')
    except Exception as e:
        bad.append(f'the fixture would not build: {type(e).__name__}: {e}')
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('nothing outside a dataset root can be reached, a dataset built now '
          'shows up, the counts are the disk\'s, and a file name is text')
    return 0


if __name__ == '__main__':
    sys.exit(main())
