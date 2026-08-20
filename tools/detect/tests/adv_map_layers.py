#!/usr/bin/env python3
"""The model layers on the map must count what the ledgers say -- and only
what the ledgers say.

map_layers.py joins two model stores (the gate's kept crops, the leash
classifier's calls) to 33K harvest manifests and bins them onto the map's
grid. Three ways that quietly goes wrong, each checked here against a
synthetic store where every count is known by hand:

  * counting: an image duplicated across two drives' manifests doubles its
    crops; a hidden region leaks back onto the map; a crop scored exactly at
    the threshold falls on the wrong side; a leash row with no score gets
    painted as "loose" when the model never said so.
  * the warm path: refresh() must SKIP when no shard moved -- the cold build
    is a minutes-long 32M-row join, and a scheduler that reruns it every
    interval turns the dashboard host into a space heater. Skipping must
    also actually skip: no rewrite of the outputs.
  * tearing: the outputs are fetched by a live page, so a crash mid-write
    must leave the previous build intact, not a truncated JSON. Every write
    goes through the .tmp + os.replace helper, and the helper must survive a
    serialisation failure without touching the final file.

And one labelling rule that is policy, not plumbing: these are MODEL numbers
feeding a picture, so each payload must carry a 'source' naming the model --
this dashboard never lets a model count masquerade as a human one.
"""

import ast
import json
import math
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))

import duckdb  # noqa: E402

import map_layers as ml  # noqa: E402

RES_LIST = (0.5, 0.15, 0.05)

# fixture images: id -> (lon, lat) or None for no geometry. Image 6 carries
# an impossible longitude, image 7 lives in the region the fixture hides.
GEOM = {1: (0.12, 0.12), 2: (0.27, 0.31), 3: (0.41, 0.12),
        4: (10.12, 20.27), 5: None, 6: (200.0, 10.0), 7: (-10.22, -10.22)}
# gate rows: (image_id, det_idx, p_dog). 2/1 sits below the cut, 2/0 exactly
# on it -- ">= keeps" is part of the contract.
GATE = [(1, 0, 0.9), (1, 1, 0.7), (2, 0, 0.5), (2, 1, 0.49),
        (3, 0, 0.6), (4, 0, 0.99), (5, 0, 0.9), (6, 0, 0.9), (7, 0, 0.9)]
# leash rows: (image_id, det_idx, p_leashed); None is a row the model never
# scored, which must land on NEITHER side
LEASH = [(1, 0, 0.8), (1, 1, 0.2), (2, 0, 0.5), (3, 0, None),
         (4, 0, 0.1), (5, 0, 0.9)]


def cell(v, res):
    return round(math.floor(v / res) * res + res / 2, 4)


def expected_cells(weights, res):
    """{(x, y): n} from raw fixture points, binned the way the map bins."""
    out = {}
    for (lon, lat), n in weights:
        k = (round(cell(lon, res), 3), round(cell(lat, res), 3))
        out[k] = out.get(k, 0) + n
    return {k: v for k, v in out.items() if v}


def got_cells(level):
    return {(round(p[0], 3), round(p[1], 3)): p[2] for p in level['points']}


def build_fixture(tmp):
    """A store small enough to count on fingers, written with duckdb so the
    guard needs nothing the builder itself does not."""
    gate_dir = os.path.join(tmp, 'gate')
    leash_dir = os.path.join(tmp, 'leash')
    man_dir = os.path.join(tmp, 'grid_runs')
    out_dir = os.path.join(tmp, 'out')
    for d in (gate_dir, leash_dir, man_dir, out_dir):
        os.makedirs(d, exist_ok=True)
    con = duckdb.connect()

    def parquet(path, sql, rows):
        con.execute(f"COPY ({sql}) TO '{path}' (FORMAT PARQUET)", rows)

    def geo(iid):
        if GEOM[iid] is None:
            return None
        return json.dumps({'type': 'Point',
                           'coordinates': list(GEOM[iid])})

    # manifests: image 1 appears on two "drives" on purpose
    mans = [
        (os.path.join(man_dir, 'm1.parquet'), 'Testland', [1, 2, 3, 5, 6]),
        (os.path.join(man_dir, 'm2.parquet'), 'Testland', [1, 4]),
        (os.path.join(man_dir, 'm3.parquet'), 'Hiddenia', [7]),
    ]
    for path, _, ids in mans:
        con.execute('CREATE OR REPLACE TEMP TABLE m'
                    '(image_id BIGINT, computed_geometry VARCHAR)')
        con.executemany('INSERT INTO m VALUES (?, ?)',
                        [(i, geo(i)) for i in ids])
        con.execute(f"COPY m TO '{path}' (FORMAT PARQUET)")
    snap = os.path.join(tmp, 'catalog.parquet')
    # mtime and size_bytes as the real catalog records them: the signature
    # reads the manifest set out of these columns rather than off the
    # snapshot's own stat, which `catalog refresh` moves every hour.
    write_snapshot(con, snap, mans, man_dir)

    con.execute('CREATE OR REPLACE TEMP TABLE g(image_id VARCHAR, '
                'det_idx INTEGER, p_dog FLOAT, model VARCHAR)')
    con.executemany('INSERT INTO g VALUES (?, ?, ?, ?)',
                    [(str(i), d, p, 'dogbin_test') for i, d, p in GATE])
    # two shards, because production has 168 and a one-file glob hides a
    # sort/list bug
    con.execute(f"COPY (SELECT * FROM g WHERE image_id IN ('1','2')) TO "
                f"'{os.path.join(gate_dir, 'gate-00000.parquet')}' "
                f"(FORMAT PARQUET)")
    con.execute(f"COPY (SELECT * FROM g WHERE image_id NOT IN ('1','2')) TO "
                f"'{os.path.join(gate_dir, 'gate-00001.parquet')}' "
                f"(FORMAT PARQUET)")
    con.execute('CREATE OR REPLACE TEMP TABLE le(image_id VARCHAR, '
                'det_idx INTEGER, p_leashed FLOAT, model VARCHAR)')
    con.executemany('INSERT INTO le VALUES (?, ?, ?, ?)',
                    [(str(i), d, p, 'leash_test') for i, d, p in LEASH])
    con.execute(f"COPY le TO "
                f"'{os.path.join(leash_dir, 'leash-00000.parquet')}' "
                f"(FORMAT PARQUET)")
    con.close()
    return dict(gate_dir=gate_dir, leash_dir=leash_dir, snapshot=snap,
                out_dir=out_dir, hide_regions=frozenset({'Hiddenia'}))


def write_snapshot(con, snap, mans, man_dir, drop=()):
    """The catalog snapshot for a manifest list. `drop` names manifests to
    keep in the catalog with their stats FROZEN -- which is what an unmounted
    drive looks like: the rows stay, the files do not."""
    rows = []
    for path, region, _ in mans:
        try:
            st = os.stat(path)
            mtime, size = float(st.st_mtime), int(st.st_size)
        except OSError:
            mtime, size = 0.0, 0
        rows.append((path, region, 'ground_animals', mtime, size))
    rows.append((os.path.join(man_dir, 'nope.parquet'), 'Testland', 'other',
                 0.0, 0))
    con.execute('CREATE OR REPLACE TEMP TABLE snap'
                '(path VARCHAR, region VARCHAR, kind VARCHAR, '
                'mtime DOUBLE, size_bytes BIGINT)')
    con.executemany('INSERT INTO snap VALUES (?, ?, ?, ?, ?)', rows)
    con.execute(f"COPY snap TO '{snap}' (FORMAT PARQUET)")


def count_checks(bad, fx):
    r = ml.refresh(force=True, **fx)
    if not r.get('built'):
        bad.append(f'a forced refresh on a fresh fixture did not build: {r}')
        return None
    dogs_p = os.path.join(fx['out_dir'], 'map_layer_dogs.json')
    leash_p = os.path.join(fx['out_dir'], 'map_layer_leash.json')
    try:
        dogs = json.load(open(dogs_p))
        leash = json.load(open(leash_p))
    except (OSError, ValueError) as e:
        bad.append(f'output unreadable: {e}')
        return None

    # dogs: images 1-4 locate (2+1+1+1 kept crops); 5 has no geometry, 6 an
    # impossible longitude, 7 sits in the hidden region -- 3 kept crops that
    # must be reported unlocated rather than silently vanish or leak on
    if dogs.get('total') != 5:
        bad.append(f"dogs total is {dogs.get('total')}, expected the 5 "
                   f"located gate-kept crops (2+1+1+1; the 0.49 crop is out, "
                   f"the 0.50 crop is in, the duplicated image counts once)")
    if dogs.get('images') != 4:
        bad.append(f"dogs images is {dogs.get('images')}, expected 4")
    if dogs.get('unlocated') != 3:
        bad.append(f"dogs unlocated is {dogs.get('unlocated')}, expected 3 "
                   f"(no geometry, lon 200, hidden region)")
    if dogs.get('source') != 'model:dogbin_test':
        bad.append(f"dogs source is {dogs.get('source')!r} -- the payload "
                   f"must name the model that produced it")
    if not dogs.get('sig') or dogs.get('sig') != r.get('sig'):
        bad.append('dogs payload does not carry the build signature')

    dog_weights = [(GEOM[i], n) for i, n in
                   ((1, 2), (2, 1), (3, 1), (4, 1))]
    for res in RES_LIST:
        lvl = (dogs.get('levels') or {}).get(str(res))
        if not lvl:
            bad.append(f'dogs levels lack the {res} grid the map renders')
            continue
        want = expected_cells(dog_weights, res)
        got = got_cells(lvl)
        if got != want:
            bad.append(f'dogs {res} grid is {got}, expected {want}')
        if lvl.get('res') != res or lvl.get('max') != max(want.values()):
            bad.append(f"dogs {res} grid header res={lvl.get('res')} "
                       f"max={lvl.get('max')} does not describe its points")
    hidden = [p for lvl in (dogs.get('levels') or {}).values()
              for p in lvl['points'] if p[0] < -5]
    if hidden:
        bad.append(f'a hidden region leaked onto the dogs layer: {hidden}')

    # leash: image 1 splits 1/1, image 2's 0.50 reads leashed, image 4 is
    # loose, image 5 locates nowhere, and image 3's unscored row is NEITHER
    if (leash.get('leashed_total'), leash.get('loose_total')) != (2, 2):
        bad.append(f"leash totals are {leash.get('leashed_total')}/"
                   f"{leash.get('loose_total')} leashed/loose, expected 2/2 "
                   f"-- an unscored row must land on neither side")
    if leash.get('unlocated') != 1:
        bad.append(f"leash unlocated is {leash.get('unlocated')}, expected "
                   f"the 1 scored row on the geometry-less image")
    if leash.get('source') != 'model:leash_test':
        bad.append(f"leash source is {leash.get('source')!r} -- the payload "
                   f"must name the model that produced it")
    for key, weights in (
            ('leashed_levels', [(GEOM[1], 1), (GEOM[2], 1)]),
            ('loose_levels', [(GEOM[1], 1), (GEOM[4], 1)])):
        for res in RES_LIST:
            lvl = (leash.get(key) or {}).get(str(res))
            if not lvl:
                bad.append(f'leash {key} lack the {res} grid')
                continue
            want = expected_cells(weights, res)
            if got_cells(lvl) != want:
                bad.append(f'leash {key} {res} grid is {got_cells(lvl)}, '
                           f'expected {want}')

    # renderer shape: [lon, lat, n] with a positive integer count
    for name, doc, keys in (('dogs', dogs, ('levels',)),
                            ('leash', leash,
                             ('leashed_levels', 'loose_levels'))):
        for key in keys:
            for res, lvl in (doc.get(key) or {}).items():
                for p in lvl['points']:
                    if (len(p) != 3 or not isinstance(p[2], int)
                            or p[2] <= 0):
                        bad.append(f'{name} {key} {res} has a point the map '
                                   f'cannot render: {p}')
                        break
    return dogs_p, leash_p


def skip_checks(bad, fx, dogs_p, leash_p):
    def stamps():
        return (os.stat(dogs_p).st_mtime_ns, os.stat(leash_p).st_mtime_ns)

    before = stamps()
    r = ml.refresh(**fx)
    if r.get('built') or r.get('reason') != 'signature match':
        bad.append(f'nothing changed, yet refresh rebuilt: {r}')
    if stamps() != before:
        bad.append('nothing changed, yet the outputs were rewritten -- the '
                   'skip must not cost a write')
    if r.get('secs', 99) > 2.0:
        bad.append(f"the warm path took {r.get('secs'):.2f}s on a toy store "
                   f"-- it is doing the join it exists to skip")
    # a shard that moved must invalidate the signature
    shard = os.path.join(fx['gate_dir'], 'gate-00000.parquet')
    st = os.stat(shard)
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    r = ml.refresh(**fx)
    if not r.get('built'):
        bad.append('a touched gate shard did not trigger a rebuild')
    if ml.refresh(**fx).get('built'):
        bad.append('the rebuild did not store the new signature')
    # force must always build
    if not ml.refresh(force=True, **fx).get('built'):
        bad.append('force=True was ignored')


def atomicity_checks(bad, dogs_p):
    with open(dogs_p, 'rb') as fh:
        before = fh.read()
    try:
        ml._atomic_write(dogs_p, {'bad': object()})   # not serialisable
    except TypeError:
        pass
    else:
        bad.append('_atomic_write serialised an object json cannot -- the '
                   'failure this check needs never happened')
    with open(dogs_p, 'rb') as fh:
        if fh.read() != before:
            bad.append('a failed write changed the final file -- a crash '
                       'mid-serialisation must leave the previous build')
    if os.path.exists(dogs_p + '.tmp'):
        bad.append('a failed write left its .tmp behind')

    # every write in the module must go through the helper; a bare
    # open(final, 'w') anywhere reintroduces the torn-page failure
    src_p = os.path.abspath(ml.__file__).replace('.pyc', '.py')
    tree = ast.parse(open(src_p).read())
    writers = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'open'):
                mode = ''
                if len(node.args) > 1 and isinstance(node.args[1],
                                                     ast.Constant):
                    mode = str(node.args[1].value)
                for kw in node.keywords:
                    if kw.arg == 'mode' and isinstance(kw.value,
                                                       ast.Constant):
                        mode = str(kw.value.value)
                if any(c in mode for c in 'wax'):
                    writers.append(fn.name)
    if [w for w in writers if w != '_atomic_write']:
        bad.append(f'open-for-write outside _atomic_write: {writers} -- '
                   f'every output write must be .tmp + os.replace')
    helper = next((f for f in ast.walk(tree)
                   if isinstance(f, ast.FunctionDef)
                   and f.name == '_atomic_write'), None)
    if helper is None:
        bad.append('_atomic_write is gone')
    else:
        # an actual os.replace CALL, not the word in the docstring -- a
        # string match here passed while the helper wrote straight through
        replaces = [n for n in ast.walk(helper)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'replace']
        if not replaces:
            bad.append('_atomic_write no longer os.replace()s -- the write '
                       'is not atomic, whatever the name says')


def manifest_checks(bad, fx, dogs_p, leash_p):
    """A manifest the catalog names and the disk has not got is not a small
    hole in the picture: every crop whose image lived in it drops out of the
    geometry join and is added to 'unlocated', which is a positive claim
    about those frames -- they have no coordinates -- and it is false. The
    ground_animals manifests live across two removable drives, so this is one
    unplugged disk, and the result is a plausible map with a fresh built_at
    and a wrong total. It must not publish.
    """
    man_dir = os.path.join(os.path.dirname(fx['snapshot']), 'grid_runs')
    mans = [(os.path.join(man_dir, 'm1.parquet'), 'Testland', None),
            (os.path.join(man_dir, 'm2.parquet'), 'Testland', None),
            (os.path.join(man_dir, 'm3.parquet'), 'Hiddenia', None)]

    def stamps():
        return (os.stat(dogs_p).st_mtime_ns, os.stat(leash_p).st_mtime_ns)

    good = ml.refresh(force=True, **fx)
    if not good.get('built'):
        bad.append(f'the complete fixture did not build: {good}')
        return
    if good.get('manifests_named') != 2 or good.get('manifests_found') != 2:
        bad.append(f"a build does not say what manifest set it read: "
                   f"named={good.get('manifests_named')} "
                   f"found={good.get('manifests_found')} (the third manifest "
                   f"is in a hidden region and is not part of the set)")
    doc = json.load(open(dogs_p))
    if (doc.get('manifests_named'), doc.get('manifests_found')) != (2, 2):
        bad.append(f"the published layer does not carry the manifest set its "
                   f"geometry came from: {doc.get('manifests_named')} / "
                   f"{doc.get('manifests_found')} -- 'unlocated' is a claim "
                   f"about exactly that set")

    # the drive goes away: the catalog keeps its rows, the files are gone
    gone = os.path.join(man_dir, 'm2.parquet')
    kept = gone + '.unmounted'
    os.rename(gone, kept)
    before = stamps()
    # With nothing else moved the signature still matches and the layers are
    # simply left alone -- which is the right answer, and is why the refusal
    # below has to be provoked by a build that would otherwise run.
    quiet = ml.refresh(**fx)
    if quiet.get('built'):
        bad.append(f'a missing manifest rebuilt the layers on its own: '
                   f'{quiet}')
    r = ml.refresh(force=True, **fx)
    if r.get('built'):
        bad.append(f'a manifest the catalog names and the disk has not got '
                   f'was published anyway: {r.get("dogs")} -- those crops '
                   f'are counted as having no coordinates, which is not true')
    if '1 of the 2' not in str(r.get('reason', '')):
        bad.append(f"the refusal does not say how much is missing: "
                   f"{r.get('reason')!r}")
    if stamps() != before:
        bad.append('the refusal still rewrote the layer files -- the point '
                   'of refusing is that the last correct build stays served')
    if (r.get('manifests_named'), r.get('manifests_found')) != (2, 1):
        bad.append(f"the refusal does not count the set: "
                   f"{r.get('manifests_named')} / {r.get('manifests_found')}")
    # ...and an operator who knows the drive is gone for good can still say so
    forced = ml.refresh(force=True, allow_missing=True, **fx)
    if not forced.get('built') or forced.get('manifests_found') != 1:
        bad.append(f'allow_missing did not publish the partial build: '
                   f'{forced}')
    os.rename(kept, gone)

    # A HIDDEN region is not part of the set, so losing it changes nothing.
    hid = os.path.join(man_dir, 'm3.parquet')
    os.rename(hid, hid + '.unmounted')
    r = ml.refresh(force=True, **fx)
    if not r.get('built'):
        bad.append(f'a missing manifest from a HIDDEN region blocked the '
                   f'build: {r.get("reason")!r} -- it was never read')
    os.rename(hid + '.unmounted', hid)
    ml.refresh(force=True, **fx)


def signature_checks(bad, fx):
    """The skip has to survive `catalog refresh`.

    catalog.parquet is rewritten unconditionally before every dashboard
    build, so a signature keyed on the snapshot's own mtime never matched
    twice: the documented "an unchanged store never rebuilds it" was dead
    code in production and every cycle paid the 32M-row join to write a
    byte-identical file. The signature is over what the join READS.
    """
    import duckdb as _dd
    man_dir = os.path.join(os.path.dirname(fx['snapshot']), 'grid_runs')
    mans = [(os.path.join(man_dir, 'm1.parquet'), 'Testland', None),
            (os.path.join(man_dir, 'm2.parquet'), 'Testland', None),
            (os.path.join(man_dir, 'm3.parquet'), 'Hiddenia', None)]
    if not ml.refresh(force=True, **fx).get('built'):
        bad.append('the fixture would not build before the signature check')
        return
    sig = ml.refresh(**fx).get('sig')
    con = _dd.connect()
    try:
        write_snapshot(con, fx['snapshot'], mans, man_dir)   # same rows again
    finally:
        con.close()
    st = os.stat(fx['snapshot'])
    os.utime(fx['snapshot'], ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    r = ml.refresh(**fx)
    if r.get('built'):
        bad.append('a catalog snapshot rewritten with the same rows forced a '
                   'full rebuild -- keyed on the file\'s stat, the skip can '
                   'never happen in production, where the catalog is '
                   'rewritten every cycle')
    if r.get('sig') != sig:
        bad.append('the signature moved with the snapshot\'s mtime')
    # ...but a manifest that really changed still invalidates it. Rewritten
    # rather than corrupted: the rebuild this must trigger has to be able to
    # READ the file, or the check passes on an exception.
    m1 = os.path.join(man_dir, 'm1.parquet')
    con = _dd.connect()
    try:
        con.execute(f"COPY (SELECT * FROM read_parquet('{m1}')) "
                    f"TO '{m1}.new' (FORMAT PARQUET)")
        os.replace(m1 + '.new', m1)
        write_snapshot(con, fx['snapshot'], mans, man_dir)
    finally:
        con.close()
    if not ml.refresh(**fx).get('built'):
        bad.append('a manifest whose size and mtime moved did not rebuild -- '
                   'the signature is blind to the files the join reads')


def build_call_checks(bad):
    """The dashboard's build() must not throw refresh()'s answer away.

    Two of its exits are returns rather than raises -- no catalog snapshot,
    and a manifest set with files missing off it -- so a discarded result
    means the layers quietly stop advancing while the page rebuilds around
    them every hour, and nothing anywhere says why. Only the raising paths
    ever reached the operator.
    """
    import ast as _ast
    import re as _re
    src_path = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')
    try:
        with open(src_path) as fh:
            src = fh.read()
        tree = _ast.parse(src)
    except (OSError, SyntaxError) as e:
        bad.append(f'could not read dashboard.py: {e}')
        return

    def is_refresh(node):
        f = getattr(node, 'func', None)
        return (isinstance(node, _ast.Call)
                and isinstance(f, _ast.Attribute) and f.attr == 'refresh'
                and isinstance(f.value, _ast.Name) and f.value.id == 'ml')

    calls, kept = 0, None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Expr) and is_refresh(node.value):
            calls += 1
            bad.append('dashboard.py calls ml.refresh() as a bare statement '
                       '-- the reason a layer did not rebuild is thrown '
                       'away, and a stalled layer looks exactly like a '
                       'skipped one')
        elif isinstance(node, _ast.Assign) and is_refresh(node.value):
            calls += 1
            if isinstance(node.targets[0], _ast.Name):
                kept = node.targets[0].id
    if not calls:
        bad.append('dashboard.py no longer refreshes the map layers')
        return
    if kept is None:
        return
    # ...and the reason has to reach a human. Read off the function that
    # holds the call, so a print somewhere else in the file cannot stand in
    # for one here.
    holder = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef):
            seg = _ast.get_source_segment(src, node) or ''
            if 'ml.refresh(' in seg and (holder is None
                                         or len(seg) < len(holder)):
                holder = seg
    if holder is None:
        return
    if not _re.search(r'print\([^\n]*' + _re.escape(kept), holder):
        bad.append(f'nothing prints the reason {kept}.refresh() gave for not '
                   f'building -- a layer that has stopped advancing has to '
                   f'look different from one that had nothing to do')
    if 'signature match' not in _re.sub(r'#[^\n]*', '', holder):
        bad.append('the normal quiet skip is not told apart from a refusal, '
                   'so either every cycle prints a line nobody reads or none '
                   'of them do')


def empty_store_checks(bad, fx):
    """No shards at all is a state, not a crash: first boot on a fresh
    clone runs the refresher before any classifier has written a row."""
    tmp = tempfile.mkdtemp(prefix='adv_map_layers_empty_')
    for d in ('gate', 'leash', 'out'):
        os.makedirs(os.path.join(tmp, d))
    try:
        r = ml.refresh(force=True, gate_dir=os.path.join(tmp, 'gate'),
                       leash_dir=os.path.join(tmp, 'leash'),
                       snapshot=fx['snapshot'],
                       out_dir=os.path.join(tmp, 'out'),
                       hide_regions=fx['hide_regions'])
    except Exception as e:  # noqa: BLE001 - that is the test
        bad.append(f'an empty store crashed the build: '
                   f'{type(e).__name__}: {e}')
        return
    if not r.get('built'):
        bad.append(f'an empty store did not build empty layers: {r}')
        return
    doc = json.load(open(os.path.join(tmp, 'out', 'map_layer_dogs.json')))
    if doc.get('total') or doc['levels']['0.5']['points']:
        bad.append('an empty gate store produced a non-empty dogs layer')


def main():
    bad = []
    tmp = tempfile.mkdtemp(prefix='adv_map_layers_')
    fx = build_fixture(tmp)
    paths = count_checks(bad, fx)
    if paths:
        skip_checks(bad, fx, *paths)
        atomicity_checks(bad, paths[0])
        manifest_checks(bad, fx, *paths)
        signature_checks(bad, fx)
    build_call_checks(bad)
    empty_store_checks(bad, fx)
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('the model layers count what the ledgers say, skip when nothing '
          'moved, never tear an output, and say which model drew them')
    return 0


if __name__ == '__main__':
    sys.exit(main())
