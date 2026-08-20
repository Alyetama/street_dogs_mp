#!/usr/bin/env python3
"""Model-result grid layers for the main page's map.

Two layers, two files under data/dashboard/, both in the exact shape the map
client already renders (a levels dict: {"<res>": {res, max, points:
[[lon, lat, n], ...]}}, cell centers on the same floor(x/res)*res + res/2
grid build_map_points() uses):

  map_layer_dogs.json   every gate-kept crop (p_dog >= P_DOG_MIN) in
                        data/gate, placed at its image's harvest coordinate.
  map_layer_leash.json  the leash classifier's calls on those crops, split
                        p_leashed >= P_LEASH_SPLIT (leashed) vs below
                        (unleashed), as two levels dicts in one file so the page
                        can draw either side or a ratio without a second
                        fetch.

Everything here is MODEL output -- the gate and leash ledgers under
data/gate and data/leash are classifier scores, not human verdicts -- so
each payload carries a 'source' field naming the model(s) that produced it.
That field is load-bearing: this dashboard never lets a model number
masquerade as a human one, and a layer file found without it should be
treated as mislabeled, not merely terse.

Read-only on every store. The only thing written is the two JSONs (atomic:
.tmp + os.replace). refresh() keys a signature on the shard lists and on the
manifest set the catalog names -- their recorded mtimes and sizes, not the
snapshot file's own stat, which is rewritten every cycle whether anything
changed or not -- stores it inside the outputs, and skips the 32M-row join
entirely when nothing moved: the cold build is about four seconds, the warm
call under a tenth of one.

And it does not publish at all from a manifest set that is not all there.
A manifest the catalog names and the disk has not got takes its images'
crops out of the geometry join and into 'unlocated', which reads as a fact
about those frames and is false; one unmounted drive is 40% of them.

    python tools/dashboard/map_layers.py [--force] [--allow-missing]
"""

import argparse
import glob
import hashlib
import json
import os
import time

import duckdb

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'data', 'dashboard')
GATE_DIR = os.path.join(REPO, 'data', 'gate')
LEASH_DIR = os.path.join(REPO, 'data', 'leash')
# the lock-free catalog snapshot, same source build_map_points() reads, so
# this never contends with the live catalog DB
SNAPSHOT = os.path.join(REPO, 'data', 'catalog.parquet')
DOGS_FILE = os.path.join(OUT, 'map_layer_dogs.json')
LEASH_FILE = os.path.join(OUT, 'map_layer_leash.json')

SCHEMA = 1
P_DOG_MIN = 0.5      # a crop the gate scored below this was not kept
P_LEASH_SPLIT = 0.5  # at or above reads as leashed, below as unleashed
# The harvest map splits these across two files because its fine grid is
# 260K cells; the gate keeps ~1/40 of the harvest's frames, so all three
# resolutions of these layers together are smaller than map_points.json
# alone and one fetch per layer is the cheaper shape.
RES_LIST = (0.5, 0.15, 0.05)


def _hide_regions():
    """The same hide_regions the dashboard honours, read the same way
    (environment > dashboard.config.json > default) but without importing
    the 13k-line server module to get one frozenset. Hidden regions never
    enter the harvest map, so a dog counted there and not here would read
    as a >100% hit rate in whatever the UI divides."""
    raw = os.environ.get('DASHBOARD_HIDE_REGIONS')
    if raw:
        vals = [x.strip() for x in raw.split(',') if x.strip()]
        return frozenset(vals)
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'dashboard.config.json')
    try:
        with open(cfg_path) as fh:
            v = json.load(fh).get('hide_regions')
    except (OSError, ValueError, AttributeError):
        v = None
    if isinstance(v, str):
        vals = [x.strip() for x in v.split(',') if x.strip()]
    elif isinstance(v, (list, tuple)):
        vals = [str(x).strip() for x in v if str(x).strip()]
    else:
        vals = []
    return frozenset(vals or ('Arctic', 'Antarctica'))


def _manifests(snapshot, hide_regions):
    """The ground_animals manifests this build is entitled to read.

    ([(path, mtime, size), ...], whether the catalog carried the two stat
    columns). Sorted, hidden regions already dropped -- what comes back IS
    the manifest set, both for the join and for the signature.
    """
    con = duckdb.connect()
    try:
        try:
            rows = con.execute(
                "SELECT DISTINCT path, region, mtime, size_bytes "
                "FROM read_parquet(?) WHERE kind='ground_animals'",
                [snapshot]).fetchall()
            dated = True
        except duckdb.Error:
            # a catalog written before it recorded a file's mtime and size
            rows = [(p, r, None, None) for p, r in con.execute(
                "SELECT DISTINCT path, region FROM read_parquet(?) "
                "WHERE kind='ground_animals'", [snapshot]).fetchall()]
            dated = False
    finally:
        con.close()
    return (sorted((p, m, sz) for p, r, m, sz in rows
                   if r not in hide_regions), dated)


def _signature(gate_shards, leash_shards, manifests, snapshot=None):
    """One hash over everything the build reads: shard names, mtimes and
    sizes for both stores, and the manifest set the catalog names -- each
    with the mtime and size the catalog recorded for it.

    NOT the catalog snapshot's own stat, which is what this keyed on first.
    `catalog refresh` rewrites catalog.parquet unconditionally and the serve
    loop runs it before every build, so that mtime moves every hour whether
    or not one row changed: keyed on it the signature never matched twice,
    the documented skip was dead code in production, and every cycle paid the
    32M-row join to write a byte-identical file. The rows are what the join
    actually reads, and an unchanged catalog hands back the same ones. (A
    catalog too old to carry mtime and size has nothing to key on, so its
    stat comes back as the proxy it always was -- passed in as `snapshot`.)
    """
    ent = []
    for p in (list(gate_shards) + list(leash_shards)
              + ([snapshot] if snapshot else [])):
        try:
            st = os.stat(p)
            ent.append((os.path.basename(p), st.st_mtime_ns, st.st_size))
        except OSError:
            ent.append((os.path.basename(p), None, None))
    # the full path here, not the basename: cell-named manifests repeat
    # across drives and the join reads both. Only the digest is ever stored.
    ent.extend(list(m) for m in manifests)
    return hashlib.sha256(
        json.dumps(ent, sort_keys=False).encode()).hexdigest()


def _stored_sig(path):
    """The signature a previous build left inside its output, or None."""
    try:
        with open(path) as fh:
            doc = json.load(fh)
        return doc.get('sig') if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def _atomic_write(path, payload):
    """All output goes through here: dump to .tmp, then one os.replace, so a
    crash mid-serialisation leaves the previous build intact instead of a
    truncated JSON the page would fail to parse. The half-written .tmp is
    removed on failure rather than left to look like work in progress."""
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w') as fh:
            json.dump(payload, fh)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def _grid(con, table, expr):
    """Levels dict for one source table, aggregated at every RES_LIST
    resolution with the exact binning build_map_points() uses, so these
    cells land pixel-for-pixel on the harvest's."""
    out = {}
    for res in RES_LIST:
        rows = con.execute(f"""
          SELECT round(floor(lon/{res})*{res}+{res / 2}, 4) x,
                 round(floor(lat/{res})*{res}+{res / 2}, 4) y,
                 CAST(sum({expr}) AS BIGINT) n
          FROM {table} GROUP BY 1, 2 HAVING sum({expr}) > 0""").fetchall()
        pts = [[r[0], r[1], r[2]] for r in rows]
        out[str(res)] = {'res': res,
                         'max': max((p[2] for p in pts), default=0),
                         'points': pts}
    return out


def _models(con, shards, where):
    """'model:a+b' from the rows a layer actually counted, never a constant
    -- a retrain writes a new model tag into the shards and the payload has
    to follow it without anyone remembering this file exists."""
    rows = con.execute(
        f"SELECT DISTINCT model FROM read_parquet(?) WHERE {where} "
        "AND model IS NOT NULL ORDER BY 1", [shards]).fetchall()
    names = [r[0] for r in rows if r[0]]
    return 'model:' + ('+'.join(names) if names else 'unknown')


def refresh(force=False, gate_dir=None, leash_dir=None, snapshot=None,
            out_dir=None, hide_regions=None, allow_missing=False):
    """Build both layer files if anything changed; report either way.

    The keyword arguments exist for the guard test, which points them at a
    synthetic fixture store -- production callers pass nothing and get the
    repo paths. Returns a dict: built, reason, sig, secs, and per-layer
    totals when a build ran.

    `allow_missing` publishes from a manifest set the catalog names and the
    disk does not have. It is off by default and should stay off: those
    crops do not vanish, they are counted as 'unlocated', which is a
    positive claim that their frames carry no coordinates -- and it is false.
    """
    t0 = time.time()
    gate_dir = gate_dir or GATE_DIR
    leash_dir = leash_dir or LEASH_DIR
    snapshot = snapshot or SNAPSHOT
    out_dir = out_dir or OUT
    if hide_regions is None:
        hide_regions = _hide_regions()
    dogs_path = os.path.join(out_dir, os.path.basename(DOGS_FILE))
    leash_path = os.path.join(out_dir, os.path.basename(LEASH_FILE))

    gate_shards = sorted(glob.glob(os.path.join(gate_dir, 'gate-*.parquet')))
    leash_shards = sorted(
        glob.glob(os.path.join(leash_dir, 'leash-*.parquet')))
    if not os.path.exists(snapshot):
        return {'built': False, 'reason': 'no catalog snapshot',
                'secs': time.time() - t0}
    named, dated = _manifests(snapshot, hide_regions)
    sig = _signature(gate_shards, leash_shards, named,
                     None if dated else snapshot)
    if (not force and _stored_sig(dogs_path) == sig
            and _stored_sig(leash_path) == sig):
        return {'built': False, 'reason': 'signature match', 'sig': sig,
                'secs': time.time() - t0}

    # THE MANIFEST SET HAS TO BE ALL THERE, or this does not publish.
    # Every gate crop whose image lived in a manifest that is not on disk
    # falls out of the geometry join and is added to 'unlocated' -- so a
    # drive that is not mounted does not produce a blank layer that reads as
    # broken, it produces a plausible map with a fresh built_at, a wrong
    # total, and a specific false claim about where those dogs are. The
    # ground_animals manifests live across two removable drives, so losing
    # one is one unplugged disk. The catalog prunes files deleted from a
    # MOUNTED drive and leaves an offline drive's rows alone, which is
    # exactly why a shortfall here means "a drive is missing" rather than
    # "somebody tidied up". Refusing keeps the previous, correct layers in
    # front of the reader.
    found = [p for p, _, _ in named if os.path.exists(p)]
    missing = len(named) - len(found)
    if not found:
        return {'built': False, 'reason': 'no ground_animals manifests',
                'sig': sig, 'manifests_named': len(named),
                'manifests_found': 0, 'secs': time.time() - t0}
    if missing and not allow_missing:
        return {'built': False,
                'reason': f'{missing:,} of the {len(named):,} ground_animals '
                          f'manifests the catalog names are not on disk -- '
                          f'mount the drive, or refresh the catalog if they '
                          f'are gone for good',
                'sig': sig, 'manifests_named': len(named),
                'manifests_found': len(found), 'secs': time.time() - t0}
    paths = found

    con = duckdb.connect()
    try:
        con.execute('PRAGMA threads=4')          # polite to the running jobs
        con.execute("SET memory_limit='8GB'")
        con.execute('INSTALL json; LOAD json;')

        # what each ledger says, one row per image. image_id stays VARCHAR
        # throughout: that is the ledgers' own type, and the manifest side is
        # cast to match rather than the other way around (lonlat_for() sets
        # the precedent) so a malformed id drops the row, never the build.
        if gate_shards:
            con.execute(
                f"""CREATE TEMP TABLE gate_w AS
                SELECT CAST(image_id AS VARCHAR) image_id, count(*) crops
                FROM read_parquet(?) WHERE p_dog >= {P_DOG_MIN}
                GROUP BY 1""", [gate_shards])
            dogs_source = _models(con, gate_shards,
                                  f'p_dog >= {P_DOG_MIN}')
        else:
            con.execute('CREATE TEMP TABLE gate_w'
                        '(image_id VARCHAR, crops BIGINT)')
            dogs_source = 'model:unknown'
        if leash_shards:
            # a row with no score lands on neither side: an unscored crop
            # painted as "loose" would be the model claiming what it never
            # said
            con.execute(
                f"""CREATE TEMP TABLE leash_w AS
                SELECT CAST(image_id AS VARCHAR) image_id,
                  count(*) FILTER (WHERE p_leashed >= {P_LEASH_SPLIT}) leashed,
                  count(*) FILTER (WHERE p_leashed < {P_LEASH_SPLIT}) loose
                FROM read_parquet(?) GROUP BY 1""", [leash_shards])
            leash_source = _models(con, leash_shards,
                                   'p_leashed IS NOT NULL')
        else:
            con.execute('CREATE TEMP TABLE leash_w'
                        '(image_id VARCHAR, leashed BIGINT, loose BIGINT)')
            leash_source = 'model:unknown'
        con.execute('CREATE TEMP TABLE want AS '
                    'SELECT image_id FROM gate_w UNION '
                    'SELECT image_id FROM leash_w')

        # One location per image. The GROUP BY is not decoration: the same
        # image can sit in manifests on two drives (cells were re-harvested
        # across drives and deduped by id downstream), and without it every
        # crop in a duplicated image would count twice.
        con.execute(
            """CREATE TEMP TABLE loc AS
            SELECT image_id, any_value(lon) lon, any_value(lat) lat FROM (
              SELECT w.image_id,
                TRY_CAST(json_extract(g.computed_geometry,
                                      '$.coordinates[0]') AS DOUBLE) lon,
                TRY_CAST(json_extract(g.computed_geometry,
                                      '$.coordinates[1]') AS DOUBLE) lat
              FROM read_parquet(?, union_by_name=true) g
              JOIN want w ON CAST(g.image_id AS VARCHAR) = w.image_id
              WHERE g.computed_geometry IS NOT NULL)
            WHERE lon BETWEEN -180 AND 180 AND lat BETWEEN -90 AND 90
            GROUP BY image_id""", [paths])

        con.execute('CREATE TEMP TABLE dpts AS '
                    'SELECT l.lon, l.lat, w.crops FROM gate_w w '
                    'JOIN loc l USING (image_id)')
        con.execute('CREATE TEMP TABLE lpts AS '
                    'SELECT l.lon, l.lat, w.leashed, w.loose FROM leash_w w '
                    'JOIN loc l USING (image_id)')

        gate_total, = con.execute(
            'SELECT coalesce(sum(crops), 0) FROM gate_w').fetchone()
        dogs_total, dogs_images = con.execute(
            'SELECT coalesce(sum(crops), 0), count(*) FROM dpts').fetchone()
        leash_all_leashed, leash_all_loose = con.execute(
            'SELECT coalesce(sum(leashed), 0), coalesce(sum(loose), 0) '
            'FROM leash_w').fetchone()
        leashed_total, loose_total = con.execute(
            'SELECT coalesce(sum(leashed), 0), coalesce(sum(loose), 0) '
            'FROM lpts').fetchone()

        built_at = time.strftime('%Y-%m-%d %H:%M')
        # How much of the manifest set the geometry came from. 'unlocated'
        # is a claim ABOUT those manifests -- these frames carry no
        # coordinates -- so the payload states what it was read against
        # rather than leaving the reader to assume it was everything.
        seen = {'manifests_named': len(named), 'manifests_found': len(found)}
        dogs_doc = {
            'schema': SCHEMA, 'layer': 'dogs_gate', 'sig': sig,
            # model output feeding a picture, and it says so
            'source': dogs_source, 'conf_min': P_DOG_MIN,
            'total': int(dogs_total), 'images': int(dogs_images),
            'unlocated': int(gate_total - dogs_total),
            'levels': _grid(con, 'dpts', 'crops'),
            'built_at': built_at, **seen}
        leash_doc = {
            'schema': SCHEMA, 'layer': 'leash', 'sig': sig,
            'source': leash_source, 'split': P_LEASH_SPLIT,
            'leashed_total': int(leashed_total),
            'loose_total': int(loose_total),
            'unlocated': int(leash_all_leashed + leash_all_loose
                             - leashed_total - loose_total),
            'leashed_levels': _grid(con, 'lpts', 'leashed'),
            'loose_levels': _grid(con, 'lpts', 'loose'),
            'built_at': built_at, **seen}
    finally:
        con.close()

    os.makedirs(out_dir, exist_ok=True)
    _atomic_write(dogs_path, dogs_doc)
    _atomic_write(leash_path, leash_doc)
    return {'built': True, 'reason': 'rebuilt', 'sig': sig,
            'secs': time.time() - t0,
            'manifests_named': len(named), 'manifests_found': len(found),
            'dogs': {'total': dogs_doc['total'],
                     'images': dogs_doc['images'],
                     'unlocated': dogs_doc['unlocated'],
                     'cells': {k: len(v['points'])
                               for k, v in dogs_doc['levels'].items()}},
            'leash': {'leashed': leash_doc['leashed_total'],
                      'loose': leash_doc['loose_total'],
                      'unlocated': leash_doc['unlocated'],
                      'cells': {k: len(v['points'])
                                for k, v in
                                leash_doc['leashed_levels'].items()}}}


def main():
    ap = argparse.ArgumentParser(
        description='build the map layers for model-found dogs and '
                    'leashed-vs-unleashed')
    ap.add_argument('--force', action='store_true',
                    help='rebuild even when the signature matches')
    ap.add_argument('--allow-missing', action='store_true',
                    help='publish even when the catalog names manifests that '
                         'are not on disk -- their crops are then reported '
                         'as having no coordinates, which is not true')
    args = ap.parse_args()
    info = refresh(force=args.force, allow_missing=args.allow_missing)
    if not info['built']:
        print(f"map_layers: skipped ({info['reason']}) "
              f"in {info['secs']:.3f}s")
        return
    d, l = info['dogs'], info['leash']
    print(f"map_layers: built in {info['secs']:.1f}s "
          f"from {info['manifests_found']:,} of the "
          f"{info['manifests_named']:,} manifests the catalog names")
    print(f"  dogs : {d['total']:,} crops on {d['images']:,} images "
          f"({d['unlocated']:,} unlocated), cells {d['cells']}")
    print(f"  leash: {l['leashed']:,} leashed / {l['loose']:,} unleashed "
          f"({l['unlocated']:,} unlocated), cells {l['cells']}")


if __name__ == '__main__':
    main()
