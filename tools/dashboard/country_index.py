#!/usr/bin/env python3
"""Map review-queue crops to countries, so the page can be filtered by one.

Every ground_animals row carries ``computed_geometry`` -- a GeoJSON Point with
the image's real lat/lon -- so a crop's country is a point-in-polygon test
against Natural Earth's 50m admin-0 boundaries (data/geo/), not a guess from
the 5-degree grid cell. A cell that size straddles several countries; using it
would put Nepalese dogs under India.

Only countries the sweep has actually REACHED appear. The list is derived from
the crops that exist, so a country with nothing to review is never offered --
an empty filter option is a dead end, not a feature.

    python tools/dashboard/country_index.py --out data/dashboard/countries.json

Writes {"generated": ts, "by_image": {image_id: ISO3}, "counts": {ISO3: n},
"names": {ISO3: display name}}. Read-only on the catalog and the crops.

Runs on a schedule (dashboard --interval), so it is incremental: image_ids
already resolved in the previous index are reused and only new crops are
looked up. A full cold build over the flag ledgers, the rolling pool and the
harvested review set is a few seconds; the incremental path is sub-second.
"""

import argparse
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CROP_RE = re.compile(r'^(\d+)_(\d+)_(\d+)\.jpg$')
SHAPEFILE = os.path.join(REPO, 'data', 'geo', 'ne_50m_admin_0_countries.shp')
DEFAULT_OUT = os.path.join(REPO, 'data', 'dashboard', 'countries.json')


def review_extra_dirs(repo):
    """The crop directories the review page serves BESIDE the rolling pool.

    dashboard.py's review_extra_dir(), read the same way and by the same
    precedence: $REVIEW_EXTRA_DIR, else the ``review_extra_dir`` key of
    tools/dashboard/dashboard.config.json, resolved against the repo.

    Read here as well as there because the incremental build DROPS a resolved
    id that is no longer in the id set (`by` is rebuilt from `ids`), so a
    build that cannot see the directory does not merely fail to add its crops
    -- it removes the ones the server had already placed.
    """
    d = os.environ.get('REVIEW_EXTRA_DIR')
    if not d:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'dashboard.config.json')
        try:
            with open(cfg_path) as fh:
                d = json.load(fh).get('review_extra_dir')
        except (OSError, ValueError, AttributeError):
            d = None
    if not isinstance(d, str) or not d:
        return []
    return [d if os.path.isabs(d) else os.path.join(repo, d)]


def crop_image_ids(repo, extra_dirs=None):
    """Every image_id that can appear in the review UI: the rolling preview
    pool, the harvested review set, and both flag ledgers (a flagged crop
    still needs a country when the reviewer filters, and the ledgers outlive
    the pool).

    THE HARVESTED SET IS NOT OPTIONAL. review_extra_dir() is the second crop
    directory review_pool_names() and crop_dir() both walk, and it is where
    build_review_set.py puts the crops chosen to spread over cells and
    confidence bands -- exactly the population a country filter exists to
    reach. Indexing only recent_crops left 1,593 of its 1,961 ids with no
    country, and the filter has no unknown-country escape (`c['country'] ==
    want`), so every country selection silently dropped the whole harvested
    set. Nothing on screen said so: the option counts are tallied over crops
    that HAVE a country, so each option's number still matched what it
    delivered.
    """
    ids = set()
    dirs = [os.path.join(repo, 'data', 'dashboard', 'recent_crops')]
    dirs += (review_extra_dirs(repo) if extra_dirs is None
             else [d for d in extra_dirs if d])
    for pool in dirs:
        try:
            names = os.listdir(pool)
        except OSError:
            continue          # absent or unreadable is not an error here
        for n in names:
            m = CROP_RE.match(n)
            if m:
                ids.add(m.group(2))
    for sub in ('hard_negatives', 'hard_positives'):
        p = os.path.join(repo, 'data', sub, 'labels.jsonl')
        try:
            with open(p) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    if r.get('image_id'):
                        ids.add(str(r['image_id']))
        except OSError:
            pass
    return ids


def lonlat_for(ids, repo, con):
    """{image_id: (lon, lat)} from the ground_animals parquets.

    duckdb reads computed_geometry as text and json_extract pulls the
    coordinates, which avoids materialising 32M GeoJSON strings in Python.
    """
    if not ids:
        return {}
    paths = [r[0] for r in con.execute(
        "SELECT path FROM files WHERE kind='ground_animals'").fetchall()]
    if not paths:
        return {}
    src = 'read_parquet([' + ','.join(
        "'" + p.replace("'", "''") + "'" for p in paths) + '])'
    con.execute('CREATE TEMP TABLE want(image_id VARCHAR)')
    con.executemany('INSERT INTO want VALUES (?)', [(i,) for i in sorted(ids)])
    rows = con.execute(
        'SELECT CAST(p.image_id AS VARCHAR), '
        "json_extract(p.computed_geometry, '$.coordinates[0]'), "
        "json_extract(p.computed_geometry, '$.coordinates[1]') "
        f'FROM {src} p JOIN want w ON CAST(p.image_id AS VARCHAR)=w.image_id '
        'WHERE p.computed_geometry IS NOT NULL').fetchall()
    out = {}
    for iid, lon, lat in rows:
        try:
            out[iid] = (float(lon), float(lat))
        except (TypeError, ValueError):
            pass
    return out


def countries_for(pts):
    """{image_id: ISO3} by point-in-polygon, plus {ISO3: name}."""
    if not pts:
        return {}, {}
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    world = gpd.read_file(SHAPEFILE)
    iso_col = next((c for c in ('ADM0_A3', 'ISO_A3', 'SOV_A3')
                    if c in world.columns), None)
    name_col = next((c for c in ('ADMIN', 'NAME', 'NAME_LONG')
                     if c in world.columns), None)
    if not iso_col or not name_col:
        raise SystemExit(f'unexpected shapefile columns: {list(world.columns)[:12]}')
    keys = list(pts)
    g = gpd.GeoDataFrame(
        {'image_id': keys},
        geometry=[Point(*pts[k]) for k in keys],
        crs=world.crs)
    # sjoin, not a per-point loop: 250 polygons x N points through the spatial
    # index instead of N x 250 shapely calls
    j = gpd.sjoin(g, world[[iso_col, name_col, 'geometry']],
                  how='left', predicate='within')
    j = j[~j.index.duplicated(keep='first')]     # a point on a border matches twice
    by = {}
    names = {}
    for iid, iso, nm in zip(j['image_id'], j[iso_col], j[name_col]):
        if isinstance(iso, str) and iso and iso != '-99':
            by[iid] = iso
            names[iso] = nm
    return by, names


def build(repo, out_path, force=False, extra_dirs=None):
    import duckdb
    prev = {}
    if not force:
        try:
            with open(out_path) as fh:
                prev = json.load(fh).get('by_image') or {}
        except (OSError, ValueError):
            prev = {}
    # ids previously looked up that resolved to NO country -- no coordinates
    # in the parquets, or a point outside every polygon. Without remembering
    # them, every incremental build re-scans the parquets and re-runs the
    # point-in-polygon for the same permanently-unresolvable ids.
    misses = set()
    if not force:
        try:
            with open(out_path) as fh:
                misses = set(json.load(fh).get('no_country') or [])
        except (OSError, ValueError):
            misses = set()
    ids = crop_image_ids(repo, extra_dirs)
    todo = {i for i in ids if i not in prev and i not in misses}
    print(f'{len(ids):,} review image_ids; {len(todo):,} need a lookup '
          f'({len(ids) - len(todo):,} reused, {len(misses):,} known-unresolvable)',
          file=sys.stderr)
    by = {i: prev[i] for i in ids if i in prev}
    names = {}
    if todo:
        con = duckdb.connect(os.path.join(repo, 'data', 'catalog.duckdb'),
                             read_only=True)
        try:
            pts = lonlat_for(todo, repo, con)
        finally:
            con.close()
        print(f'  {len(pts):,} of {len(todo):,} had coordinates',
              file=sys.stderr)
        new, names = countries_for(pts)
        by.update(new)
    # names for reused ids come from the previous file
    try:
        with open(out_path) as fh:
            names = {**(json.load(fh).get('names') or {}), **names}
    except (OSError, ValueError):
        pass
    counts = {}
    for iso in by.values():
        counts[iso] = counts.get(iso, 0) + 1
    # keep only misses still reachable, so the list cannot grow without bound
    # as the pool rotates
    misses = {i for i in (misses | (todo - set(by))) if i in ids}
    doc = {'generated': int(time.time()), 'by_image': by,
           'counts': counts, 'names': names,
           'no_country': sorted(misses)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(doc, fh)
    os.replace(tmp, out_path)
    print(f'{len(by):,} crops in {len(counts)} countries -> {out_path}',
          file=sys.stderr)
    return doc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=REPO)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--force', action='store_true',
                    help='re-resolve every id instead of reusing the index')
    ap.add_argument('--extra-dir', action='append', default=None,
                    metavar='DIR',
                    help='another crop directory the review page serves. '
                    'Defaults to $REVIEW_EXTRA_DIR / the dashboard config, '
                    'which is what the server itself uses.')
    a = ap.parse_args()
    doc = build(a.repo, a.out, a.force, a.extra_dir)
    top = sorted(doc['counts'].items(), key=lambda kv: -kv[1])[:12]
    for iso, n in top:
        print(f'  {iso}  {doc["names"].get(iso, iso)[:28]:<30}{n:>6}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
