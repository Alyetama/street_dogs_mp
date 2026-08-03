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
looked up. A full cold build over the flag ledger plus the rolling pool is a
few seconds; the incremental path is sub-second.
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


def crop_image_ids(repo):
    """Every image_id that can appear in the review UI: the rolling preview
    pool plus both flag ledgers (a flagged crop still needs a country when the
    reviewer filters, and the ledgers outlive the pool)."""
    ids = set()
    pool = os.path.join(repo, 'data', 'dashboard', 'recent_crops')
    try:
        for n in os.listdir(pool):
            m = CROP_RE.match(n)
            if m:
                ids.add(m.group(2))
    except OSError:
        pass
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


def build(repo, out_path, force=False):
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
    ids = crop_image_ids(repo)
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
    a = ap.parse_args()
    doc = build(a.repo, a.out, a.force)
    top = sorted(doc['counts'].items(), key=lambda kv: -kv[1])[:12]
    for iso, n in top:
        print(f'  {iso}  {doc["names"].get(iso, iso)[:28]:<30}{n:>6}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
