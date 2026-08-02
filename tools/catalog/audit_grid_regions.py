#!/usr/bin/env python3
"""
Audit the grid's region labels against real country polygons.

``original_global_grid_5deg.csv`` assigns each 5-degree cell a region using
coarse lat/lon boxes applied in priority order. Those boxes are wrong in places:
the ``Africa`` box runs to lon 55 / lat 40 and is evaluated before ``Middle
East`` (which is only 12 cells, lon 55..65), so the whole Arabian peninsula and
the Levant -- Kuwait, Riyadh, Baghdad, Doha, Jerusalem -- end up filed as Africa.

This intersects every cell with Natural Earth country polygons, attributes the
land area inside the cell to each country, maps countries to the project's own
region names, and reports where that disagrees with the CSV.

Two things it is careful about:

*Straddling cells.* At 5 degrees a single cell can hold two continents -- e.g.
35..40E / 15..20N contains Sudan and Eritrea alongside Saudi Arabia and Yemen,
split by the Red Sea. No single label is right there. Rather than silently
picking one, every cell reports ``share`` (the dominant region's fraction of the
cell's land), so ambiguous cells are visible and can be judged separately.

*Ocean cells.* Mostly-water cells keep whatever the CSV says: basin labels
(Pacific/Atlantic/Indian Ocean, Arctic) are a convention this audit has no
opinion about. But an ocean LABEL is not proof of ocean: the original grid filed
the 100%-land Volga cell as Indian Ocean. So the pass-through applies only while
the cell's land fraction is below --ocean-land-frac (default 0.60); a
mostly-land cell is audited like any other regardless of its label.

*Areas are equal-area.* All areas are computed in EPSG:6933 (km2), not in raw
degrees -- a degree of longitude shrinks toward the poles, and degree-area was
within rounding distance of flipping near-threshold verdicts at high latitude.

READ-ONLY: reads the grid CSV, the shapefile and (optionally) the catalog; the
only thing it writes is its own report.

    python tools/catalog/audit_grid_regions.py \\
        --grid original_global_grid_5deg.csv \\
        --shapefile data/geo/ne_50m_admin_0_countries.shp \\
        --out data/geo/region_audit.csv
"""

import argparse
import os
import sys
from collections import defaultdict

import geopandas as gpd
import pandas as pd
import pyproj
from shapely.geometry import box
from shapely.ops import transform as shp_transform

# Equal-area projection for all area arithmetic (km2). Raw EPSG:4326 degree
# areas overweight low latitudes and nearly flipped near-threshold verdicts.
_TO_EA = pyproj.Transformer.from_crs(4326, 6933, always_xy=True).transform


def area_km2(geom):
    """Geometry area in km2, computed in EPSG:6933."""
    return shp_transform(_TO_EA, geom).area / 1e6


# Country ISO_A3 -> project region, for cases the UN subregion gets wrong for
# this project's taxonomy. Everything else falls through to SUBREGION_MAP.
COUNTRY_OVERRIDE = {
    # Russia gets its own project region. Without this, Natural Earth tags the
    # Asian half SUBREGION='Central Asia', which would rename Siberia to Central
    # Asia. Note ISO_A3 is '-99' for Russia, so this only resolves via ADM0_A3.
    # Cells whose land is European Russia then fall out as 'taxonomy' (Europe vs
    # Russia & North Asia) rather than as errors -- a convention, not a mistake.
    'RUS': 'Russia & North Asia',
    'GRL': 'Greenland',
    # UN M49 files Iran and Afghanistan under "Southern Asia"; this project's
    # South Asia box is 60..100E / 5..35N (the subcontinent), and its Middle
    # East cells sit at 55..65E -- i.e. Iran was always meant to be Middle East.
    'IRN': 'Middle East',
    'AFG': 'Central Asia',
    'AUS': 'Australia',
    'NZL': 'New Zealand & Pacific',
    # M49 files Turkey and Cyprus under Western Asia -> Middle East already;
    # these entries are redundant belt-and-braces, kept for explicitness.
    'TUR': 'Middle East',
    'CYP': 'Middle East',
    # Christmas Island and Cocos (Keeling): NE tags them South-Eastern Asia
    # (geography); UN M49 files both under "Australia and New Zealand". The
    # project follows M49/sovereignty for territories (as with Andaman).
    # Keyed by ISO_A3 -- their ISO codes are valid ('CXR'/'CCK'), so an
    # ADM0_A3 key ('IOA') would never be consulted and was dead code.
    'CXR': 'Australia',
    'CCK': 'Australia',
}

# Subunit-level fixes where Natural Earth's SUBREGION follows geography but the
# territory's administration (and this project's taxonomy) follows the sovereign.
SUBUNIT_OVERRIDE = {
    # Cases where NE's SUBREGION follows geography but M49 (and the project,
    # which follows sovereignty for territories) files them with the sovereign.
    'Andaman Is.': 'South Asia',  # India; NE says South-Eastern Asia
    'Nicobar Is.': 'South Asia',  # ditto
    'Hawaii': 'North America',  # USA; NE says Polynesia; M49: N. America
    'Easter Island': 'South America',  # Chile; NE says Polynesia
    'Isla Sala y Gomez': 'South America',  # Chile
}

# NE tags a handful of remote islands SUBREGION='Seven seas (open ocean)', a
# label with no M49 counterpart. Mapped per M49 by their administering state so
# country_region() can never silently return None for real land.
SEVEN_SEAS_BY_ADM0 = {
    'SGS': 'South America',  # South Georgia & S. Sandwich (M49: South America)
    'IOT': 'Africa',  # British Indian Ocean Territory (M49: E. Africa)
    'SHN': 'Africa',  # Ascension / St Helena (M49: W. Africa)
    'ZAF': 'Africa',  # Prince Edward Islands (South Africa)
    'ATF': 'Africa',  # French Southern Territories (M49: E. Africa)
    'HMD': 'Australia',  # Heard & McDonald (M49: Australia and NZ)
    'ATA': 'Antarctica',  # South Orkney Is.
}

SUBREGION_MAP = {
    'Northern Africa': 'Africa',
    'Eastern Africa': 'Africa',
    'Middle Africa': 'Africa',
    'Southern Africa': 'Africa',
    'Western Africa': 'Africa',
    'Western Asia': 'Middle East',
    'Central Asia': 'Central Asia',
    'Southern Asia': 'South Asia',
    'Eastern Asia': 'East Asia',
    'South-Eastern Asia': 'Southeast Asia',
    'Eastern Europe': 'Europe',
    'Northern Europe': 'Europe',
    'Southern Europe': 'Europe',
    'Western Europe': 'Europe',
    'Northern America': 'North America',
    'Central America': 'Central America & Caribbean',
    'Caribbean': 'Central America & Caribbean',
    'South America': 'South America',
    'Australia and New Zealand': 'Australia',
    'Melanesia': 'New Zealand & Pacific',
    'Micronesia': 'New Zealand & Pacific',
    'Polynesia': 'New Zealand & Pacific',
    'Antarctica': 'Antarctica',
}

# Real continent behind each project region. A disagreement WITHIN a continent
# (Europe vs Russia & North Asia) is a naming convention the project is entitled
# to choose; a disagreement ACROSS continents (Kuwait filed as Africa) is a
# straightforward error. Only the latter is reported as MISASSIGNED.
REGION_CONTINENT = {
    'Africa': {'Africa'},
    'Europe': {'Europe'},
    'Middle East': {'Asia'},
    'Central Asia': {'Asia'},
    'South Asia': {'Asia'},
    'East Asia': {'Asia'},
    'Southeast Asia': {'Asia'},
    'Russia & North Asia': {'Europe', 'Asia'},  # spans the Urals
    'North America': {'North America'},
    'Central America & Caribbean': {'North America'},
    'South America': {'South America'},
    'Greenland': {'North America'},
    'Australia': {'Oceania'},
    'New Zealand & Pacific': {'Oceania'},
    'Antarctica': {'Antarctica'},
}

# Labels this audit will not second-guess: they describe open water or polar
# caps, which country polygons cannot speak to.
OCEAN_LABELS = {
    'Pacific Ocean', 'Atlantic Ocean', 'Indian Ocean', 'Arctic', 'Antarctica'
}


def country_region(row):
    """Project region for one Natural Earth country row."""
    su = row.get('SUBUNIT')
    if su in SUBUNIT_OVERRIDE:
        return SUBUNIT_OVERRIDE[su]
    iso = row.get('ISO_A3')
    if not iso or iso == '-99':  # NE uses '-99' as a null ISO code
        iso = row.get('ADM0_A3')
    if iso in COUNTRY_OVERRIDE:
        return COUNTRY_OVERRIDE[iso]
    if row.get('SUBREGION') == 'Seven seas (open ocean)':
        return SEVEN_SEAS_BY_ADM0.get(row.get('ADM0_A3'))
    sub = SUBREGION_MAP.get(row.get('SUBREGION'))
    if sub:
        return sub
    cont = row.get('CONTINENT')
    return {
        'Africa': 'Africa',
        'Europe': 'Europe',
        'Asia': 'Middle East',
        'North America': 'North America',
        'South America': 'South America',
        'Oceania': 'New Zealand & Pacific',
        'Antarctica': 'Antarctica'
    }.get(cont)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grid', default='original_global_grid_5deg.csv')
    p.add_argument(
        '--shapefile',
        default='data/geo/ne_50m_admin_0_map_subunits.shp',
        help='Use the map_subunits layer, not admin_0_countries: '
        'the latter welds overseas departments onto the sovereign '
        "(France's polygon swallows French Guiana and reports it as "
        'Europe) and keeps Russia as one Europe-tagged shape.')
    p.add_argument('--out', default='data/geo/region_audit.csv')
    p.add_argument('--min-share',
                   type=float,
                   default=0.60,
                   help='Below this dominant-region land share a cell is '
                   'reported as straddling rather than mis-assigned '
                   '(default 0.60).')
    p.add_argument('--ocean-land-frac',
                   type=float,
                   default=0.60,
                   help='An ocean/polar label is kept only while the cell is '
                   'less than this fraction land; a mostly-land cell is '
                   'audited like any other (default 0.60).')
    p.add_argument(
        '--catalog',
        default='data/catalog.duckdb',
        help='Optional: attach harvested rows/images per cell so the '
        'blast radius of a rename is visible.')
    args = p.parse_args()

    grid = pd.read_csv(args.grid)
    world = gpd.read_file(args.shapefile)
    world['proj_region'] = world.apply(country_region, axis=1)
    world = world[world['proj_region'].notna()]
    sindex = world.sindex

    rows = []
    for i, g in grid.iterrows():
        cell = box(g.sw_lon, g.sw_lat, g.ne_lon, g.ne_lat)
        hit = world.iloc[list(sindex.query(cell, predicate='intersects'))]
        by = defaultdict(float)
        bycont = defaultdict(float)
        for _, c in hit.iterrows():
            try:
                inter = c.geometry.intersection(cell)
            except Exception:
                continue
            if not inter.is_empty:
                a = area_km2(inter)
                by[c['proj_region']] += a
                bycont[c['CONTINENT']] += a
        land = sum(by.values())
        cell_km2 = area_km2(cell)
        name = (f"{g.region.replace('&', 'and').replace(' ', '_')}"
                f"_{g.sw_lon}_{g.sw_lat}_{g.ne_lon}_{g.ne_lat}")
        if land <= 0:
            rows.append(
                dict(cell=name,
                     assigned=g.region,
                     dominant='',
                     share=0.0,
                     land_frac=0.0,
                     land_km2=0.0,
                     verdict='ocean-no-land',
                     breakdown=''))
            continue
        dom, dom_area = max(by.items(), key=lambda kv: kv[1])
        share = dom_area / land
        land_frac = land / cell_km2
        assigned_share = by.get(g.region, 0.0) / land
        domcont = max(bycont.items(), key=lambda kv: kv[1])[0]
        ok_conts = REGION_CONTINENT.get(g.region, set())
        if g.region in OCEAN_LABELS and land_frac < args.ocean_land_frac:
            verdict = 'ocean-label-kept'
        elif dom == g.region:
            verdict = 'ok'
        elif domcont not in ok_conts and assigned_share < 1e-9:
            # The assigned region owns NONE of the land and the land is on the
            # wrong continent: an error even when no region clears min_share.
            verdict = 'MISASSIGNED'
        elif share < args.min_share:
            verdict = 'straddles'
        elif domcont not in ok_conts:
            verdict = 'MISASSIGNED'  # wrong continent outright
        else:
            verdict = 'taxonomy'  # right continent, different sub-label
        rows.append(
            dict(cell=name,
                 assigned=g.region,
                 dominant=dom,
                 continent=domcont,
                 share=round(share, 3),
                 land_frac=round(land_frac, 6),
                 land_km2=round(land, 1),
                 verdict=verdict,
                 breakdown='; '.join(
                     f'{k} {v / land:.0%}'
                     for k, v in sorted(by.items(), key=lambda kv: -kv[1]))))
    out = pd.DataFrame(rows)

    if args.catalog and os.path.exists(args.catalog):
        try:
            import duckdb
            con = duckdb.connect(args.catalog, read_only=True)
            f = con.execute('SELECT cell, sum(n_rows) r FROM files '
                            'GROUP BY 1').df()
            im = con.execute('SELECT cell, sum(n_images) i FROM images '
                             'GROUP BY 1').df()
            con.close()
            out = out.merge(f, on='cell', how='left').merge(im,
                                                            on='cell',
                                                            how='left')
            out['r'] = out['r'].fillna(0).astype('int64')
            out['i'] = out['i'].fillna(0).astype('int64')
            out = out.rename(columns={'r': 'rows', 'i': 'images'})
        except Exception as e:
            print(f'(catalog attach skipped: {e})', file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.',
                exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f'{len(out):,} cells audited -> {args.out}\n')
    print(out['verdict'].value_counts().to_string())
    bad = out[out.verdict == 'MISASSIGNED']
    if len(bad):
        cols = [
            c for c in ('cell', 'assigned', 'dominant', 'share', 'rows',
                        'images') if c in bad.columns
        ]
        srt = 'rows' if 'rows' in bad.columns else 'share'
        print(f'\n--- MISASSIGNED ({len(bad)}), worst first ---')
        print(
            bad.sort_values(
                srt, ascending=False)[cols].head(30).to_string(index=False))
        if 'rows' in bad.columns:
            print(f"\n  affected: {bad['rows'].sum():,} parquet rows · "
                  f"{bad['images'].sum():,} jpgs")
    tax = out[out.verdict == 'taxonomy']
    if len(tax):
        print(f'\n--- TAXONOMY ({len(tax)}): right continent, different '
              'sub-label. A convention choice, not an error ---')
        print(
            tax.groupby([
                'assigned', 'dominant'
            ]).size().sort_values(ascending=False).head(10).to_string())
    strad = out[out.verdict == 'straddles']
    if len(strad):
        print(
            f'\n--- STRADDLING ({len(strad)}): no single label is correct ---')
        cols = [
            c for c in ('cell', 'assigned', 'dominant', 'share', 'breakdown')
            if c in strad.columns
        ]
        print(strad.sort_values('share')[cols].head(15).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
