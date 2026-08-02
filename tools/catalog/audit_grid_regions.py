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

*Ocean cells.* Cells with no land keep whatever the CSV says. The ocean/polar
labels encode a basin convention this audit has no opinion about, and rewriting
them from country polygons would be nonsense.

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
from shapely.geometry import box

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
    # Turkey and Cyprus straddle the Europe/Asia line; the grid already labels
    # Istanbul's cell Europe, so keep them with Europe rather than churn.
    'TUR': 'Middle East',
    'CYP': 'Middle East',
}

# Subunit-level fixes where Natural Earth's SUBREGION follows geography but the
# territory's administration (and this project's taxonomy) follows the sovereign.
SUBUNIT_OVERRIDE = {
    'Andaman Is.':
    'South Asia',  # Indian territory, NE says South-Eastern Asia
    'Nicobar Is.': 'South Asia',  # ditto
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
                by[c['proj_region']] += inter.area
                bycont[c['CONTINENT']] += inter.area
        land = sum(by.values())
        name = (f"{g.region.replace('&', 'and').replace(' ', '_')}"
                f"_{g.sw_lon}_{g.sw_lat}_{g.ne_lon}_{g.ne_lat}")
        if land <= 0:
            rows.append(
                dict(cell=name,
                     assigned=g.region,
                     dominant='',
                     share=0.0,
                     land_frac=0.0,
                     verdict='ocean-no-land',
                     breakdown=''))
            continue
        dom, dom_area = max(by.items(), key=lambda kv: kv[1])
        share = dom_area / land
        domcont = max(bycont.items(), key=lambda kv: kv[1])[0]
        ok_conts = REGION_CONTINENT.get(g.region, set())
        if g.region in OCEAN_LABELS:
            verdict = 'ocean-label-kept'
        elif dom == g.region:
            verdict = 'ok'
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
                 land_frac=round(land / cell.area, 3),
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
