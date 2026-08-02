# Region assignment — sources, mapping, and where judgement was applied

This documents how every grid cell's region was decided, so the assignment is
citable rather than assumed. It separates what comes from an external authority
from what is a project-specific choice.

## External sources (citable)

**Geometry — which country's land lies in which cell**

> Natural Earth (v5.1.1). *Admin 0 – Map Subunits*, 1:50m Cultural Vectors.
> Public domain. https://www.naturalearthdata.com/downloads/50m-cultural-vectors/

Used for: the polygons themselves, the `CONTINENT` attribute, and the
`SUBREGION` attribute. Downloaded file:
`ne_50m_admin_0_map_subunits.shp` (VERSION.txt reports `5.1.1`).

The **map_subunits** layer is used rather than `admin_0_countries` because the
latter welds overseas departments onto the sovereign state: France there is a
single polygon covering metropolitan France *and* French Guiana, tagged
`CONTINENT=Europe`, which would place South American cells in Europe. Subunits
separate them, and additionally split Russia into a Europe-tagged western
polygon and an Asia-tagged eastern one.

**Region taxonomy — which sub-region a country belongs to**

> United Nations Statistics Division. *Standard Country or Area Codes for
> Statistical Use (Series M, No. 49)* — the "M49" geoscheme.
> https://unstats.un.org/unsd/methodology/m49/

Natural Earth's `SUBREGION` field **approximates** M49 — 16 of 297 joinable
subunits deviate from the UN table (verified against
https://unstats.un.org/unsd/methodology/m49/ on 2026-08-01/02). Every deviation
is either neutralized by an explicit override siding with M49 (Hawaii, Easter
I., Christmas I., Cocos, the "Seven seas" subunits, Andaman/Nicobar) or
verified to decide no cell's dominant land. Russia's override is NOT
M49-siding — it implements the project's own Russia & North Asia region and is
the largest documented departure (14 cells). The column below headed "M49
sub-region" uses NE's SUBREGION vocabulary, which mixes M49 sub-region and
intermediate-region names (e.g. `Caribbean`, `Eastern Africa`) and NE spelling
(`South-Eastern Asia` vs UN `South-eastern Asia`).

## Where the two disagree with this project

**This project's 19 regions are not an M49 taxonomy.** M49 has no "Middle East"
(it has *Western Asia*), no "Russia & North Asia", and no standalone
"Greenland". So a mapping from M49 to these region names necessarily involves
choices. Those choices are listed below rather than buried in code.

### Rules that follow M49 exactly

| M49 subregion | project region |
|---|---|
| Northern / Eastern / Middle / Southern / Western Africa | Africa |
| Western Asia | Middle East |
| Central Asia | Central Asia |
| Southern Asia | South Asia |
| Eastern Asia | East Asia |
| South-Eastern Asia | Southeast Asia |
| Eastern / Northern / Southern / Western Europe | Europe |
| Northern America | North America |
| Central America, Caribbean | Central America & Caribbean |
| South America | South America |
| Australia and New Zealand | Australia |
| Melanesia, Micronesia, Polynesia | New Zealand & Pacific |
| Antarctica | Antarctica |

Turkey and Cyprus need no override: M49 already files them under *Western Asia*,
which maps to Middle East.

### Rules that DEPART from M49 — project-specific judgement

| entity | M49 says | used here | why |
|---|---|---|---|
| Russia (all subunits) | Eastern Europe | **Russia & North Asia** | The project defines a dedicated Russia region; M49 files the whole Russian Federation under Eastern Europe. (NE's *Central Asia* tag on Siberia is Natural Earth's artefact, not M49's, and is neutralized separately.) |
| Iran | Southern Asia | **Middle East** | The project's original Middle East cells sat at lon 55–65, i.e. Iran — the intent was evidently Iran = Middle East. Conventional usage also places Iran in the Middle East. |
| Afghanistan | Southern Asia | **Central Asia** | Judgement call; contestable. |
| Greenland | Northern America | **Greenland** | The project defines Greenland as its own region. |
| New Zealand | Australia and New Zealand | **New Zealand & Pacific** | The project separates Australia from New Zealand & Pacific; M49 groups them. |

Andaman & Nicobar were previously mislisted here: M49 files India (including
these territories) under *Southern Asia* → South Asia, so the project
**conforms** to M49; the "South-Eastern Asia" label was Natural Earth's tag,
corrected by a SUBUNIT override.

### Thresholds — arbitrary, chosen here

| parameter | value | meaning |
|---|---|---|
| `--min-share` | 0.60 | below this the dominant region holds too little of the cell's land to call it; reported as `straddles` instead of an error |
| `--min-land-km2` | 100 | cells with less land than this are not renamed by the audit path — the "dominant region" would rest on a sliver of reef. `--reconcile` intentionally ignores it: disk↔CSV coherence outranks the guard (affected 2 zero-data cells) |

## Impact of the judgement calls

Measured against the **UN M49 table itself** (not Natural Earth's copy of it),
joined by ISO alpha-3 — `data/geo/un_m49.csv`, 249 entries; 301 of 308 NE subunits join by ISO code, 297 carry a usable M49
sub-region (the comparison denominator), and the project mapping resolves all
308:

Of the **74 cells relabelled** across the three migrations, **51 match UN M49
exactly**. The other **23** rest on the documented departures above:

| cells | used here | UN M49 says |
|---|---|---|
| 14 | Russia & North Asia | Europe |
| 5 | Middle East | South Asia |
| 4 | Greenland | North America |

No relabelled cell depends on an *undocumented* departure. The 14 are Russian
cells (M49 files the Russian Federation under Eastern Europe); the 5 are
Iranian; the 4 are Greenlandic. All are reversible — the journals
`runs/region_fix_aug01.json`, `runs/region_fix_aug02.json` and
`runs/region_fix_aug02_reconcile.json` record every rename and every CSV row
change, and removing the overrides from `audit_grid_regions.py` reproduces the
strict-M49 assignment.

**No cell's renaming was decided by recalled knowledge.** Every assignment comes
from intersecting the cell with the Natural Earth polygons above; the only human
input is the mapping table in this document.

## Reproducing

```bash
python tools/catalog/audit_grid_regions.py \
    --grid original_global_grid_5deg.csv \
    --shapefile data/geo/ne_50m_admin_0_map_subunits.shp \
    --out data/geo/region_audit.csv
```

`region_audit.csv` carries, per cell: assigned region, dominant region, that
region's share of the cell's land, the land fraction of the cell, the verdict,
and a full per-region land breakdown — so any individual decision can be checked
without re-running anything.
