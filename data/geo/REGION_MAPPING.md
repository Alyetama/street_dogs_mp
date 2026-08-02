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

Natural Earth's `SUBREGION` field encodes M49, so M49 is applied via that field.

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
| Russia (all subunits) | Eastern Europe / Central Asia | **Russia & North Asia** | The project defines a dedicated Russia region; leaving it to M49 would put European Russia in Europe and, worse, file Siberia as *Central Asia* (an artefact of how NE tags the Asian subunit). |
| Iran | Southern Asia | **Middle East** | The project's original Middle East cells sat at lon 55–65, i.e. Iran — the intent was evidently Iran = Middle East. Conventional usage also places Iran in the Middle East. |
| Afghanistan | Southern Asia | **Central Asia** | Judgement call; contestable. |
| Greenland | Northern America | **Greenland** | The project defines Greenland as its own region. |
| New Zealand | Australia and New Zealand | **New Zealand & Pacific** | The project separates Australia from New Zealand & Pacific; M49 groups them. |
| Andaman & Nicobar Is. | South-Eastern Asia | **South Asia** | Indian territory. M49 classifies them geographically; the project's regions otherwise follow sovereignty for territories. |

### Thresholds — arbitrary, chosen here

| parameter | value | meaning |
|---|---|---|
| `--min-share` | 0.60 | below this the dominant region holds too little of the cell's land to call it; reported as `straddles` instead of an error |
| `--min-land-km2` | 100 | cells with less land than this are not renamed — the "dominant region" would rest on a sliver of reef |

## Impact of the judgement calls

Of the **51 cells renamed**, **39 follow UN M49 exactly**. The remaining **12**
depend on the departures above:

| cells | used here | strict M49 would say |
|---|---|---|
| 7 | Russia & North Asia | Europe |
| 5 | Middle East | South Asia |

The 7 are cells of European Russia (the Volga region, ~63.2M rows); the 5 are
Iranian cells. Both are reversible — `runs/region_fix_aug01.json` journals every
rename, and re-running the audit with the overrides removed produces the strict
M49 assignment.

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
