# Regions — how the grid is organized, and on whose authority

This is the canonical description of how every image in this project is
assigned to a named world region. It covers the grid itself, the assignment
logic, the external references behind it, the places where the project
deliberately departs from those references, and how to reproduce or challenge
any single assignment.

Related files:

| file | role |
|---|---|
| `original_global_grid_5deg.csv` | the grid: one row per cell, with its region label (the operative source of truth) |
| `data/geo/REGION_MAPPING.md` | the mapping table + departures, prose form |
| `data/geo/region_mapping.json` | same, machine-readable |
| `data/geo/region_audit.csv` | per-cell audit: dominant land region, share, verdict, full breakdown |
| `tools/catalog/audit_grid_regions.py` | recomputes the audit from the polygons |
| `tools/catalog/fix_grid_regions.py` | applies audit verdicts to disk + CSV (journalled, undoable) |
| `docs/CHANGELOG_REGION_AUDIT.md` | full history of the 2026-08-01/02 corrections |
| `data/geo/un_m49.csv` | UN M49 + ISO 3166 table, used to verify NE and measure conformance |

---

## 1. The grid

The world is divided into **2,592 cells of 5° × 5°** (72 × 36 minus none —
every cell exists, including open ocean). A cell is identified by its
south-west and north-east corners in integer degrees:

```
(sw_lon, sw_lat, ne_lon, ne_lat)      e.g. (45, 25, 50, 30)
```

Each cell carries exactly one **region** label out of 19:

> Africa · Antarctica · Arctic · Atlantic Ocean · Australia ·
> Central America & Caribbean · Central Asia · East Asia · Europe · Greenland ·
> Indian Ocean · Middle East · New Zealand & Pacific · North America ·
> Pacific Ocean · Russia & North Asia · South America · South Asia ·
> Southeast Asia

On disk, a cell's directory name is the sanitized region (`&`→`and`, space→`_`)
plus the bbox:

```
Middle_East_45_25_50_30/
├── all_data_Middle_East_45_25_50_30_000.parquet      (every Mapillary image)
├── ground_animals_Middle_East_45_25_50_30_000.parquet (ground-animal subset)
├── ground_animal_images/<image_id>.jpg                (downloaded jpgs)
└── validated_images_Middle_East_45_25_50_30.txt       (download ledger)
```

The cell name is embedded in the *filenames*, not just the directory — any
relabelling must rename both, or the parquets become invisible to the
pipeline's globs.

The region label is purely organizational: it decides which folder a cell's
data lives in and what `--region` selects. It never affects which images are
harvested — harvesting is driven by the bbox.

## 2. How regions are assigned

### 2.1 Original scheme (superseded)

Regions were originally assigned by coarse lat/lon boxes applied in priority
order. Two boxes were drawn wrong — `Africa` extended to lon 55/lat 40 and was
evaluated before `Middle East` (which held only 12 cells at lon 55–65) — so
everything from the Sinai to Iran was filed as Africa, Iceland as Greenland,
Colombia as Central America, and the Volga region as Central Asia. This was
found on 2026-08-01 (a user noticed Kuwait under Africa) and corrected; the
full forensic record is in `docs/CHANGELOG_REGION_AUDIT.md`.

### 2.2 Current scheme (operative)

Each cell's label is decided by **which region owns the most land inside the
cell**, computed by intersecting the cell rectangle with country polygons:

```
cell ∩ country polygons  →  land area per country
country → region         (mapping in §3)
label = region with the largest land share
```

with these guards:

| rule | value | rationale |
|---|---|---|
| dominant share < 0.60 | verdict `straddles`, cell left as-is | at 5° a cell can genuinely contain two continents (e.g. 45–50E / 10–15N is 52% Middle East, 48% Africa across the Red Sea); no single label is correct, so the incumbent label stands |
| land < 100 km² in the cell | not relabelled by the audit path | a label should not be decided by a sliver of reef in an otherwise open-ocean cell. Exception: `--reconcile` aligns disk to the CSV unconditionally — coherence outranks the guard — so 2 sub-guard cells relabelled in the CSV before the guard existed (9.6 and 17.2 km², zero data, geographically correct labels) were aligned rather than reverted |
| ocean/polar label, land fraction < 0.60 | label kept | basin labels (Pacific/Atlantic/Indian Ocean, Arctic, Antarctica) are a convention for mostly-water cells — but a label is not proof of ocean: the original grid filed the 100%-land Volga cell as *Indian Ocean*, so a mostly-land cell is audited like any other regardless of its label (Hawaii at ~3% land keeps Pacific Ocean; the Volga cell did not keep Indian Ocean) |
| wrong sub-label, same continent | verdict `taxonomy`, left as-is | e.g. European Russia as *Europe* vs *Russia & North Asia* is a naming convention, not an error; only cross-continent disagreements were treated as mistakes |

All areas are computed in an equal-area projection (EPSG:6933) — raw degree
areas overweight low latitudes and sat within rounding distance of flipping
near-threshold verdicts.

Applied 2026-08-01: 51 cells relabelled (5,961 renames across 6 drives,
journal `runs/region_fix_aug01.json`). Applied 2026-08-02 after independent
adversarial verification: 20 further cells that had been hidden behind ocean
labels (incl. the Volga cell as *Indian Ocean*, 18.3M rows; the Caucasus; a
Siberia/Tibet block as *Pacific Ocean*; six mostly-land (62–78%) Arctic cells) plus 3 cells
reconciled from an earlier CSV/disk divergence (journals
`runs/region_fix_aug02*.json`). Post-fix audit: **0 misassigned**,
76 `taxonomy`, 7 `straddles`; disk↔CSV reconcile reports zero drift; catalog
integrity identical before/after every migration (68,384 files ·
3,048,780,701 rows · 32,582,319 jpgs).

## 3. The mapping, and its authorities

### 3.1 Citable sources

**Geometry — where each country's land is:**

> Natural Earth (v5.1.1). *Admin 0 – Map Subunits*, 1:50m Cultural Vectors.
> Public domain.
> https://www.naturalearthdata.com/downloads/50m-cultural-vectors/
> (local copy: `data/geo/ne_50m_admin_0_map_subunits.shp`)

The *map_subunits* layer is used rather than *admin_0_countries* because the
country layer welds overseas territories onto the sovereign (France's single
polygon includes French Guiana, tagged `CONTINENT=Europe`), which misplaces
whole cells. Subunits separate territories and split Russia at the Urals.

**Taxonomy — which sub-region a country belongs to:**

> United Nations Statistics Division. *Standard Country or Area Codes for
> Statistical Use* (Series M, No. 49) — the "M49" geoscheme.
> https://unstats.un.org/unsd/methodology/m49/

M49 reaches the pipeline through Natural Earth's `SUBREGION` attribute, which
**approximates** it: an adversarial cross-check against the UN's own table
found 16 of 297 comparable subunits where NE disagrees with M49 (Asian Russia
tagged `Central Asia`; Hawaii `Polynesia` vs M49 Northern America; Easter I.,
Christmas I., Cocos, Corsica, Chatham, Andaman/Nicobar; plus seven "Seven seas
(open ocean)" subunits — those have no direct M49 row of their own and are
mapped via their administering state's M49 entry; an eighth, South Orkney, has
no M49 counterpart on either side and maps to Antarctica). Every divergence is
neutralized by an explicit override or by the standard mapping (Chatham needs
none: Melanesia → New Zealand & Pacific coincides with the NZ rule), or is
verified to affect no cell's dominant land; where overrides exist they side
with **M49/sovereignty**, the project's convention for territories.

### 3.2 M49 sub-region → project region (follows the standard)

| UN M49 sub-region | project region |
|---|---|
| Northern / Eastern / Middle / Southern / Western Africa | Africa |
| Western Asia | Middle East |
| Central Asia | Central Asia |
| Southern Asia | South Asia |
| Eastern Asia | East Asia |
| South-Eastern Asia | Southeast Asia |
| Eastern / Northern / Southern / Western Europe | Europe |
| Northern America | North America |
| Central America · Caribbean | Central America & Caribbean |
| South America | South America |
| Australia and New Zealand | Australia |
| Melanesia · Micronesia · Polynesia | New Zealand & Pacific |
| Antarctica | Antarctica |

Turkey and Cyprus need no special handling: M49 files both under *Western
Asia* → Middle East.

### 3.3 Documented departures from M49 (project judgement)

The project's 19 regions are **not** an M49 taxonomy — M49 has no "Middle
East", no "Russia & North Asia", no standalone "Greenland" — so six explicit
overrides exist. These are choices, recorded so they can be challenged:

| entity | M49 says | project uses | why | renamed cells that rest on it |
|---|---|---|---|---|
| Russia (all subunits) | Eastern Europe | Russia & North Asia | dedicated Russia region; also neutralizes NE's "Central Asia" tag on Siberia | 14 |
| Iran | Southern Asia | Middle East | the original grid's Middle East cells sat at lon 55–65 — i.e. Iran; matches common usage | 5 |
| Afghanistan | Southern Asia | Central Asia | judgement call, contestable | 0 |
| Greenland | Northern America | Greenland | dedicated region | 4 |
| New Zealand | Australia and New Zealand | New Zealand & Pacific | project splits what M49 groups | 0 |
Andaman & Nicobar were previously mislisted here as a departure: UN M49 files
India (including these territories) under *Southern Asia* → South Asia, so the
project **conforms**; "South-Eastern Asia" is Natural Earth's tag, corrected
below.

Corrections **toward** M49 where Natural Earth deviates from it (not
departures): Hawaii → North America, Easter I. / Isla Sala y Gomez → South
America, Christmas I. / Cocos → Australia, and the eight "Seven seas" subunits
mapped by administering state (South Georgia & S. Sandwich → South America;
BIOT, St Helena/Ascension, Prince Edward, French Southern Territories →
Africa; Heard & McDonald → Australia; S. Orkney → Antarctica). None currently
decides a cell's dominant land; they exist so the chain can never silently
drop land or follow NE against the UN table.

**Of the 74 relabelled cells, 51 match UN M49 exactly; 23 rest on the
departures above** — 14 Russian, 5 Iranian, 4 Greenlandic. Measured against the
UN table itself (`data/geo/un_m49.csv`), not Natural Earth's copy. No cell
depends on an undocumented departure. Strict M49 compliance is recoverable:
delete the overrides in `audit_grid_regions.py` and re-run.

## 4. Reproducing and challenging

Recompute the whole audit (read-only, ~2 min):

```bash
python tools/catalog/audit_grid_regions.py \
    --grid original_global_grid_5deg.csv \
    --shapefile data/geo/ne_50m_admin_0_map_subunits.shp \
    --out data/geo/region_audit.csv
```

Check one cell without running anything: `data/geo/region_audit.csv` carries,
per cell, the assigned region, the dominant land region, its share, the land
fraction, the verdict, and the full per-region land breakdown (e.g.
`Middle East 53%; Africa 47%`).

Apply verdicts to disk (dry-run by default, journalled, `--undo`-able):

```bash
python tools/catalog/fix_grid_regions.py --roots <every grid_runs root> [--execute]
```

## 5. Known limits

- **5° is coarser than geography.** 7 cells straddle continents with no
  correct single label; they keep their incumbent labels and are listed in the
  audit as `straddles`.
- **Ocean labels are convention, not derivation.** Basin boundaries
  (Pacific/Atlantic/Indian/Arctic) were never recomputed.
- **Natural Earth's `SUBREGION` is a secondary source for M49.** Where NE and
  the UN table disagree, the UN table governs; discrepancies found are either
  neutralized by an override or must be fixed in `SUBREGION_MAP`.
- **The 76 `taxonomy` cells** (right continent, different sub-label) reflect
  the incumbent convention, not the polygon derivation. Renaming them is a
  policy decision, not a correctness fix.
