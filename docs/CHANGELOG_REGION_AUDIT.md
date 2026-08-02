# Region audit — change log

Running record of everything touched while investigating the grid-region
mis-assignment. Nothing here renames or deletes project data; anything that
would is listed under "Proposed, NOT executed".

## Trigger

User observed Kuwait filed under `Africa`. Confirmed: the cell
`(sw_lon 45, sw_lat 25, ne_lon 50, ne_lat 30)` in `original_global_grid_5deg.csv`
is labelled `Africa`, and that box covers Kuwait, Bahrain, Qatar and eastern
Saudi Arabia.

## Root cause

`original_global_grid_5deg.csv` assigns regions with coarse lat/lon boxes applied
in priority order. Two of those boxes are wrong:

- `Africa` spans lon -20..55, lat -40..40 — far past the Red Sea, so it absorbs
  the whole Arabian peninsula and the Levant.
- `Middle East` is only **12 cells**, lon 55..65 / lat 10..40 — roughly the UAE
  and Oman only.

Africa is evaluated first, so everything from the Sinai to Iran lands in Africa.

## Provenance note (important)

The first pass at scoping this used city coordinates and expected regions
recalled from memory, plus a hand-invented filter
(`region=='Africa' AND sw_lon>=35 AND sw_lat>=12`). That produced "19 cells,
20.6% of Africa's rows" — a heuristic, not ground truth, and it is superseded by
the polygon-based audit below. Recorded here so the earlier numbers are not
mistaken for measurements.

## Files ADDED (no existing file modified)

| path | what | reversible |
|---|---|---|
| `data/geo/ne_50m_admin_0_countries.*` | Natural Earth 1:50m country polygons, downloaded from naturalearth.s3.amazonaws.com (public domain). ~800 KB zip. | delete the dir |
| `data/geo/ne_countries.zip` | the downloaded archive | delete |
| `tools/catalog/audit_grid_regions.py` | read-only audit: assigns every grid cell a region by land-area overlap with real country polygons and reports disagreements with the CSV | delete |
| `data/geo/region_audit.csv` | audit output, one row per cell | regenerate |
| `docs/CHANGELOG_REGION_AUDIT.md` | this file | — |

## Files MODIFIED

_(none yet)_

## Proposed, NOT executed

Renaming regions would rename cell directories across all six data roots and
invalidate: the DuckDB catalog, `coverage_missing*` shards, the audit
checkpoints, and every `.backfill_progress.json` sidecar. Nothing on disk has
been touched. Any migration needs explicit approval and a dry run first.

## Known limitation — cells that straddle continents

At 5° resolution some cells legitimately contain two continents. Example:
`35..40E, 15..20N` holds Sudan and Eritrea (Africa) *and* Saudi Arabia and Yemen
(Asia), split by the Red Sea. No single label is correct for those. The audit
reports the land share per region so these are visible rather than silently
forced one way.

## Audit method (final)

Every 5° cell is intersected with Natural Earth **map_subunits** polygons; land
area inside the cell is attributed per subunit, mapped to the project's region
names, and compared with the CSV.

Two corrections were needed to the first version of the tool, both of which had
inflated the error count:

1. **`admin_0_countries` welds overseas departments onto the sovereign.**
   France's polygon there is one shape covering metropolitan France *and* French
   Guiana, tagged `CONTINENT=Europe` — so South American cells were reported as
   "should be Europe". Madeira likewise. `map_subunits` separates them
   (French Guiana → South America, Madeira → Africa). Same layer also splits
   Russia into a Europe-tagged western polygon and an Asia-tagged eastern one,
   so Moscow no longer reads as misfiled.
2. **`ISO_A3` is the literal string `-99`** for several countries (France among
   them), which defeated the `ISO_A3 or ADM0_A3` fallback in `country_region()`.

Verdicts are split so naming conventions are not confused with errors:

- `MISASSIGNED` — the cell's dominant land is on a **different continent** than
  the assigned region implies. Unambiguous.
- `taxonomy` — right continent, different sub-label (e.g. Europe vs
  Russia & North Asia). The project's choice to make; not reported as an error.
- `straddles` — dominant region holds < 60% of the cell's land, so no single
  label is correct at 5° resolution.
- `ocean-*` — no land, or an ocean/polar label the audit will not second-guess.

## Results

| verdict | cells |
|---|---|
| ok | 721 |
| ocean-no-land | 1,228 |
| ocean-label-kept | 421 |
| taxonomy | 159 |
| **MISASSIGNED** | **55** |
| straddles | 8 |

**55 cells are on the wrong continent — 81,406,099 parquet rows and 374,523 jpgs.**

| assigned | should be | cells | rows | images |
|---|---|---|---|---|
| Central Asia | Europe | 6 | 44,337,862 | 6,381 |
| Africa | Middle East | 17 | 15,231,339 | 91,522 |
| Central America & Caribbean | South America | 9 | 14,871,833 | 220,824 |
| Europe | Middle East | 5 | 3,638,875 | 13,033 |
| Greenland | Europe | 6 | 1,830,698 | 24,911 |
| Europe | Africa | 2 | 778,861 | 5,032 |
| Southeast Asia | New Zealand & Pacific | 6 | 374,266 | 10,600 |
| South Asia | Southeast Asia | 2 | 340,811 | 2,215 |
| Australia | Southeast Asia | 2 | 1,554 | 5 |

Notable individual cases:

- `Africa_45_25_50_30` — Kuwait/Bahrain/Qatar, the cell that started this.
- `Greenland_-25_60_-20_65` and 5 more — **Iceland filed as Greenland**.
- `Central_Asia_45_55_50_60` (15.9M rows) — the Volga region of European Russia,
  filed as Central Asia because the grid's Central Asia box runs to lon 45.
- `Central_America_and_Caribbean_-80_5_-75_10` — Colombia, filed as Central
  America.

## Proposed, NOT executed

No rename has been performed. Renaming would touch cell directories on all six
data roots plus the catalog, the `coverage_missing*` shards, audit checkpoints
and every `.backfill_progress.json`. Needs explicit approval and a dry run.

Note the fix is not purely cosmetic in one direction: correcting a label does
not move any image, but it does change which `--region` flag reaches a cell, so
a half-done rename is worse than none.

---

# EXECUTED — migration applied

## Two further corrections to the audit before executing

Both were found by inspecting the marginal cases in the dry run rather than
trusting the aggregate:

3. **Andaman and Nicobar Islands are Indian territory** but Natural Earth tags
   them `SUBREGION='South-Eastern Asia'`. That would have renamed two Indian
   cells to Southeast Asia. Added `SUBUNIT_OVERRIDE` keeping them South Asia.
4. **Three cells were essentially all ocean** — `Australia_115_-15_120_-10` had
   *10 km²* of land (a scrap of Indonesia) deciding its label, and two others
   were similar with zero harvested data. Renaming on that evidence is not
   accuracy, it is noise. `fix_grid_regions.py --min-land-km2` (default 100)
   now skips them.

Final change set: **51 cells** (down from the 56 the raw audit proposed).

## What ran

    python tools/catalog/fix_grid_regions.py --roots <all six grid_runs> \
        --stamp aug01 --execute

- **5,961 renames** — 161 cell directories across 6 roots, 5,800 files inside
  them. All `os.rename` within a filesystem: atomic, no copy, no window where
  data is in neither place. Zero collisions.
- **54 rows** of `original_global_grid_5deg.csv` relabelled. Backup at
  `data/geo/original_global_grid_5deg.csv.bak`.
- Journal: `runs/region_fix_aug01.json`. Reverse with
  `--undo runs/region_fix_aug01.json` (verified working on a sandbox first).
- Catalog rebuilt (`refresh` + `images`).

## Verification

Integrity — identical before and after, so nothing was lost or duplicated:

| | before | after |
|---|---|---|
| parquet files | 68,384 | 68,384 |
| rows | 3,048,780,701 | 3,048,780,701 |
| jpgs | 32,582,319 | 32,582,319 |

Re-running the audit against the corrected grid: **MISASSIGNED 0** (was 54),
`ok` 806 → 860. The Kuwait cell now resolves as
`Middle_East_45_25_50_30`, with its parquets renamed to match
(`all_data_Middle_East_45_25_50_30_000.parquet`), so the globs in
`batch_chunks_mp_api.py` and `coverage_audit.py` still find them.

Region totals after: Middle East 12 → **30 cells / 26.5M rows**; Africa down to
132 cells / 61.7M rows.

## Deliberately NOT changed

- **76 `taxonomy` cells** — right continent, different sub-label (European
  Russia as Europe vs Russia & North Asia). A convention the project is entitled
  to pick; renaming them would churn 76 cells for no gain in accuracy.
- **7 `straddles` cells** — two continents in one 5° box, e.g.
  `Africa_45_10_50_15` at 53% Middle East / 47% Africa. No label is right.
- **3 near-landless cells**, per the guard above.
- **Ocean and polar labels** — country polygons cannot speak to basin
  conventions.

## Still stale after this

- `data/missing_worklist/*.parquet` (11 files) and any `coverage_missing*`
  shards are keyed by the OLD parent regions. Regenerate before using them;
  they are derived, so nothing is lost by rebuilding.
- `.backfill_progress.json` sidecars are keyed by parent region name — a
  backfill resumed against a renamed region will restart that region from
  offset 0 rather than lose anything.

---

# Provenance

Asked whether the regions rest on a citable authority or on recalled knowledge.
Answer, precisely:

**External, citable:**
- Geometry — *Natural Earth v5.1.1, Admin 0 Map Subunits, 1:50m Cultural
  Vectors*, public domain. https://www.naturalearthdata.com/downloads/50m-cultural-vectors/
- Taxonomy — *UN M49 Standard Country or Area Codes for Statistical Use*, UN
  Statistics Division. https://unstats.un.org/unsd/methodology/m49/ (encoded in
  Natural Earth's `SUBREGION` attribute).

**Project-specific judgement:** this project's 19 regions are not an M49
taxonomy — M49 has no "Middle East" (it has *Western Asia*), no
"Russia & North Asia", no standalone "Greenland" — so a mapping requires
choices. Every choice is now written down in `data/geo/REGION_MAPPING.md` and
machine-readable in `data/geo/region_mapping.json`, including the ones that
contradict M49 (Russia, Iran, Afghanistan, Greenland, New Zealand, Andaman &
Nicobar) with the reason for each.

**Measured impact:** of the 51 renamed cells, **39 follow UN M49 exactly**; 12
depend on the departures — 7 European-Russia cells (M49: Europe) and 5 Iranian
cells (M49: South Asia).

No cell was renamed on the basis of recalled knowledge. Every assignment came
from intersecting the cell with the Natural Earth polygons; the only human input
is the documented mapping table.

---

# 2026-08-02 — cleanup + canonical doc

- **Deleted `grid_runs/*/covered_countries.txt` on every root** (user request):
  2,404 on crucial, 115 on weasel, 904 on capybara, 0 elsewhere — 3,423 files,
  ~80 KB. One-line sidecars ("No countries found (Ocean/Sea)" etc.) from an old
  step; verified zero remain on all six roots.
- Added `docs/REGIONS.md` — canonical description of the grid, the assignment
  logic, citations (Natural Earth v5.1.1 map_subunits; UN M49), the documented
  departures, thresholds, and how to reproduce or challenge any assignment.
- Independent verification workflow (UN-source cross-check, disk consistency,
  point-lattice re-derivation, hostile code review) launched; results pending.
