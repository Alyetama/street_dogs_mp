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

## Results *(superseded — pre-equal-area tool output; current figures in the 2026-08-02 sections)*

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
4. **Three cells were skipped by the land guard** — `Australia_115_-15_120_-10`
   had *10 km²* of land (a scrap of Indonesia) deciding its label. CORRECTION
   (2026-08-02 verification): one of the three, `Southeast_Asia_135_5_140_10`
   (Palau, ~120 km²), was above the documented 100 km² threshold and held
   25,530 rows / 172 jpgs — it was skipped only because the guard read a
   3-decimal-rounded land_frac that rounded 120 km² to zero. The guard now
   reads an unrounded equal-area land_km2 column, and the cell has since been
   renamed (2026-08-02 section below).

Final change set: **51 cells** (of the 54 the audit proposed; 3 guard-skipped).

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
- **3 near-landless cells**, per the guard above. *(Correction 2026-08-02: 2 of these were later dir-renamed by `--reconcile` to match their aug01 CSV relabels — coherence outranks the guard; zero data, labels geographically correct.)*
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

**Measured impact** *(superseded — aug01 scope only; final: 74 cells, 51
M49-exact, 23 departures = 14 Russia + 5 Iran + 4 Greenland; see the
2026-08-02 self-review section)*: of the 51 renamed cells, 39 follow UN M49
exactly; 12 depend on the departures — 7 European-Russia cells (M49: Europe) and 5 Iranian
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

---

# 2026-08-02 — adversarial verification, and the fixes it forced

Four independent agents attacked the executed migration: (1) cross-check of
Natural Earth's SUBREGION against the UN's own M49 table, (2) disk↔CSV
consistency sweep over all six roots (8,641 dirs / 204,658 files), (3) an
independent re-derivation of all 2,592 cells by 0.2° point-lattice sampling +
an EPSG:6933 equal-area recomputation, (4) hostile review of both tools.
Verdict: **every one of the 51 executed renames was independently confirmed
correct** — but the surrounding machinery had real defects.

## Defects found and fixed

1. **CSV/disk divergence (critical).** The aug01 run relabelled 54 CSV rows but
   renamed only 51 cells' dirs: the land guard applied to the dir plan, not the
   CSV rewrite. 3 cells diverged; one (Palau) held real data. Fixed structurally
   — `eligible()` now computes the acted-on set once and both consumers use it —
   and healed on disk with the new `--reconcile` mode (9 dirs, journal
   `region_fix_aug02_reconcile.json`).
2. **Ocean-label blind spot (critical).** The audit's OCEAN_LABELS short-circuit
   ran before any land test, so an ocean-labelled cell was never audited even at
   100% land — the same error class the migration existed to fix, structurally
   invisible to it. The gate is now `ocean label AND land_frac < 0.60`.
   Re-audit surfaced **20 hidden misassignments** (23.2M rows, 48,476 jpgs):
   the Volga cell as *Indian Ocean* (18.3M rows), the Caucasus as Indian Ocean
   (35,524 jpgs), a 10-cell Siberia/Tibet block as *Pacific Ocean*, six
   all-land Arctic cells (N. Greenland / Ellesmere). All renamed
   (75 dirs / 2,006 files, journal `region_fix_aug02.json`). Hawaii (~3% land)
   and Svalbard (<60%) correctly keep their basin labels.
3. **land_frac rounding (major).** The guard reconstructed km² from a 3-decimal
   land_frac; 100 km² ≡ 0.000325 always rounded to zero. Audit now emits an
   unrounded equal-area `land_km2` column; the guard consumes it directly.
4. **Areas now equal-area (EPSG:6933)** instead of raw degrees.
5. **Undo hardened:** CSV row changes are journalled (write-ahead, before any
   rename executes) and restored by `--undo`; undo reports reversed vs skipped
   honestly. Proven live: a botched invocation (zsh passed six roots as one
   string → CSV-only change) was fully reverted by `--undo` before the correct
   re-run.
6. **NE↔M49 divergences neutralized:** 16/297 subunits where NE's SUBREGION
   contradicts the UN table now carry explicit M49-siding overrides (Hawaii →
   North America, Easter I. → South America, Christmas/Cocos → Australia, the
   eight "Seven seas" subunits by administering state). None decides a cell's
   dominant land today; the chain can no longer silently drop land
   (country_region() returning None is now impossible for real land).
7. **Verdict ordering:** a cell whose assigned region owns none of its land is
   MISASSIGNED even below the 0.60 dominance threshold (this is what caught the
   Caucasus cell at 57.7% dominance).

## Evidence preservation

- `data/geo/region_audit_prefix.csv` — the pre-fix audit as the aug01-era tool
  produced it (54 MISASSIGNED), frozen. The CURRENT tool on the `.bak` grid
  yields 74 MISASSIGNED (it also sees the ocean-label blind spot) — equal to
  the final relabel set exactly, a stronger confirmation than byte-identity.
- `data/geo/region_audit_aug01_postfix.csv` — the state between migrations.
- `data/geo/region_audit.csv` — current (0 MISASSIGNED).

## Final state

- Audit: **0 MISASSIGNED**, 76 taxonomy, 7 straddles, 178 ocean-label-kept.
- `--reconcile` dry-run: **zero** disk↔CSV drift across all six roots.
- Catalog integrity across all three migrations: identical
  (68,384 files · 3,048,780,701 rows · 32,582,319 jpgs).
- Totals: 74 cells relabelled (54 aug01 CSV / 51 dirs + 20 aug02 + 3 reconciled).
- Still stale by design: `coverage_missing*` shards and `data/missing_worklist/`
  are keyed by old cell names in file names AND `safe_region_id` row values —
  regenerate before next use.

---

# 2026-08-02 (later) — self-review corrections

A second hostile review (3 agents) fact-checked every claim in the four docs
against the artifacts and attacked the conformance method. The data layer held:
all 74 relabels reproduce from scratch and none needs reverting. Defects were
in documentation and two code paths; fixed:

- **`run_missing_downloads.sh` rewritten to loop over `data/missing_worklist/`**
  instead of a hardcoded region list, which had silently stranded the 734
  Middle East images (no `Middle_East` invocation existed) and kept three dead
  `Indian_Ocean` calls.
- **Stale departure counts corrected everywhere** (7/5/0 and 39-of-51/12 →
  final 14 Russia / 5 Iran / 4 Greenland = 23 of 74, 51 M49-exact), including
  machine-readable `region_mapping.json`, whose `cells_affected` fields a
  consumer would have summed to 12.
- **Christmas/Cocos override was dead code** — keyed `'IOA'` (ADM0) but both
  islands carry valid ISO codes (`CXR`/`CCK`), so the code actually sided with
  Natural Earth against M49; contained only by the ocean gate. Re-keyed; audit
  verdicts unchanged (both cells ocean-gated at ~0.04% land).
- **`runs/region_fix_aug01.json` carried no CSV entries** (CSV journalling
  postdates it) — the advertised `--undo` would have reversed 5,961 renames
  while leaving 54 CSV rows on new labels, recreating the drift the saga
  fixed. The 54 entries were reconstructed from `region_audit_prefix.csv` and
  appended. Undo order across journals: reconcile → aug02 → aug01.
- **Andaman & Nicobar removed from the departures list** — M49 files India
  under Southern Asia → South Asia, so the project conforms; the deviation was
  Natural Earth's tag.
- **Russia's M49 position corrected** to "Eastern Europe" (the *Central Asia*
  tag on Siberia is NE's artefact, not M49's) and Russia removed from the
  "M49-siding overrides" list — it is the largest documented departure.
- **"Byte-reproducibly" corrected** for `region_audit_prefix.csv`: it is the
  aug01-era tool's frozen output; the current tool on the `.bak` yields 74
  (= the final relabel set).
- Wording: "six all-land Arctic cells" → mostly-land (62–78%); the Red Sea
  straddle example updated to equal-area figures (52/48); "56" → 54;
  `journal_flush` now fsyncs file and directory (power-loss durability);
  `data/geo/region_audit.csv` committed alongside the two snapshots.
- Known-latent, accepted: REGION_CONTINENT keys on NE's CONTINENT field with no
  'Seven seas' entry (no live effect — every affected cell audits
  ok/ocean-kept); `undo()` resolves the journal's relative grid path against
  the CWD — run it from the repo root; the 0.60 ocean gate leaves 5 contiguous
  Greenland-coast cells (0.42–0.56 land, dominant Greenland) labelled Arctic
  beside 4 renamed ones — a threshold consequence, disclosed; parked worklist
  copies at `data/missing_worklist_stale_aug02/` and
  `data/missing_worklist_after/`.
