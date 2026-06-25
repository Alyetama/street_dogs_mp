# Street Dogs Mapillary Pipeline

A pipeline for harvesting Mapillary ground-animal photos worldwide. It scans geographic grid cells, extracts image metadata, keeps the images Mapillary tags with an `animal--ground-animal` detection, and downloads the matching photos — then audits coverage and backfills anything the first pass missed.

The work runs in **three independent stages**, each a standalone script:

| Stage | Script | What it does |
| --- | --- | --- |
| **1. Extract** | `batch_chunks_mp_api.py` | Scan grid cells → metadata → detections → Parquet + downloaded jpgs. |
| **2. Audit** | `coverage_audit.py` | Verify every image (up to a cutoff date) was captured; list what's missing. |
| **3. Backfill** | `backfill_missing.py` | Fetch metadata + download the in-scope missing images. |

## Table of contents

- [Setup](#setup)
- [Repository layout](#repository-layout)
- [Stage 1 — Extract](#stage-1--extract)
- [Stage 2 — Completeness audit](#stage-2--completeness-audit)
- [Stage 3 — Backfill](#stage-3--backfill)
- [Browsing results](#browsing-results)
- [Data catalog](#data-catalog)
- [Helper scripts](#helper-scripts)
- [Troubleshooting](#troubleshooting)

## Setup

**Requirements:** Python 3.14+, a Mapillary access token, the dependencies in `requirements.txt`, and enough disk for checkpoints, Parquet partitions, and image downloads.

```bash
python -m pip install -r requirements.txt
```

**Token(s).** Create a `.env` file in the project root:

```env
MLY_KEY=your_default_token
# optional numbered tokens, selectable with --token <n> (and used round-robin by the audit/backfill stages):
MLY_KEY_1=your_first_token
MLY_KEY_2=your_second_token
```

> Run every script **from the repository root** — their default paths are relative to the working directory.

## Repository layout

The repo root holds only the three pipeline scripts (`batch_chunks_mp_api.py`, `coverage_audit.py`, `backfill_missing.py`) and the `browse.py` web app. Everything else lives under `tools/`. Configuration (`.env`, `proxies.txt`), the master grid `original_global_grid_5deg.csv`, and the live data directories (`grid_runs/`, `coverage_missing/`, `coverage_missing_inscope/`, `progress_files/`) stay at the root.

<details>
<summary><strong>Full directory map</strong></summary>

| Location | Contents |
| --- | --- |
| *(root)* | The three pipeline scripts (`batch_chunks_mp_api.py`, `coverage_audit.py`, `backfill_missing.py`) and the `browse.py` web app. |
| `tools/grid/` | Grid prep & visualization: `split_regions.py`, `generate_countries.py`, `visualize_region_tiles.py`. |
| `tools/progress/` | Run monitoring & lookup: `progress_tracker.py`, `find_location_folder.py`, `scan_regions.py`. |
| `tools/maintenance/` | Run integrity & ledgers: `audit_markers.py`, `audit_silent_skips.py`, `generate_rerun_commands.py`, `check_zst_health.py`, `generate_ledger.py`. |
| `tools/coverage/` | Completeness-audit helpers: `convert_missing_csv_to_parquet.py`, `split_missing_from_csv.py`, `validate_missing_sample.py`, `check_grid_data.py`. |
| `tools/repair/` | Image-gap / manifest repair: `diagnose_images.py`, `fetch_missing_images.py`, `rebuild_manifest_from_images.py`, `drop_dead_from_manifest.py`, `cleanup_offending_regions.py`, `deduplicate_parquets.py`. |
| `tools/catalog/` | DuckDB inventory of every Parquet (and downloaded image) across all drives: `catalog.py`. |
| `runners/` | Convenience shell scripts: `monitor_and_verify.sh`, `pull_all.sh`. |
| `data/` | Working artifacts; `data/manifests/` (missing/orphan/dead manifest CSVs) and `data/grids/` (per-region input grid CSVs). |
| `logs/` | Run logs from helper scripts. |
| `archived/` | Superseded scripts kept for reference. |

</details>

---

## Stage 1 — Extract

`batch_chunks_mp_api.py` processes a CSV of regions and, for each row, runs a resumable six-phase workflow:

1. Split the region into mercantile tiles; keep land tiles only.
2. Query Mapillary image sequences for each land tile.
3. Expand sequences into image IDs.
4. Fetch image metadata.
5. Fetch detections and filter for `animal--ground-animal`.
6. Write compressed checkpoints + Parquet, then (optionally) download images and stamp capture timestamps as EXIF / file mtime.

It's built for large runs: multiprocessing across regions, threaded API/download work within each region, compressed checkpoints, chunked Parquet, and resume markers.

<details>
<summary><strong>Input CSV format</strong></summary>

One row per region, with these columns:

| Column | Description |
| --- | --- |
| `sw_lon`, `sw_lat` | Southwest corner (lon, lat). |
| `ne_lon`, `ne_lat` | Northeast corner (lon, lat). |
| `region` | Human-readable region name, used in output paths. |

```csv
sw_lon,sw_lat,ne_lon,ne_lat,region
-74.10,40.60,-73.70,40.90,sample_region
```

</details>

### Quick start

```bash
# All regions in a CSV (default local multiprocessing):
python batch_chunks_mp_api.py regions.csv

# One CSV row by zero-based index:
python batch_chunks_mp_api.py regions.csv --row-index 0

# Specific sub-grid cells within a region (handy for backfilling):
python batch_chunks_mp_api.py regions.csv --sub-indices 4,12,15

# Metadata + detections only, no image downloads:
python batch_chunks_mp_api.py regions.csv --no-download-images

# Download later from existing ground_animals_*.parquet:
python batch_chunks_mp_api.py regions.csv --download-only

# Send images to another disk; optionally buffer on a fast SSD first:
python batch_chunks_mp_api.py regions.csv --image-dir /mnt/hdd/images --temp-dir /tmp/ssd_spool

# Pick a numbered token:
python batch_chunks_mp_api.py regions.csv --token 1
```

<details>
<summary><strong>All CLI options</strong></summary>

| Option | Default | Purpose |
| --- | ---: | --- |
| `--outer-max-workers` | `5` | Regions processed in parallel. |
| `--search-max-workers` | `150 / outer` | Threads per region for bbox/sequence calls. |
| `--entity-max-workers` | `520 / outer` | Threads per region for metadata/detection calls. |
| `--download-max-workers` | `10` | Threads for image downloads. |
| `--sub-grid-step` | `1.0` | Degree step for splitting regions into sub-grids. |
| `--sub-indices` | all | Comma-separated sub-grid indices to process. |
| `--parent-dir` | `grid_runs` | Root for per-region outputs and checkpoints. |
| `--image-dir` | = parent-dir | Separate directory for image downloads. |
| `--temp-dir` | unset | Fast SSD buffer for downloads before moving to `--image-dir`. |
| `--parquet-chunk-size` | `100000` | Rows per `all_data_*.parquet` partition. |
| `--proxy-file` | unset | Proxy list (`ip:port`, `ip:port:user:pass`, or full URL). |
| `--exclude-ledger` | unset | Text file of image IDs to skip. |
| `--token` | unset | Use `MLY_KEY_<n>` instead of `MLY_KEY`. |
| `--no-download-images` | off | Fetch metadata/detections only. |
| `--download-only` | off | Skip API work; only download from existing Parquets. |

Run `python batch_chunks_mp_api.py --help` for the full reference.

</details>

<details>
<summary><strong>Output layout</strong></summary>

Outputs are written under `grid_runs/<safe_region_id>/` (the safe ID is built from the region name and bounding box):

| File | Description |
| --- | --- |
| `*_checkpoint_<sub_id>.{json,jsonl}.zst` | Compressed topology / metadata / detection checkpoints. |
| `all_data_<region>_<part>.parquet` | Metadata for all images. |
| `ground_animals_<region>_<part>.parquet` | Metadata for images with ground-animal detections. |
| `ground_animal_images/` | Downloaded `.jpg` files. |
| `validated_images_<region>.txt` | Ledger of images that passed validity checks. |
| `failed_downloads_<region>.txt` | Image IDs that failed download. |
| `covered_countries.txt` | Countries intersecting the region (from `tools/grid/generate_countries.py`). |
| `.completed_<sub_id>` / `.empty_<sub_id>` | Resume markers (done / no results). |

With `--image-dir`, jpgs go to `<image-dir>/<region>/ground_animal_images/` while checkpoints and Parquet stay under `--parent-dir`.

</details>

<details>
<summary><strong>Resume, interruption &amp; SLURM</strong></summary>

Re-running the same command resumes: it reuses checkpoints, skips completed sub-grids, avoids reprocessing image IDs already in Parquet, and skips images recorded in the validation ledger. `Ctrl+C` force-exits quickly and may leave the in-progress file corrupt; the next run cleans interrupted JSONL checkpoints automatically.

`submit_mp_jobs.slurm` is a starting point for cluster runs — update the CSV path, array range, conda env, and command first. The SLURM path processes one CSV row per task, using `SLURM_ARRAY_TASK_ID` unless `--row-index` is given.

```bash
python batch_chunks_mp_api.py regions.csv \
  --slurm \
  --search-max-workers "$((SLURM_CPUS_PER_TASK * 4))" \
  --entity-max-workers "$((SLURM_CPUS_PER_TASK * 16))" \
  --no-download-images
```

</details>

---

## Stage 2 — Completeness audit

`coverage_audit.py` verifies the extraction captured **every** Mapillary image (up to a cutoff date) per region, and lists what's missing. It enumerates via Mapillary's **vector tiles** rather than the `/images?bbox` search API — the bbox API silently caps at ~2000 results per query and 500s on dense city cells, making it useless for a completeness guarantee. A single z14 vector tile returns every image point (id, sequence_id, captured_at) with no such limit.

Subcommands run in sequence, or all at once via `audit`:

| Subcommand | Purpose |
| --- | --- |
| `audit GRID.csv --dirs …` | The normal entry point: `enumerate → retry → diff → datefilter`. |
| `enumerate GRID.csv --dirs …` | Fetch every z14 land tile; write per-region checkpoint + meta sidecar. Resumable. |
| `check` | Read meta sidecars only (instant); report per-region counts and failed tiles. |
| `retry` | Re-fetch only each region's `failed_tiles` and merge them in. Repeatable. |
| `diff GRID.csv --dirs …` | `missing = coverage_ids − all_data_ids` → per-region Parquet shards. |
| `datefilter` | Keep rows captured on/before `--cutoff` → `coverage_missing_inscope/<Parent>.parquet`. |

```bash
# Full audit of one region, reading data spread across drives:
python coverage_audit.py audit original_global_grid_5deg.csv \
    --dirs grid_runs /mnt/hdd/grid_runs --region Europe

# Individual stages:
python coverage_audit.py enumerate original_global_grid_5deg.csv --region Europe -w 64
python coverage_audit.py diff original_global_grid_5deg.csv --dirs grid_runs --region Europe
python coverage_audit.py datefilter --cutoff 2026-05-31
```

<details>
<summary><strong>All CLI options</strong></summary>

| Option | Default | Purpose |
| --- | ---: | --- |
| `--region` | all | One parent region; accepts the original or sanitized name (`"Middle East"` or `Middle_East`). |
| `--data-dir` | `.` | Where coverage checkpoints/meta and the budget sidecar live (the data drive). |
| `--dirs` | — | Base dirs holding `all_data_*.parquet` (for `diff`). |
| `-w` / `--workers` | `64` | Concurrent tile fetches. |
| `--proxies` | unset | Rotating proxy list for tile/API requests (per-IP throttle aware). |
| `--daily-tile-limit` | `50000` | Per-token daily tile cap, tracked in `.tile_request_budget.json`. |
| `--wait` | off | When the daily budget is exhausted, wait for the reset and resume (timezone-agnostic). |
| `--wait-interval` | `1800` | Seconds between budget-reset probes. |

</details>

**Daily limit.** `tiles.mapillary.com` allows ~50k requests/day per token (16 tokens ≈ 800k tiles/day). A continental enumeration spans multiple days — just re-run the same command (or pass `--wait`); completed regions are skipped and a cut-off region resumes from its pending tiles. A live token-budget bar shows spend while it runs.

---

## Stage 3 — Backfill

`backfill_missing.py` fetches the in-scope missing set from the audit. For each image it gets metadata + detections in one Graph API call, writes append-only Parquet chunks matching the main schema, and downloads the ground-animal jpgs. It writes **both** an `all_data_*` parquet (every image, for later stats) and a `ground_animals_*` parquet (animals only).

```bash
python backfill_missing.py \
    --inscope coverage_missing_inscope --region Europe \
    --out-dir /mnt/weasel/grid_runs --image-dir /mnt/jpgs \
    --processes 3 --entity-workers 520 --download-workers 10
```

<details>
<summary><strong>All CLI options</strong></summary>

| Option | Default | Purpose |
| --- | ---: | --- |
| `--inscope` | `coverage_missing_inscope` | Datefilter output dir (or a single parquet). |
| `--region` | all | Parent region; accepts the sanitized name. |
| `--out-dir` | `grid_runs` | Where backfill parquets are written. |
| `--image-dir` | = `--out-dir` | Separate drive for downloaded jpgs. |
| `--no-download` | off | Write parquets only; download later. |
| `--download-only` | off | Skip metadata; only download jpgs from existing `ground_animals_*` parquets. Add `--watch` to keep draining as new parquets appear. |
| `--processes` | `1` | Fan out across N processes (disjoint token/row slices) to beat the single-process JSON-parse ceiling. |
| `--entity-workers` | `520` | Threads for metadata/detection calls (entity API is 60k/min per token). |
| `--download-workers` | `10` | Threads for image downloads. |
| `--batch` | `50000` | Rows fetched per flush. |
| `--proxies` | unset | Rotating proxies for metadata fetches. |

</details>

Resumable: the in-scope parquet is processed sequentially with a per-parent row offset in a `.backfill_progress*.json` sidecar; per-cell part numbers continue past existing `*_backfill_*.parquet`.

> A finished region's offset is at the end, so re-running won't re-download skipped images. To download images for an already-backfilled region, use `--download-only` (it reads the existing `ground_animals_*` parquets under `--out-dir`).

### Supporting tools

- **`tools/coverage/validate_missing_sample.py`** — before a big backfill, sample the in-scope set and probe `/detections` to estimate how many images are still live vs gone and what fraction are ground animals, then extrapolate the download volume.

  ```bash
  python tools/coverage/validate_missing_sample.py --inscope coverage_missing_inscope --region Europe -n 2000
  ```

- **`tools/coverage/convert_missing_csv_to_parquet.py`** / **`split_missing_from_csv.py`** — one-off bridges from the legacy CSV `diff` output to the current Parquet layout (`convert` → per-parent `from_csv.parquet`; `split` → per-cell shards a later `audit` recognizes as done).

  ```bash
  python tools/coverage/convert_missing_csv_to_parquet.py --csv data/manifests/coverage_missing.csv --out-dir coverage_missing
  python tools/coverage/split_missing_from_csv.py coverage_missing/Europe/from_csv.parquet
  ```

---

## Browsing results

`browse.py` — a single-file Flask app for interactively browsing pipeline output across one or more `--dirs`. Requires `flask` + `polars` (both in `requirements.txt`).

```bash
python browse.py --dirs grid_runs /mnt/hdd/grid_runs --port 8080 --host 0.0.0.0
```

Features: a region sidebar; location search (geocodes a city/country via Nominatim and filters by bounding-box overlap); tabs for *All Data*, *Animal Detections*, and *Downloaded Images*; paginated listings; an image lightbox sortable by name or capture date; map/heatmap visualization of parquet coordinates on Leaflet; per-file download links; and session-based login (`ADMIN_USER` / `ADMIN_PASS` set in the script).

---

## Data catalog

`tools/catalog/catalog.py` builds a [DuckDB](https://duckdb.org)-backed inventory of every Parquet file (and, optionally, every downloaded image) across all drives, so you can answer "what do we have, where, how much" instantly — even when a drive is unmounted — without re-globbing tens of thousands of files. It reads only Parquet **footers** (row counts; never a full scan) and parses each cell's region + bounding box from its directory name, so it never touches image data and writes only its own catalog (`data/catalog.duckdb` + a `catalog.parquet` snapshot).

```bash
python tools/catalog/catalog.py refresh     # build / update (incremental; skips unchanged files)
python tools/catalog/catalog.py images       # inventory downloaded jpgs (add --with-size for bytes)
python tools/catalog/catalog.py summary      # totals by drive / kind / region
python tools/catalog/catalog.py sql "SELECT region, sum(n_rows) FILTER (WHERE kind='ground_animals') AS dogs FROM files GROUP BY 1 ORDER BY dogs DESC"
# or open it directly:  duckdb data/catalog.duckdb
```

`refresh` is incremental (a file is re-read only when its size/mtime changes) and offline-drive aware (unmounted drives keep their rows, marked offline; files deleted from a mounted drive are pruned). The catalog files live under `data/` and are gitignored.

Which roots to scan is **not hardcoded**: pass `--dirs <path> …`, or list your drives one-per-line in a gitignored `data/catalog_dirs.txt` (override with `--dirs-file`). The script's only built-in default is the generic `grid_runs`.

---

## Helper scripts

All run from the repository root. Use `--help` on any script for its full options.

<details>
<summary><strong>Grid preparation</strong> — split the grid, list countries, visualize tiles</summary>

- **`tools/grid/split_regions.py`** — split `global_grid_5deg.csv` into per-region CSVs under `regions/pending/`.
- **`tools/grid/generate_countries.py`** — write a `covered_countries.txt` into each region folder (Natural Earth 110 m).
- **`tools/grid/visualize_region_tiles.py`** — render a static map of a region's tiles (green=land, red=water) into its folder. Needs a `Name_SWLon_SWLat_NELon_NELat` folder name.

```bash
python tools/grid/generate_countries.py --dirs grid_runs /mnt/hdd/grid_runs
python tools/grid/visualize_region_tiles.py "Sample_Region_-74.1_40.6_-73.7_40.9" --zoom 14
```

</details>

<details>
<summary><strong>Progress &amp; navigation</strong> — monitor runs, locate regions</summary>

- **`tools/progress/progress_tracker.py`** — Rich progress table by parent region (completion %, data points, ground-animal counts, download rate); saves a timestamped CSV to `progress_files/`. Key flags: `--dirs` (required), `-w/--workers`.
- **`tools/progress/find_location_folder.py`** — geocode a place and find which region folders overlap it.
- **`tools/progress/scan_regions.py`** — scan dirs for a region prefix and recommend the exact `--parent-dir` / `--image-dir` flags; flags data/images split across unexpected drives.

```bash
python tools/progress/progress_tracker.py regions.csv --dirs grid_runs /mnt/hdd/grid_runs
python tools/progress/find_location_folder.py "Japan" --dirs grid_runs /mnt/hdd/grid_runs
python tools/progress/scan_regions.py South_America --dirs grid_runs /mnt/hdd/grid_runs
```

</details>

<details>
<summary><strong>Run integrity &amp; maintenance</strong> — markers, silent skips, checkpoints, ledgers</summary>

- **`tools/maintenance/audit_markers.py`** — delete orphaned `.completed_*` markers whose checkpoints are missing (which would otherwise silently skip the sub-grid).
- **`tools/maintenance/audit_silent_skips.py`** — detect sub-grids marked `.completed_` but whose checkpoints hold fewer records than expected; deletes the marker to force a re-run. `--dry-run` to report only.
- **`tools/maintenance/generate_rerun_commands.py`** — emit ready-to-run `batch_chunks` commands targeting only the incomplete cells of each region (via `--row-index` / `--sub-indices`).
- **`tools/maintenance/check_zst_health.py`** — test all `.zst` with `zstd -t`; list/delete corrupt files. `--clear-completed` drops the matching `.completed_` marker so the sub-grid re-runs.
- **`tools/maintenance/generate_ledger.py`** — build/append an exclude ledger of downloaded image IDs (pass to extract via `--exclude-ledger`).
- **`tools/coverage/check_grid_data.py`** — report which CSV rows already have Parquet data on disk.

```bash
python tools/maintenance/audit_silent_skips.py --dry-run --substring North_America
python tools/maintenance/generate_rerun_commands.py regions.csv --substring "South America" --output-script rerun_sa.sh
python tools/maintenance/check_zst_health.py --delete-all --clear-completed --ignore-recent 1.5
python tools/maintenance/generate_ledger.py --image-dir /mnt/hdd/grid_runs --output global_exclude_ledger.txt
```

</details>

<details>
<summary><strong>Image-gap repair</strong> — fix on-disk jpgs vs manifest drift</summary>

These fix `% Images Downloaded` anomalies when on-disk jpgs and `ground_animals_*.parquet` manifests drift apart (expired URLs, or lost parquet rows whose images survived).

- **`tools/repair/diagnose_images.py`** — compare parquet `image_id`s vs jpgs on disk; report orphaned images and missing downloads. Run this first.
- **`tools/repair/fetch_missing_images.py`** — download images that are in a manifest but missing on disk, fetching a fresh signed URL from the Graph API on failure; records permanently-gone IDs. Subcommands: `scan`, `download`, `all`.
- **`tools/repair/rebuild_manifest_from_images.py`** — treat images as source of truth (the `% > 100` case): re-query detections for orphan jpgs and rebuild the missing `ground_animals_*_recovered_*.parquet` rows.
- **`tools/repair/drop_dead_from_manifest.py`** — remove permanently-dead IDs (`data/dead_images.txt`) from manifests so they stop inflating the gap; archives removed rows first; idempotent; dry-run by default (`--execute`).
- **`tools/repair/cleanup_offending_regions.py`** — delete parquet/checkpoint/markers (but keep images + ledger) for a hard-coded list of regions, forcing clean re-extraction; dry-run by default.
- **`tools/repair/deduplicate_parquets.py`** — multiprocessed pass removing duplicate `image_id` rows; rewrites only files that had duplicates.
- **`runners/monitor_and_verify.sh`** — wait for a manifest rebuild to finish, then regenerate the progress report.

```bash
python tools/repair/diagnose_images.py original_global_grid_5deg.csv --dirs grid_runs --region "South Asia"
python tools/repair/fetch_missing_images.py all original_global_grid_5deg.csv --dirs grid_runs --proxy-file proxies.txt -w 24
python tools/repair/drop_dead_from_manifest.py --dirs grid_runs /mnt/hdd/grid_runs --execute
python tools/repair/deduplicate_parquets.py --parent-dir grid_runs --substring North_America
```

</details>

---

## Troubleshooting

<details>
<summary><strong>Common problems &amp; fixes</strong></summary>

| Symptom | Fix |
| --- | --- |
| **Missing token** | Ensure `.env` has `MLY_KEY` (or the `MLY_KEY_<n>` selected by `--token`). |
| **No images downloaded** | Don't pass `--no-download-images`; confirm `ground_animals_*.parquet` has non-null `thumb_original_url`. |
| **Rate limits / network errors** | Lower `--search-max-workers` / `--entity-max-workers` / `--download-max-workers`, or supply `--proxy-file`. |
| **Disk filling up** | Use `--image-dir` for images, or `--no-download-images` first then `--download-only` later. |
| **Slow HDD writes during downloads** | Point `--temp-dir` at a fast SSD; the pipeline buffers there and moves files in the background. |
| **Reprocess one sub-grid** | `--sub-indices <n>` targets only that cell. |
| **Interrupted run** | Re-run the same command — checkpoints, Parquet, and markers resume the work. |

</details>
