# Street Dogs Mapillary Pipeline

A multiprocessing Mapillary ingestion pipeline for scanning geographic grid cells, extracting Mapillary image metadata, identifying images with Mapillary's `animal--ground-animal` detections, and optionally downloading the matching images.

The supported entry point for this repository is **`batch_chunks_mp_api_v3.py`**. Earlier `batch_chunks_mp_api*.py` scripts are historical versions and should be ignored for normal use.

## What the main script does

`batch_chunks_mp_api_v3.py` processes a CSV of geographic regions and, for each row, performs a resumable six-phase workflow:

1. Splits the region into Mapillary/mercantile tiles and keeps land tiles only.
2. Queries Mapillary image sequences for each land tile.
3. Expands sequences into image IDs.
4. Fetches image metadata.
5. Fetches detection records and filters for `animal--ground-animal`.
6. Writes compressed checkpoints and Parquet outputs, then optionally downloads matching images and applies capture timestamps as EXIF/file modification times.

The pipeline is designed for large runs: it uses local multiprocessing across regions, threaded API/download work inside each region, compressed checkpoints, chunked Parquet outputs, and resume markers for interrupted jobs.

## Requirements

- Python 3.14+.
- A Mapillary access token.
- Dependencies from `requirements.txt`.
- Enough disk space for compressed checkpoint files, Parquet partitions, and optional image downloads.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Mapillary token setup

Create a `.env` file in the project root:

```env
MLY_KEY=your_default_mapillary_access_token
```

You can also define numbered tokens and select them with `--token`:

```env
MLY_KEY_1=your_first_token
MLY_KEY_2=your_second_token
```

Example using `MLY_KEY_1`:

```bash
python batch_chunks_mp_api_v3.py regions.csv --token 1
```

## Input CSV format

The required positional argument is a CSV file containing one row per region. The main script expects these columns:

| Column | Description |
| --- | --- |
| `sw_lon` | Southwest longitude. |
| `sw_lat` | Southwest latitude. |
| `ne_lon` | Northeast longitude. |
| `ne_lat` | Northeast latitude. |
| `region` | Human-readable region name used in output paths. |

Example:

```csv
sw_lon,sw_lat,ne_lon,ne_lat,region
-74.10,40.60,-73.70,40.90,sample_region
```

## Quick start

Run all regions in a CSV using the default local multiprocessing settings:

```bash
python batch_chunks_mp_api_v3.py regions.csv
```

Run one specific CSV row by zero-based index:

```bash
python batch_chunks_mp_api_v3.py regions.csv --row-index 0
```

Fetch metadata and detections without downloading images:

```bash
python batch_chunks_mp_api_v3.py regions.csv --no-download-images
```

Download images later from existing `ground_animals_*.parquet` files:

```bash
python batch_chunks_mp_api_v3.py regions.csv --download-only
```

Send image downloads to a separate disk or mount point:

```bash
python batch_chunks_mp_api_v3.py regions.csv --image-dir /path/to/image_storage
```

Use a fast SSD as a write buffer before moving images to a slow HDD:

```bash
python batch_chunks_mp_api_v3.py regions.csv --image-dir /mnt/hdd/images --temp-dir /tmp/ssd_spool
```

Process only specific sub-grid indices within a region (useful for backfilling):

```bash
python batch_chunks_mp_api_v3.py regions.csv --sub-indices 4,12,15
```

## Common options

| Option | Default | Purpose |
| --- | ---: | --- |
| `--zoom-level` | `14` | Mercantile tile zoom used while scanning bounding boxes. |
| `--outer-max-workers` | `5` | Number of regions processed in parallel in local mode. |
| `--search-max-workers` | `150 / outer` | Threads per region for bbox/sequence search API calls. Dynamically scaled from `--outer-max-workers` by default. |
| `--entity-max-workers` | `520 / outer` | Threads per region for metadata and detection API calls. Dynamically scaled from `--outer-max-workers` by default. |
| `--download-max-workers` | `10` | Threads used specifically for image downloads. |
| `--sub-grid-step` | `1.0` | Degree step used to split large input regions into smaller sub-grids. |
| `--sub-indices` | unset | Comma-separated sub-grid indices to process (e.g., `4` or `4,12,15`). Processes all sub-grids when unset. |
| `--parent-dir` | `grid_runs` | Root directory for per-region outputs and checkpoints. |
| `--image-dir` | unset | Separate directory for image downloads (e.g., a different disk). |
| `--temp-dir` | unset | Fast SSD temp directory to buffer downloads before moving to `--image-dir`. |
| `--api-chunk-size` | `5000` | Batch size for threaded API work. |
| `--parquet-chunk-size` | `100000` | Rows per `all_data_*.parquet` output partition. |
| `--proxy-file` | unset | Optional proxy list file. Supports `ip:port`, `ip:port:user:password`, and full proxy URLs. |
| `--exclude-ledger` | unset | Text file of image IDs to skip. |
| `--token` | unset | Selects `MLY_KEY_<n>` instead of `MLY_KEY`. |
| `--slurm` | `False` | Runs one region based on `SLURM_ARRAY_TASK_ID` or `--row-index`. |
| `--download-only` | `False` | Skip API fetching and ONLY download images from existing ground_animals Parquets. |

For the complete CLI reference, run:

```bash
python batch_chunks_mp_api_v3.py --help
```

## Output layout

By default, outputs are written under `grid_runs/<safe_region_id>/`, where the safe region ID is built from the CSV row's region name and bounding box.

Typical files include:

| Output | Description |
| --- | --- |
| `topology_checkpoint_<sub_id>.json.zst` | Compressed image-to-sequence topology checkpoint for a sub-grid. |
| `metadata_checkpoint_<sub_id>.jsonl.zst` | Compressed JSONL metadata checkpoint. |
| `animal_detections_checkpoint_<sub_id>.jsonl.zst` | Compressed JSONL animal-detection checkpoint. |
| `ground_animals_<sub_id>.json.zst` | Compressed detection features for sub-grids where animals were found. |
| `all_data_<safe_region_id>_<part>.parquet` | Chunked Parquet metadata for all images. |
| `ground_animals_<safe_region_id>_<part>.parquet` | Chunked Parquet metadata for images with ground-animal detections. |
| `ground_animal_images/` | Downloaded `.jpg` files when image downloading is enabled. |
| `validated_images_<safe_region_id>.txt` | Ledger of images that passed local validity checks. |
| `failed_downloads_<safe_region_id>.txt` | Image IDs that failed download attempts. |
| `covered_countries.txt` | Countries intersecting the region bounding box (written by `generate_countries.py`). |
| `.completed_<sub_id>` | Resume marker indicating a sub-grid completed. |
| `.empty_<sub_id>` | Resume marker indicating a sub-grid had no usable topology results. |

If `--image-dir` is provided, image files are written to:

```text
<image-dir>/<safe_region_id>/ground_animal_images/
```

while checkpoints and Parquet outputs remain under `--parent-dir`.

## Resume and interruption behavior

The pipeline is resumable. Re-running the same command will reuse existing compressed checkpoints, skip completed sub-grids, avoid reprocessing image IDs already present in Parquet output, and avoid re-downloading images recorded in the validation ledger.

In local multiprocessing mode, `Ctrl+C` force-exits quickly and may leave the file currently being written in a corrupt state. On the next run, the script attempts to clean interrupted compressed JSONL checkpoints. In SLURM mode, the script installs signal handlers for array jobs.

## SLURM usage

`submit_mp_jobs.slurm` is included as a starting point for cluster runs. Update the CSV path, array range, conda environment, and command before submitting.

The SLURM execution path processes a single CSV row per task. It uses `SLURM_ARRAY_TASK_ID` unless `--row-index` is passed.

Example command inside a job script:

```bash
python batch_chunks_mp_api_v3.py regions.csv \
  --slurm \
  --search-max-workers "$((SLURM_CPUS_PER_TASK * 4))" \
  --entity-max-workers "$((SLURM_CPUS_PER_TASK * 16))" \
  --parent-dir grid_runs \
  --no-download-images
```

## Web browser

### `browse.py`

A single-file Flask web application for interactively browsing pipeline output data stored under one or more `--dirs` directories. Requires `flask` and `polars` (already in `requirements.txt`).

```bash
python browse.py --dirs grid_runs /mnt/hdd/grid_runs
python browse.py --dirs grid_runs --port 8080 --host 0.0.0.0
```

| Option | Default | Purpose |
| --- | ---: | --- |
| `--dirs` | `grid_runs` | One or more base directories to scan for region folders. |
| `--host` | `127.0.0.1` | Address to bind the server to. |
| `--port` | `5000` | Port to listen on. |

**Features:**

- **Region sidebar** — lists all parent regions derived from folder names. Click a region to filter the file listing.
- **Location search** — geocodes any city or country via Nominatim and filters the sidebar to folders whose bounding boxes intersect the result.
- **Data type tabs** — switch between *All Data* (`all_data_*.parquet`), *Animal Detections* (`ground_animals_*.parquet`), and *Downloaded Images* (`ground_animal_images/*.jpg`).
- **Paginated listing** — 25 files per page for parquets, 50 thumbnail images per page for the images tab.
- **Image lightbox** — click any thumbnail to view the full image with previous/next navigation. Images can be sorted by name, newest first, or oldest first (capture timestamp is taken from the file's modification time, which the pipeline sets to the Mapillary capture date).
- **Map visualization** — click the 🗺 button on any file row or region to plot coordinates from a parquet file on an interactive Leaflet map, rendered as a heatmap for large datasets. A *Map all* button in the type bar maps every parquet in the current region or search result for the active tab type.
- **Download buttons** — every file row and image thumbnail has a direct download link.
- **Login page** — session-based authentication; credentials are set in the script (`ADMIN_USER` / `ADMIN_PASS`).

---

## Completeness audit & backfill

A second pipeline verifies that the main extraction actually captured **every**
Mapillary image (up to a cutoff date) for each grid region, lists what is
missing, and backfills it. It enumerates via Mapillary's vector tiles rather
than the `/images?bbox` search API, because the bbox API silently caps at ~2000
results per query and returns HTTP 500 on dense city cells — making it unusable
for a completeness guarantee. A single z14 vector tile returns every image point
(id, sequence_id, captured_at) with neither limit.

### `coverage_audit.py`

The audit driver. Subcommands run in sequence (or all at once via `audit`):

| Subcommand | Purpose |
| --- | --- |
| `audit GRID.csv --dirs ...` | One command: `enumerate → retry → diff → datefilter`. The normal entry point. |
| `enumerate GRID.csv --dirs ...` | Fetch every z14 land tile of each region as a vector tile; write a per-region checkpoint (`coverage_checkpoint_<safe>.json.zst`) and tiny `coverage_meta_<safe>.json` sidecar. Resumable. |
| `check` | Read only the meta sidecars (instant); report per-region image counts and failed-tile counts. |
| `retry` | Re-fetch only the `failed_tiles` of each region and merge them in. Safe to run repeatedly. |
| `diff GRID.csv --dirs ...` | `missing = coverage_ids − all_data_ids`. Writes per-parent-region Parquet shards of missing image_ids (carrying `captured_at`). |
| `datefilter` | Keep missing rows captured on/before `--cutoff`. Local (no API) when `captured_at` is present; falls back to the Graph API only for rows lacking it. Writes `coverage_missing_inscope/<Parent>.parquet`. |

```bash
# Full audit of one parent region, reading data spread across several drives:
python coverage_audit.py audit original_global_grid_5deg.csv \
    --dirs grid_runs /mnt/hdd/grid_runs --region Europe

# Individual stages:
python coverage_audit.py enumerate original_global_grid_5deg.csv --region Europe -w 64
python coverage_audit.py diff original_global_grid_5deg.csv --dirs grid_runs --region Europe
python coverage_audit.py datefilter --cutoff 2026-05-31
```

| Option | Default | Purpose |
| --- | ---: | --- |
| `--region` | all | Restrict to one parent region. Accepts the original or sanitized name (`"Middle East"` or `Middle_East`). |
| `--data-dir` | `.` | Where coverage checkpoints/meta and budget sidecar live (the parquet/data drive, not image drives). |
| `--dirs` | — | Base dirs holding `all_data_*.parquet` (for `diff`). |
| `-w` / `--workers` | `64` | Concurrent tile fetches. |
| `--proxies` | unset | Rotating proxy list for tile/API requests (per-IP throttle aware). |
| `--daily-tile-limit` | `50000` | Per-token daily tile cap. Requests round-robin across every `MLY_KEY*` token, each with its own budget (tracked in `.tile_request_budget.json`). |
| `--wait` | `False` | When the daily budget is exhausted, probe and wait for it to reset, then resume (timezone-agnostic). |
| `--wait-interval` | `1800` | Seconds between budget-reset probes. |

**Daily limit:** `tiles.mapillary.com` allows ~50,000 requests/day per token, so
16 tokens ≈ 800k tiles/day. A full continental enumeration spans multiple days —
just re-run the same command (or pass `--wait`); completed regions are skipped
and a region cut off mid-way resumes from its pending tiles.

### `backfill_missing.py`

Step 4 — fetches the in-scope missing set. For every image it gets metadata +
detections in one Graph API call (`fields=…,detections.value`), writes
append-only Parquet chunks matching the main pipeline schema, and downloads the
ground-animal jpgs. It writes **both** an `all_data_*` parquet (every image, for
later stats) and a `ground_animals_*` parquet (animals only).

```bash
python backfill_missing.py \
    --inscope coverage_missing_inscope --region Europe \
    --out-dir /mnt/weasel/grid_runs --image-dir /mnt/jpgs \
    --processes 3 --entity-workers 520 --download-workers 10
```

| Option | Default | Purpose |
| --- | ---: | --- |
| `--inscope` | `coverage_missing_inscope` | Datefilter output dir (or a single parquet). |
| `--region` | all | Parent region; accepts the sanitized name. |
| `--out-dir` | `grid_runs` | Where backfill parquets are written. |
| `--image-dir` | = `--out-dir` | Separate drive for downloaded jpgs. |
| `--no-download` | `False` | Write parquets only; download later. |
| `--download-only` | `False` | Skip metadata; only download jpgs from existing `ground_animals_*` parquets. Pair with `--watch` to keep draining as new parquets appear. |
| `--processes` | `1` | Fan out across N OS processes (`--shard I/N`), each with a disjoint slice of tokens/rows — needed to beat the single-process GIL/JSON-parse ceiling. |
| `--entity-workers` | `520` | Threads for metadata/detection calls. Entity API is 60,000/min per token; requests round-robin and are rate-capped per token. |
| `--download-workers` | `10` | Threads for image downloads. |
| `--batch` | `50000` | Rows fetched per flush. |
| `--proxies` | unset | Rotating proxies for metadata fetches. |

Resumable: the in-scope parquet is processed sequentially with a per-parent row
offset recorded in a `.backfill_progress*.json` sidecar; per-cell part numbers
continue past existing `*_backfill_*.parquet`.

### `validate_missing_sample.py`

Before committing to a backfill, samples the in-scope set (every k-th row, so it
spans all cells) and probes `/detections` to estimate how many images are still
live vs gone (400/404) and what fraction are ground animals — then extrapolates
to the full set so you can size the download volume.

```bash
python tools/coverage/validate_missing_sample.py --inscope coverage_missing_inscope \
    --region Europe -n 2000 [--proxies proxies.txt]
```

### `convert_missing_csv_to_parquet.py` / `split_missing_from_csv.py`

One-off bridges from the older CSV-based `diff` output to the current Parquet
layout. `convert_missing_csv_to_parquet.py` streams a legacy `coverage_missing.csv`
into per-parent `<out-dir>/<Parent>/from_csv.parquet`. `split_missing_from_csv.py`
then splits one per-parent parquet into the per-grid-cell shards `diff` writes,
so a later `audit` re-run recognizes those cells as already processed.

```bash
python tools/coverage/convert_missing_csv_to_parquet.py --csv data/manifests/coverage_missing.csv --out-dir coverage_missing
python tools/coverage/split_missing_from_csv.py coverage_missing/Europe/from_csv.parquet
```

---

## Helper scripts

Helper scripts are grouped by purpose. **Run them from the repository root** (their default paths are relative to the working directory):

| Location | Contents |
| --- | --- |
| *(root)* | Core entry points: `batch_chunks_mp_api_v3.py`, `coverage_audit.py`, `backfill_missing.py`, plus `progress_tracker.py`, `browse.py`, `split_regions.py`, `generate_countries.py`. |
| `tools/coverage/` | Completeness-audit helpers: `convert_missing_csv_to_parquet.py`, `split_missing_from_csv.py`, `validate_missing_sample.py`, `check_grid_data.py`. |
| `tools/repair/` | Image-gap / manifest repair + maintenance: `diagnose_images.py`, `fetch_missing_images.py`, `rebuild_manifest_from_images.py`, `drop_dead_from_manifest.py`, `cleanup_offending_regions.py`, `deduplicate_parquets.py`. |
| `runners/` | Convenience shell scripts: `monitor_and_verify.sh`, `pull_all.sh`. |
| `data/` | Working artifacts (`dead_images.txt`, `dead_manifest_rows.parquet`, `rebuild_done_regions.txt`), plus `data/manifests/` (missing/orphan/dead manifest CSVs) and `data/grids/` (per-region input grid CSVs). |
| `logs/` | Run logs from helper scripts (`*.log`). |
| `archived/` | Superseded scripts kept for reference. |

Kept at the repo root: the core scripts, configuration (`.env`, `proxies.txt`), the master grid `original_global_grid_5deg.csv`, and the live data directories (`grid_runs/`, `coverage_missing/`, `coverage_missing_inscope/`, `progress_files/`). `proxies.txt`, `original_global_grid_5deg.csv`, and `coverage_missing_inscope/` are read by long-running jobs, so they stay put.

### Grid preparation

#### `split_regions.py`

Splits `global_grid_5deg.csv` into individual per-region CSV files placed under `regions/pending/`. Each file contains the rows for a single named region and is ready to pass directly to `batch_chunks_mp_api_v3.py`.

```bash
python split_regions.py
```

#### `generate_countries.py`

For every region folder found under one or more `--dirs` directories, writes a `covered_countries.txt` file listing the countries whose boundaries intersect that region's bounding box. Uses Natural Earth 110 m country data.

```bash
python generate_countries.py --dirs grid_runs /mnt/hdd/grid_runs
```

---

### Progress and navigation

#### `progress_tracker.py`

Displays a Rich-formatted progress table grouped by parent region, showing completion percentage, total data points, ground-animal counts, and image download rate. Accepts multiple base directories so runs spread across several drives can be reported together. Saves a timestamped CSV report to `progress_files/`.

```bash
python progress_tracker.py regions.csv --dirs grid_runs /mnt/hdd/grid_runs
```

| Option | Default | Purpose |
| --- | ---: | --- |
| `--dirs` | required | One or more base directories to scan for grid runs. |
| `--sub-grid-step` | `1.0` | Must match the `--sub-grid-step` used in the main script. |
| `-w` / `--workers` | `2 × CPU` | Concurrent threads for scanning. |

#### `find_location_folder.py`

Geocodes a city or country name via Nominatim and finds which region folders in your grid runs overlap with it. Useful for quickly locating data for a specific place across multiple drives.

```bash
python find_location_folder.py "Japan" --dirs grid_runs /mnt/hdd/grid_runs
python find_location_folder.py "Paris" --dirs grid_runs
```

#### `scan_regions.py`

Scans one or more base directories for folders matching a region prefix, then recommends the exact `--parent-dir` and `--image-dir` flags to pass to the main script. Also reports per-directory breakdowns and flags any regions whose data or images are split across unexpected directories.

```bash
python scan_regions.py South_America --dirs grid_runs /mnt/hdd/grid_runs
```

---

### Ledger management

#### `generate_ledger.py`

Builds or appends to an exclude ledger (a plain-text file of image IDs) by scanning a directory tree for `.jpg` files. Pass the resulting file to the main script via `--exclude-ledger` to skip images that have already been downloaded.

```bash
python generate_ledger.py --image-dir /mnt/hdd/grid_runs --output global_exclude_ledger.txt
python generate_ledger.py --image-dir /mnt/hdd/grid_runs --output global_exclude_ledger.txt --substring North_America
```

| Option | Default | Purpose |
| --- | ---: | --- |
| `--image-dir` | required | Base directory containing grid run folders with images. |
| `--output` | `global_exclude_ledger.txt` | Ledger file to create or append to. |
| `--substring` | unset | Only include images from folders whose path contains this string. |

---

### Checkpoint maintenance

#### `compress_checkpoints.py`

One-time migration utility. Scans `grid_runs/` for uncompressed `.json`, `.jsonl`, and `.csv` files and gzip-compresses them in place. Only needed if you have checkpoints produced before the pipeline switched to zstd.

```bash
python compress_checkpoints.py
```

#### `convert_to_zstd.py`

Converts `.json.gz` and `.jsonl.gz` checkpoint files to `.zst` format, with optional byte-level verification and automatic deletion of the original `.gz` files after a confirmed match.

```bash
python convert_to_zstd.py regions.csv --parent-dirs grid_runs /mnt/hdd/grid_runs
python convert_to_zstd.py regions.csv --parent-dirs grid_runs --compare --delete-gz
```

| Option | Default | Purpose |
| --- | ---: | --- |
| `--parent-dirs` | `grid_runs` | One or more directories to scan for `.gz` files. |
| `--compare` | `False` | Verify the decompressed `.zst` stream matches the original `.gz` byte-for-byte. |
| `--delete-gz` | `False` | Delete `.gz` files after processing (only after a verified match if `--compare` is set). |
| `--overwrite` | `False` | Re-convert even if a `.zst` file already exists. |
| `--ram-gb` | `8.0` | Memory budget for read/write chunks. |
| `--workers` | all cores | Zstandard compression threads. |

#### `check_zst_health.py`

Tests all `.zst` files under `grid_runs/` using `zstd -t`. Lists corrupted files and optionally deletes them interactively or in bulk. When `--clear-completed` is set, deletes the corresponding `.completed_<sub_id>` marker instead of a region-level text file, so the main script will re-process the affected sub-grid on the next run.

```bash
python check_zst_health.py
python check_zst_health.py --delete-all --clear-completed --ignore-recent 1.5
```

---

### Audit and repair

#### `audit_markers.py`

Scans `grid_runs/` for orphaned `.completed_*` resume markers — markers whose corresponding `metadata_checkpoint_*.jsonl.zst` or `animal_detections_checkpoint_*.jsonl.zst` files are missing. Orphaned markers would otherwise convince the main script that a sub-grid finished successfully, causing it to be skipped silently on the next run. Any orphaned markers found are deleted automatically so the pipeline will re-process those sub-grids.

```bash
python audit_markers.py
```

#### `audit_silent_skips.py`

A multiprocessed auditor that detects *silent skips* — sub-grids marked as `.completed_` but whose checkpoint files contain fewer records than expected. For each completed sub-grid it loads the topology checkpoint to get the expected image count, then counts lines in the metadata and animal-detection checkpoints. If either count falls short, the `.completed_` marker is deleted to force a backfill rerun. Use `--dry-run` to report discrepancies without deleting anything.

```bash
python audit_silent_skips.py
python audit_silent_skips.py --dry-run --substring North_America
```

| Option | Default | Purpose |
| --- | ---: | --- |
| `--workers` | all cores | Parallel CPU workers for checkpoint parsing. |
| `--dry-run` | `False` | Report discrepancies without deleting markers. |
| `--substring` | unset | Only audit sub-grids whose path contains this string (e.g., `North_America`). |

#### `generate_rerun_commands.py`

Reads a grid CSV and checks every region directory for missing `.completed_*` or `.empty_*` markers. For each region with incomplete sub-grids, it generates a ready-to-run `batch_chunks_mp_api_v3.py` command using `--row-index` and `--sub-indices` to target only the missing cells. All commands are printed to the terminal and saved to a bash script that can be executed directly.

```bash
python generate_rerun_commands.py regions.csv
python generate_rerun_commands.py regions.csv --substring "South America" --output-script rerun_sa.sh
```

| Option | Default | Purpose |
| --- | ---: | --- |
| `--parent-dir` | `grid_runs` | Directory containing region output folders. |
| `--substring` | unset | Filter to rows whose region name contains this string. |
| `--sub-grid-step` | `1.0` | Must match the `--sub-grid-step` used in the main script. |
| `--output-script` | `run_missing.sh` | Name of the generated bash script. |

#### `check_grid_data.py`

Reports which CSV row indices already have corresponding Parquet data on disk
across one or more `--dirs`, so you can see at a glance which regions still need
to be run.

```bash
python tools/coverage/check_grid_data.py regions/running/north_america.csv --dirs grid_runs /mnt/hdd/grid_runs
```

---

### Image gap repair

These scripts address the `% Images Downloaded` anomalies that arise when the
on-disk jpgs and the `ground_animals_*.parquet` manifests drift out of sync —
either because signed download URLs expired, or because parquet rows were lost
while their images survived.

#### `diagnose_images.py`

Compares `ground_animals_*` parquet `image_id`s against the `.jpg` files on disk
for each region, reporting **orphaned images** (file exists, not in any parquet)
and **missing downloads** (in parquet, no file). Run it first to understand a
region whose progress shows >100%.

```bash
python tools/repair/diagnose_images.py original_global_grid_5deg.csv \
    --dirs grid_runs /mnt/hdd/grid_runs --region "South Asia"
```

#### `fetch_missing_images.py`

Finds image_ids that are in a manifest but have no jpg on disk (the true missing
set), then downloads them — trying the stored `thumb_original_url` first and, on
failure, fetching a **fresh** signed URL from the Graph API (rotating across all
`MLY_KEY*` tokens). Images the API reports gone (400/404) are recorded as
permanently dead. Subcommands: `scan`, `download`, `all`.

```bash
python tools/repair/fetch_missing_images.py all original_global_grid_5deg.csv --dirs grid_runs ... \
    --out data/manifests/missing_manifest.csv --proxy-file proxies.txt -w 24
```

#### `rebuild_manifest_from_images.py`

Treats the **images as the source of truth** to fix the `% > 100` case (more jpgs
on disk than manifest rows). For every orphan image_id it re-queries Mapillary
detections, keeps only those still classified `animal--ground-animal`, and writes
the reconstructed rows to `ground_animals_<region>_recovered_<NNN>.parquet` (reusing
the all_data row verbatim when present, else re-fetching metadata), so the manifest
count rises to match the images.

#### `drop_dead_from_manifest.py`

Removes permanently-dead image_ids (listed in `dead_images.txt`) from every
`ground_animals_*` parquet so they stop inflating the missing gap. Archives the
removed rows to `dead_manifest_rows.parquet` first (restorable), rewrites affected
files atomically, and is idempotent. Default is dry-run; pass `--execute`.

```bash
python tools/repair/drop_dead_from_manifest.py --dirs grid_runs /mnt/hdd/grid_runs            # dry-run
python tools/repair/drop_dead_from_manifest.py --dirs grid_runs /mnt/hdd/grid_runs --execute
```

#### `cleanup_offending_regions.py`

Deletes parquet, checkpoint, and `.completed_/.empty_` marker files (but **keeps**
downloaded images and the validated-images ledger) for a hard-coded list of region
dirs with orphaned images, forcing a clean re-extraction of those regions. Default
is dry-run; pass `--execute`.

```bash
python tools/repair/cleanup_offending_regions.py --dirs DIR1 DIR2 DIR3 --execute
```

#### `monitor_and_verify.sh`

Waits for a running manifest rebuild to finish, then regenerates the
`progress_tracker.py` report to confirm every region is back at ≤100%.

```bash
./runners/monitor_and_verify.sh
```

---

### Data maintenance

#### `deduplicate_parquets.py`

Multiprocessed pass over every `*.parquet` that removes duplicate `image_id` rows
(keeping the first occurrence) and rewrites only files that actually had
duplicates, with zstd compression matching the main script.

```bash
python tools/repair/deduplicate_parquets.py --parent-dir grid_runs --substring North_America
```

---

### Visualization

#### `visualize_region_tiles.py`

Generates a static map image showing mercantile tiles for a region, colored green (land) or red (water). Saves the PNG into the region's output folder. Requires a folder name in the format `Name_SWLon_SWLat_NELon_NELat`.

```bash
python visualize_region_tiles.py "Sample_Region_-74.1_40.6_-73.7_40.9"
python visualize_region_tiles.py "Sample_Region_-74.1_40.6_-73.7_40.9" --zoom 14 --parent_dir grid_runs
```

## Troubleshooting

- **Missing token:** ensure `.env` contains `MLY_KEY` or the numbered `MLY_KEY_<n>` selected by `--token`.
- **No images downloaded:** confirm that `--no-download-images` was not used and that `ground_animals_*.parquet` files contain non-null `thumb_original_url` values.
- **Rate limits or network failures:** reduce `--search-max-workers`, reduce `--entity-max-workers`, reduce `--download-max-workers`, or provide `--proxy-file` if appropriate for your environment.
- **Large output directories:** increase available disk space, use `--image-dir` for images, or run with `--no-download-images` first and download later with `--download-only`.
- **Slow HDD writes during downloads:** use `--temp-dir` pointing to a fast SSD; the pipeline writes there first and moves files to `--image-dir` in a background thread.
- **Need to reprocess a specific sub-grid:** use `--sub-indices` with the sub-grid number (zero-based) to target only that cell without reprocessing the entire region.
- **Interrupted run:** re-run the same command. Existing checkpoints, Parquet chunks, and completion markers are used to resume work where possible.
