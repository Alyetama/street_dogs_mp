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

## Helper scripts

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

#### `check_gz_health.py`

Concurrently tests all `.gz` files under `grid_runs/` using `gzip -t`. Lists corrupted files and optionally deletes them interactively or in bulk.

```bash
python check_gz_health.py
python check_gz_health.py --delete-all --substring Pacific_Ocean --ignore-recent 2.0
```

| Option | Default | Purpose |
| --- | ---: | --- |
| `-d` / `--delete-all` | `False` | Delete all corrupted files without prompting. |
| `-s` / `--substring` | unset | Only check files whose path contains this string. |
| `-i` / `--ignore-recent` | `0` | Skip files modified within the last N hours. |
| `-c` / `--clear-completed` | `False` | Remove the region from `completed_regions.txt` if corruption is found. |
| `-e` / `--exclude-ext` | unset | Skip files ending with specific sub-extensions (e.g., `.csv.gz`). |
| `-w` / `--workers` | CPU count | Concurrent workers. |

#### `check_zst_health.py`

Same as `check_gz_health.py` but for `.zst` files, using `zstd -t`. When `--clear-completed` is set, deletes the corresponding `.completed_<sub_id>` marker instead of a region-level text file, so the main script will re-process the affected sub-grid on the next run.

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
