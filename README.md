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

Process existing JSON checkpoints into Parquet files without fetching new Mapillary API data:

```bash
python batch_chunks_mp_api_v3.py regions.csv --parquet-only
```

Send image downloads to a separate disk or mount point:

```bash
python batch_chunks_mp_api_v3.py regions.csv --image-dir /path/to/image_storage
```

## Common options

| Option | Default | Purpose |
| --- | ---: | --- |
| `--zoom-level` | `14` | Mercantile tile zoom used while scanning bounding boxes. |
| `--outer-max-workers` | `5` | Number of regions processed in parallel in local mode. |
| `--inner-max-workers` | `20` | Threads per region for Mapillary API calls. |
| `--download-max-workers` | `10` | Threads used specifically for image downloads. |
| `--sub-grid-step` | `1.0` | Degree step used to split large input regions into smaller sub-grids. |
| `--parent-dir` | `grid_runs` | Root directory for per-region outputs and checkpoints. |
| `--api-chunk-size` | `2500` | Batch size for threaded API work. |
| `--parquet-chunk-size` | `100000` | Rows per `all_data_*.parquet` output partition. |
| `--proxy-file` | unset | Optional proxy list file. Supports `ip:port`, `ip:port:user:password`, and full proxy URLs. |
| `--exclude-ledger` | unset | Text file of image IDs to skip. |
| `--token` | unset | Selects `MLY_KEY_<n>` instead of `MLY_KEY`. |
| `--slurm` | `False` | Runs one region based on `SLURM_ARRAY_TASK_ID` or `--row-index`. |
| `--download-only` | `False` | Skip API fetching and ONLY download images from existing ground_animals Parquets. |
| `--parquet-only` | `False` | Skip all API fetching and ONLY process existing JSON files into Parquet chunks. |

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
  --inner-max-workers "$((SLURM_CPUS_PER_TASK * 4))" \
  --parent-dir grid_runs \
  --no-download-images
```

## Helper scripts

This repository also contains utility scripts for preparing grids, checking/compressing outputs, and visualization, including `generate_countries.py`, `split_regions.py`, `scan_regions.py`, `check_gz_health.py`, `convert_to_zstd.py`, `compress_checkpoints.py`, `progress_tracker.py`, and `visualize_region_tiles.py`. These are supporting tools; the primary pipeline entry point remains `batch_chunks_mp_api_v3.py`.

## Troubleshooting

- **Missing token:** ensure `.env` contains `MLY_KEY` or the numbered `MLY_KEY_<n>` selected by `--token`.
- **No images downloaded:** confirm that `--no-download-images` was not used and that `ground_animals_*.parquet` files contain non-null `thumb_original_url` values.
- **Rate limits or network failures:** reduce `--inner-max-workers`, reduce `--download-max-workers`, or provide `--proxy-file` if appropriate for your environment.
- **Large output directories:** increase available disk space, use `--image-dir` for images, or run with `--no-download-images` first and download later with `--download-only`.
- **Interrupted run:** re-run the same command. Existing checkpoints, Parquet chunks, and completion markers are used to resume work where possible.
