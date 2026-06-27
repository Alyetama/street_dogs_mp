"""
Diagnostic: compare ground_animals parquet image_ids vs image files on disk.
Run this to find orphaned images (file exists, not in parquet) and missing
downloads (in parquet, no file) for regions where % > 100.

Usage (run from the repo root):
    python tools/repair/diagnose_images.py original_global_grid_5deg.csv \
        --dirs grid_runs /mnt/hdd/grid_runs ... \
        --region "South Asia"
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from rich.console import Console
from rich.table import Table


def sanitize_folder_name(name):
    safe_name = name.replace('&', 'and')
    return re.sub(r'[^\w\-_\.]', '_', safe_name).strip('_')


def diagnose_region_row(row, dirs):
    parent_region = row['region']
    unique_region_id = f"{parent_region}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"
    safe_region_id = sanitize_folder_name(unique_region_id)

    region_dirs = []
    for d in dirs:
        p = os.path.join(d, safe_region_id)
        if os.path.exists(p):
            region_dirs.append(p)

    if not region_dirs:
        return None

    animal_files = []
    image_dirs_found = []

    for r_dir in region_dirs:
        try:
            for entry in os.scandir(r_dir):
                if (entry.is_file() and entry.name.startswith(
                        f'ground_animals_{safe_region_id}_')
                        and entry.name.endswith('.parquet')):
                    animal_files.append(entry.path)
        except Exception:
            pass

        img_dir = os.path.join(r_dir, 'ground_animal_images')
        if os.path.exists(img_dir):
            image_dirs_found.append(img_dir)

    if not animal_files and not image_dirs_found:
        return None

    # Unique image_ids from all parquet files
    parquet_ids = set()
    if animal_files:
        try:
            parquet_ids = set(
                pl.scan_parquet(list(set(animal_files))).select(
                    'image_id').collect()['image_id'].to_list())
        except Exception as e:
            parquet_ids = set()

    # Unique image filenames on disk (strip .jpg → image_id)
    disk_ids = set()
    for img_dir in image_dirs_found:
        try:
            for entry in os.scandir(img_dir):
                if entry.is_file() and entry.name.lower().endswith('.jpg'):
                    disk_ids.add(entry.name[:-4])
        except Exception:
            pass

    orphaned = disk_ids - parquet_ids  # on disk, not in parquet
    missing = parquet_ids - disk_ids  # in parquet, no file

    if len(disk_ids) == 0 and len(parquet_ids) == 0:
        return None

    pct = (len(disk_ids) / len(parquet_ids) * 100) if parquet_ids else 0.0

    return {
        'region_id': unique_region_id,
        'safe_id': safe_region_id,
        'parquet_ids': len(parquet_ids),
        'disk_ids': len(disk_ids),
        'orphaned': len(orphaned),
        'missing': len(missing),
        'pct': pct,
        'region_dirs': region_dirs,
        'image_dirs': image_dirs_found,
        'parquet_files': len(animal_files),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('grid_csv')
    parser.add_argument('--dirs', nargs='+', required=True)
    parser.add_argument(
        '--region',
        type=str,
        default=None,
        help='Filter to a specific parent region (e.g. "South Asia")')
    parser.add_argument(
        '--min-pct',
        type=float,
        default=100.1,
        help='Only show regions above this %% (default: 100.1)')
    parser.add_argument('--top',
                        type=int,
                        default=20,
                        help='Show top N worst offenders (default: 20)')
    parser.add_argument('-w', '--workers', type=int, default=16)
    args = parser.parse_args()

    console = Console()
    df_grid = pl.read_csv(args.grid_csv)
    rows = list(df_grid.iter_rows(named=True))

    if args.region:
        rows = [r for r in rows if r['region'] == args.region]
        console.print(
            f"Filtered to {len(rows)} rows for region: [cyan]{args.region}[/]")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(diagnose_region_row, row, args.dirs): row
            for row in rows
        }
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    # Filter to regions exceeding threshold
    over = [r for r in results if r['pct'] >= args.min_pct]
    over.sort(key=lambda x: x['orphaned'], reverse=True)

    if not over:
        console.print(f"\n[green]No regions above {args.min_pct}% found.[/]")
        return

    console.print(
        f"\nFound [bold red]{len(over)}[/] regions above {args.min_pct}% - showing top {args.top}:\n"
    )

    table = Table(show_lines=True)
    table.add_column("Region ID", style="cyan", no_wrap=False, max_width=60)
    table.add_column("Parquet IDs", justify="right")
    table.add_column("Disk Files", justify="right")
    table.add_column("Orphaned\n(disk, not parquet)",
                     justify="right",
                     style="red")
    table.add_column("Missing\n(parquet, no file)",
                     justify="right",
                     style="yellow")
    table.add_column("% Img", justify="right", style="magenta")
    table.add_column("Pq files", justify="right")
    table.add_column("Dirs found", justify="right")

    for r in over[:args.top]:
        table.add_row(
            r['region_id'],
            f"{r['parquet_ids']:,}",
            f"{r['disk_ids']:,}",
            f"{r['orphaned']:,}",
            f"{r['missing']:,}",
            f"{r['pct']:.1f}%",
            str(r['parquet_files']),
            str(len(r['region_dirs'])),
        )

    console.print(table)

    # Summary by orphaned vs missing
    total_orphaned = sum(r['orphaned'] for r in over)
    total_missing = sum(r['missing'] for r in over)
    console.print(
        f"\nTotal orphaned images (file exists, not in parquet): [red]{total_orphaned:,}[/]"
    )
    console.print(
        f"Total missing downloads (parquet entry, no file):    [yellow]{total_missing:,}[/]"
    )

    # Show which dirs each top region is in
    if over:
        console.print(f"\n[bold]Dirs breakdown for top offender:[/]")
        top = over[0]
        console.print(f"  Region: {top['region_id']}")
        console.print(f"  Data dirs: {top['region_dirs']}")
        console.print(f"  Image dirs: {top['image_dirs']}")


if __name__ == '__main__':
    main()
