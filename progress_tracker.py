import argparse
import csv
import datetime
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn, TimeRemainingColumn)
from rich.table import Table


def sanitize_folder_name(name):
    safe_name = name.replace('&', 'and')
    return re.sub(r'[^\w\-_\.]', '_', safe_name).strip('_')


def get_expected_subgrids(west, south, east, north, step=1.0):
    """Calculates the exact number of sub-grids that the main script processes."""
    count = 0
    cur_lat = south
    while cur_lat < north:
        cur_lon = west
        while cur_lon < east:
            count += 1
            cur_lon += step
        cur_lat += step
    return count


def count_parquet_rows(files):
    """Uses Polars Lazy API to count unique images rapidly."""
    if not files:
        return 0
    try:
        return pl.scan_parquet(files).select(
            'image_id').unique().collect().height
    except Exception:
        return 0


def process_region_row(row, dirs, sub_grid_step):
    """Worker function to process a single region concurrently."""
    parent_region = row['region']
    unique_region_id = f"{parent_region}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"
    safe_region_id = sanitize_folder_name(unique_region_id)

    region_dirs = []
    for d in dirs:
        possible_path = os.path.join(d, safe_region_id)
        if os.path.exists(possible_path):
            region_dirs.append(possible_path)

    if not region_dirs:
        return parent_region, 1, 0, 0, 0, 0

    expected_subgrids = get_expected_subgrids(row['sw_lon'], row['sw_lat'],
                                              row['ne_lon'], row['ne_lat'],
                                              sub_grid_step)

    empty_markers = 0
    completed_markers = 0
    all_data_files = []
    animal_files = []
    image_count = 0

    for r_dir in region_dirs:
        try:
            for entry in os.scandir(r_dir):
                if entry.is_file():
                    name = entry.name
                    if name.startswith('.empty_'):
                        empty_markers += 1
                    elif name.startswith('.completed_'):
                        completed_markers += 1
                    elif name.startswith(f'all_data_{safe_region_id}_'
                                         ) and name.endswith('.parquet'):
                        all_data_files.append(entry.path)
                    elif name.startswith(f'ground_animals_{safe_region_id}_'
                                         ) and name.endswith('.parquet'):
                        animal_files.append(entry.path)
        except Exception:
            pass

        image_folder = os.path.join(r_dir, 'ground_animal_images')
        if os.path.exists(image_folder):
            try:
                image_count += sum(
                    1 for entry in os.scandir(image_folder)
                    if entry.is_file() and entry.name.lower().endswith('.jpg'))
            except Exception:
                pass

    total_markers = empty_markers + completed_markers
    completed_val = 1 if total_markers >= expected_subgrids else 0

    data_points = count_parquet_rows(list(set(all_data_files)))
    animals = count_parquet_rows(list(set(animal_files)))

    return parent_region, 1, completed_val, data_points, animals, image_count


def main():
    default_workers = min(32, (os.cpu_count() or 4) * 2)

    parser = argparse.ArgumentParser(description="Mapillary Progress Tracker")
    parser.add_argument('grid_csv',
                        type=str,
                        help="Path to the master grid CSV file")
    parser.add_argument(
        '--dirs',
        nargs='+',
        required=True,
        help="One or more base directories to search for grid runs")
    parser.add_argument(
        '--sub-grid-step',
        type=float,
        default=1.0,
        help="Step size used in the main script (default: 1.0)")
    parser.add_argument(
        '-w',
        '--workers',
        type=int,
        default=default_workers,
        help=f"Number of concurrent threads (default: {default_workers})")

    args = parser.parse_args()
    console = Console()

    if not os.path.exists(args.grid_csv):
        console.print(
            f"[bold red]Error:[/] Could not find CSV file: {args.grid_csv}")
        return

    df_grid = pl.read_csv(args.grid_csv)
    rows = list(df_grid.iter_rows(named=True))

    tracker = defaultdict(
        lambda: {
            "total_regions": 0,
            "completed_regions": 0,
            "total_data_points": 0,
            "total_animals": 0,
            "total_images": 0
        })

    console.print("\n")
    with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn(
                "[progress.percentage]{task.completed}/{task.total} regions"),
            TimeRemainingColumn(),
            console=console) as progress:

        task_id = progress.add_task("Analyzing Parquet & File Chunks...",
                                    total=len(rows))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_region_row, row, args.dirs,
                                args.sub_grid_step) for row in rows
            ]

            for future in as_completed(futures):
                try:
                    parent_region, total_reg, comp_reg, dp, anim, imgs = future.result(
                    )
                    tracker[parent_region]["total_regions"] += total_reg
                    tracker[parent_region]["completed_regions"] += comp_reg
                    tracker[parent_region]["total_data_points"] += dp
                    tracker[parent_region]["total_animals"] += anim
                    tracker[parent_region]["total_images"] += imgs
                except Exception as e:
                    console.print(f"[bold red]Error processing region:[/] {e}")

                progress.advance(task_id)

    table = Table(title="Mapillary Extraction Progress",
                  title_style="bold magenta")
    table.add_column("Parent Region", style="cyan", no_wrap=True)
    table.add_column("% Completed", justify="right", style="green")
    table.add_column("Total Data Points", justify="right", style="blue")
    table.add_column("Ground Animals", justify="right", style="yellow")
    table.add_column("Images Collected", justify="right", style="magenta")
    table.add_column("% Img Downloaded", justify="right", style="bright_cyan")

    csv_data = []
    grand_total_regions = 0
    grand_completed_regions = 0
    grand_total_data_points = 0
    grand_total_animals = 0
    grand_total_images = 0

    for parent_region, stats in sorted(tracker.items()):
        total = stats["total_regions"]
        completed = stats["completed_regions"]

        pct_completed = (completed / total) * 100 if total > 0 else 0.0
        has_animals = stats['total_animals'] > 0

        if has_animals:
            pct_images_downloaded = (stats['total_images'] /
                                     stats['total_animals']) * 100
            img_collected_str = f"{stats['total_images']:,}"
            pct_images_str = f"{pct_images_downloaded:.1f}%"

            # Only include in grand total if animals exist
            grand_total_animals += stats['total_animals']
            grand_total_images += stats['total_images']
        else:
            pct_images_downloaded = 0.0
            img_collected_str = "-"
            pct_images_str = "-"

        grand_total_regions += total
        grand_completed_regions += completed
        grand_total_data_points += stats['total_data_points']

        table.add_row(parent_region,
                      f"{pct_completed:.1f}% ({completed}/{total})",
                      f"{stats['total_data_points']:,}",
                      f"{stats['total_animals']:,}", img_collected_str,
                      pct_images_str)

        csv_data.append({
            "Parent Region":
            parent_region,
            "% Completed":
            round(pct_completed, 2),
            "Total Regions":
            total,
            "Completed Regions":
            completed,
            "Total Data Points":
            stats['total_data_points'],
            "Ground Animals":
            stats['total_animals'],
            "Images Collected":
            stats['total_images'] if has_animals else "-",
            "% Images Downloaded":
            round(pct_images_downloaded, 2) if has_animals else "-"
        })

    grand_pct_completed = (grand_completed_regions / grand_total_regions
                           ) * 100 if grand_total_regions > 0 else 0.0

    if grand_total_animals > 0:
        grand_pct_images = (grand_total_images / grand_total_animals) * 100
        grand_img_str = f"{grand_total_images:,}"
        grand_pct_str = f"{grand_pct_images:.1f}%"
    else:
        grand_pct_images = 0.0
        grand_img_str = "-"
        grand_pct_str = "-"

    table.add_section()
    table.add_row(
        "GLOBAL TOTAL",
        f"{grand_pct_completed:.1f}% ({grand_completed_regions}/{grand_total_regions})",
        f"{grand_total_data_points:,}",
        f"{grand_total_animals:,}",
        grand_img_str,
        grand_pct_str,
        style="bold white on default")

    csv_data.append({
        "Parent Region":
        "GLOBAL TOTAL",
        "% Completed":
        round(grand_pct_completed, 2),
        "Total Regions":
        grand_total_regions,
        "Completed Regions":
        grand_completed_regions,
        "Total Data Points":
        grand_total_data_points,
        "Ground Animals":
        grand_total_animals,
        "Images Collected":
        grand_total_images if grand_total_animals > 0 else "-",
        "% Images Downloaded":
        round(grand_pct_images, 2) if grand_total_animals > 0 else "-"
    })

    console.print("\n")
    console.print(table)

    output_dir = "progress_files"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"progress_report_{timestamp}.csv")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

    console.print(
        f"\n[\u2713] Detailed report successfully saved to [bold green]{output_path}[/]"
    )


if __name__ == "__main__":
    main()
