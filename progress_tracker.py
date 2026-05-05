import argparse
import csv
import datetime
import glob
import os
import re
from collections import defaultdict

import polars as pl
from rich.console import Console
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
    """Uses Polars Lazy API to count unique images rapidly without blowing up RAM."""
    if not files:
        return 0
    try:
        return pl.scan_parquet(files).select(
            'image_id').unique().collect().height
    except Exception:
        return 0


def main():
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

    args = parser.parse_args()

    console = Console()

    if not os.path.exists(args.grid_csv):
        console.print(
            f"[bold red]Error:[/] Could not find CSV file: {args.grid_csv}")
        return

    df_grid = pl.read_csv(args.grid_csv)

    tracker = defaultdict(
        lambda: {
            "total_regions": 0,
            "completed_regions": 0,
            "total_data_points": 0,
            "total_animals": 0,
            "total_images": 0
        })

    with console.status(
            "[bold cyan]Scanning directories and analyzing Parquet chunks...",
            spinner="dots"):
        for row in df_grid.iter_rows(named=True):
            mother_region = row['region']
            unique_region_id = f"{mother_region}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"
            safe_region_id = sanitize_folder_name(unique_region_id)

            tracker[mother_region]["total_regions"] += 1

            region_dirs = []
            for d in args.dirs:
                possible_path = os.path.join(d, safe_region_id)
                if os.path.exists(possible_path):
                    region_dirs.append(possible_path)

            if not region_dirs:
                continue

            expected_subgrids = get_expected_subgrids(row['sw_lon'],
                                                      row['sw_lat'],
                                                      row['ne_lon'],
                                                      row['ne_lat'],
                                                      args.sub_grid_step)

            empty_markers = 0
            completed_markers = 0
            all_data_files = []
            animal_files = []
            image_count = 0

            for r_dir in region_dirs:
                empty_markers += len(glob.glob(os.path.join(r_dir,
                                                            '.empty_*')))
                completed_markers += len(
                    glob.glob(os.path.join(r_dir, '.completed_*')))

                all_data_files.extend(
                    glob.glob(
                        os.path.join(r_dir,
                                     f'all_data_{safe_region_id}_*.parquet')))
                animal_files.extend(
                    glob.glob(
                        os.path.join(
                            r_dir,
                            f'ground_animals_{safe_region_id}_*.parquet')))

                image_folder = os.path.join(r_dir, 'ground_animal_images')
                if os.path.exists(image_folder):
                    image_count += sum(1 for f in os.listdir(image_folder)
                                       if f.lower().endswith('.jpg'))

            total_markers = empty_markers + completed_markers
            if total_markers >= expected_subgrids:
                tracker[mother_region]["completed_regions"] += 1

            tracker[mother_region]["total_data_points"] += count_parquet_rows(
                list(set(all_data_files)))

            tracker[mother_region]["total_animals"] += count_parquet_rows(
                list(set(animal_files)))

            tracker[mother_region]["total_images"] += image_count

    table = Table(title="Mapillary Extraction Progress",
                  title_style="bold magenta")
    table.add_column("Mother Region", style="cyan", no_wrap=True)
    table.add_column("% Completed", justify="right", style="green")
    table.add_column("% Incomplete", justify="right", style="red")
    table.add_column("Total Data Points", justify="right", style="blue")
    table.add_column("Ground Animals", justify="right", style="yellow")
    table.add_column("Images Collected", justify="right", style="magenta")

    csv_data = []

    grand_total_regions = 0
    grand_completed_regions = 0
    grand_total_data_points = 0
    grand_total_animals = 0
    grand_total_images = 0

    for mother_region, stats in sorted(tracker.items()):
        total = stats["total_regions"]
        completed = stats["completed_regions"]

        pct_completed = (completed / total) * 100 if total > 0 else 0.0
        pct_incomplete = 100.0 - pct_completed

        grand_total_regions += total
        grand_completed_regions += completed
        grand_total_data_points += stats['total_data_points']
        grand_total_animals += stats['total_animals']
        grand_total_images += stats['total_images']

        table.add_row(mother_region,
                      f"{pct_completed:.1f}% ({completed}/{total})",
                      f"{pct_incomplete:.1f}%",
                      f"{stats['total_data_points']:,}",
                      f"{stats['total_animals']:,}",
                      f"{stats['total_images']:,}")

        csv_data.append({
            "Mother Region": mother_region,
            "% Completed": round(pct_completed, 2),
            "% Incomplete": round(pct_incomplete, 2),
            "Total Regions": total,
            "Completed Regions": completed,
            "Total Data Points": stats['total_data_points'],
            "Ground Animals": stats['total_animals'],
            "Images Collected": stats['total_images']
        })

    grand_pct_completed = (grand_completed_regions / grand_total_regions
                           ) * 100 if grand_total_regions > 0 else 0.0
    grand_pct_incomplete = 100.0 - grand_pct_completed

    table.add_section()

    table.add_row(
        "GLOBAL TOTAL",
        f"{grand_pct_completed:.1f}% ({grand_completed_regions}/{grand_total_regions})",
        f"{grand_pct_incomplete:.1f}%",
        f"{grand_total_data_points:,}",
        f"{grand_total_animals:,}",
        f"{grand_total_images:,}",
        style="bold white on default")

    csv_data.append({
        "Mother Region": "GLOBAL TOTAL",
        "% Completed": round(grand_pct_completed, 2),
        "% Incomplete": round(grand_pct_incomplete, 2),
        "Total Regions": grand_total_regions,
        "Completed Regions": grand_completed_regions,
        "Total Data Points": grand_total_data_points,
        "Ground Animals": grand_total_animals,
        "Images Collected": grand_total_images
    })

    console.print("\n")
    console.print(table)

    output_dir = "progress_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
