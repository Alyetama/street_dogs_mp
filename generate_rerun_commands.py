import argparse
import os
import re

import polars as pl
from rich.console import Console


def sanitize_folder_name(name):
    safe_name = name.replace('&', 'and')
    return re.sub(r'[^\w\-_\.]', '_', safe_name).strip('_')


def get_expected_subgrids(west, south, east, north, step=1.0):
    """Calculates exactly how many internal subgrids should exist."""
    sub_bboxes = 0
    cur_lat = south
    while cur_lat < north:
        cur_lon = west
        while cur_lon < east:
            sub_bboxes += 1
            cur_lon += step
        cur_lat += step
    return sub_bboxes


def main():
    parser = argparse.ArgumentParser(
        description=
        "Generate terminal commands for missing rows and sub-indices.")
    parser.add_argument('csv_file',
                        type=str,
                        help="Path to your grid CSV file")
    parser.add_argument('--parent-dir',
                        type=str,
                        default='grid_runs',
                        help="Directory containing the output folders")
    parser.add_argument(
        '--substring',
        type=str,
        default="",
        help="Optional region filter (e.g., 'North America' or 'North_America')"
    )
    parser.add_argument(
        '--sub-grid-step',
        type=float,
        default=1.0,
        help="Must match what you used in batch_chunks (default: 1.0)")
    parser.add_argument(
        '--output-script',
        type=str,
        default="run_missing.sh",
        help="Name of the bash script to generate (default: run_missing.sh)")
    args = parser.parse_args()

    console = Console()

    try:
        df = pl.read_csv(args.csv_file).with_row_index("row_index")
    except Exception as e:
        console.print(f"[bold red]Failed to read CSV: {e}[/bold red]")
        return

    if args.substring:
        safe_substring = args.substring.replace(" ", "_").replace("&", "and")
        df = df.filter(
            pl.col("region").str.replace_all(" ", "_").str.replace_all(
                "&", "and").str.contains(safe_substring))

    console.print(
        f"[cyan]Scanning {df.height} matching rows from your CSV...[/cyan]")

    if df.is_empty():
        console.print(
            "[bold red][X] Found 0 rows! Check your spelling or CSV contents.[/bold red]"
        )
        return

    missing_details = []

    with console.status(
            "[bold yellow]Scanning for missing .completed_ or .empty_ markers...[/bold yellow]"
    ):
        for row in df.iter_rows(named=True):
            index = row['row_index']

            raw_id = f"{row['region']}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"
            safe_id = sanitize_folder_name(raw_id)
            region_dir = os.path.join(args.parent_dir, safe_id)

            if not os.path.exists(region_dir):
                missing_details.append(
                    (index, safe_id, ["ALL (Folder missing)"]))
                continue

            expected_count = get_expected_subgrids(row['sw_lon'],
                                                   row['sw_lat'],
                                                   row['ne_lon'],
                                                   row['ne_lat'],
                                                   args.sub_grid_step)

            missing_subs = []

            for i in range(expected_count):
                sub_id = f"{safe_id}_sub_{i}"

                completed_marker = os.path.join(region_dir,
                                                f'.completed_{sub_id}')
                empty_marker = os.path.join(region_dir, f'.empty_{sub_id}')

                if not os.path.exists(completed_marker) and not os.path.exists(
                        empty_marker):
                    missing_subs.append(f"sub_{i}")

            if missing_subs:
                missing_details.append((index, safe_id, missing_subs))

    if not missing_details:
        console.print(
            f"\n[bold green][\u2713] All {df.height} scanned regions have 100% of their sub-grid markers![/bold green]"
        )
        return

    console.print(
        f"\n[bold red][!] Found {len(missing_details)} regions missing one or more sub-grid markers.[/bold red]\n"
    )

    # Hardcoded base command string provided by user
    base_cmd = (
        f'python batch_chunks_mp_api_v3.py "{args.csv_file}" '
        '--outer-max-workers 1 --search-max-workers 150 --entity-max-workers 520 '
        '--api-chunk-size 5000 --parquet-chunk-size 100000 '
        '--parent-dir "/media/biodiv/crucial/street_dogs_mp_crucial/grid_runs" '
        '--image-dir "/home/biodiv/capybara/street_dogs_mp_capybara/grid_runs" '
        '--no-download-images')

    commands = []

    for idx, reg_id, subs in missing_details:
        if "ALL (Folder missing)" in subs:
            cmd = f'{base_cmd} --row-index {idx}'
        else:
            sub_nums = [s.replace('sub_', '') for s in subs]
            sub_str = ",".join(sub_nums)
            cmd = f'{base_cmd} --row-index {idx} --sub-indices {sub_str} --token \n'

        commands.append(cmd)

    console.print("[cyan]Generated Commands:[/cyan]")
    for c in commands:
        print(c)

    with open(args.output_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(
            "# Auto-generated script for rerunning missing Mapillary data\n\n")
        for c in commands:
            f.write(c + "\n")

    console.print(
        f"\n[bold green][\u2713] Saved {len(commands)} commands to {args.output_script}[/bold green]"
    )
    console.print(
        f"[cyan]You can run them all sequentially by executing: bash {args.output_script}[/cyan]"
    )


if __name__ == "__main__":
    main()
