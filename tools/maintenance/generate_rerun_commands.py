"""Emit batch_chunks_mp_api.py commands for the cells that are still short of
their sub-grid markers.

The cells of the master grid are split across several data roots, so
completeness has to be judged over ALL of them (a cell that lives on another
drive is complete, not missing). Pass every root with --dirs; with no --dirs
the roots in data/catalog_dirs.txt are used, falling back to --parent-dir.
Each emitted command targets the root where that cell already lives, so a
rerun tops the existing cell up instead of starting a second copy elsewhere.
"""

import argparse
import os
import re

import polars as pl
from rich.console import Console

CATALOG_DIRS = os.path.join('data', 'catalog_dirs.txt')


def sanitize_folder_name(name):
    safe_name = name.replace('&', 'and')
    return re.sub(r'[^\w\-_\.]', '_', safe_name).strip('_')


def read_catalog_dirs(path=CATALOG_DIRS):
    """Data roots listed in data/catalog_dirs.txt ([] if it is not there)."""
    out = []
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln and not ln.startswith('#'):
                    out.append(ln.rstrip('/'))
    except OSError:
        pass
    return out


def resolve_roots(args, console):
    """Every root to SEARCH, with --parent-dir first (it is the write target)."""
    roots = [args.parent_dir]
    extra = args.dirs if args.dirs else read_catalog_dirs()
    src = '--dirs' if args.dirs else CATALOG_DIRS
    for d in extra:
        if d not in roots:
            roots.append(d)
    if len(roots) == 1:
        console.print(
            f"[yellow][!] scanning only {args.parent_dir}. Cells that live on "
            f"another drive will look missing and be re-harvested in full -- "
            f"pass --dirs with every data root.[/yellow]")
    else:
        console.print(f"[cyan]Searching {len(roots)} data roots "
                      f"(from {src}):[/cyan] {', '.join(roots)}")
    return roots


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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv_file',
                        type=str,
                        help="Path to your grid CSV file")
    parser.add_argument('--parent-dir',
                        type=str,
                        default='grid_runs',
                        help="Where a cell that exists NOWHERE gets harvested "
                        "to; also searched")
    parser.add_argument('--dirs',
                        nargs='+',
                        default=None,
                        help="EVERY grid_runs root that may hold a cell "
                        f"(default: the roots in {CATALOG_DIRS}). Omitting "
                        "one makes its cells look missing.")
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help="Optional separate image directory to put in the generated "
        "commands (omitted when unset)")
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

    roots = resolve_roots(args, console)
    missing_details = []

    with console.status(
            "[bold yellow]Scanning for missing .completed_ or .empty_ markers...[/bold yellow]"
    ):
        for row in df.iter_rows(named=True):
            index = row['row_index']

            raw_id = f"{row['region']}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"
            safe_id = sanitize_folder_name(raw_id)
            # The cell may live on any root; markers are only ever written next
            # to the data, so completeness is their UNION across roots.
            region_dirs = [
                os.path.join(r, safe_id) for r in roots
                if os.path.isdir(os.path.join(r, safe_id))
            ]

            if not region_dirs:
                missing_details.append(
                    (index, safe_id, ["ALL (Folder missing)"], args.parent_dir))
                continue

            # Top the cell up where it already is, not on the default drive.
            target_dir = os.path.dirname(region_dirs[0])

            expected_count = get_expected_subgrids(row['sw_lon'],
                                                   row['sw_lat'],
                                                   row['ne_lon'],
                                                   row['ne_lat'],
                                                   args.sub_grid_step)

            missing_subs = []

            for i in range(expected_count):
                sub_id = f"{safe_id}_sub_{i}"

                done = any(
                    os.path.exists(os.path.join(rd, f'.completed_{sub_id}'))
                    or os.path.exists(os.path.join(rd, f'.empty_{sub_id}'))
                    for rd in region_dirs)

                if not done:
                    missing_subs.append(f"sub_{i}")

            if missing_subs:
                missing_details.append(
                    (index, safe_id, missing_subs, target_dir))

    if not missing_details:
        console.print(
            f"\n[bold green][\u2713] All {df.height} scanned regions have 100% of their sub-grid markers![/bold green]"
        )
        return

    console.print(
        f"\n[bold red][!] Found {len(missing_details)} regions missing one or more sub-grid markers.[/bold red]\n"
    )

    def base_cmd(parent_dir):
        return (
            f'python batch_chunks_mp_api.py "{args.csv_file}" '
            '--outer-max-workers 1 --search-max-workers 150 '
            '--entity-max-workers 520 '
            '--api-chunk-size 5000 --parquet-chunk-size 100000 '
            f'--parent-dir "{parent_dir}" '
            + (f'--image-dir "{args.image_dir}" ' if args.image_dir else '')
            + '--no-download-images')

    commands = []

    for idx, reg_id, subs, parent_dir in missing_details:
        cmd = f'{base_cmd(parent_dir)} --row-index {idx}'
        if "ALL (Folder missing)" not in subs:
            sub_nums = [s.replace('sub_', '') for s in subs]
            # NOTE: no --token here. batch_chunks' --token takes an integer key
            # index; a bare flag makes argparse exit 2 before any work starts.
            cmd += f' --sub-indices {",".join(sub_nums)}'

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
