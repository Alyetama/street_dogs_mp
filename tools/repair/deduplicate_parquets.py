import argparse
import glob
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
from rich.console import Console
from rich.table import Table
from tqdm import tqdm


def deduplicate_single_parquet(filepath):
    """Worker function to read, deduplicate, and overwrite a single Parquet file."""
    try:
        # Extract the region folder name for aggregation logging
        region_name = os.path.basename(os.path.dirname(filepath))

        df = pl.read_parquet(filepath)
        original_count = df.height

        if original_count == 0:
            return region_name, 0, 0

        # Deduplicate based on 'image_id' (keeping the first occurrence)
        df_unique = df.unique(subset=['image_id'], keep='first')
        new_count = df_unique.height
        duplicates_removed = original_count - new_count

        # Only spend time overwriting the file if duplicates actually existed
        if duplicates_removed > 0:
            # Overwrite the file natively with zstd compression matching the main script
            df_unique.write_parquet(filepath, compression='zstd')

        return region_name, original_count, duplicates_removed

    except Exception as e:
        return None, -1, f"Error processing {filepath}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Scan and deduplicate ground_animal Parquet files.")
    parser.add_argument('--parent-dir',
                        type=str,
                        default='grid_runs',
                        help="Base directory containing the runs")
    parser.add_argument(
        '--substring',
        type=str,
        default="",
        help="Optional substring to filter regions (e.g., 'North_America')")
    parser.add_argument('--workers',
                        type=int,
                        default=multiprocessing.cpu_count(),
                        help="Number of CPU cores to use")
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[cyan]Scanning {args.parent_dir} for ground_animals Parquet files...[/cyan]"
    )

    # Find all ground_animals parquet files across all subdirectories
    search_pattern = os.path.join(args.parent_dir, '*',
                                  'ground_animals_*.parquet')
    all_files = glob.glob(search_pattern)

    # Filter by substring if one was provided
    if args.substring:
        safe_substring = args.substring.replace(" ", "_")
        target_files = [f for f in all_files if safe_substring in f]
        console.print(
            f"[cyan]Filtering for '{args.substring}' -> Found {len(target_files)} files.[/cyan]"
        )
    else:
        target_files = all_files
        console.print(f"[cyan]Found {len(target_files)} total files.[/cyan]")

    if not target_files:
        console.print(
            "[bold red][X] No parquet files found matching your criteria![/bold red]"
        )
        return

    # Trackers for the final report
    region_stats = defaultdict(lambda: {
        'total_rows': 0,
        'duplicates_removed': 0,
        'files_touched': 0
    })
    total_duplicates_purged = 0

    with console.status(
            f"[bold yellow]Firing up {args.workers} CPU cores for deduplication...[/bold yellow]"
    ):
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(deduplicate_single_parquet, filepath): filepath
                for filepath in target_files
            }

            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Processing Parquets"):
                region_name, original_count, duplicates = future.result()

                if original_count == -1:
                    tqdm.write(f"[bold red]{duplicates}[/bold red]"
                               )  # This prints the error string
                    continue

                region_stats[region_name]['total_rows'] += original_count

                if duplicates > 0:
                    region_stats[region_name][
                        'duplicates_removed'] += duplicates
                    region_stats[region_name]['files_touched'] += 1
                    total_duplicates_purged += duplicates

    # --- Print the Final Report ---
    console.print(
        "\n[bold green][\u2713] Deduplication Complete![/bold green]")

    if total_duplicates_purged == 0:
        console.print(
            "[bold cyan]All files are already 100% clean. Zero duplicates found.[/bold cyan]"
        )
        return

    # Create a nice summary table using Rich
    table = Table(title="Deduplication Summary by Region",
                  title_style="bold magenta")
    table.add_column("Region Name", style="cyan", no_wrap=True)
    table.add_column("Total Rows Scanned", justify="right", style="white")
    table.add_column("Duplicates Purged", justify="right", style="bold red")
    table.add_column("Affected Files", justify="right", style="yellow")

    # Sort regions alphabetically for clean reading
    for region in sorted(region_stats.keys()):
        stats = region_stats[region]
        if stats['duplicates_removed'] > 0:
            table.add_row(region, f"{stats['total_rows']:,}",
                          f"-{stats['duplicates_removed']:,}",
                          f"{stats['files_touched']} files")

    console.print(table)
    console.print(
        f"\n[bold red]Grand Total: Purged {total_duplicates_purged:,} duplicate rows from your dataset![/bold red]"
    )


if __name__ == "__main__":
    main()
