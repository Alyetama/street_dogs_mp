"""Drop duplicate image_id rows from the ground_animals manifest chunks.

Every rewrite is atomic: the deduplicated copy is written to ``<name>.tmp``,
fsynced, row-counted, and only then ``os.replace``d over the original. A kill,
a power cut or a full disk therefore leaves either the untouched original or a
complete replacement -- never a truncated manifest. (An in-place
``write_parquet(filepath)`` truncates the target the moment it opens it, so an
interrupted run used to destroy the whole chunk, unique rows included.)

    python tools/repair/deduplicate_parquets.py --parent-dir grid_runs --dry-run
    python tools/repair/deduplicate_parquets.py --parent-dir grid_runs
"""

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


def durable_replace(df, final):
    """Write ``df`` beside ``final`` and atomically swap it in.

    The temp name ends in ``.tmp`` so it can never be picked up by a
    ``ground_animals_*.parquet`` glob, and it is removed if anything fails, so
    an aborted run leaves no partial file behind either.
    """
    tmp = final + '.tmp'
    try:
        df.write_parquet(tmp, compression='zstd')
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        # Read the footer back: refuse to swap in a file that did not land with
        # exactly the rows we meant to keep.
        written = pl.scan_parquet(tmp).select(pl.len()).collect().item()
        if written != df.height:
            raise IOError(
                f'wrote {written} rows, expected {df.height} -- not swapping')
        os.replace(tmp, final)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    dfd = os.open(os.path.dirname(final) or '.', os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def deduplicate_single_parquet(filepath, dry_run=False):
    """Worker function to read, deduplicate, and rewrite a single Parquet file."""
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

        # Only spend time rewriting the file if duplicates actually existed
        if duplicates_removed > 0 and not dry_run:
            assert new_count + duplicates_removed == original_count, filepath
            durable_replace(df_unique, filepath)

        return region_name, original_count, duplicates_removed

    except Exception as e:
        return None, -1, f"Error processing {filepath}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Count the duplicates without rewriting a single file")
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
                executor.submit(deduplicate_single_parquet, filepath,
                                args.dry_run): filepath
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
    console.print("\n[bold green][\u2713] Deduplication "
                  f"{'Dry Run' if args.dry_run else 'Complete'}![/bold green]")

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
    verb = 'Would purge' if args.dry_run else 'Purged'
    console.print(f"\n[bold red]Grand Total: {verb} "
                  f"{total_duplicates_purged:,} duplicate rows from your "
                  f"dataset![/bold red]")
    if args.dry_run:
        console.print(
            "[yellow]DRY RUN: nothing was written. Re-run without --dry-run "
            "to apply.[/yellow]")


if __name__ == "__main__":
    main()
