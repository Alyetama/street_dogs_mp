import argparse
import compression.zstd as zstd
import glob
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import orjson
from rich.console import Console
from tqdm import tqdm

PARENT_DIR = "grid_runs"


def count_jsonl_lines(filepath):
    """Fast line counting for compressed zstd files."""
    if not os.path.exists(filepath):
        return 0
    count = 0
    try:
        with zstd.open(filepath, 'rb') as f:
            for line in f:
                if line.strip(): count += 1
    except Exception:
        pass
    return count


def audit_single_file(topo_path):
    """Pure CPU-bound worker function to analyze a single sub-grid."""
    parent_dir = os.path.dirname(topo_path)
    filename = os.path.basename(topo_path)

    sub_id = filename.replace('topology_checkpoint_',
                              '').replace('.json.zst', '')
    completed_marker = os.path.join(parent_dir, f'.completed_{sub_id}')

    if not os.path.exists(completed_marker):
        return None  # Already known to be incomplete

    try:
        with zstd.open(topo_path, 'rb') as f:
            topology_data = orjson.loads(f.read())
            expected_count = len(topology_data.keys())
    except Exception:
        expected_count = 0

    if expected_count == 0:
        return None

    meta_path = os.path.join(parent_dir,
                             f'metadata_checkpoint_{sub_id}.jsonl.zst')
    anim_path = os.path.join(
        parent_dir, f'animal_detections_checkpoint_{sub_id}.jsonl.zst')

    meta_count = count_jsonl_lines(meta_path)
    anim_count = count_jsonl_lines(anim_path)

    if meta_count < expected_count or anim_count < expected_count:
        return {
            'sub_id': sub_id,
            'expected': expected_count,
            'meta': meta_count,
            'anim': anim_count,
            'marker_path': completed_marker
        }

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Multiprocessed Auditor for Mapillary Data")
    parser.add_argument(
        '--workers',
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPU cores to use (Defaults to all available cores)")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help=
        "Flag to only print discrepancies WITHOUT deleting the .completed_ markers"
    )
    # ADDED: Substring filter
    parser.add_argument(
        '--substring',
        type=str,
        default="",
        help="Optional substring to filter regions (e.g., 'North_America')")
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[cyan]Scanning {PARENT_DIR} for silently skipped data...[/cyan]")
    console.print(
        f"[cyan]Firing up [bold]{args.workers}[/bold] parallel CPU workers![/cyan]"
    )

    if args.dry_run:
        console.print(
            "[yellow][!] DRY RUN ENABLED: No markers will be deleted.[/yellow]"
        )

    topology_files = glob.glob(
        os.path.join(PARENT_DIR, '*', 'topology_checkpoint_*.json.zst'))

    # ADDED: Filter the file list based on the substring
    if args.substring:
        topology_files = [f for f in topology_files if args.substring in f]
        console.print(
            f"[cyan]Filtering for regions containing: '{args.substring}'[/cyan]"
        )

    if not topology_files:
        console.print(
            "[bold red]No topology checkpoints found matching your criteria![/bold red]"
        )
        return

    corrupted_or_missing = 0

    # Execute the heavy parsing across all CPU cores
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(audit_single_file, topo_path): topo_path
            for topo_path in topology_files
        }

        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Auditing Checkpoints"):
            result = future.result()

            # If the worker returned data, it found a discrepancy!
            if result:
                sub_id = result['sub_id']
                expected = result['expected']
                meta = result['meta']
                anim = result['anim']

                missing_meta = expected - meta
                missing_anim = expected - anim

                tqdm.write(f"\n[!] Discrepancy found in {sub_id}")
                tqdm.write(
                    f"    -> Expected: {expected:,} | "
                    f"Metadata: {meta:,} \033[1;31m(-{missing_meta:,})\033[0m | "
                    f"Detections: {anim:,} \033[1;31m(-{missing_anim:,})\033[0m"
                )
                tqdm.write(f"    {result['marker_path']}")

                if not args.dry_run:
                    try:
                        os.remove(result['marker_path'])
                        tqdm.write(
                            f"    [-] Deleted .completed_ marker to force a backfill rerun."
                        )
                    except OSError as e:
                        tqdm.write(f"    [X] Failed to delete marker: {e}")

                corrupted_or_missing += 1

    # --- Print Summary ---
    if corrupted_or_missing == 0:
        console.print(
            "\n[bold green][\u2713] Audit complete. All points are intact![/bold green]"
        )
    else:
        action = "require backfilling" if not args.dry_run else "would require backfilling (Dry Run)"
        console.print(
            f"\n[bold yellow][!] Audit complete. Found {corrupted_or_missing} regions that {action}.[/bold yellow]"
        )
        if not args.dry_run:
            console.print(
                "[i] Run `get_rerun_indexes.py` to get your CSV indexes, then rerun your main script.[/i]"
            )


if __name__ == "__main__":
    main()
