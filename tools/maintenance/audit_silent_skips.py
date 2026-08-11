"""Find sub-grids that carry a .completed_ marker but not the data it claims.

Driven by the MARKERS, not by the topology checkpoints: the marker is what makes
batch_chunks skip the sub-grid forever, so every marker has to be accounted for.
A sub-grid whose topology checkpoint is missing or unreadable cannot be checked
at all -- it is reported as UNCHECKABLE rather than counted as healthy (its
marker is never deleted automatically; the data next to it is usually intact and
a forced re-harvest would be the more expensive mistake).
"""

import argparse
import compression.zstd as zstd
import glob
import multiprocessing
import os
from collections import defaultdict
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


def audit_single_file(marker_path):
    """Pure CPU-bound worker: analyse the sub-grid behind one .completed_ marker.

    Returns None when the sub-grid checks out, or a dict whose ``kind`` is
    'discrepancy' (the data is short of what the topology expected) or
    'uncheckable' (no readable topology, so nothing can be said about it).
    """
    parent_dir = os.path.dirname(marker_path)
    sub_id = os.path.basename(marker_path)[len('.completed_'):]
    topo_path = os.path.join(parent_dir,
                             f'topology_checkpoint_{sub_id}.json.zst')

    if not os.path.exists(topo_path):
        return {
            'kind': 'uncheckable',
            'sub_id': sub_id,
            'why': 'topology checkpoint missing',
            'marker_path': marker_path
        }

    try:
        with zstd.open(topo_path, 'rb') as f:
            topology_data = orjson.loads(f.read())
            expected_count = len(topology_data.keys())
    except Exception as e:
        # NOT the same thing as "nothing was expected here": swallowing this
        # into expected_count = 0 passed the total-loss case as healthy.
        return {
            'kind': 'uncheckable',
            'sub_id': sub_id,
            'why': f'topology unreadable ({type(e).__name__})',
            'marker_path': marker_path
        }

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
            'kind': 'discrepancy',
            'sub_id': sub_id,
            'expected': expected_count,
            'meta': meta_count,
            'anim': anim_count,
            'marker_path': marker_path
        }

    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--dirs',
        nargs='+',
        default=[PARENT_DIR],
        help=f"grid_runs roots to audit (default: {PARENT_DIR}). Cells on a "
        "root you leave out are not audited and not claimed to be healthy.")
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
    console.print(f"[cyan]Scanning {', '.join(args.dirs)} for silently "
                  f"skipped data...[/cyan]")
    console.print(
        f"[cyan]Firing up [bold]{args.workers}[/bold] parallel CPU workers![/cyan]"
    )

    if args.dry_run:
        console.print(
            "[yellow][!] DRY RUN ENABLED: No markers will be deleted.[/yellow]"
        )

    # Enumerate the MARKERS: a marker with no topology checkpoint next to it is
    # exactly the case a topology-driven glob could never see.
    markers = []
    for root in args.dirs:
        markers += glob.glob(os.path.join(root, '*', '.completed_*'))

    # ADDED: Filter the file list based on the substring
    if args.substring:
        markers = [f for f in markers if args.substring in f]
        console.print(
            f"[cyan]Filtering for regions containing: '{args.substring}'[/cyan]"
        )

    if not markers:
        console.print(
            "[bold red]No .completed_ markers found matching your criteria![/bold red]"
        )
        return

    flagged_subs = []
    uncheckable = []

    # Execute the heavy parsing across all CPU cores
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(audit_single_file, marker): marker
            for marker in markers
        }

        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Auditing sub-grids"):
            result = future.result()
            if not result:
                continue

            if result['kind'] == 'uncheckable':
                uncheckable.append(result)
                continue

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
                f"Detections: {anim:,} \033[1;31m(-{missing_anim:,})\033[0m")
            tqdm.write(f"    {result['marker_path']}")

            if not args.dry_run:
                try:
                    os.remove(result['marker_path'])
                    tqdm.write(
                        f"    [-] Deleted .completed_ marker to force a backfill rerun."
                    )
                except OSError as e:
                    tqdm.write(f"    [X] Failed to delete marker: {e}")

            flagged_subs.append(result)

    # --- Print Summary ---
    # One region holds up to 25 sub-grids, so a count of flagged sub-grids is
    # NOT a count of regions. Report both, and never call a run clean while
    # something in it could not be checked.
    regions = defaultdict(int)
    for r in flagged_subs:
        regions[os.path.basename(os.path.dirname(r['marker_path']))] += 1

    console.print(f"\n[cyan]Audited {len(markers):,} completed sub-grids "
                  f"across {', '.join(args.dirs)}.[/cyan]")

    if flagged_subs:
        action = ("require backfilling" if not args.dry_run else
                  "would require backfilling (Dry Run)")
        console.print(
            f"[bold yellow][!] {len(flagged_subs)} sub-grid(s) in "
            f"{len(regions)} region(s) {action}.[/bold yellow]")
        for reg, n in sorted(regions.items(), key=lambda x: -x[1])[:20]:
            console.print(f"    {reg}  ({n} sub-grid(s))")
        if not args.dry_run:
            console.print(
                "[i] Run `get_rerun_indexes.py` to get your CSV indexes, then rerun your main script.[/i]"
            )

    if uncheckable:
        console.print(
            f"[bold red][?] {len(uncheckable)} completed sub-grid(s) could NOT "
            f"be checked -- no readable topology to compare against. Their "
            f"markers were left alone; look at them by hand.[/bold red]")
        for r in uncheckable[:20]:
            console.print(f"    {r['sub_id']}  ({r['why']})")
        if len(uncheckable) > 20:
            console.print(f"    ... and {len(uncheckable) - 20} more")

    if not flagged_subs and not uncheckable:
        console.print(
            "\n[bold green][\u2713] Audit complete. All points are intact![/bold green]"
        )


if __name__ == "__main__":
    main()
