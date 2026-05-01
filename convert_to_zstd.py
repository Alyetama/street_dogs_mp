import argparse
import compression.zstd as zstd
import gzip
import os
import re

import pandas as pd
from tqdm import tqdm


def sanitize_folder_name(name):
    """Matches the exact folder naming logic from the main pipeline."""
    safe_name = name.replace('&', 'and')
    return re.sub(r'[^\w\-_\.]', '_', safe_name).strip('_')


def get_target_files(grid_csv, parent_dirs):
    """Parses the CSV and extracts .json.gz / .jsonl.gz files strictly from valid region folders."""
    df = pd.read_csv(grid_csv)
    target_files = []

    for _, row in df.iterrows():
        overall_region = row['region']
        unique_region_id = f"{overall_region}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"
        safe_region_id = sanitize_folder_name(unique_region_id)

        for p_dir in parent_dirs:
            region_dir = os.path.join(p_dir, safe_region_id)
            if os.path.exists(region_dir):
                try:
                    for file in os.listdir(region_dir):
                        if file.endswith('.json.gz') or file.endswith(
                                '.jsonl.gz'):
                            target_files.append(os.path.join(region_dir, file))
                except OSError:
                    pass

    return target_files


def process_file(gz_filepath, do_compare, delete_gz, chunk_size, overwrite,
                 workers):
    """
    Handles Conversion, Comparison, and Deletion as a continuous pipeline.
    """
    zst_filepath = gz_filepath[:-3] + '.zst'
    status_log = []

    # ==========================================
    # 1. CONVERSION PHASE
    # ==========================================
    is_existing = os.path.exists(zst_filepath)
    if not is_existing or overwrite:
        temp_filepath = zst_filepath + ".tmp"
        try:
            options = {zstd.CompressionParameter.nb_workers: workers}

            with gzip.open(gz_filepath,
                           'rb') as f_in, zstd.open(temp_filepath,
                                                    'wb',
                                                    options=options) as f_out:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)

            os.replace(temp_filepath, zst_filepath)
            status_log.append("Overwritten" if is_existing else "Converted")
        except Exception as e:
            if os.path.exists(temp_filepath): os.remove(temp_filepath)
            return f"Error converting: {e}"
    else:
        status_log.append("Skipped (Already ZST)")

    # ==========================================
    # 2. VERIFICATION PHASE (If --compare is used)
    # ==========================================
    if do_compare:
        is_match = True
        try:
            with gzip.open(gz_filepath,
                           'rb') as f_gz, zstd.open(zst_filepath,
                                                    'rb') as f_zst:
                while True:
                    chunk_gz = f_gz.read(chunk_size)
                    chunk_zst = f_zst.read(chunk_size)

                    if chunk_gz != chunk_zst:
                        is_match = False
                        break

                    if not chunk_gz:
                        break
        except Exception as e:
            return f"{' | '.join(status_log)} | Error comparing: {e}"

        if is_match:
            status_log.append("Verified Exact Match")
        else:
            return f"{' | '.join(status_log)} | MISMATCH DETECTED (Corrupted ZST!)"

    # ==========================================
    # 3. DELETION PHASE (If --delete-gz is used)
    # ==========================================
    if delete_gz:
        if not os.path.exists(zst_filepath):
            return f"{' | '.join(status_log)} | ERROR: Aborted Deletion! ZST file missing."
        if os.path.getsize(zst_filepath) == 0:
            return f"{' | '.join(status_log)} | ERROR: Aborted Deletion! ZST file is 0 bytes."

        try:
            os.remove(gz_filepath)
            status_log.append("Deleted GZ")
        except OSError as e:
            status_log.append(
                f"{' | '.join(status_log)} | Failed to delete GZ: {e}")

    return " | ".join(status_log)


def main():
    parser = argparse.ArgumentParser(
        description=
        "Safely convert and verify Mapillary .gz checkpoint files to .zst format sequentially."
    )
    parser.add_argument('grid_csv_file',
                        type=str,
                        help="Path to the global_grid_5deg.csv file.")
    parser.add_argument(
        '--parent-dirs',
        type=str,
        nargs='+',
        default=['grid_runs'],
        help=
        "One or more parent directories (e.g. grid_runs grid_runs_anvil_pear)")

    parser.add_argument(
        '--workers',
        type=int,
        default=os.cpu_count(),
        help=
        "Number of internal CPU threads for Zstandard compression (defaults to all cores)."
    )
    parser.add_argument('--ram-gb',
                        type=float,
                        default=8.0,
                        help="Maximum total RAM to use in GB (default: 8.0)")

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="Overwrite existing .zst files instead of skipping them.")
    parser.add_argument(
        '--compare',
        action='store_true',
        help=
        "Verify the decompressed stream of the .zst file perfectly matches the .gz file. (Runs after conversion if file doesn't exist)."
    )
    parser.add_argument(
        '--delete-gz',
        action='store_true',
        help=
        "Delete the original .gz file after processing. If combined with --compare, it only deletes if mathematically verified."
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable detailed terminal output for every file processed.")

    args = parser.parse_args()

    if not os.path.exists(args.grid_csv_file):
        print(f"[!] CSV File '{args.grid_csv_file}' does not exist.")
        return

    mode_desc = "Conversion Pipeline"
    if args.compare: mode_desc += " + Strict Verification"
    if args.delete_gz: mode_desc += " + Deletion"

    chunk_size = int((args.ram_gb * 1024 * 1024 * 1024) / 3)
    chunk_mb = chunk_size / (1024 * 1024)

    print(f"Mode: {mode_desc}")
    print(f"Zstd Threads: {args.workers}")
    print(
        f"RAM Limit: {args.ram_gb} GB -> Allocated {chunk_mb:.1f} MB chunks.")
    print(
        f"Parsing '{args.grid_csv_file}' and scanning directories: {', '.join(args.parent_dirs)}..."
    )

    files_to_convert = get_target_files(args.grid_csv_file, args.parent_dirs)

    if not files_to_convert:
        print(
            "No matching .json.gz or .jsonl.gz files found in the specified regions. Exiting."
        )
        return

    print(
        f"Found {len(files_to_convert)} files. Starting sequential processing...\n"
    )

    counts = {
        'converted': 0,
        'overwritten': 0,
        'verified': 0,
        'deleted': 0,
        'skipped': 0,
        'errors': 0,
        'mismatch': 0
    }

    with tqdm(total=len(files_to_convert), desc="Total Progress",
              unit="file") as pbar:
        for filepath in files_to_convert:
            filename = os.path.basename(filepath)
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

            if args.verbose:
                tqdm.write(
                    f"\n[Target] {filename} (Size: {file_size_mb:.2f} MB)")

            result = process_file(filepath, args.compare, args.delete_gz,
                                  chunk_size, args.overwrite, args.workers)

            if "Error" in result or "ERROR" in result:
                counts['errors'] += 1
                if args.verbose: tqdm.write(f"   └── [!] {result}")
            elif "MISMATCH" in result:
                counts['mismatch'] += 1
                if args.verbose: tqdm.write(f"   └── [X] {result}")
            elif "Skipped" in result:
                counts['skipped'] += 1
                if args.verbose: tqdm.write(f"   └── [-] {result}")
            else:
                if "Converted" in result: counts['converted'] += 1
                if "Overwritten" in result: counts['overwritten'] += 1
                if "Verified Exact Match" in result: counts['verified'] += 1
                if "Deleted GZ" in result: counts['deleted'] += 1
                if args.verbose: tqdm.write(f"   └── [\u2713] {result}")

            pbar.update(1)

    print("\n" + "=" * 45)
    print("Execution Summary:")
    print(f"  Files Converted:        {counts['converted']}")
    print(f"  Files Overwritten:      {counts['overwritten']}")
    if args.compare:
        print(f"  Files Verified Match:   {counts['verified']}")
    if args.delete_gz:
        print(f"  Original .gz Deleted:   {counts['deleted']}")
    print(f"  Skipped:                {counts['skipped']}")
    if args.compare:
        print(f"  Data Mismatches:        {counts['mismatch']}")
    print(f"  System Errors:          {counts['errors']}")
    print("=" * 45)


if __name__ == "__main__":
    main()
