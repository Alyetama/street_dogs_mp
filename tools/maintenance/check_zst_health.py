import argparse
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TARGET_DIR = "grid_runs"


def check_file(file_path):
    """
    Runs 'zstd -t' on the file to test its integrity.
    Returns the file_path if corrupted, otherwise returns None.
    """
    try:
        result = subprocess.run(
            ['zstd', '-t', '-q', str(file_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return file_path
    except Exception:
        return file_path
    return None


def main():
    default_workers = os.cpu_count() or 4

    parser = argparse.ArgumentParser(
        description="Concurrently scan and manage corrupted .zst files.")
    parser.add_argument(
        '-d',
        '--delete-all',
        action='store_true',
        help="Automatically delete all corrupted files without prompting.")
    parser.add_argument(
        '-w',
        '--workers',
        type=int,
        default=default_workers,
        help=
        f"Number of concurrent workers (default: {default_workers} based on your CPU count)."
    )
    parser.add_argument(
        '-s',
        '--substring',
        type=str,
        default="",
        help=
        "Only check files whose path contains this substring (e.g., 'Southeast_Asia'). Default is all files."
    )
    parser.add_argument(
        '-i',
        '--ignore-recent',
        type=float,
        default=0.0,
        help=
        "Ignore files modified within the last X hours (e.g., 1.5 to ignore files changed in the last 1.5 hours)."
    )
    parser.add_argument(
        '-c',
        '--clear-completed',
        action='store_true',
        help=
        "Delete the corresponding .completed_<sub_id> marker file if a corrupted sub-grid file is found."
    )
    parser.add_argument(
        '-e',
        '--exclude-ext',
        type=str,
        nargs='*',
        default=[],
        help=
        "Exclude files ending with specific sub-extensions (e.g., '.json.zst' '.jsonl.zst'). Can specify multiple."
    )

    args = parser.parse_args()

    target_path = Path(TARGET_DIR)
    if not target_path.is_dir():
        print(f"Error: Directory '{TARGET_DIR}' not found in current path.")
        return

    current_time = time.time()
    cutoff_time = current_time - (args.ignore_recent * 3600)

    zst_files = []

    for f in target_path.rglob("*.zst"):
        if args.substring and args.substring not in str(f):
            continue

        if args.ignore_recent > 0 and f.stat().st_mtime >= cutoff_time:
            continue

        if args.exclude_ext and any(
                str(f).endswith(ext) for ext in args.exclude_ext):
            continue

        zst_files.append(f)

    total_files = len(zst_files)

    if total_files == 0:
        print(f"No .zst files found matching the criteria in '{TARGET_DIR}'.")
        return

    print(f"Found {total_files} .zst files matching criteria.")
    print(f"Starting concurrent scan with {args.workers} workers...")

    corrupted_files = []
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {executor.submit(check_file, f): f for f in zst_files}

        for future in as_completed(future_to_file):
            processed += 1
            result = future.result()

            if result:
                corrupted_files.append(result)
                print(f"\r\033[K[!] Corrupted file detected: {result}")

            print(f"Scanning: {processed}/{total_files} files...",
                  end="\r",
                  flush=True)

    print(f"\n\nScan complete. Found {len(corrupted_files)} corrupted files.")

    if not corrupted_files:
        print("All files are healthy!")
        return

    print("-" * 48)

    # --- UPDATED MARKER DELETION LOGIC ---
    if args.clear_completed:
        removed_count = 0
        deleted_markers = set()

        for file_path in corrupted_files:
            try:
                # Extract the sub_id number from the corrupted file name
                match = re.search(r'_sub_(\d+)', file_path.name)
                if match:
                    region_name = file_path.parent.name
                    sub_id = f"{region_name}_sub_{match.group(1)}"
                    marker_file = file_path.parent / f".completed_{sub_id}"

                    # Ensure we don't try to delete the same marker twice if multiple files in the sub-grid are corrupted
                    if marker_file.exists(
                    ) and marker_file not in deleted_markers:
                        os.remove(marker_file)
                        deleted_markers.add(marker_file)
                        removed_count += 1
                        print(
                            f"[-] Removed completion marker: {marker_file.name}"
                        )
            except Exception as e:
                pass

        if removed_count == 0:
            print("[-] No completion markers needed to be removed.")

        print("-" * 48)
    # -------------------------------------

    delete_all = args.delete_all

    for file_path in corrupted_files:
        if delete_all:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
            continue

        while True:
            choice = input(f"Delete '{file_path}'? [y/n/A (yes to all)]: "
                           ).strip().lower()

            if choice in ('y', 'yes'):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
                break
            elif choice in ('a', 'all'):
                delete_all = True
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
                break
            elif choice in ('n', 'no', ''):
                print(f"Skipped: {file_path}")
                break
            else:
                print(
                    "Invalid input. Please answer y (yes), n (no), or a (yes to all)."
                )


if __name__ == "__main__":
    main()
