import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TARGET_DIR = "grid_runs"


def check_file(file_path):
    """
    Runs 'gzip -t' on the file.
    Returns the file_path if corrupted, otherwise returns None.
    """
    try:
        result = subprocess.run(['gzip', '-t', str(file_path)],
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
        description="Concurrently scan and manage corrupted .gz files.")
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
        "Only check files whose path contains this substring (e.g., 'Pacific_Ocean'). Default is all files."
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
        "Remove the region entry from completed_regions.txt if any corrupted file is found within it."
    )

    args = parser.parse_args()

    target_path = Path(TARGET_DIR)
    if not target_path.is_dir():
        print(f"Error: Directory '{TARGET_DIR}' not found in current path.")
        return

    current_time = time.time()
    cutoff_time = current_time - (args.ignore_recent * 3600)

    gz_files = []

    for f in target_path.rglob("*.gz"):
        if args.substring and args.substring not in str(f):
            continue

        if args.ignore_recent > 0 and f.stat().st_mtime >= cutoff_time:
            continue

        gz_files.append(f)

    total_files = len(gz_files)

    if total_files == 0:
        print(f"No .gz files found matching the criteria in '{TARGET_DIR}'.")
        return

    print(f"Found {total_files} .gz files matching criteria.")
    print(f"Starting concurrent scan with {args.workers} workers...")

    corrupted_files = []
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {executor.submit(check_file, f): f for f in gz_files}

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

    if args.clear_completed:
        corrupted_regions = set()
        for file_path in corrupted_files:
            try:
                region_name = file_path.relative_to(target_path).parts[0]
                corrupted_regions.add(region_name)
            except ValueError:
                pass

        if corrupted_regions:
            completed_txt = target_path / "completed_regions.txt"
            if completed_txt.exists():
                with open(completed_txt, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                removed_count = 0

                for line in lines:
                    region_entry = line.strip()

                    normalized_entry = region_entry.replace(" ", "_")

                    if normalized_entry in corrupted_regions:
                        removed_count += 1
                        print(
                            f"[-] Removed from completed list: {region_entry}")
                    else:
                        new_lines.append(line)

                if removed_count > 0:
                    with open(completed_txt, 'w') as f:
                        f.writelines(new_lines)
            else:
                print(
                    f"[!] Warning: {completed_txt} not found. Cannot update completed regions."
                )

        print("-" * 48)

    delete_all = args.delete_all

    for file_path in corrupted_files:
        if delete_all:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            continue

        while True:
            choice = input(f"Delete '{file_path}'? [y/n/A (yes to all)]: "
                           ).strip().lower()

            if choice in ('y', 'yes'):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                break
            elif choice in ('a', 'all'):
                delete_all = True
                os.remove(file_path)
                print(f"Deleted: {file_path}")
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
