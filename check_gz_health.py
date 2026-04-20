import argparse
import os
import subprocess
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

    args = parser.parse_args()

    target_path = Path(TARGET_DIR)
    if not target_path.is_dir():
        print(f"Error: Directory '{TARGET_DIR}' not found in current path.")
        return

    gz_files = list(target_path.rglob("*.gz"))
    total_files = len(gz_files)

    if total_files == 0:
        print(f"No .gz files found in '{TARGET_DIR}'.")
        return

    print(f"Found {total_files} .gz files.")
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
