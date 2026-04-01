import gzip
import os
import shutil

from tqdm import tqdm

PARENT_DIR = 'grid_runs'


def compress_existing_files():
    if not os.path.exists(PARENT_DIR):
        print(f"Directory '{PARENT_DIR}' not found. Nothing to compress.")
        return

    target_extensions = ('.json', '.jsonl', '.csv')
    files_to_compress = []

    print("Scanning for uncompressed files...")

    for root, dirs, files in os.walk(PARENT_DIR):
        for file in files:
            if file.endswith(target_extensions) and not file.endswith('.gz'):
                files_to_compress.append(os.path.join(root, file))

    if not files_to_compress:
        print("No uncompressed files found! You are good to go.")
        return

    print(f"Found {len(files_to_compress)} files. Starting compression...")
    files_compressed = 0

    for original_path in tqdm(files_to_compress,
                              desc="Compressing",
                              unit="file"):
        gz_path = original_path + '.gz'

        try:
            with open(original_path, 'rb') as f_in:
                with gzip.open(gz_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(original_path)
            files_compressed += 1
        except Exception as e:
            tqdm.write(
                f"Failed to compress {os.path.basename(original_path)}: {e}")

    print(f"\nDone! Successfully compressed {files_compressed} files.")


if __name__ == "__main__":
    compress_existing_files()
