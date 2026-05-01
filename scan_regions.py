import argparse
import os
from collections import Counter, defaultdict


def has_data_file(region_path):
    """Scans maxdepth 1 for data files. Exits instantly on first match."""
    valid_exts = ('.parquet', '.zst', '.csv', '.gz')
    try:
        for entry in os.scandir(region_path):
            if entry.is_file() and entry.name.endswith(valid_exts):
                return True
    except OSError:
        pass
    return False


def has_image_file(region_path):
    """Recursively scans for image files. Exits instantly on first match."""
    valid_exts = ('.jpg', '.jpeg', '.png')
    try:
        for entry in os.scandir(region_path):
            if entry.is_file() and entry.name.lower().endswith(valid_exts):
                return True
            elif entry.is_dir():
                if has_image_file(entry.path):
                    return True
    except OSError:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description=
        "Scan directories for region data/images and recommend script flags.")
    parser.add_argument(
        "prefix",
        type=str,
        help="The region prefix pattern to search for (e.g., South_America)")
    parser.add_argument(
        "--dirs",
        "-d",
        type=str,
        nargs="+",
        required=True,
        help="Space-separated list of base directories to search in")

    args = parser.parse_args()

    prefix = args.prefix
    search_dirs = args.dirs

    region_data_locs = defaultdict(list)
    region_image_locs = defaultdict(list)
    all_regions = set()

    for base_dir in search_dirs:
        base_dir = base_dir.rstrip('/')

        if not os.path.exists(base_dir):
            print(
                f"[Warning] Directory does not exist and will be skipped: {base_dir}"
            )
            continue

        try:
            for entry in os.scandir(base_dir):
                if entry.is_dir() and entry.name.startswith(prefix):
                    region = entry.name
                    all_regions.add(region)

                    if has_data_file(entry.path):
                        region_data_locs[region].append(base_dir)
                    if has_image_file(entry.path):
                        region_image_locs[region].append(base_dir)
        except OSError:
            continue

    if not all_regions:
        print(
            f"\nNo regions found starting with '{prefix}' in the provided directories."
        )
        return

    data_counts = Counter()
    image_counts = Counter()

    for locs in region_data_locs.values():
        for l in locs:
            data_counts[l] += 1

    for locs in region_image_locs.values():
        for l in locs:
            image_counts[l] += 1

    primary_data_dir = data_counts.most_common(
        1)[0][0] if data_counts else None
    primary_image_dir = image_counts.most_common(
        1)[0][0] if image_counts else None

    print(f"\n=======================================================")
    print(f" RECOMMENDED FLAGS FOR: {prefix}*")
    print(f"=======================================================")

    flags = []
    if primary_data_dir:
        flags.append(f'--parent-dir "{primary_data_dir}"')
    if primary_image_dir:
        flags.append(f'--image-dir "{primary_image_dir}"')

    if flags:
        print("\nPass these exact paths to your main script:\n")
        print(" " + " ".join(flags))
    else:
        print(
            "\nNo Data or Images were found inside any matching region folders."
        )

    print(f"\n=======================================================")
    print(f" DIRECTORY BREAKDOWN")
    print(f"=======================================================")

    if data_counts:
        print("\n[DATA LOCATIONS]")
        for d, c in data_counts.items():
            print(f"  -> {c} region(s) have data inside: {d}")

    if image_counts:
        print("\n[IMAGE LOCATIONS]")
        for d, c in image_counts.items():
            print(f"  -> {c} region(s) have images inside: {d}")

    print(f"\n=======================================================")
    print(f" OUTLIER DETECTION")
    print(f"=======================================================")

    outliers = []
    for region in sorted(all_regions):
        d_locs = region_data_locs.get(region, [])
        i_locs = region_image_locs.get(region, [])

        if len(d_locs) > 1:
            outliers.append(
                f"[!] DUPLICATE: {region} has Data files in multiple directories: {d_locs}"
            )
        if len(i_locs) > 1:
            outliers.append(
                f"[!] DUPLICATE: {region} has Image files in multiple directories: {i_locs}"
            )

        if d_locs and primary_data_dir not in d_locs:
            outliers.append(
                f"[!] SPLIT: {region} Data is in {d_locs[0]} (Differs from recommended --parent-dir)"
            )
        if i_locs and primary_image_dir not in i_locs:
            outliers.append(
                f"[!] SPLIT: {region} Images are in {i_locs[0]} (Differs from recommended --image-dir)"
            )

    if not outliers:
        print(
            "  \u2713 Clean. All regions perfectly align with the recommended flags."
        )
    else:
        for out in sorted(list(set(outliers))):
            print(out)

    print("")


if __name__ == "__main__":
    main()
