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
            if entry.is_file():
                name_lower = entry.name.lower()
                if name_lower.endswith(
                        valid_exts) and not name_lower.endswith('_tiles.png'):
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
    print(f" PER-REGION LAYOUT  (data/images spanning drives is normal)")
    print(f"=======================================================\n")

    def drive_label(d):
        """Short, recognizable drive name from a full grid_runs path."""
        parts = d.rstrip('/').split('/')
        try:
            return parts[parts.index('biodiv') + 1]
        except (ValueError, IndexError):
            return os.path.basename(d.rstrip('/')) or d

    for region in sorted(all_regions):
        d = ', '.join(
            drive_label(x) for x in region_data_locs.get(region, [])) or '-'
        i = ', '.join(
            drive_label(x) for x in region_image_locs.get(region, [])) or '-'
        print(f"  {region:<46} data: {d:<24} images: {i}")

    # The one case still worth flagging: images present but NO data/manifest on
    # any drive (genuine orphan images). Data on one drive + images on another
    # is the normal multi-drive layout and is no longer treated as an outlier.
    orphans = [
        r for r in sorted(all_regions)
        if region_image_locs.get(r) and not region_data_locs.get(r)
    ]
    if orphans:
        print(f"\n  [!] {len(orphans)} region(s) have images but no manifest "
              f"on any drive:")
        for r in orphans:
            print(f"      - {r}")

    print("")


if __name__ == "__main__":
    main()
