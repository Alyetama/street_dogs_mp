import argparse
import os

import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description=
        "Extract covered countries directly into region folders across multiple drives."
    )
    parser.add_argument(
        '--dirs',
        nargs='+',
        required=True,
        help="One or more base directories containing grid_runs folders")
    args = parser.parse_args()

    print("Loading country boundaries from Natural Earth...")
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    try:
        world = gpd.read_file(url)
    except Exception as e:
        print(f"Failed to load Natural Earth data: {e}")
        return

    name_col = 'ADMIN' if 'ADMIN' in world.columns else 'NAME' if 'NAME' in world.columns else 'name'

    processed_count = 0

    for base_dir in args.dirs:
        if not os.path.exists(base_dir):
            print(
                f"\n[!] Warning: Directory '{base_dir}' not found. Skipping..."
            )
            continue

        folders = [
            f for f in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, f))
        ]

        if not folders:
            continue

        for folder in tqdm(
                folders,
                desc=f"Scanning {os.path.basename(os.path.normpath(base_dir))}"
        ):
            folder_path = os.path.join(base_dir, folder)

            parts = folder.split('_')
            try:
                ne_lat = float(parts[-1])
                ne_lon = float(parts[-2])
                sw_lat = float(parts[-3])
                sw_lon = float(parts[-4])
            except (ValueError, IndexError):
                continue

            bounding_box = box(sw_lon, sw_lat, ne_lon, ne_lat)
            intersecting_countries = world[world.intersects(bounding_box)]

            country_names = list(set(
                intersecting_countries[name_col].tolist()))

            if not country_names:
                country_names = ["No countries found (Ocean/Sea)"]

            out_file = os.path.join(folder_path, 'covered_countries.txt')

            country_names.sort()
            with open(out_file, 'w', encoding='utf-8') as f:
                for country in country_names:
                    f.write(f"{country}\n")

            processed_count += 1

    print(
        f"\n[\u2713] Successfully generated country lists for {processed_count} regions across all drives."
    )


if __name__ == '__main__':
    main()
