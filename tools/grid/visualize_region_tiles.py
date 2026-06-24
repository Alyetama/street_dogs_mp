import argparse
import os

import contextily as cx
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mercantile
from global_land_mask import globe


def visualize_region_tiles(region_dir, zoom=14, parent_dir=None):
    """
    Generates a visualization of land/water tiles for a given region.

    Args:
        region_dir (str): Folder name containing coordinates (e.g., 'NZ_-160_-50_165_-45')
        zoom (int): Mercantile zoom level for tile calculation.
        parent_dir (str, optional): Parent folder to save the output. If None, saves to current directory.
    """
    # Parse coordinates from the directory string
    parts = region_dir.split('_')
    try:
        # Expected format ends in _lon_lat_lon_lat
        sw_lon, sw_lat, ne_lon, ne_lat = map(float, parts[-4:])
    except (ValueError, IndexError):
        print(f"Error: Could not parse coordinates from '{region_dir}'.")
        print("Ensure format is 'Name_SWLon_SWLat_NELon_NELat'")
        return

    tiles = list(mercantile.tiles(sw_lon, sw_lat, ne_lon, ne_lat, zoom))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(sw_lon, ne_lon)
    ax.set_ylim(sw_lat, ne_lat)

    land_count = 0
    water_count = 0

    for t in tiles:
        bbox = mercantile.bounds(t.x, t.y, t.z)

        # Check if the center is land
        center_lat_tile = (bbox.south + bbox.north) / 2.0
        center_lon_tile = (bbox.west + bbox.east) / 2.0
        is_land = globe.is_land(center_lat_tile, center_lon_tile)

        if is_land:
            facecolor = '#00FF00'  # Neon Green
            alpha = 0.3
            land_count += 1
        else:
            facecolor = '#FF0000'  # Bright Red
            alpha = 0.15
            water_count += 1

        rect = patches.Rectangle((bbox.west, bbox.south),
                                 bbox.east - bbox.west,
                                 bbox.north - bbox.south,
                                 linewidth=0.5,
                                 edgecolor='black',
                                 facecolor=facecolor,
                                 alpha=alpha)
        ax.add_patch(rect)

    main_rect = patches.Rectangle((sw_lon, sw_lat),
                                  ne_lon - sw_lon,
                                  ne_lat - sw_lat,
                                  linewidth=3,
                                  edgecolor='yellow',
                                  facecolor='none')
    ax.add_patch(main_rect)

    try:
        cx.add_basemap(ax,
                       crs="EPSG:4326",
                       source=cx.providers.Esri.WorldImagery)
    except Exception as e:
        print(f"Note: Could not fetch background map. Error: {e}")

    plt.title(
        f"Tile Filtering: {region_dir}\n{land_count} Land (Green) | {water_count} Water (Red)",
        color='white',
        backgroundcolor='black')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    if parent_dir:
        output_dir = os.path.join(parent_dir, region_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{region_dir}_tiles.png")
    else:
        output_file = f"{region_dir}_tiles.png"

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mercantile tiles over a region.")

    parser.add_argument(
        "region_dir",
        type=str,
        help="The region directory string (e.g., 'Area_170_-50_175_-45')")
    parser.add_argument("-z",
                        "--zoom",
                        type=int,
                        default=14,
                        help="Zoom level for visualization (default: 14)")
    parser.add_argument(
        "-p",
        "--parent_dir",
        type=str,
        default=None,
        help=
        "Parent folder where the region directory is located (e.g., 'grid_runs')"
    )

    args = parser.parse_args()

    visualize_region_tiles(args.region_dir,
                           zoom=args.zoom,
                           parent_dir=args.parent_dir)


if __name__ == "__main__":
    main()
