import argparse
import os
import re

from geopy.geocoders import Nominatim
from rich.console import Console


def parse_bbox_from_folder(folder_name):
    """
    Extracts the bounding box from folder names like:
    North_America_-60_25_-55_30 -> West: -60, South: 25, East: -55, North: 30
    """
    match = re.search(
        r'_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$',
        folder_name)
    if match:
        w, s, e, n = map(float, match.groups())
        return w, s, e, n
    return None


def bboxes_intersect(w1, s1, e1, n1, w2, s2, e2, n2):
    """
    Returns True if two bounding boxes overlap or intersect at any point.
    """
    if e1 < w2 or w1 > e2:
        return False
    if n1 < s2 or s1 > n2:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description=
        "Find which grid_runs folder(s) contain a specific location.")
    parser.add_argument(
        "location",
        type=str,
        nargs='+',
        help="Name of the city or country (e.g., 'Paris', 'Japan')")
    parser.add_argument("--dirs",
                        nargs='+',
                        help="List of base directories to scan")

    args = parser.parse_args()
    query_location = " ".join(args.location)

    console = Console()
    console.print(
        f"[cyan]Geocoding boundaries for: '{query_location}'...[/cyan]")

    geolocator = Nominatim(user_agent="mapillary_grid_locator")

    try:
        location = geolocator.geocode(query_location)
        if not location:
            console.print(
                f"[bold red][!] Could not find data for '{query_location}'. Check your spelling.[/bold red]"
            )
            return

        bbox_raw = location.raw.get('boundingbox')
        if not bbox_raw:
            console.print(
                f"[bold red][!] API did not return bounding box data for this location.[/bold red]"
            )
            return

        loc_s, loc_n, loc_w, loc_e = map(float, bbox_raw)
        console.print(
            f"[bold green][\u2713] Found Location Bounds: West {loc_w:.2f}, East {loc_e:.2f}, South {loc_s:.2f}, North {loc_n:.2f}[/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red][!] Geocoding failed: {e}[/bold red]")
        return

    console.print(
        f"\n[cyan]Scanning directories for intersecting regions...[/cyan]")
    found_matches = []

    for base_dir in args.dirs:
        if not os.path.exists(base_dir):
            continue

        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)
            if os.path.isdir(folder_path):
                grid_bbox = parse_bbox_from_folder(folder_name)
                if grid_bbox:
                    grid_w, grid_s, grid_e, grid_n = grid_bbox

                    if bboxes_intersect(loc_w, loc_s, loc_e, loc_n, grid_w,
                                        grid_s, grid_e, grid_n):
                        found_matches.append(folder_path)

    if found_matches:
        console.print(
            f"\n[bold green][\u2713] '{query_location}' intersects with {len(found_matches)} folder(s):[/bold green]"
        )
        for match in sorted(found_matches):
            console.print(f"  [bold cyan]->[/bold cyan] {match}")
    else:
        console.print(
            f"\n[bold red][X] No grid folders found overlapping with '{query_location}'.[/bold red]"
        )


if __name__ == "__main__":
    main()
