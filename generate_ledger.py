import argparse
import os
from pathlib import Path

from rich.console import Console


def main():
    parser = argparse.ArgumentParser(
        description=
        "Generate an exclude ledger from existing downloaded images.")
    parser.add_argument(
        '--image-dir',
        required=True,
        help=
        "Base directory containing the grid runs (e.g., your Capybara or Bobcat drive)"
    )
    parser.add_argument('--output',
                        default="global_exclude_ledger.txt",
                        help="Output text file name")
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[cyan]Scanning for .jpg files in {args.image_dir}...[/cyan]")

    image_ids = set()

    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_ids.add(Path(file).stem)

    if not image_ids:
        console.print(
            "[yellow]No .jpg files found in the specified directory.[/yellow]")
        return

    console.print(
        f"[\u2713] Found [green]{len(image_ids):,}[/green] unique images. Writing to ledger..."
    )

    with open(args.output, 'w') as f:
        for img_id in image_ids:
            f.write(f"{img_id}\n")

    console.print(
        f"[bold green][\u2713] Successfully saved {len(image_ids):,} IDs to {args.output}[/bold green]"
    )


if __name__ == "__main__":
    main()
