import argparse
import os
from pathlib import Path

from rich.console import Console


def main():
    parser = argparse.ArgumentParser(
        description=
        "Generate or append to an exclude ledger from existing downloaded images."
    )
    parser.add_argument('--image-dir',
                        required=True,
                        help="Base directory containing the grid runs")
    parser.add_argument('--output',
                        default="global_exclude_ledger.txt",
                        help="Output text file name")
    args = parser.parse_args()

    console = Console()

    existing_ids = set()
    if os.path.exists(args.output):
        console.print(
            f"[cyan]Found existing ledger '{args.output}'. Reading contents...[/cyan]"
        )
        with open(args.output, 'r') as f:
            existing_ids = {line.strip() for line in f if line.strip()}
        console.print(f"[\u2713] Loaded {len(existing_ids):,} existing IDs.")

    console.print(
        f"\n[cyan]Scanning for .jpg files in {args.image_dir}...[/cyan]")
    scanned_ids = set()

    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                scanned_ids.add(Path(file).stem)

    if not scanned_ids:
        console.print(
            "[yellow]No .jpg files found in the specified directory.[/yellow]")
        return

    new_ids = scanned_ids - existing_ids

    if not new_ids:
        console.print(
            f"\n[\u2713] All [green]{len(scanned_ids):,}[/green] scanned images are already in the ledger. Nothing to append."
        )
        return

    console.print(
        f"[\u2713] Found [green]{len(new_ids):,}[/green] NEW unique images. Appending to ledger..."
    )

    with open(args.output, 'a') as f:
        for img_id in new_ids:
            f.write(f"{img_id}\n")

    total_size = len(existing_ids) + len(new_ids)
    console.print(
        f"[bold green]\n[\u2713] Successfully appended {len(new_ids):,} IDs.[/bold green]"
    )
    console.print(
        f"[i] The ledger '{args.output}' now contains a total of {total_size:,} image IDs.[/i]"
    )


if __name__ == "__main__":
    main()
