import os
from pathlib import Path

from rich.console import Console

TARGET_DIR = "grid_runs"


def main():
    console = Console()
    target_path = Path(TARGET_DIR)

    if not target_path.is_dir():
        console.print(
            f"[bold red]Error:[/] Directory '{TARGET_DIR}' not found.")
        return

    console.print(
        f"[cyan]Scanning for orphaned completion markers in {TARGET_DIR}...[/cyan]"
    )

    marker_files = list(target_path.rglob(".completed_*"))

    if not marker_files:
        console.print("[\u2713] No completion markers found.")
        return

    orphaned_markers = []

    for marker_path in marker_files:
        # Extract the sub_id (e.g., getting 'South_Asia_sub_4' from '.completed_South_Asia_sub_4')
        sub_id = marker_path.name.replace('.completed_', '')
        parent_dir = marker_path.parent

        # Define the expected data files for this marker
        expected_metadata = parent_dir / f"metadata_checkpoint_{sub_id}.jsonl.zst"
        expected_animals = parent_dir / f"animal_detections_checkpoint_{sub_id}.jsonl.zst"

        # If either of these files is completely missing, the marker is an orphan!
        if not expected_metadata.exists() or not expected_animals.exists():
            orphaned_markers.append(marker_path)

    if not orphaned_markers:
        console.print(
            f"[\u2713] All [green]{len(marker_files)}[/green] markers are perfectly healthy and have their data files!"
        )
        return

    console.print(
        f"\n[bold red][!] Found {len(orphaned_markers)} orphaned markers missing their data files.[/bold red]"
    )

    # Automatically delete the orphaned markers
    deleted_count = 0
    for marker in orphaned_markers:
        try:
            os.remove(marker)
            console.print(f"[-] Deleted invalid marker: {marker.name}")
            deleted_count += 1
        except Exception as e:
            console.print(f"[bold red]Failed to delete {marker.name}: {e}[/]")

    console.print(
        f"\n[bold green][\u2713] Successfully cleaned {deleted_count} orphaned markers.[/bold green]"
    )
    console.print(
        "[i] You can now safely re-run your main batch_chunks script. It will see these missing markers and automatically backfill the missing data![/i]"
    )


if __name__ == "__main__":
    main()
