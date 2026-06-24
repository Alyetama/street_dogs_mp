from pathlib import Path

import pandas as pd

df = pd.read_csv("global_grid_5deg.csv")

base_dir = Path("regions")
pending_dir = base_dir / "pending"
running_dir = base_dir / "running"
finished_dir = base_dir / "finished"

for folder in [pending_dir, running_dir, finished_dir]:
    folder.mkdir(parents=True, exist_ok=True)

for region, group in df.groupby("region"):
    safe_region = (region.replace(" ",
                                  "_").replace("/",
                                               "_").replace("&",
                                                            "and").lower())

    filename = pending_dir / f"{safe_region}.csv"
    group.to_csv(filename, index=False)
