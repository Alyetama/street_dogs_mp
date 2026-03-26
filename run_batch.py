import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from mapillary_module import process_region

# Configuration
load_dotenv()
MLY_KEY = os.environ.get('MLY_KEY')
DB_PATH = 'mapillary_data.db'
CSV_PATH = 'global_grid_5deg.csv'

def get_completed_runs(conn):
    """Checks the database to see which runs are already completely finished."""
    try:
        # We query the database for regions we have already successfully inserted data for
        df = pd.read_sql_query("SELECT DISTINCT grid_region, grid_west, grid_south FROM all_data", conn)
        # Create a set of unique identifiers to easily check against later
        return set(df.apply(lambda row: f"{row['grid_region']}_{row['grid_west']}_{row['grid_south']}", axis=1))
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        # Table doesn't exist yet, which means no runs are completed
        return set()

def main():
    if not MLY_KEY:
        raise ValueError("MLY_KEY not found in environment variables.")

    grid_df = pd.read_csv(CSV_PATH)
    conn = sqlite3.connect(DB_PATH)
    
    completed_runs = get_completed_runs(conn)

    print(f"Loaded {len(grid_df)} total locations. Found {len(completed_runs)} already completed in database.")

    # Wrap the loop in tqdm to track overall progress across the CSV
    for index, row in tqdm(grid_df.iterrows(), total=len(grid_df), desc="Overall Progress (Grids)"):
        west = row['sw_lon']
        south = row['sw_lat']
        east = row['ne_lon']
        north = row['ne_lat']
        region = row['region']
        
        clean_region = str(region).replace(" ", "_").replace("/", "_")
        
        # Check if we should skip this run
        run_identifier = f"{region}_{west}_{south}"
        if run_identifier in completed_runs:
            # We use tqdm.write so the print statement doesn't break the progress bar
            tqdm.write(f"⏭️ Skipping {clean_region} ({west}, {south}) - Already in database.")
            continue

        run_name = f"{clean_region}_{west}_{south}_{east}_{north}"
        
        # Print a clean divider using tqdm.write to preserve the progress bar layout
        tqdm.write(f"\n{'='*50}\n▶️ STARTING RUN: {run_name}\n{'='*50}")

        try:
            all_df, animals_df = process_region(
                west=west, south=south, east=east, north=north, 
                run_name=run_name, mly_key=MLY_KEY, max_workers=50
            )

            if not all_df.empty:
                for df in [all_df, animals_df]:
                    if not df.empty:
                        df['grid_west'] = west
                        df['grid_south'] = south
                        df['grid_east'] = east
                        df['grid_north'] = north
                        df['grid_region'] = region

                all_df.to_sql('all_data', conn, if_exists='append', index=False)
                if not animals_df.empty:
                    animals_df.to_sql('ground_animals', conn, if_exists='append', index=False)
                    
            tqdm.write(f"✅ Successfully committed data for {run_name} to SQLite.")

        except Exception as e:
            tqdm.write(f"❌ ERROR on {run_name}: {str(e)}")
            error_data = [{
                'grid_west': west,
                'grid_south': south,
                'grid_east': east,
                'grid_north': north,
                'grid_region': region,
                'error_message': str(e)
            }]
            pd.DataFrame(error_data).to_sql('errored_runs', conn, if_exists='append', index=False)

    conn.close()
    print("\nBatch process complete.")

if __name__ == "__main__":
    main()

