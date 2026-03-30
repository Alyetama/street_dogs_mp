import json
import multiprocessing
import os
import re
import sqlite3
import threading
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from datetime import datetime

import mercantile
import pandas as pd
import piexif
import requests
from dotenv import load_dotenv
from global_land_mask import globe
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
from visualize_region_tiles import visualize_region_tiles

load_dotenv()

MLY_KEY = os.environ.get('MLY_KEY')
ZOOM_LEVEL = 14
GRID_CSV_FILE = 'global_grid_5deg.csv'

# --- TUNED CONCURRENCY SETTINGS ---
OUTER_MAX_WORKERS = 5
INNER_MAX_WORKERS = 30
PARENT_DIR = 'grid_runs'
DB_NAME = os.path.join(PARENT_DIR, 'mapillary_db.sqlite')
TRACKER_FILE = os.path.join(PARENT_DIR, 'completed_regions.txt')

# Global variable for the multiprocessing lock
db_lock = None


def init_worker(lock):
    """Initializes the shared lock for each process in the ProcessPool."""
    global db_lock
    db_lock = lock


def sanitize_folder_name(name):
    """Sanitizes names and ensures they are safe for OS paths."""
    safe_name = name.replace('&', 'and')
    return re.sub(r'[^\w\-_\.]', '_', safe_name).strip('_')


# --- SQLite Helper ---
def append_to_sqlite(df, table_name, run_name, region_id):
    """Safely appends a DataFrame to the global SQLite database."""
    if df.empty:
        return

    df_sql = df.copy()

    for col in df_sql.columns:
        if df_sql[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df_sql[col] = df_sql[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

    df_sql['run_name'] = run_name
    df_sql['region_id'] = region_id

    with db_lock:
        with sqlite3.connect(DB_NAME, timeout=60) as conn:
            df_sql.to_sql(table_name, conn, if_exists='append', index=False)


# --- API Fetcher Helpers ---
def get_sequences_for_bbox(bbox, session):
    bbox_str = f'{bbox.west},{bbox.south},{bbox.east},{bbox.north}'
    url = f'https://graph.mapillary.com/images?access_token={MLY_KEY}&fields=id,sequence&bbox={bbox_str}'
    try:
        return {
            img['sequence']
            for img in session.get(url, timeout=10).json().get('data', [])
            if 'sequence' in img
        }
    except Exception:
        return set()


def get_images_for_sequence(seq, session):
    url = f'https://graph.mapillary.com/image_ids?access_token={MLY_KEY}&sequence_id={seq}'
    try:
        return [
            obj['id']
            for obj in session.get(url, timeout=10).json().get('data', [])
        ]
    except Exception:
        return []


def fetch_image_data(image_id, fields_str, session):
    url = f'https://graph.mapillary.com/{image_id}?access_token={MLY_KEY}&fields={fields_str}'
    try:
        return image_id, session.get(url, timeout=10).json()
    except Exception:
        return image_id, None


def fetch_animal_detections(image_id, seq_id, session):
    url = f'https://graph.mapillary.com/{image_id}/detections?access_token={MLY_KEY}&fields=geometry,value'
    features = []
    try:
        dets_data = session.get(url, timeout=10).json().get('data', [])
        for det in dets_data:
            if det.get('value') == 'animal--ground-animal':
                features.append({
                    'type': 'Feature',
                    'properties': {
                        'image_id': image_id,
                        'sequence_id': seq_id,
                        'object_value': det['value']
                    },
                    'geometry': det['geometry']
                })
        return image_id, features
    except Exception:
        return image_id, []


def download_single_image(image_id, url, captured_at, output_folder_name):
    filepath = os.path.join(output_folder_name, f"{image_id}.jpg")
    temp_filepath = filepath + ".tmp"
    if os.path.exists(filepath): return image_id, True
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        with open(temp_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        os.rename(temp_filepath, filepath)

        if pd.notna(captured_at):
            try:
                dt = datetime.strptime(str(captured_at), "%Y-%m-%d %H:%M:%S")
                exif_time = dt.strftime("%Y:%m:%d %H:%M:%S")
                try:
                    exif_dict = piexif.load(filepath)
                except Exception:
                    exif_dict = {
                        "0th": {},
                        "Exif": {},
                        "GPS": {},
                        "1st": {},
                        "Interop": {}
                    }
                exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_time.encode(
                    'utf-8')
                exif_dict["Exif"][piexif.ExifIFD.
                                  DateTimeOriginal] = exif_time.encode('utf-8')
                exif_dict["Exif"][
                    piexif.ExifIFD.DateTimeDigitized] = exif_time.encode(
                        'utf-8')
                piexif.insert(piexif.dump(exif_dict), filepath)
                os.utime(filepath, (dt.timestamp(), dt.timestamp()))
            except Exception:
                pass
        return image_id, True
    except Exception:
        if os.path.exists(temp_filepath): os.remove(temp_filepath)
        return image_id, False


def build_mapillary_dataframe(detections_dict):
    records = []
    for image_id, metadata in detections_dict.items():
        if metadata is not None:
            row = metadata.copy()
            row['image_id'] = image_id
            if 'detections' in row and isinstance(row['detections'], dict):
                row['detections'] = json.dumps(row['detections'].get(
                    'data', []))
            records.append(row)
    df = pd.json_normalize(records)
    if not df.empty:
        cols = ['image_id'] + [c for c in df.columns if c != 'image_id']
        df = df[cols]
    return df


# --- Phase Handlers with Verbose TQDM ---
def get_image_topology(west, south, east, north, region_dir, region_id,
                       session, pos, short_name):
    checkpoint_file = os.path.join(region_dir,
                                   f'topology_checkpoint_{region_id}.json')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)

    tiles = list(mercantile.tiles(west, south, east, north, ZOOM_LEVEL))
    all_bboxes = [mercantile.bounds(t.x, t.y, t.z) for t in tiles]

    land_bboxes = []
    for bbox in all_bboxes:
        center_lat = (bbox.south + bbox.north) / 2.0
        center_lon = (bbox.west + bbox.east) / 2.0
        if globe.is_land(center_lat, center_lon):
            land_bboxes.append(bbox)

    if len(land_bboxes) < len(all_bboxes):
        tqdm.write(
            f"[{short_name}] Filtered out {len(all_bboxes) - len(land_bboxes)} water tiles. Kept {len(land_bboxes)} land tiles."
        )

    unique_sequences = set()

    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_sequences_for_bbox, bbox, session): bbox
            for bbox in land_bboxes
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc=f"[{short_name}] 1/5 BBoxes",
                           position=pos,
                           leave=False):
            unique_sequences.update(future.result())

    image_to_sequence_map = {}
    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_images_for_sequence, seq, session): seq
            for seq in unique_sequences
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc=f"[{short_name}] 2/5 Sequences",
                           position=pos,
                           leave=False):
            seq_id = futures[future]
            for img_id in future.result():
                image_to_sequence_map[img_id] = seq_id

    with open(checkpoint_file, 'w') as f:
        json.dump(image_to_sequence_map, f)
    return image_to_sequence_map


def get_all_image_data_fast(image_ids, fields_str, region_dir, region_id,
                            session, pos, short_name):
    checkpoint_file = os.path.join(region_dir,
                                   f'metadata_checkpoint_{region_id}.jsonl')
    results_dict = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    results_dict[record['image_id']] = record['data']

    missing_ids = [
        img_id for img_id in image_ids if img_id not in results_dict
    ]
    if not missing_ids: return results_dict

    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_image_data, img_id, fields_str, session):
            img_id
            for img_id in missing_ids
        }
        with open(checkpoint_file, 'a') as f:
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc=f"[{short_name}] 3/5 Metadata",
                               position=pos,
                               leave=False):
                image_id, data = future.result()
                if data is not None:
                    results_dict[image_id] = data
                    with write_lock:
                        f.write(
                            json.dumps({
                                'image_id': image_id,
                                'data': data
                            }) + '\n')
                        f.flush()
    return results_dict


def get_all_animal_detections_fast(image_to_seq_map, region_dir, region_id,
                                   session, pos, short_name):
    checkpoint_file = os.path.join(
        region_dir, f'animal_detections_checkpoint_{region_id}.jsonl')
    results_dict = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    results_dict[record['image_id']] = record['features']

    image_ids = list(image_to_seq_map.keys())
    missing_ids = [
        img_id for img_id in image_ids if img_id not in results_dict
    ]
    if not missing_ids: return results_dict

    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_animal_detections, img_id,
                            image_to_seq_map[img_id], session): img_id
            for img_id in missing_ids
        }
        with open(checkpoint_file, 'a') as f:
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc=f"[{short_name}] 4/5 Detections",
                               position=pos,
                               leave=False):
                image_id, features = future.result()
                results_dict[image_id] = features
                with write_lock:
                    f.write(
                        json.dumps({
                            'image_id': image_id,
                            'features': features
                        }) + '\n')
                    f.flush()
    return results_dict


def download_mapillary_images(df, output_folder_name, pos, short_name):
    os.makedirs(output_folder_name, exist_ok=True)
    download_tasks = [(row['image_id'], row.get('thumb_original_url'),
                       row.get('captured_at')) for _, row in df.iterrows()
                      if pd.notna(row.get('thumb_original_url'))]

    if not download_tasks: return

    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_single_image, img_id, url, cap_at,
                            output_folder_name): img_id
            for img_id, url, cap_at in download_tasks
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc=f"[{short_name}] 5/5 Downloads",
                           position=pos,
                           leave=False):
            pass


# --- The Master Process Worker ---
def process_region(west, south, east, north, unique_region_id, run_name):
    worker_name = multiprocessing.current_process().name
    match = re.search(r'\d+', worker_name)
    pos = int(match.group()) if match else 1

    safe_region_id = sanitize_folder_name(unique_region_id)
    short_name = unique_region_id.replace('_', ' ')

    region_dir = os.path.join(PARENT_DIR, safe_region_id)
    os.makedirs(region_dir, exist_ok=True)
    output_folder_name = os.path.join(region_dir, 'ground_animal_images')

    try:
        visualize_region_tiles(safe_region_id,
                               zoom=ZOOM_LEVEL,
                               parent_dir=PARENT_DIR)
    except Exception as e:
        tqdm.write(f"[{short_name}] Note: Map visualization failed: {e}")

    session = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504])
    session.mount(
        'https://',
        HTTPAdapter(max_retries=retries,
                    pool_connections=INNER_MAX_WORKERS,
                    pool_maxsize=INNER_MAX_WORKERS))

    image_to_seq_map = get_image_topology(west, south, east, north, region_dir,
                                          safe_region_id, session, pos,
                                          short_name)

    fields_str = 'id,computed_geometry,captured_at,sequence,is_pano,camera_type,computed_compass_angle,creator,height,width,detections,make,model,thumb_256_url,thumb_1024_url,thumb_2048_url,thumb_original_url'
    detections_data = get_all_image_data_fast(list(image_to_seq_map.keys()),
                                              fields_str, region_dir,
                                              safe_region_id, session, pos,
                                              short_name)

    animal_detections_dict = get_all_animal_detections_fast(
        image_to_seq_map, region_dir, safe_region_id, session, pos, short_name)

    ground_animal_features = []
    extracted_image_ids = set()
    for img_id, features in animal_detections_dict.items():
        if features:
            extracted_image_ids.add(img_id)
            ground_animal_features.extend(features)

    with open(
            os.path.join(region_dir, f'ground_animals_{safe_region_id}.json'),
            'w') as f:
        json.dump(ground_animal_features, f, indent=4)

    # --- DataFrames & SQL Injection ---
    all_results_df = build_mapillary_dataframe(detections_data)

    if all_results_df.empty:
        ground_animals_df = pd.DataFrame(columns=['image_id'])
    else:
        if 'captured_at' in all_results_df.columns:
            all_results_df['captured_at'] = pd.to_datetime(
                all_results_df['captured_at'], unit='ms',
                errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        ground_animals_df = all_results_df[all_results_df['image_id'].isin(
            extracted_image_ids)].copy()

    all_results_df.to_csv(os.path.join(region_dir,
                                       f'all_data_{safe_region_id}.csv'),
                          index=False)
    ground_animals_df.to_csv(os.path.join(
        region_dir, f'ground_animals_{safe_region_id}.csv'),
                             index=False)

    append_to_sqlite(all_results_df, 'all_data', run_name, unique_region_id)
    append_to_sqlite(ground_animals_df, 'ground_animals', run_name,
                     unique_region_id)

    download_mapillary_images(ground_animals_df, output_folder_name, pos,
                              short_name)

    return f"Completed '{unique_region_id}' ({len(extracted_image_ids)} images)."


if __name__ == "__main__":
    if not MLY_KEY:
        raise ValueError("MLY_KEY environment variable is missing.")

    print('\033[?25l', end="")

    os.makedirs(PARENT_DIR, exist_ok=True)

    df_grid = pd.read_csv(GRID_CSV_FILE)

    GLOBAL_RUN_NAME = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    completed_regions = set()
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            completed_regions = {line.strip() for line in f if line.strip()}

    tasks_to_run = []
    for _, row in df_grid.iterrows():
        unique_region_id = f"{row['region']}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"
        if unique_region_id not in completed_regions:
            tasks_to_run.append((row['sw_lon'], row['sw_lat'], row['ne_lon'],
                                 row['ne_lat'], unique_region_id))

    if not tasks_to_run:
        print(
            f"All {len(df_grid)} regions have already been completed according to {TRACKER_FILE}."
        )
    else:
        tqdm.write(
            f"Skipped {len(df_grid) - len(tasks_to_run)} completed regions.")
        tqdm.write(
            f"Initiating multi-region processing. {len(tasks_to_run)} tasks remaining. Run ID: {GLOBAL_RUN_NAME}"
        )

        try:
            m = multiprocessing.Manager()
            master_lock = m.Lock()

            with ProcessPoolExecutor(max_workers=OUTER_MAX_WORKERS,
                                     initializer=init_worker,
                                     initargs=(master_lock, )) as executor:
                futures = {}
                for task in tasks_to_run:
                    sw_lon, sw_lat, ne_lon, ne_lat, region_id = task
                    future = executor.submit(process_region, sw_lon, sw_lat,
                                             ne_lon, ne_lat, region_id,
                                             GLOBAL_RUN_NAME)
                    futures[future] = region_id

                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc="Total Progress",
                                   position=0,
                                   leave=True):
                    region_id = futures[future]
                    try:
                        result_msg = future.result()
                        tqdm.write(f"[\u2713] {result_msg}")

                        with open(TRACKER_FILE, 'a') as f:
                            f.write(f"{region_id}\n")

                    except Exception as exc:
                        tqdm.write(f"[X] Region '{region_id}' failed: {exc}")
        finally:
            print('\033[?25h', end="")

