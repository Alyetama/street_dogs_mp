import gc
import gzip
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

load_dotenv()

MLY_KEY = os.environ.get('MLY_KEY')
ZOOM_LEVEL = 14
GRID_CSV_FILE = 'Indian_Ocean.csv'
VISUALIZE = False
DOWNLOAD_IMAGES = True

OUTER_MAX_WORKERS = 10
INNER_MAX_WORKERS = 20
SUB_GRID_STEP = 1.0

PARENT_DIR = 'grid_runs'
DB_NAME = os.path.join(PARENT_DIR, 'mapillary_db.sqlite')
TRACKER_FILE = os.path.join(PARENT_DIR, 'completed_regions.txt')

db_lock = None


def init_worker(lock):
    global db_lock
    db_lock = lock


def sanitize_folder_name(name):
    safe_name = name.replace('&', 'and')
    return re.sub(r'[^\w\-_\.]', '_', safe_name).strip('_')


def clean_jsonl_file(filepath):
    """Removes corrupt lines from an interrupted .jsonl.gz file."""
    if not os.path.exists(filepath):
        return
    valid_lines = []
    is_corrupt = False
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError:
                is_corrupt = True

    if is_corrupt:
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            for line in valid_lines:
                f.write(line.strip() + '\n')


# --- SQLite Helper ---
def append_to_sqlite(df, table_name, run_name, region_id):
    if df.empty: return
    df_sql = df.copy()
    complex_cols = ['computed_geometry', 'creator', 'detections']
    for col in complex_cols:
        if col in df_sql.columns:
            df_sql[col] = df_sql[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
    df_sql['run_name'] = run_name
    df_sql['region_id'] = region_id

    with db_lock:
        with sqlite3.connect(DB_NAME, timeout=60) as conn:
            df_sql.to_sql(table_name, conn, if_exists='append', index=False)

    del df_sql


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


def build_mapillary_dataframe_from_records(records):
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    expected_columns = [
        'image_id', 'computed_geometry', 'captured_at', 'sequence', 'is_pano',
        'camera_type', 'computed_compass_angle', 'creator', 'height', 'width',
        'detections', 'make', 'model', 'thumb_256_url', 'thumb_1024_url',
        'thumb_2048_url', 'thumb_original_url'
    ]
    for col in expected_columns:
        if col not in df.columns: df[col] = None
    return df[expected_columns]


# --- Phase Handlers with Embedded TQDM Progress ---
def get_image_topology(west, south, east, north, region_dir, sub_id, session,
                       pos, desc_prefix):
    checkpoint_file = os.path.join(region_dir,
                                   f'topology_checkpoint_{sub_id}.json.gz')
    if os.path.exists(checkpoint_file):
        try:
            with gzip.open(checkpoint_file, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass

    tiles = list(mercantile.tiles(west, south, east, north, ZOOM_LEVEL))
    all_bboxes = [mercantile.bounds(t.x, t.y, t.z) for t in tiles]
    land_bboxes = [
        bbox for bbox in all_bboxes
        if globe.is_land((bbox.south + bbox.north) /
                         2.0, (bbox.west + bbox.east) / 2.0)
    ]

    if not land_bboxes: return {}

    unique_sequences = set()
    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_sequences_for_bbox, bbox, session): bbox
            for bbox in land_bboxes
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc=f"{desc_prefix} 1/5 BBoxes",
                           position=pos,
                           leave=False,
                           mininterval=2.0):
            unique_sequences.update(future.result())

    image_to_sequence_map = {}
    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_images_for_sequence, seq, session): seq
            for seq in unique_sequences
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc=f"{desc_prefix} 2/5 Sequences",
                           position=pos,
                           leave=False,
                           mininterval=2.0):
            seq_id = futures[future]
            for img_id in future.result():
                image_to_sequence_map[img_id] = seq_id

    temp_checkpoint = checkpoint_file + '.tmp'
    with gzip.open(temp_checkpoint, 'wt', encoding='utf-8') as f:
        json.dump(image_to_sequence_map, f)

    os.replace(temp_checkpoint, checkpoint_file)

    return image_to_sequence_map


def fetch_metadata_to_jsonl(image_ids, fields_str, region_dir, sub_id, session,
                            pos, desc_prefix):
    checkpoint_file = os.path.join(region_dir,
                                   f'metadata_checkpoint_{sub_id}.jsonl.gz')

    clean_jsonl_file(checkpoint_file)

    completed_ids = set()
    if os.path.exists(checkpoint_file):
        with gzip.open(checkpoint_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    completed_ids.add(json.loads(line)['image_id'])

    missing_ids = [
        img_id for img_id in image_ids if img_id not in completed_ids
    ]
    if not missing_ids: return

    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_image_data, img_id, fields_str, session):
            img_id
            for img_id in missing_ids
        }
        with gzip.open(checkpoint_file, 'at', encoding='utf-8') as f:
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc=f"{desc_prefix} 3/5 Metadata",
                               position=pos,
                               leave=False,
                               mininterval=2.0):
                image_id, data = future.result()
                if data is not None:
                    with write_lock:
                        f.write(
                            json.dumps({
                                'image_id': image_id,
                                'data': data
                            }) + '\n')


def fetch_detections_to_jsonl(image_to_seq_map, region_dir, sub_id, session,
                              pos, desc_prefix):
    checkpoint_file = os.path.join(
        region_dir, f'animal_detections_checkpoint_{sub_id}.jsonl.gz')

    clean_jsonl_file(checkpoint_file)

    completed_ids = set()
    if os.path.exists(checkpoint_file):
        with gzip.open(checkpoint_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    completed_ids.add(json.loads(line)['image_id'])

    missing_ids = [
        img_id for img_id in image_to_seq_map.keys()
        if img_id not in completed_ids
    ]
    if not missing_ids: return

    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_animal_detections, img_id, image_to_seq_map[img_id], session):
            img_id
            for img_id in missing_ids
        }
        with gzip.open(checkpoint_file, 'at', encoding='utf-8') as f:
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc=f"{desc_prefix} 4/5 Detections",
                               position=pos,
                               leave=False,
                               mininterval=2.0):
                image_id, features = future.result()
                with write_lock:
                    f.write(
                        json.dumps({
                            'image_id': image_id,
                            'features': features
                        }) + '\n')


# --- The Master Process Worker ---
def process_region(west, south, east, north, unique_region_id, run_name):
    worker_name = multiprocessing.current_process().name
    match = re.search(r'\d+', worker_name)
    pos = int(match.group()) if match else 1

    safe_region_id = sanitize_folder_name(unique_region_id)
    region_run_id = unique_region_id.replace('_', ' ')
    region_dir = os.path.join(PARENT_DIR, safe_region_id)
    os.makedirs(region_dir, exist_ok=True)
    output_folder_name = os.path.join(region_dir, 'ground_animal_images')
    os.makedirs(output_folder_name, exist_ok=True)

    session = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504])
    session.mount(
        'https://',
        HTTPAdapter(max_retries=retries,
                    pool_connections=INNER_MAX_WORKERS,
                    pool_maxsize=INNER_MAX_WORKERS))

    # 1. Generate Sub-BBoxes
    sub_bboxes = []
    cur_lat = south
    while cur_lat < north:
        next_lat = min(cur_lat + SUB_GRID_STEP, north)
        cur_lon = west
        while cur_lon < east:
            next_lon = min(cur_lon + SUB_GRID_STEP, east)
            sub_bboxes.append((cur_lon, cur_lat, next_lon, next_lat))
            cur_lon += SUB_GRID_STEP
        cur_lat += SUB_GRID_STEP

    total_animals_found = 0

    # 2. Iterate through sub-regions sequentially
    for i, (sw_lon, sw_lat, ne_lon, ne_lat) in enumerate(sub_bboxes):
        sub_id = f"{safe_region_id}_sub_{i}"
        desc_prefix = f"[{region_run_id} C{i+1}/{len(sub_bboxes)}]"

        # Step A: Topology
        image_to_seq_map = get_image_topology(sw_lon, sw_lat, ne_lon, ne_lat,
                                              region_dir, sub_id, session, pos,
                                              desc_prefix)
        if not image_to_seq_map:
            continue

        # Step B: Metadata & Detections
        fields_str = 'id,computed_geometry,captured_at,sequence,is_pano,camera_type,computed_compass_angle,creator,height,width,detections,make,model,thumb_256_url,thumb_1024_url,thumb_2048_url,thumb_original_url'
        fetch_metadata_to_jsonl(list(image_to_seq_map.keys()), fields_str,
                                region_dir, sub_id, session, pos, desc_prefix)
        fetch_detections_to_jsonl(image_to_seq_map, region_dir, sub_id,
                                  session, pos, desc_prefix)

        # Step C: Parse Data & Clear Memory
        animal_checkpoint = os.path.join(
            region_dir, f'animal_detections_checkpoint_{sub_id}.jsonl.gz')
        metadata_checkpoint = os.path.join(
            region_dir, f'metadata_checkpoint_{sub_id}.jsonl.gz')

        extracted_image_ids = set()
        ground_animal_features = []

        if os.path.exists(animal_checkpoint):
            with gzip.open(animal_checkpoint, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    record = json.loads(line)
                    features = record.get('features', [])
                    if features:
                        extracted_image_ids.add(record['image_id'])
                        ground_animal_features.extend(features)

        if ground_animal_features:
            final_json_path = os.path.join(region_dir,
                                           f'ground_animals_{sub_id}.json.gz')
            temp_json_path = final_json_path + '.tmp'

            with gzip.open(temp_json_path, 'wt', encoding='utf-8') as f:
                json.dump(ground_animal_features, f)

            os.replace(temp_json_path, final_json_path)

        del ground_animal_features
        total_animals_found += len(extracted_image_ids)

        # Step D: Process Metadata and Database Append
        all_csv_path = os.path.join(region_dir,
                                    f'all_data_{safe_region_id}.csv.gz')
        animals_csv_path = os.path.join(
            region_dir, f'ground_animals_{safe_region_id}.csv.gz')

        records = []
        download_tasks = []
        chunk_size = 10000

        def process_metadata_chunk(chunk_records):
            df = build_mapillary_dataframe_from_records(chunk_records)
            if df.empty: return

            if 'captured_at' in df.columns:
                df['captured_at'] = pd.to_datetime(
                    df['captured_at'], unit='ms',
                    errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

            animals_df = df[df['image_id'].isin(extracted_image_ids)].copy()

            df.to_csv(all_csv_path,
                      mode='a',
                      header=not os.path.exists(all_csv_path),
                      index=False,
                      compression='gzip')

            if not animals_df.empty:
                animals_df.to_csv(animals_csv_path,
                                  mode='a',
                                  header=not os.path.exists(animals_csv_path),
                                  index=False,
                                  compression='gzip')
                for _, row in animals_df.iterrows():
                    if pd.notna(row.get('thumb_original_url')):
                        download_tasks.append(
                            (row['image_id'], row['thumb_original_url'],
                             row['captured_at']))

            append_to_sqlite(df, 'all_data', run_name, unique_region_id)
            if not animals_df.empty:
                append_to_sqlite(animals_df, 'ground_animals', run_name,
                                 unique_region_id)
            del df
            del animals_df
            gc.collect()

        if os.path.exists(metadata_checkpoint):
            with gzip.open(metadata_checkpoint, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    record = json.loads(line)
                    row = record['data'].copy()
                    row['image_id'] = record['image_id']
                    if 'detections' in row and isinstance(
                            row['detections'], dict):
                        row['detections'] = row['detections'].get('data', [])
                    records.append(row)

                    if len(records) >= chunk_size:
                        process_metadata_chunk(records)
                        records = []

            if records:
                process_metadata_chunk(records)
                records = []

        # Step E: Download
        if DOWNLOAD_IMAGES and download_tasks:
            with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(download_single_image, img_id, url, cap_at, output_folder_name):
                    img_id
                    for img_id, url, cap_at in download_tasks
                }
                for _ in tqdm(as_completed(futures),
                              total=len(futures),
                              desc=f"{desc_prefix} 5/5 Downloads",
                              position=pos,
                              leave=False,
                              mininterval=2.0):
                    pass

        del download_tasks
        del image_to_seq_map
        gc.collect()

    return f"Completed '{unique_region_id}' ({total_animals_found} animals found across all chunks)."


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

    tasks_to_run = [
        (row['sw_lon'], row['sw_lat'], row['ne_lon'], row['ne_lat'],
         f"{row['region']}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"
         ) for _, row in df_grid.iterrows() if
        f"{row['region']}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"
        not in completed_regions
    ]

    if not tasks_to_run:
        print(f"All {len(df_grid)} regions have already been completed.")
    else:
        tqdm.write(
            f"Initiating multi-region processing. {len(tasks_to_run)} tasks remaining. Run ID: {GLOBAL_RUN_NAME}"
        )

        try:
            m = multiprocessing.Manager()
            master_lock = m.Lock()

            with ProcessPoolExecutor(max_workers=OUTER_MAX_WORKERS,
                                     initializer=init_worker,
                                     initargs=(master_lock, )) as executor:
                futures = {
                    executor.submit(process_region, sw_lon, sw_lat, ne_lon, ne_lat, region_id, GLOBAL_RUN_NAME):
                    region_id
                    for sw_lon, sw_lat, ne_lon, ne_lat, region_id in
                    tasks_to_run
                }

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
