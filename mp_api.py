import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import mercantile
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

load_dotenv()

# --- HARDCODED CONSTANTS ---
WEST, SOUTH, EAST, NORTH = [
    -73.30618501253828, -39.872011050803245, -73.17619937533453,
    -39.77387065582631
]
REGION_NAME = 'valdivia'
OUTPUT_FOLDER_NAME = os.path.join(REGION_NAME, 'ground_animal_images')

MLY_KEY = os.environ['MLY_KEY']
MAX_WORKERS = 50

# --- GLOBAL SESSION SETUP ---
session = requests.Session()
retries = Retry(total=5,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504])
session.mount(
    'https://',
    HTTPAdapter(max_retries=retries,
                pool_connections=MAX_WORKERS,
                pool_maxsize=MAX_WORKERS))


def get_sequences_for_bbox(bbox):
    bbox_str = f'{bbox.west},{bbox.south},{bbox.east},{bbox.north}'
    url = (f'https://graph.mapillary.com/images?access_token={MLY_KEY}'
           f'&fields=id,sequence&bbox={bbox_str}')
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        return {img['sequence'] for img in data if 'sequence' in img}
    except Exception:
        return set()


def get_images_for_sequence(seq):
    url = (f'https://graph.mapillary.com/image_ids?access_token={MLY_KEY}'
           f'&sequence_id={seq}')
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return [obj['id'] for obj in response.json().get('data', [])]
    except Exception:
        return []


def get_image_topology():
    checkpoint_file = f'topology_checkpoint_{REGION_NAME}.json'

    if os.path.exists(checkpoint_file):
        print(
            f"\n--- Phase 1 & 2: Loading topology checkpoint from '{checkpoint_file}' ---"
        )
        with open(checkpoint_file, 'r') as f:
            return json.load(f)

    print("\n--- Phase 1: Generating bounding box tiles... ---")
    # Using the exact order that works perfectly for the 49 tiles
    tiles = list(mercantile.tiles(WEST, SOUTH, EAST, NORTH, 14))
    bbox_list = [mercantile.bounds(t.x, t.y, t.z) for t in tiles]

    unique_sequences = set()

    print(f"Fetching sequences across {len(bbox_list)} tiles...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_sequences_for_bbox, bbox): bbox
            for bbox in bbox_list
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="BBoxes"):
            unique_sequences.update(future.result())

    print(f"Found {len(unique_sequences)} unique sequences.")

    print(
        f"\n--- Phase 2: Fetching images for {len(unique_sequences)} sequences ---"
    )
    image_to_sequence_map = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_images_for_sequence, seq): seq
            for seq in unique_sequences
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Sequences"):
            seq_id = futures[future]
            for img_id in future.result():
                image_to_sequence_map[img_id] = seq_id

    with open(checkpoint_file, 'w') as f:
        json.dump(image_to_sequence_map, f)

    return image_to_sequence_map


def fetch_image_data(image_id, fields_str):
    url = (f'https://graph.mapillary.com/{image_id}?'
           f'access_token={MLY_KEY}&fields={fields_str}')
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return image_id, response.json()
    except Exception:
        return image_id, None


def get_all_image_data_fast(image_ids, fields_str):
    checkpoint_file = f'metadata_checkpoint_{REGION_NAME}.jsonl'
    results_dict = {}

    if os.path.exists(checkpoint_file):
        print(
            f"\n--- Phase 3: Loading metadata checkpoint from '{checkpoint_file}' ---"
        )
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    results_dict[record['image_id']] = record['data']

    missing_ids = [
        img_id for img_id in image_ids if img_id not in results_dict
    ]

    if not missing_ids:
        print("All metadata already fetched from checkpoint!")
        return results_dict

    print(
        f"Fetching metadata for {len(missing_ids)} remaining images out of {len(image_ids)} total..."
    )

    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_image_data, img_id, fields_str): img_id
            for img_id in missing_ids
        }

        with open(checkpoint_file, 'a') as f:
            for future in tqdm(as_completed(futures),
                               total=len(missing_ids),
                               desc="Fetching Metadata"):
                image_id, data = future.result()
                if data is not None:
                    results_dict[image_id] = data

                    record = {'image_id': image_id, 'data': data}
                    with write_lock:
                        f.write(json.dumps(record) + '\n')
                        f.flush()

    return results_dict


def fetch_animal_detections(image_id, seq_id):
    url = (f'https://graph.mapillary.com/{image_id}/detections?'
           f'access_token={MLY_KEY}&fields=geometry,value')
    features = []
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        dets_data = response.json().get('data', [])

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


def get_all_animal_detections_fast(image_to_seq_map):
    checkpoint_file = f'animal_detections_checkpoint_{REGION_NAME}.jsonl'
    results_dict = {}

    if os.path.exists(checkpoint_file):
        print(
            f"\n--- Phase 4: Loading animal detections checkpoint from '{checkpoint_file}' ---"
        )
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    results_dict[record['image_id']] = record['features']

    image_ids = list(image_to_seq_map.keys())
    missing_ids = [
        img_id for img_id in image_ids if img_id not in results_dict
    ]

    if not missing_ids:
        print("All animal detections already fetched from checkpoint!")
        return results_dict

    print(
        f"Scanning for animals in {len(missing_ids)} remaining images out of {len(image_ids)} total..."
    )

    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_animal_detections, img_id, image_to_seq_map[img_id]):
            img_id
            for img_id in missing_ids
        }

        with open(checkpoint_file, 'a') as f:
            for future in tqdm(as_completed(futures),
                               total=len(missing_ids),
                               desc="Scanning Detections"):
                image_id, features = future.result()
                results_dict[image_id] = features

                record = {'image_id': image_id, 'features': features}
                with write_lock:
                    f.write(json.dumps(record) + '\n')
                    f.flush()

    return results_dict


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


def download_mapillary_images(df):
    os.makedirs(OUTPUT_FOLDER_NAME, exist_ok=True)

    download_tasks = []
    for index, row in df.iterrows():
        url = row.get('thumb_original_url')
        if pd.notna(url) and isinstance(url, str):
            download_tasks.append((row['image_id'], url))

    if not download_tasks:
        print("No images to download.")
        return

    print(f"\nFound {len(download_tasks)} images to download.")

    def download_single_image(image_id, url):
        filepath = os.path.join(OUTPUT_FOLDER_NAME, f"{image_id}.jpg")
        temp_filepath = filepath + ".tmp"

        if os.path.exists(filepath):
            return image_id, True

        try:
            response = requests.get(url, stream=True, timeout=15)
            response.raise_for_status()

            with open(temp_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            os.rename(temp_filepath, filepath)
            return image_id, True
        except Exception:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return image_id, False

    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_single_image, img_id, url): img_id
            for img_id, url in download_tasks
        }

        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Downloading Images"):
            img_id, success = future.result()
            if success:
                success_count += 1

    print(
        f"\nFinished! Successfully downloaded {success_count} out of {len(download_tasks)} images to '{OUTPUT_FOLDER_NAME}'."
    )


if __name__ == "__main__":
    if not MLY_KEY:
        raise ValueError(
            "MLY_KEY environment variable is missing. Please set it in your .env file."
        )

    # --- Phase 1 & 2: Topology ---
    image_to_seq_map = get_image_topology()
    all_image_ids = list(image_to_seq_map.keys())
    print(f"Total images found in region: {len(all_image_ids)}")

    # Setup fields
    fields_list = [
        'id', 'computed_geometry', 'captured_at', 'sequence', 'is_pano',
        'camera_type', 'computed_compass_angle', 'creator', 'height', 'width',
        'detections', 'make', 'model', 'thumb_256_url', 'thumb_1024_url',
        'thumb_2048_url', 'thumb_original_url'
    ]
    fields_str_global = ','.join(fields_list)

    # --- Phase 3: Metadata (for all_data.csv) ---
    detections_data = get_all_image_data_fast(all_image_ids, fields_str_global)
    print(
        f"Successfully retrieved rich data for {len(detections_data)} images.")

    # --- Phase 4: Animal Detections (The Correct API Endpoint) ---
    animal_detections_dict = get_all_animal_detections_fast(image_to_seq_map)

    ground_animal_features = []
    extracted_image_ids = set()

    for img_id, features in animal_detections_dict.items():
        if features:  # If the list is not empty, animals were found
            extracted_image_ids.add(img_id)
            ground_animal_features.extend(features)

    print(
        f"\nTotal 'animal--ground-animal' detections found: {len(extracted_image_ids)}"
    )

    with open(f'ground_animals_{REGION_NAME}.json', 'w') as f:
        json.dump(ground_animal_features, f, indent=4)
    print(f"Results saved to 'ground_animals_{REGION_NAME}.json'")

    # --- Phase 5: DataFrames ---
    print("\n--- Phase 5: Building CSVs ---")
    all_results_df = build_mapillary_dataframe(detections_data)

    all_results_df.to_csv(f'all_data_{REGION_NAME}.csv', index=False)
    print(
        f"Created 'all_data_{REGION_NAME}.csv' with {all_results_df.shape[0]} rows."
    )

    ground_animals_df = all_results_df[all_results_df['image_id'].isin(
        extracted_image_ids)].copy()
    ground_animals_df.to_csv(f'ground_animals_{REGION_NAME}.csv', index=False)
    print(
        f"Created 'ground_animals_{REGION_NAME}.csv' with {ground_animals_df.shape[0]} rows."
    )

    # --- Phase 6: Download ---
    download_mapillary_images(ground_animals_df)

    # --- Phase 7: Cleanup & Organization ---
    print(
        f"\n--- Phase 7: Moving all data files to the '{REGION_NAME}' folder ---"
    )
    os.makedirs(REGION_NAME, exist_ok=True)

    files_to_move = [
        f'topology_checkpoint_{REGION_NAME}.json',
        f'metadata_checkpoint_{REGION_NAME}.jsonl',
        f'animal_detections_checkpoint_{REGION_NAME}.jsonl',
        f'ground_animals_{REGION_NAME}.json', f'all_data_{REGION_NAME}.csv',
        f'ground_animals_{REGION_NAME}.csv'
    ]

    for file_name in files_to_move:
        if os.path.exists(file_name):
            destination = os.path.join(REGION_NAME, file_name)
            # Prevent crash if the file is already there from a previous run
            if os.path.exists(destination):
                os.remove(destination)
            shutil.move(file_name, REGION_NAME)
            print(f"Moved: {file_name}")

    print(
        f"\nAll done! Data is in '{REGION_NAME}/' and images are in '{OUTPUT_FOLDER_NAME}/'."
    )
