import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import mercantile
import pandas as pd
import piexif
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


def setup_session(max_workers):
    session = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504])
    session.mount(
        'https://',
        HTTPAdapter(max_retries=retries,
                    pool_connections=max_workers,
                    pool_maxsize=max_workers))
    return session


def get_sequences_for_bbox(bbox, session, mly_key):
    bbox_str = f'{bbox.west},{bbox.south},{bbox.east},{bbox.north}'
    url = f'https://graph.mapillary.com/images?access_token={mly_key}&fields=id,sequence&bbox={bbox_str}'
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        return {img['sequence'] for img in data if 'sequence' in img}
    except Exception:
        return set()


def get_images_for_sequence(seq, session, mly_key):
    url = f'https://graph.mapillary.com/image_ids?access_token={mly_key}&sequence_id={seq}'
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return [obj['id'] for obj in response.json().get('data', [])]
    except Exception:
        return []


def get_image_topology(west, south, east, north, run_name, session, mly_key,
                       max_workers):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_file = os.path.join('checkpoints',
                                   f'topology_checkpoint_{run_name}.json')

    if os.path.exists(checkpoint_file):
        print(
            f"\n--- Loading topology checkpoint from '{checkpoint_file}' ---")
        with open(checkpoint_file, 'r') as f:
            return json.load(f)

    print("\n--- Generating bounding box tiles... ---")
    tiles = list(mercantile.tiles(west, south, east, north, 14))
    bbox_list = [mercantile.bounds(t.x, t.y, t.z) for t in tiles]

    unique_sequences = set()
    print(f"Fetching sequences across {len(bbox_list)} tiles...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_sequences_for_bbox, bbox, session, mly_key):
            bbox
            for bbox in bbox_list
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="BBoxes"):
            unique_sequences.update(future.result())

    print(f"Fetching images for {len(unique_sequences)} sequences...")
    image_to_sequence_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_images_for_sequence, seq, session, mly_key):
            seq
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


def fetch_image_data(image_id, fields_str, session, mly_key):
    url = f'https://graph.mapillary.com/{image_id}?access_token={mly_key}&fields={fields_str}'
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return image_id, response.json()
    except Exception:
        return image_id, None


def get_all_image_data_fast(image_ids, fields_str, run_name, session, mly_key,
                            max_workers):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_file = os.path.join('checkpoints',
                                   f'metadata_checkpoint_{run_name}.jsonl')
    results_dict = {}

    if os.path.exists(checkpoint_file):
        print(
            f"\n--- Loading metadata checkpoint from '{checkpoint_file}' ---")
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    results_dict[record['image_id']] = record['data']

    missing_ids = [
        img_id for img_id in image_ids if img_id not in results_dict
    ]
    if not missing_ids:
        return results_dict

    print(
        f"Fetching metadata for {len(missing_ids)} remaining images out of {len(image_ids)} total..."
    )
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_image_data, img_id, fields_str, session, mly_key):
            img_id
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


def fetch_animal_detections(image_id, seq_id, session, mly_key):
    url = f'https://graph.mapillary.com/{image_id}/detections?access_token={mly_key}&fields=geometry,value'
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


def get_all_animal_detections_fast(image_to_seq_map, run_name, session,
                                   mly_key, max_workers):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_file = os.path.join(
        'checkpoints', f'animal_detections_checkpoint_{run_name}.jsonl')
    results_dict = {}

    if os.path.exists(checkpoint_file):
        print(
            f"\n--- Loading animal detections checkpoint from '{checkpoint_file}' ---"
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
        return results_dict

    print(
        f"Scanning for animals in {len(missing_ids)} remaining images out of {len(image_ids)} total..."
    )
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_animal_detections, img_id, image_to_seq_map[img_id], session, mly_key):
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


def download_mapillary_images(df, output_folder, max_workers):
    os.makedirs(output_folder, exist_ok=True)
    download_tasks = []

    for _, row in df.iterrows():
        url = row.get('thumb_original_url')
        captured_at = row.get('captured_at')
        if pd.notna(url) and isinstance(url, str):
            download_tasks.append((row['image_id'], url, captured_at))

    if not download_tasks:
        print("No images to download.")
        return

    print(f"\nFound {len(download_tasks)} images to download.")

    def download_single_image(image_id, url, captured_at):
        filepath = os.path.join(output_folder, f"{image_id}.jpg")
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

            # EXIF Timestamp Injection
            if pd.notna(captured_at):
                try:
                    dt = datetime.strptime(str(captured_at),
                                           "%Y-%m-%d %H:%M:%S")
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

                    exif_dict["0th"][
                        piexif.ImageIFD.DateTime] = exif_time.encode('utf-8')
                    exif_dict["Exif"][
                        piexif.ExifIFD.DateTimeOriginal] = exif_time.encode(
                            'utf-8')
                    exif_dict["Exif"][
                        piexif.ExifIFD.DateTimeDigitized] = exif_time.encode(
                            'utf-8')

                    piexif.insert(piexif.dump(exif_dict), filepath)
                    os.utime(filepath, (dt.timestamp(), dt.timestamp()))
                except Exception:
                    pass

            return image_id, True
        except Exception:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return image_id, False

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_image, img_id, url, cap_at): img_id
            for img_id, url, cap_at in download_tasks
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Downloading Images"):
            if future.result()[1]:
                success_count += 1
    print(
        f"Downloaded {success_count}/{len(download_tasks)} images to '{output_folder}'."
    )


def process_region(west,
                   south,
                   east,
                   north,
                   run_name,
                   mly_key,
                   max_workers=50):
    """Main function executed by the master script."""
    session = setup_session(max_workers)
    output_folder = os.path.join(run_name, 'ground_animal_images')

    image_to_seq_map = get_image_topology(west, south, east, north, run_name,
                                          session, mly_key, max_workers)
    all_image_ids = list(image_to_seq_map.keys())
    print(f"Total images found in region: {len(all_image_ids)}")

    if not all_image_ids:
        print("No images found in this region. Returning empty.")
        return pd.DataFrame(), pd.DataFrame()

    fields_str = 'id,computed_geometry,captured_at,sequence,is_pano,camera_type,computed_compass_angle,creator,height,width,detections,make,model,thumb_256_url,thumb_1024_url,thumb_2048_url,thumb_original_url'

    detections_data = get_all_image_data_fast(all_image_ids, fields_str,
                                              run_name, session, mly_key,
                                              max_workers)
    animal_detections_dict = get_all_animal_detections_fast(
        image_to_seq_map, run_name, session, mly_key, max_workers)

    extracted_image_ids = set()
    for img_id, features in animal_detections_dict.items():
        if features:
            extracted_image_ids.add(img_id)

    print(f"\nBuilding DataFrames...")
    all_results_df = build_mapillary_dataframe(detections_data)

    if 'captured_at' in all_results_df.columns:
        all_results_df['captured_at'] = pd.to_datetime(
            all_results_df['captured_at'], unit='ms',
            errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

    ground_animals_df = all_results_df[all_results_df['image_id'].isin(
        extracted_image_ids)].copy()

    download_mapillary_images(ground_animals_df, output_folder, max_workers)

    # Cleanup: Move checkpoint files from the 'checkpoints' folder to the final 'run_name' folder
    os.makedirs(run_name, exist_ok=True)
    checkpoints = [
        os.path.join('checkpoints', f'topology_checkpoint_{run_name}.json'),
        os.path.join('checkpoints', f'metadata_checkpoint_{run_name}.jsonl'),
        os.path.join('checkpoints',
                     f'animal_detections_checkpoint_{run_name}.jsonl')
    ]
    for cp in checkpoints:
        if os.path.exists(cp):
            dest = os.path.join(run_name, os.path.basename(cp))
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(cp, run_name)

    return all_results_df, ground_animals_df
