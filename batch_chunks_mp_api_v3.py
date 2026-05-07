import argparse
import compression.zstd as zstd
import gc
import glob
import itertools
import multiprocessing
import os
import queue
import random
import re
import shutil
import signal
import threading
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from datetime import datetime

import mercantile
import orjson
import piexif
import polars as pl
import requests
from dotenv import load_dotenv
from global_land_mask import globe
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

load_dotenv()

# --- Global defaults (Overridden by argparse in __main__) ---
MLY_KEY = None
ZOOM_LEVEL = 14
GRID_CSV_FILE = None
VISUALIZE = False
DOWNLOAD_IMAGES = True
DOWNLOAD_ONLY = False
DOWNLOAD_MAX_WORKERS = 10
OUTER_MAX_WORKERS = 5
INNER_MAX_WORKERS = 20
SUB_GRID_STEP = 1.0
PARENT_DIR = 'grid_runs'
IMAGE_DIR = None
TEMP_DIR = None
API_CHUNK_SIZE = 2500
PARQUET_CHUNK_SIZE = 100000
PROXY_LIST = []
EXCLUDE_SET = set()

ZSTD_OPTIONS = {zstd.CompressionParameter.nb_workers: 2}

# Global event for handling graceful shutdowns across threads/processes
shutdown_event = threading.Event()


def init_worker(config, event):
    """Initializes worker processes with the parsed CLI config and shutdown event (Local MP Mode)."""
    global MLY_KEY, ZOOM_LEVEL, VISUALIZE, DOWNLOAD_IMAGES, DOWNLOAD_ONLY, DOWNLOAD_MAX_WORKERS
    global INNER_MAX_WORKERS, SUB_GRID_STEP, PARENT_DIR, IMAGE_DIR, TEMP_DIR
    global API_CHUNK_SIZE, PARQUET_CHUNK_SIZE, PROXY_LIST, EXCLUDE_SET
    global shutdown_event

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    shutdown_event = event

    MLY_KEY = config['MLY_KEY']
    ZOOM_LEVEL = config['ZOOM_LEVEL']
    VISUALIZE = config['VISUALIZE']
    DOWNLOAD_IMAGES = config['DOWNLOAD_IMAGES']
    DOWNLOAD_ONLY = config['DOWNLOAD_ONLY']
    DOWNLOAD_MAX_WORKERS = config['DOWNLOAD_MAX_WORKERS']
    INNER_MAX_WORKERS = config['INNER_MAX_WORKERS']
    SUB_GRID_STEP = config['SUB_GRID_STEP']
    PARENT_DIR = config['PARENT_DIR']
    IMAGE_DIR = config.get('IMAGE_DIR')
    TEMP_DIR = config.get('TEMP_DIR')
    API_CHUNK_SIZE = config['API_CHUNK_SIZE']
    PARQUET_CHUNK_SIZE = config['PARQUET_CHUNK_SIZE']
    PROXY_LIST = config.get('PROXY_LIST', [])

    ledger_path = config.get('EXCLUDE_LEDGER')
    if ledger_path and os.path.exists(ledger_path):
        with open(ledger_path, 'r') as f:
            EXCLUDE_SET = {line.strip() for line in f if line.strip()}


def slurm_signal_handler(signum, frame):
    """Catches scancel (SIGTERM) or Ctrl+C (SIGINT) in SLURM mode to exit safely."""
    tqdm.write(
        f"\n[!] Shutdown signal ({signum}) received! INSTANT FORCE-KILL INITIATED."
    )
    os._exit(1)


def chunked_iterable(iterable, size):
    """Yields chunks of a specified size from an iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def sanitize_folder_name(name):
    safe_name = name.replace('&', 'and')
    return re.sub(r'[^\w\-_\.]', '_', safe_name).strip('_')


def clean_jsonl_file(filepath):
    """Removes corrupt lines from an interrupted .jsonl.zst file and deletes 0-byte ghosts."""
    if not os.path.exists(filepath):
        return

    if os.path.getsize(filepath) == 0:
        os.remove(filepath)
        return

    is_corrupt = False
    try:
        with zstd.open(filepath, 'rb') as f:
            for line in f:
                pass
    except (EOFError, OSError, zstd.ZstdError):
        is_corrupt = True

    if is_corrupt:
        temp_filepath = filepath + ".tmp"
        valid_lines = 0
        try:
            with zstd.open(filepath, 'rb') as f_in, \
                 zstd.open(temp_filepath, 'wb', options=ZSTD_OPTIONS) as f_out:
                try:
                    for line in f_in:
                        if not line.strip(): continue
                        try:
                            orjson.loads(line)
                            f_out.write(line)
                            valid_lines += 1
                        except orjson.JSONDecodeError:
                            pass
                except (EOFError, OSError, zstd.ZstdError):
                    pass

            if valid_lines == 0:
                os.remove(filepath)
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
            else:
                os.replace(temp_filepath, filepath)
        except Exception:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)


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


def is_valid_image(filepath, image_id, valid_set, lock, ledger_path):
    """Fast validation that remembers healthy files between script runs."""
    img_id_str = str(image_id)

    if img_id_str in valid_set:
        return True

    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) < 100:
        return False

    try:
        with open(filepath, 'rb') as f:
            if f.read(2) == b'\xff\xd8':
                with lock:
                    valid_set.add(img_id_str)
                    with open(ledger_path, 'a') as lf:
                        lf.write(f"{img_id_str}\n")
                return True
    except Exception:
        pass

    return False


def background_hdd_mover(move_queue):
    """Dedicated background thread to sequentially move files from fast SSD to slow HDD."""
    while True:
        item = move_queue.get()
        if item is None:
            break
        temp_path, final_path = item
        try:
            shutil.move(temp_path, final_path)
        except Exception:
            pass
        move_queue.task_done()


def download_single_image(image_id, url, output_folder_name, session,
                          valid_set, lock, ledger_path, temp_folder_name,
                          move_queue):
    """Network I/O download with Keep-Alive, Proxies, and API Fallback for expired URLs."""
    global PROXY_LIST, MLY_KEY
    filepath = os.path.join(output_folder_name, f"{image_id}.jpg")

    if is_valid_image(filepath, image_id, valid_set, lock, ledger_path):
        return image_id, filepath, True, False

    def attempt_download(target_url, max_tries):
        last_err = "Unknown Error"
        for attempt in range(max_tries):
            proxy_dict = None
            if PROXY_LIST:
                proxy_url = random.choice(PROXY_LIST)
                proxy_dict = {"http": proxy_url, "https": proxy_url}

            try:
                response = session.get(target_url,
                                       timeout=(5, 15),
                                       proxies=proxy_dict)
                response.raise_for_status()

                if temp_folder_name and move_queue:
                    temp_filepath = os.path.join(temp_folder_name,
                                                 f"{image_id}.jpg")
                    with open(temp_filepath, 'wb') as f:
                        f.write(response.content)
                    move_queue.put((temp_filepath, filepath))
                else:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)

                with lock:
                    valid_set.add(str(image_id))
                    with open(ledger_path, 'a') as lf:
                        lf.write(f"{str(image_id)}\n")

                return True, None

            except requests.exceptions.HTTPError as e:
                last_err = f"HTTP Error {e.response.status_code if e.response else 'Unknown'}"
                if e.response is not None and e.response.status_code in [
                        400, 401, 403, 404
                ]:
                    return False, last_err
            except Exception as e:
                last_err = str(e) or type(e).__name__
                if temp_folder_name:
                    try:
                        err_tmp = os.path.join(temp_folder_name,
                                               f"{image_id}.jpg")
                        if os.path.exists(err_tmp): os.remove(err_tmp)
                    except Exception:
                        pass

        return False, last_err

    max_attempts = 3 if PROXY_LIST else 2
    success, error = attempt_download(url, max_attempts)

    if success:
        return image_id, filepath, True, True

    tqdm.write(
        f"    [!] Download failed for {image_id} ({error}). Fetching fresh URL..."
    )

    fallback_url = f'https://graph.mapillary.com/{image_id}?access_token={MLY_KEY}&fields=thumb_original_url'
    try:
        fb_response = session.get(fallback_url, timeout=10)
        fb_response.raise_for_status()
        new_url = fb_response.json().get('thumb_original_url')

        if new_url:
            success, fallback_error = attempt_download(new_url, max_attempts)
            if success:
                tqdm.write(
                    f"    [\u2713] Successfully recovered {image_id} using fresh URL."
                )
                return image_id, filepath, True, True
            else:
                error = f"Fallback download failed: {fallback_error}"
        else:
            error = "API returned no thumb_original_url."
    except Exception as e:
        error = f"Failed to fetch fresh URL: {e}"

    if os.path.exists(filepath):
        os.remove(filepath)

    tqdm.write(
        f"\033[91m    [X] Fatal Download Error ({image_id}): {error}\033[0m")
    return image_id, filepath, False, False


def apply_exif_data(filepath, captured_at):
    """Pure CPU-bound task applying metadata timestamps to downloaded images."""
    if not captured_at: return
    try:
        val = str(captured_at)
        if '-' in val:
            dt = datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
        else:
            dt = datetime.fromtimestamp(int(val) / 1000.0)

        exif_time = dt.strftime("%Y:%m:%d %H:%M:%S")

        exif_dict = {
            "0th": {},
            "Exif": {},
            "GPS": {},
            "1st": {},
            "Interop": {}
        }
        exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_time.encode('utf-8')
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_time.encode(
            'utf-8')
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_time.encode(
            'utf-8')

        piexif.insert(piexif.dump(exif_dict), filepath)
        os.utime(filepath, (dt.timestamp(), dt.timestamp()))
    except Exception as e:
        tqdm.write(f"\033[91m    [X] EXIF Error ({filepath}): {e}\033[0m")


def build_mapillary_dataframe_from_records(records):
    """Builds a strictly typed Polars DataFrame safe from Float/Int type drifts."""
    if not records: return pl.DataFrame()

    MAPILLARY_SCHEMA = {
        'image_id': pl.Utf8,
        'captured_at': pl.Int64,
        'sequence': pl.Utf8,
        'is_pano': pl.Boolean,
        'camera_type': pl.Utf8,
        'computed_compass_angle': pl.Float64,
        'height': pl.Float64,
        'width': pl.Float64,
        'make': pl.Utf8,
        'model': pl.Utf8,
        'thumb_256_url': pl.Utf8,
        'thumb_1024_url': pl.Utf8,
        'thumb_2048_url': pl.Utf8,
        'thumb_original_url': pl.Utf8
    }

    for record in records:
        if record.get('computed_geometry'):
            record['computed_geometry'] = orjson.dumps(
                record['computed_geometry']).decode('utf-8')
        if record.get('creator'):
            record['creator'] = orjson.dumps(record['creator']).decode('utf-8')
        if record.get('detections'):
            record['detections'] = orjson.dumps(
                record['detections']).decode('utf-8')

    df = pl.DataFrame(records,
                      schema_overrides=MAPILLARY_SCHEMA,
                      infer_schema_length=10000)

    expected_columns = list(MAPILLARY_SCHEMA.keys()) + [
        'computed_geometry', 'creator', 'detections'
    ]

    for col in expected_columns:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    df = df.with_columns([
        pl.col('computed_geometry').cast(pl.Utf8),
        pl.col('creator').cast(pl.Utf8),
        pl.col('detections').cast(pl.Utf8),
        pl.col('height').cast(pl.Int64, strict=False),
        pl.col('width').cast(pl.Int64, strict=False)
    ])

    return df.select(expected_columns)


def get_image_topology(west, south, east, north, region_dir, sub_id, session,
                       pos, desc_prefix):
    checkpoint_file = os.path.join(region_dir,
                                   f'topology_checkpoint_{sub_id}.json.zst')
    if os.path.exists(checkpoint_file):
        try:
            with zstd.open(checkpoint_file, 'rb') as f:
                return orjson.loads(f.read())
        except (orjson.JSONDecodeError, zstd.ZstdError):
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
        with tqdm(total=len(land_bboxes),
                  desc=f"{desc_prefix} 1/6 BBoxes",
                  position=pos,
                  leave=False,
                  mininterval=2.0) as pbar:
            for chunk in chunked_iterable(land_bboxes, API_CHUNK_SIZE):
                if shutdown_event.is_set(): break
                futures = {
                    executor.submit(get_sequences_for_bbox, bbox, session):
                    bbox
                    for bbox in chunk
                }
                for future in as_completed(futures):
                    if shutdown_event.is_set():
                        for f in futures:
                            f.cancel()
                        break

                    unique_sequences.update(future.result())
                    pbar.update(1)
                    del futures[future]
                gc.collect()

    image_to_sequence_map = {}
    with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
        with tqdm(total=len(unique_sequences),
                  desc=f"{desc_prefix} 2/6 Sequences",
                  position=pos,
                  leave=False,
                  mininterval=2.0) as pbar:
            for chunk in chunked_iterable(unique_sequences, API_CHUNK_SIZE):
                if shutdown_event.is_set(): break
                futures = {
                    executor.submit(get_images_for_sequence, seq, session): seq
                    for seq in chunk
                }
                for future in as_completed(futures):
                    if shutdown_event.is_set():
                        for f in futures:
                            f.cancel()
                        break

                    seq_id = futures[future]
                    for img_id in future.result():
                        image_to_sequence_map[img_id] = seq_id
                    pbar.update(1)
                    del futures[future]
                gc.collect()

    if not shutdown_event.is_set():
        temp_checkpoint = checkpoint_file + '.tmp'
        with zstd.open(temp_checkpoint, 'wb', options=ZSTD_OPTIONS) as f:
            f.write(orjson.dumps(image_to_sequence_map))
        os.replace(temp_checkpoint, checkpoint_file)

    return image_to_sequence_map


def fetch_metadata_to_jsonl(image_ids, fields_str, region_dir, sub_id, session,
                            pos, desc_prefix):
    checkpoint_file = os.path.join(region_dir,
                                   f'metadata_checkpoint_{sub_id}.jsonl.zst')

    clean_jsonl_file(checkpoint_file)

    completed_ids = set()
    if os.path.exists(checkpoint_file):
        with zstd.open(checkpoint_file, 'rb') as f:
            for line in f:
                if line.strip():
                    match = re.search(rb'"image_id"\s*:\s*"([^"]+)"', line)
                    if match:
                        completed_ids.add(match.group(1).decode('utf-8'))

    missing_ids = [
        img_id for img_id in image_ids if img_id not in completed_ids
    ]
    if not missing_ids: return

    write_lock = threading.Lock()
    with zstd.open(checkpoint_file, 'ab', options=ZSTD_OPTIONS) as f:
        with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
            with tqdm(total=len(missing_ids),
                      desc=f"{desc_prefix} 3/6 Metadata",
                      position=pos,
                      leave=False,
                      mininterval=2.0) as pbar:
                for chunk in chunked_iterable(missing_ids, API_CHUNK_SIZE):
                    if shutdown_event.is_set(): break
                    futures = {
                        executor.submit(fetch_image_data, img_id, fields_str, session):
                        img_id
                        for img_id in chunk
                    }
                    for future in as_completed(futures):
                        if shutdown_event.is_set():
                            for f in futures:
                                f.cancel()
                            break

                        image_id, data = future.result()
                        if data is not None:
                            with write_lock:
                                f.write(
                                    orjson.dumps({
                                        'image_id': image_id,
                                        'data': data
                                    }) + b'\n')
                        pbar.update(1)

                        del futures[future]
                        del data
                        del future

                    f.flush()
                    gc.collect()


def fetch_detections_to_jsonl(image_to_seq_map, region_dir, sub_id, session,
                              pos, desc_prefix):
    checkpoint_file = os.path.join(
        region_dir, f'animal_detections_checkpoint_{sub_id}.jsonl.zst')

    clean_jsonl_file(checkpoint_file)

    completed_ids = set()
    if os.path.exists(checkpoint_file):
        with zstd.open(checkpoint_file, 'rb') as f:
            for line in f:
                if line.strip():
                    match = re.search(rb'"image_id"\s*:\s*"([^"]+)"', line)
                    if match:
                        completed_ids.add(match.group(1).decode('utf-8'))

    missing_ids = [
        img_id for img_id in image_to_seq_map.keys()
        if img_id not in completed_ids
    ]
    if not missing_ids: return

    write_lock = threading.Lock()
    with zstd.open(checkpoint_file, 'ab', options=ZSTD_OPTIONS) as f:
        with ThreadPoolExecutor(max_workers=INNER_MAX_WORKERS) as executor:
            with tqdm(total=len(missing_ids),
                      desc=f"{desc_prefix} 4/6 Detections",
                      position=pos,
                      leave=False,
                      mininterval=2.0) as pbar:
                for chunk in chunked_iterable(missing_ids, API_CHUNK_SIZE):
                    if shutdown_event.is_set(): break
                    futures = {
                        executor.submit(fetch_animal_detections, img_id, image_to_seq_map[img_id], session):
                        img_id
                        for img_id in chunk
                    }
                    for future in as_completed(futures):
                        if shutdown_event.is_set():
                            for f in futures:
                                f.cancel()
                            break

                        image_id, features = future.result()
                        with write_lock:
                            f.write(
                                orjson.dumps({
                                    'image_id': image_id,
                                    'features': features
                                }) + b'\n')
                        pbar.update(1)
                        del futures[future]
                        del features
                        del future

                    f.flush()
                    gc.collect()


# --- The Master Process Worker ---
def process_region(west,
                   south,
                   east,
                   north,
                   unique_region_id,
                   run_name,
                   pos=None):
    if shutdown_event.is_set():
        return f"Aborted '{unique_region_id}' safely."

    if pos is None:
        worker_name = multiprocessing.current_process().name
        match = re.search(r'\d+', worker_name)
        pos = int(match.group()) if match else 1

    safe_region_id = sanitize_folder_name(unique_region_id)
    region_run_id = unique_region_id.replace('_', ' ')
    region_dir = os.path.join(PARENT_DIR, safe_region_id)
    os.makedirs(region_dir, exist_ok=True)
    if IMAGE_DIR:
        output_folder_name = os.path.join(IMAGE_DIR, safe_region_id,
                                          'ground_animal_images')
    else:
        output_folder_name = os.path.join(region_dir, 'ground_animal_images')
    os.makedirs(output_folder_name, exist_ok=True)

    if TEMP_DIR:
        temp_folder_name = os.path.join(TEMP_DIR, safe_region_id,
                                        'ground_animal_images')
        os.makedirs(temp_folder_name, exist_ok=True)

        move_queue = queue.Queue(maxsize=2000)
        mover_thread = threading.Thread(target=background_hdd_mover,
                                        args=(move_queue, ),
                                        daemon=True)
        mover_thread.start()
    else:
        temp_folder_name = None
        move_queue = None
        mover_thread = None

    valid_ledger_path = os.path.join(region_dir,
                                     f'validated_images_{safe_region_id}.txt')
    validated_set = set()
    if os.path.exists(valid_ledger_path):
        with open(valid_ledger_path, 'r') as f:
            validated_set = {line.strip() for line in f if line.strip()}
    valid_lock = threading.Lock()

    # =========================================================
    #                    DOWNLOAD-ONLY MODE
    # =========================================================
    if DOWNLOAD_ONLY:
        animal_files = glob.glob(
            os.path.join(region_dir,
                         f'ground_animals_{safe_region_id}_*.parquet'))

        if not animal_files:
            return f"Skipped '{unique_region_id}': No ground_animals Parquet found for download-only mode."

        tqdm.write(
            f"[{datetime.now().strftime('%H:%M:%S')}] [{region_run_id}] Loading Parquet chunks for Download-Only mode..."
        )

        try:
            dfs = [
                pl.read_parquet(
                    f,
                    columns=['image_id', 'thumb_original_url', 'captured_at'])
                for f in animal_files
            ]
            df = pl.concat(dfs)

            if 'captured_at' not in df.columns:
                df = df.with_columns(pl.lit(None).alias('captured_at'))

            download_tasks = []
            for row in df.iter_rows(named=True):
                if row.get('thumb_original_url') is not None:
                    if str(row['image_id']) in EXCLUDE_SET:
                        continue
                    expected_filepath = os.path.join(output_folder_name,
                                                     f"{row['image_id']}.jpg")
                    if not is_valid_image(expected_filepath, row['image_id'],
                                          validated_set, valid_lock,
                                          valid_ledger_path):
                        download_tasks.append(
                            (row['image_id'], row['thumb_original_url'],
                             row['captured_at']))

        except Exception as e:
            return f"Error reading Parquet for '{unique_region_id}': {e}"

        if not download_tasks:
            return f"Completed '{unique_region_id}' (No images to download)."

        tqdm.write(
            f"[{datetime.now().strftime('%H:%M:%S')}] [{region_run_id}] Proceeding to download {len(download_tasks)} images..."
        )

        dl_session = requests.Session()
        dl_session.mount(
            'https://',
            HTTPAdapter(pool_connections=DOWNLOAD_MAX_WORKERS,
                        pool_maxsize=DOWNLOAD_MAX_WORKERS))

        # ---------------------------------------------------------
        # PHASE 1: Pure Network Download (No EXIF in this mode)
        # ---------------------------------------------------------
        with ThreadPoolExecutor(max_workers=DOWNLOAD_MAX_WORKERS) as executor:
            with tqdm(total=len(download_tasks),
                      desc=f"[{region_run_id}] Downloads",
                      position=pos,
                      leave=False,
                      mininterval=2.0) as pbar:
                for chunk in chunked_iterable(download_tasks, API_CHUNK_SIZE):
                    if shutdown_event.is_set(): break

                    futures = {
                        executor.submit(download_single_image, img_id, url, output_folder_name, dl_session, validated_set, valid_lock, valid_ledger_path, temp_folder_name, move_queue):
                        cap_at
                        for img_id, url, cap_at in chunk
                    }

                    for future in as_completed(futures):
                        if shutdown_event.is_set():
                            for f in futures:
                                f.cancel()
                            break

                        img_id, filepath, success, newly_downloaded = future.result(
                        )

                        if not success:
                            failed_tracker_path = os.path.join(
                                region_dir,
                                f'failed_downloads_{safe_region_id}.txt')
                            with open(failed_tracker_path, 'a') as f:
                                f.write(f"{img_id}\n")

                        pbar.update(1)
                        del futures[future]
                    gc.collect()

        if mover_thread:
            move_queue.join()
            move_queue.put(None)
            mover_thread.join()

        return f"Completed '{unique_region_id}' (Processed {len(download_tasks)} images via Download-Only mode)."
    # =========================================================

    session = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504])
    session.mount(
        'https://',
        HTTPAdapter(max_retries=retries,
                    pool_connections=INNER_MAX_WORKERS,
                    pool_maxsize=INNER_MAX_WORKERS))

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

    # ====================================================================
    # ARCHITECTURE FIX: GLOBALLY TRACK BUFFERS ACROSS ALL SUB-GRIDS
    # ====================================================================
    total_animals_found = 0
    records = []
    download_tasks = []
    global_extracted_image_ids = set()

    seen_image_ids = set()
    existing_all_files = glob.glob(
        os.path.join(region_dir, f'all_data_{safe_region_id}_*.parquet'))

    if existing_all_files:
        existing_all_files.sort()
        tqdm.write(
            f"\n[{region_run_id}] Scanning existing Parquet files to prevent duplicates..."
        )
        for f in existing_all_files:
            chunk_num = f.split('_')[-1].replace('.parquet', '')
            all_size_mb = os.path.getsize(f) / (1024 * 1024)

            anim_f = f.replace('all_data_', 'ground_animals_')
            anim_size_mb = os.path.getsize(anim_f) / (
                1024 * 1024) if os.path.exists(anim_f) else 0.0

            tqdm.write(
                f"    [\u2713] Found Parquet Chunk {chunk_num} -> All Data: {all_size_mb:.2f} MB | Animals: {anim_size_mb:.2f} MB"
            )

            seen_image_ids.update(
                pl.read_parquet(f, columns=['image_id'])['image_id'].to_list())

    part_index = len(existing_all_files)
    cumulative_bypassed = 0

    existing_anim_files = glob.glob(
        os.path.join(region_dir, f'ground_animals_{safe_region_id}_*.parquet'))
    anim_part_index = len(existing_anim_files)
    animal_df_buffer = []

    def process_metadata_chunk(chunk_records):
        nonlocal part_index, cumulative_bypassed
        original_count = len(chunk_records)
        df = build_mapillary_dataframe_from_records(chunk_records)
        if df.is_empty(): return

        if 'captured_at' in df.columns:
            df = df.with_columns(
                pl.col('captured_at').cast(pl.Int64, strict=False).cast(
                    pl.Datetime("ms")).dt.strftime('%Y-%m-%d %H:%M:%S'))

        df = df.unique(subset=['image_id'], keep='last')

        df = df.filter(~pl.col('image_id').is_in(seen_image_ids))

        if df.is_empty():
            cumulative_bypassed += original_count
            tqdm.write(
                f"    [i] Bypassed {cumulative_bypassed:,} total records so far (safely stored on disk)..."
            )
            return

        seen_image_ids.update(df['image_id'].to_list())

        animals_df = df.filter(
            pl.col('image_id').is_in(global_extracted_image_ids))

        all_pq_path = os.path.join(
            region_dir, f'all_data_{safe_region_id}_{part_index:03d}.parquet')

        df.write_parquet(all_pq_path, compression='zstd')

        all_size_mb = os.path.getsize(all_pq_path) / (1024 * 1024)
        anim_size_mb = 0.0

        if not animals_df.is_empty():
            anim_pq_path = os.path.join(
                region_dir,
                f'ground_animals_{safe_region_id}_{part_index:03d}.parquet')

            animals_df.write_parquet(anim_pq_path, compression='zstd')
            anim_size_mb = os.path.getsize(anim_pq_path) / (1024 * 1024)

            if DOWNLOAD_IMAGES:
                for row in animals_df.iter_rows(named=True):
                    if row.get('thumb_original_url') is not None:
                        if str(row['image_id']) in EXCLUDE_SET:
                            continue
                        expected_filepath = os.path.join(
                            output_folder_name, f"{row['image_id']}.jpg")
                        if not is_valid_image(expected_filepath,
                                              row['image_id'], validated_set,
                                              valid_lock, valid_ledger_path):
                            download_tasks.append(
                                (row['image_id'], row['thumb_original_url'],
                                 row['captured_at']))

        tqdm.write(
            f"    [\u2713] Saved Parquet Chunk {part_index:03d} -> All Data: {all_size_mb:.2f} MB | Animals: {anim_size_mb:.2f} MB"
        )

        part_index += 1
        del df
        del animals_df
        gc.collect()

    # ====================================================================

    subgrids_processed = 0

    for i, (sw_lon, sw_lat, ne_lon, ne_lat) in enumerate(sub_bboxes):
        if shutdown_event.is_set():
            return f"Aborted '{unique_region_id}' safely."

        sub_id = f"{safe_region_id}_sub_{i}"
        desc_prefix = f"[{region_run_id} C{i+1}/{len(sub_bboxes)}]"

        empty_marker = os.path.join(region_dir, f'.empty_{sub_id}')
        completed_marker = os.path.join(region_dir, f'.completed_{sub_id}')

        if os.path.exists(completed_marker):
            continue

        if os.path.exists(empty_marker):
            continue

        subgrids_processed += 1

        image_to_seq_map = get_image_topology(sw_lon, sw_lat, ne_lon, ne_lat,
                                              region_dir, sub_id, session, pos,
                                              desc_prefix)
        if not image_to_seq_map or shutdown_event.is_set():
            if shutdown_event.is_set():
                return f"Aborted '{unique_region_id}' safely."

            open(empty_marker, 'a').close()
            continue

        fields_str = 'id,computed_geometry,captured_at,sequence,is_pano,camera_type,computed_compass_angle,creator,height,width,detections,make,model,thumb_256_url,thumb_1024_url,thumb_2048_url,thumb_original_url'
        fetch_metadata_to_jsonl(list(image_to_seq_map.keys()), fields_str,
                                region_dir, sub_id, session, pos, desc_prefix)
        if shutdown_event.is_set():
            return f"Aborted '{unique_region_id}' safely."

        fetch_detections_to_jsonl(image_to_seq_map, region_dir, sub_id,
                                  session, pos, desc_prefix)
        if shutdown_event.is_set():
            return f"Aborted '{unique_region_id}' safely."

        animal_checkpoint = os.path.join(
            region_dir, f'animal_detections_checkpoint_{sub_id}.jsonl.zst')
        metadata_checkpoint = os.path.join(
            region_dir, f'metadata_checkpoint_{sub_id}.jsonl.zst')

        anim_size_mb = os.path.getsize(animal_checkpoint) / (
            1024 * 1024) if os.path.exists(animal_checkpoint) else 0
        meta_size_mb = os.path.getsize(metadata_checkpoint) / (
            1024 * 1024) if os.path.exists(metadata_checkpoint) else 0

        if anim_size_mb > 0 or meta_size_mb > 0:
            tqdm.write(
                f"{desc_prefix} Parsing Data -> Metadata: {meta_size_mb:.2f} MB | Animals: {anim_size_mb:.2f} MB"
            )

        extracted_image_ids = set()
        ground_animal_features = []

        if os.path.exists(animal_checkpoint):
            try:
                with zstd.open(animal_checkpoint, 'rb') as f:
                    for line in f:
                        if shutdown_event.is_set(): break
                        if not line.strip(): continue
                        record = orjson.loads(line)
                        features = record.get('features', [])
                        if features:
                            global_extracted_image_ids.add(record['image_id'])
                            ground_animal_features.extend(features)
            except Exception as e:
                tqdm.write(
                    f"\033[91m\n[!] SEVERE CORRUPTION IN: {animal_checkpoint}\033[0m"
                )
                tqdm.write(
                    f"\033[91m    -> Auto-deleting unrecoverable file. Will skip for now. Re-run script later to backfill.\033[0m"
                )
                os.remove(animal_checkpoint)

        if shutdown_event.is_set():
            return f"Aborted '{unique_region_id}' safely."

        if ground_animal_features:
            final_json_path = os.path.join(
                region_dir, f'ground_animals_{sub_id}.json.zst')
            temp_json_path = final_json_path + '.tmp'

            with zstd.open(temp_json_path, 'wb', options=ZSTD_OPTIONS) as f:
                f.write(orjson.dumps(ground_animal_features))

            os.replace(temp_json_path, final_json_path)

        if os.path.exists(metadata_checkpoint):
            try:
                with zstd.open(metadata_checkpoint, 'rb') as f:
                    for line in f:
                        if shutdown_event.is_set(): break
                        if not line.strip(): continue
                        record = orjson.loads(line)

                        row = record.get('data')
                        if not row: continue

                        row = row.copy()
                        row['image_id'] = record['image_id']
                        if 'detections' in row and isinstance(
                                row['detections'], dict):
                            row['detections'] = row['detections'].get(
                                'data', [])
                        records.append(row)

                        if len(records) >= PARQUET_CHUNK_SIZE:
                            process_metadata_chunk(records)
                            records = []
            except Exception as e:
                tqdm.write(
                    f"\n[!] SEVERE CORRUPTION IN: {metadata_checkpoint}")
                tqdm.write(
                    f"    -> Auto-deleting unrecoverable file. Will skip for now. Re-run script later to backfill."
                )
                os.remove(metadata_checkpoint)
                records = []

        if records:
            process_metadata_chunk(records)
            records = []

        if not shutdown_event.is_set():
            open(completed_marker, 'a').close()

    # ====================================================================
    # OUTSIDE THE SUB-GRID LOOP (FINAL WRAP UP)
    # ====================================================================
    if shutdown_event.is_set():
        return f"Aborted '{unique_region_id}' safely."

    if records:
        process_metadata_chunk(records)
        records = []

    final_anim_files = glob.glob(
        os.path.join(region_dir, f'ground_animals_{safe_region_id}_*.parquet'))
    if final_anim_files:
        total_animals_found = pl.scan_parquet(final_anim_files).select(
            'image_id').unique().collect().height
    else:
        total_animals_found = len(global_extracted_image_ids)

    if DOWNLOAD_IMAGES:
        exif_tasks = []

        if final_anim_files:
            queued_ids = {t[0] for t in download_tasks}

            df_dl = pl.scan_parquet(final_anim_files).select(
                ['image_id', 'thumb_original_url', 'captured_at']).collect()

            if 'captured_at' not in df_dl.columns:
                df_dl = df_dl.with_columns(pl.lit(None).alias('captured_at'))

            for row in df_dl.iter_rows(named=True):
                img_id = row['image_id']
                if img_id not in queued_ids and row.get(
                        'thumb_original_url') is not None and str(
                            img_id) not in EXCLUDE_SET:
                    expected_filepath = os.path.join(output_folder_name,
                                                     f"{img_id}.jpg")

                    if not is_valid_image(expected_filepath, img_id,
                                          validated_set, valid_lock,
                                          valid_ledger_path):
                        download_tasks.append(
                            (img_id, row['thumb_original_url'],
                             row['captured_at']))
                    else:
                        if row['captured_at']:
                            try:
                                val = str(row['captured_at'])
                                if '-' in val:
                                    dt = datetime.strptime(
                                        val, "%Y-%m-%d %H:%M:%S")
                                else:
                                    dt = datetime.fromtimestamp(
                                        int(val) / 1000.0)

                                if int(os.path.getmtime(
                                        expected_filepath)) != int(
                                            dt.timestamp()):
                                    exif_tasks.append((expected_filepath,
                                                       row['captured_at']))
                            except Exception:
                                pass

        if download_tasks:
            dl_session = requests.Session()
            dl_session.mount(
                'https://',
                HTTPAdapter(pool_connections=DOWNLOAD_MAX_WORKERS,
                            pool_maxsize=DOWNLOAD_MAX_WORKERS))

            with ThreadPoolExecutor(
                    max_workers=DOWNLOAD_MAX_WORKERS) as executor:
                with tqdm(total=len(download_tasks),
                          desc=f"[{region_run_id}] 5/6 Downloads",
                          position=pos,
                          leave=False,
                          mininterval=2.0) as pbar:
                    for chunk in chunked_iterable(download_tasks,
                                                  API_CHUNK_SIZE):
                        if shutdown_event.is_set(): break
                        futures = {
                            executor.submit(download_single_image, img_id, url, output_folder_name, dl_session, validated_set, valid_lock, valid_ledger_path, temp_folder_name, move_queue):
                            cap_at
                            for img_id, url, cap_at in chunk
                        }
                        for future in as_completed(futures):
                            if shutdown_event.is_set():
                                for f in futures:
                                    f.cancel()
                                break

                            cap_at = futures[future]
                            img_id, filepath, success, newly_downloaded = future.result(
                            )

                            if success and newly_downloaded:
                                exif_tasks.append((filepath, cap_at))
                            elif not success:
                                failed_tracker_path = os.path.join(
                                    region_dir,
                                    f'failed_downloads_{safe_region_id}.txt')
                                with open(failed_tracker_path, 'a') as f:
                                    f.write(f"{img_id}\n")

                            pbar.update(1)
                            del futures[future]
                        gc.collect()

            # --- CRITICAL: Wait for background moves to finish BEFORE checking EXIF ---
            if mover_thread:
                move_queue.join()

        if exif_tasks and not shutdown_event.is_set():
            with ThreadPoolExecutor(
                    max_workers=DOWNLOAD_MAX_WORKERS) as executor:
                with tqdm(total=len(exif_tasks),
                          desc=f"[{region_run_id}] 6/6 EXIF Data",
                          position=pos,
                          leave=False,
                          mininterval=2.0) as pbar:
                    for chunk in chunked_iterable(exif_tasks, API_CHUNK_SIZE):
                        if shutdown_event.is_set(): break
                        futures = [
                            executor.submit(apply_exif_data, fp, cat)
                            for fp, cat in chunk
                        ]
                        for future in as_completed(futures):
                            if shutdown_event.is_set():
                                for f in futures:
                                    f.cancel()
                                break
                            pbar.update(1)
                        gc.collect()

    del download_tasks
    gc.collect()

    if mover_thread:
        move_queue.put(None)
        mover_thread.join()

    return f"Completed '{unique_region_id}' ({total_animals_found} animals found across all chunks)."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mapillary Ground Animal API Downloader")
    parser.add_argument(
        'grid_csv_file',
        type=str,
        help="Path to the input CSV file (e.g., indian_ocean.csv)")
    parser.add_argument(
        '--slurm',
        action='store_true',
        help=
        "Run in SLURM Array mode (processes 1 task mapping to SLURM_ARRAY_TASK_ID)"
    )
    parser.add_argument('--zoom-level',
                        type=int,
                        default=14,
                        help="Zoom level for bounding boxes (default: 14)")
    parser.add_argument('--visualize',
                        action='store_true',
                        help="Enable visualizer flag (default: False)")
    parser.add_argument('--no-download-images',
                        dest='download_images',
                        action='store_false',
                        help="Disable downloading images (default: True)")
    parser.add_argument(
        '--download-only',
        action='store_true',
        help=
        "Skip API fetching/generation and ONLY download images from existing ground_animals Parquets."
    )
    parser.add_argument(
        '--download-max-workers',
        type=int,
        default=10,
        help="Max threads specifically for image downloading (default: 10)")
    parser.add_argument('--outer-max-workers',
                        type=int,
                        default=5,
                        help="Max parallel regions (default: 5)")
    parser.add_argument('--inner-max-workers',
                        type=int,
                        default=20,
                        help="Max threads per region (default: 20)")
    parser.add_argument('--sub-grid-step',
                        type=float,
                        default=1.0,
                        help="Step size for internal chunking (default: 1.0)")
    parser.add_argument('--parent-dir',
                        type=str,
                        default='grid_runs',
                        help="Output directory (default: grid_runs)")
    parser.add_argument(
        '--token',
        type=int,
        default=None,
        help="Token index to use (e.g., 1 for MLY_KEY_1). Defaults to MLY_KEY")
    parser.add_argument(
        '--api-chunk-size',
        type=int,
        default=2500,
        help="Chunk size for multithreaded API requests (default: 2500)")
    parser.add_argument(
        '--parquet-chunk-size',
        type=int,
        default=100000,
        help="Rows per Parquet partition file (default: 100000)")
    parser.add_argument(
        '--proxy-file',
        type=str,
        default=None,
        help="Path to a txt file containing proxies (e.g., free-proxy-list.txt)"
    )
    parser.add_argument(
        '--exclude-ledger',
        type=str,
        default=None,
        help=
        "Path to a txt file containing image IDs to skip (e.g., completed_ledger.txt)"
    )
    parser.add_argument('--image-dir',
                        type=str,
                        default=None,
                        help="Output directory specifically for images (HDD)")
    parser.add_argument(
        '--row-index',
        type=int,
        default=None,
        help=
        "Process only a specific 0-indexed row from the CSV (e.g., 0 for the first data row)."
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default=None,
        help="Fast SSD temp directory to spool downloads before moving to HDD")

    args = parser.parse_args()

    GRID_CSV_FILE = args.grid_csv_file
    ZOOM_LEVEL = args.zoom_level
    VISUALIZE = args.visualize
    DOWNLOAD_IMAGES = args.download_images
    DOWNLOAD_ONLY = args.download_only
    DOWNLOAD_MAX_WORKERS = args.download_max_workers
    OUTER_MAX_WORKERS = args.outer_max_workers
    INNER_MAX_WORKERS = args.inner_max_workers
    SUB_GRID_STEP = args.sub_grid_step
    PARENT_DIR = args.parent_dir
    IMAGE_DIR = args.image_dir
    TEMP_DIR = args.temp_dir
    API_CHUNK_SIZE = args.api_chunk_size
    PARQUET_CHUNK_SIZE = args.parquet_chunk_size

    proxy_list = []
    if args.proxy_file and os.path.exists(args.proxy_file):
        with open(args.proxy_file, 'r') as pf:
            for line in pf:
                line = line.strip()
                if not line: continue

                parts = line.split(':')

                if len(parts) == 4:
                    ip, port, user, password = parts
                    formatted_proxy = f"http://{user}:{password}@{ip}:{port}"
                    proxy_list.append(formatted_proxy)

                elif len(parts) == 2:
                    ip, port = parts
                    formatted_proxy = f"http://{ip}:{port}"
                    proxy_list.append(formatted_proxy)

                elif line.startswith(('http', 'socks')):
                    proxy_list.append(line)

        print(f"Loaded {len(proxy_list)} proxies from {args.proxy_file}")

    token_env_name = f"MLY_KEY_{args.token}" if args.token else "MLY_KEY"
    MLY_KEY = os.environ.get(token_env_name)

    if not MLY_KEY:
        raise ValueError(f"{token_env_name} environment variable is missing.")

    print('\033[?25l', end="")
    os.makedirs(PARENT_DIR, exist_ok=True)

    df_grid = pl.read_csv(GRID_CSV_FILE)
    GLOBAL_RUN_NAME = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.slurm:
        # ==========================================
        #           SLURM EXECUTION PATH
        # ==========================================
        tqdm.set_lock(threading.RLock())
        signal.signal(signal.SIGINT, slurm_signal_handler)
        signal.signal(signal.SIGTERM, slurm_signal_handler)

        if args.row_index is not None:
            task_id = args.row_index
        else:
            task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

        if task_id >= df_grid.height:
            print(
                f"Task ID {task_id} is out of bounds for grid with {df_grid.height} regions."
            )
            exit(0)

        row = df_grid[task_id].to_dicts()[0]
        unique_region_id = f"{row['region']}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"

        tqdm.write(
            f"Task {task_id}: Initiating single-region processing for {unique_region_id}. Run ID: {GLOBAL_RUN_NAME}"
        )

        try:
            PROXY_LIST = proxy_list

            if args.exclude_ledger and os.path.exists(args.exclude_ledger):
                with open(args.exclude_ledger, 'r') as f:
                    EXCLUDE_SET = {line.strip() for line in f if line.strip()}

            result_msg = process_region(row['sw_lon'],
                                        row['sw_lat'],
                                        row['ne_lon'],
                                        row['ne_lat'],
                                        unique_region_id,
                                        GLOBAL_RUN_NAME,
                                        pos=1)

            if "Aborted" in result_msg:
                tqdm.write(f"[-] {result_msg}")
            else:
                tqdm.write(f"[\u2713] {result_msg}")

        except Exception as exc:
            tqdm.write(f"[X] Region '{unique_region_id}' failed: {exc}")
        finally:
            print('\033[?25h', end="")

    else:
        # ==========================================
        #        LOCAL MULTIPROCESSING PATH
        # ==========================================
        if args.row_index is not None:
            if args.row_index < 0 or args.row_index >= df_grid.height:
                print(
                    f"Error: Row index {args.row_index} is out of bounds for grid with {df_grid.height} regions."
                )
                exit(0)
            df_grid = df_grid.slice(args.row_index, 1)

        tasks_to_run = [(
            row['sw_lon'], row['sw_lat'], row['ne_lon'], row['ne_lat'],
            f"{row['region']}_{row['sw_lon']}_{row['sw_lat']}_{row['ne_lon']}_{row['ne_lat']}"
        ) for row in df_grid.iter_rows(named=True)]

        if not tasks_to_run:
            print(f"All {df_grid.height} regions have already been completed.")
            print('\033[?25h', end="")
            exit(0)

        tqdm.write(
            f"Initiating multi-region processing. {len(tasks_to_run)} tasks remaining. Run ID: {GLOBAL_RUN_NAME}"
        )

        try:
            m = multiprocessing.Manager()
            shutdown_event = m.Event()

            worker_config = {
                'MLY_KEY': MLY_KEY,
                'ZOOM_LEVEL': ZOOM_LEVEL,
                'VISUALIZE': VISUALIZE,
                'DOWNLOAD_IMAGES': DOWNLOAD_IMAGES,
                'DOWNLOAD_ONLY': args.download_only,
                'DOWNLOAD_MAX_WORKERS': args.download_max_workers,
                'INNER_MAX_WORKERS': INNER_MAX_WORKERS,
                'SUB_GRID_STEP': SUB_GRID_STEP,
                'PARENT_DIR': PARENT_DIR,
                'IMAGE_DIR': args.image_dir,
                'TEMP_DIR': args.temp_dir,
                'API_CHUNK_SIZE': API_CHUNK_SIZE,
                'PARQUET_CHUNK_SIZE': PARQUET_CHUNK_SIZE,
                'PROXY_LIST': proxy_list,
                'EXCLUDE_LEDGER': args.exclude_ledger
            }

            with ProcessPoolExecutor(max_workers=OUTER_MAX_WORKERS,
                                     initializer=init_worker,
                                     initargs=(worker_config,
                                               shutdown_event)) as executor:
                outer_futures = {
                    executor.submit(process_region, sw_lon, sw_lat, ne_lon, ne_lat, region_id, GLOBAL_RUN_NAME):
                    region_id
                    for sw_lon, sw_lat, ne_lon, ne_lat, region_id in
                    tasks_to_run
                }

                try:
                    for future in tqdm(as_completed(outer_futures),
                                       total=len(outer_futures),
                                       desc="Total Progress",
                                       position=0,
                                       leave=True):
                        region_id = outer_futures[future]
                        try:
                            result_msg = future.result()
                            if "Aborted" in result_msg:
                                tqdm.write(f"[-] {result_msg}")
                            else:
                                tqdm.write(f"[\u2713] {result_msg}")
                        except Exception as exc:
                            tqdm.write(
                                f"[X] Region '{region_id}' failed: {exc}")
                except KeyboardInterrupt:
                    tqdm.write(
                        "\n\n[!] Ctrl+C detected! INSTANT FORCE-KILL INITIATED."
                    )
                    tqdm.write(
                        "[!] Warning: Terminating abruptly. The current file being written may be corrupted."
                    )
                    os._exit(1)

        finally:
            print('\033[?25h', end="")
