"""
Step 4 -- targeted backfill of the audit's in-scope missing images.

Reads the datefilter output (`coverage_missing_inscope/<Parent>.parquet`), and
for every image fetches metadata + detections in ONE Graph API call
(`fields=...,detections.value`), spread across ALL tokens under a per-token
rate budget. It writes, per grid cell, append-only Parquet chunks matching the
main pipeline's schema, and downloads the ground-animal jpgs:

    <out-dir>/<cell>/all_data_<cell>_backfill_<NNN>.parquet        (every image)
    <out-dir>/<cell>/ground_animals_<cell>_backfill_<NNN>.parquet  (ground animals)
    <image-dir>/<cell>/ground_animal_images/<image_id>.jpg         (jpgs)

`--out-dir` (parquets) and `--image-dir` (jpgs, defaults to --out-dir) can point
at different drives. `--no-download` writes parquets only. Schema + download/exif
reuse `batch_chunks_mp_api.py`; tokens + session + graceful Ctrl+C reuse
`coverage_audit.py`. Image downloads are never proxied (only metadata fetches
may be, via --proxies).

TOKENS / RATE: only the entity API (graph.mapillary.com/:id) is used -- limited
to 60,000/min PER TOKEN. Requests round-robin across every token and each is
capped at that rate (so 42 tokens -> ~2.5M/min ceiling). The search API
(/images?bbox, 10,000/min) is NOT used here. Worker defaults mirror
batch_chunks_mp_api.py (entity 520, download 10); downloads stay modest
because the real ground-animal volume is bandwidth-bound.

Resumable: the in-scope parquet is cell-contiguous and processed sequentially;
a `<inscope>/.backfill_progress.json` sidecar records the per-parent row offset.
Per-cell part numbers continue past existing `*_backfill_*.parquet`.

    python backfill_missing.py \
        --inscope coverage_missing_inscope --region Europe \
        --out-dir /path/to/grid_runs --image-dir /path/to/images \
        --entity-workers 520 --download-workers 10 [--proxies proxies.txt]
"""

import argparse
import gc
import glob
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import orjson
import polars as pl
from tqdm import tqdm

import batch_chunks_mp_api as bc
import coverage_audit as ca

GROUND = 'animal--ground-animal'
FIELDS = (
    'id,computed_geometry,captured_at,sequence,is_pano,camera_type,'
    'computed_compass_angle,creator,height,width,detections.value,make,'
    'model,thumb_256_url,thumb_1024_url,thumb_2048_url,thumb_original_url')
PARQUET_CHUNK = 100000
ENTITY_RATE_PER_MIN = 60000  # Mapillary entity API cap, per token


class TokenRateLimiter:
    """Round-robins tokens under a per-token requests-per-minute cap.

    Spreads usage across every token and never lets a single token exceed
    ``per_min`` requests within a fixed 60s window (the Mapillary entity-API
    limit). If every token has hit its cap for the current window, ``acquire``
    briefly sleeps until the window rolls. Thread-safe.
    """

    def __init__(self, keys, per_min=ENTITY_RATE_PER_MIN):
        self.keys = list(keys)
        self.per_min = per_min
        self._lock = threading.Lock()
        self._win = time.monotonic()
        self._counts = {k: 0 for k in self.keys}
        self._i = 0

    def acquire(self):
        """Return the next token with budget left this minute (may sleep)."""
        while True:
            with self._lock:
                now = time.monotonic()
                if now - self._win >= 60:
                    self._win = now
                    self._counts = {k: 0 for k in self.keys}
                n = len(self.keys)
                for _ in range(n):
                    k = self.keys[self._i % n]
                    self._i += 1
                    if self._counts[k] < self.per_min:
                        self._counts[k] += 1
                        return k
                sleep = max(0.01, 60 - (now - self._win))
            time.sleep(min(sleep, 1.0))


def fetch_meta(iid, limiter, session, proxypool):
    """Fetch one image's metadata + detection values in a single call.

    Args:
        iid: Image id.
        limiter: A ``TokenRateLimiter``.
        session: A requests Session.
        proxypool: Optional ``ProxyPool`` (metadata only).

    Returns:
        ``(iid, data_or_None, status)`` with status in {'ok','gone','err'}.
    """
    url = (f'https://graph.mapillary.com/{iid}'
           f'?access_token={limiter.acquire()}&fields={FIELDS}')
    pid, proxies = (None, None)
    if proxypool is not None:
        pid, proxies = proxypool.acquire()
    try:
        r = session.get(url, timeout=20, proxies=proxies)
        if r.status_code in (400, 404):
            return iid, None, 'gone'
        if r.status_code != 200:
            return iid, None, 'err'
        return iid, orjson.loads(r.content), 'ok'
    except Exception:
        return iid, None, 'err'
    finally:
        if proxypool is not None:
            proxypool.release(pid)


def prep_record(iid, data):
    """Normalize an API record to the pipeline's row shape.

    Renames ``id`` -> ``image_id`` and flattens ``detections`` to the list of
    detection dicts (as the main pipeline stores it).
    """
    row = dict(data)
    row['image_id'] = str(data.get('id', iid))
    det = data.get('detections')
    if isinstance(det, dict):
        row['detections'] = det.get('data', [])
    return row


def is_ground(row):
    """Return True iff a record has a ground-animal detection."""
    for d in row.get('detections') or []:
        if d.get('value') == GROUND:
            return True
    return False


def next_part(cdir, cell, tag):
    """Next backfill part index for a cell+shard (continues past existing)."""
    n = -1
    for p in glob.glob(
            os.path.join(cdir, f'all_data_{cell}_backfill_{tag}*.parquet')):
        try:
            n = max(n, int(os.path.basename(p).split('_')[-1].split('.')[0]))
        except Exception:
            pass
    return n + 1


def parse_shard(s):
    """Parse '--shard I/N' -> (i, n) 0-based, default (0, 1)."""
    if not s:
        return 0, 1
    try:
        i, n = (int(x) for x in s.split('/'))
    except Exception:
        raise SystemExit(f"--shard must be 'I/N' (e.g. 0/3), got {s!r}")
    if not 0 <= i < n:
        raise SystemExit(f"--shard 'I/N' needs 0 <= I < N, got {s!r}")
    return i, n


def load_progress(path):
    """Load the per-parent row-offset resume sidecar."""
    try:
        return orjson.loads(open(path, 'rb').read())
    except Exception:
        return {}


def save_progress(path, prog):
    """Atomically persist the resume sidecar."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(orjson.dumps(prog))
    os.replace(tmp, path)


def resolve_files(inscope, region):
    """Resolve the inscope arg to ``[(parent, file)]``."""
    if os.path.isdir(inscope):
        if region:
            safe_region = ca.sanitize_folder_name(region)
            files = [os.path.join(inscope, f'{safe_region}.parquet')]
        else:
            files = sorted(glob.glob(os.path.join(inscope, '*.parquet')))
    else:
        files = [inscope]
    return [(os.path.basename(f)[:-len('.parquet')], f) for f in files
            if os.path.exists(f)]


def backfill_parent(parent,
                    inscope_file,
                    args,
                    limiter,
                    meta_session,
                    dl_session,
                    proxypool,
                    prog,
                    prog_path,
                    position=0):
    """Backfill one parent region's in-scope missing images.

    Streams the cell-contiguous parquet from the saved offset, fetching metadata
    in batches under the rate budget, buffering per cell, and flushing
    all_data/ground_animals chunks + downloads on cell change or buffer fill.
    """
    lf = pl.scan_parquet(inscope_file)
    total = int(lf.select(pl.len()).collect()['len'][0])
    si, sn = args.shard
    tag = f's{si}_' if sn > 1 else ''
    pkey = f'{parent}#{si}/{sn}' if sn > 1 else parent
    start = si * total // sn
    end = (si + 1) * total // sn
    offset = max(int(prog.get(pkey, 0)), start)
    if offset >= end:
        tqdm.write(f"✓ {pkey}: already complete.")
        return
    tqdm.write(f"{pkey} · rows [{start:,},{end:,}) · resuming at {offset:,}")

    state = {
        'cell': None,
        'cdir': None,
        'part': 0,
        'buf': [],
        'animals': set(),
        'n_all': 0,
        'n_anim': 0,
        'gone': 0,
        'err': 0
    }

    def flush():
        """Write the current cell buffer to parquet(s) and download animals."""
        buf = state['buf']
        if not buf:
            return
        cell, cdir = state['cell'], state['cdir']
        df = bc.build_mapillary_dataframe_from_records(buf)
        if not df.is_empty():
            if 'captured_at' in df.columns:
                df = df.with_columns(
                    pl.col('captured_at').cast(pl.Int64, strict=False).cast(
                        pl.Datetime('ms')).dt.strftime('%Y-%m-%d %H:%M:%S'))
            df = df.unique(subset=['image_id'], keep='last')
            part = state['part']
            df.write_parquet(os.path.join(
                cdir, f'all_data_{cell}_backfill_{tag}{part:03d}.parquet'),
                             compression='zstd')
            state['n_all'] += df.height
            animals = df.filter(pl.col('image_id').is_in(state['animals']))
            if not animals.is_empty():
                animals.write_parquet(os.path.join(
                    cdir,
                    f'ground_animals_{cell}_backfill_{tag}{part:03d}.parquet'),
                                      compression='zstd')
                state['n_anim'] += animals.height
                if not args.no_download:
                    _download(animals, cell)
            state['part'] += 1
        state['buf'] = []
        state['animals'] = set()
        gc.collect()

    def _download(animals, cell):
        """Download the ground-animal jpgs for a flushed chunk (never proxied).

        Images go under ``--image-dir`` (falling back to ``--out-dir``), so jpgs
        can live on a different drive than the parquets.
        """
        idir = os.path.join(args.image_dir or args.out_dir, cell)
        img_dir = os.path.join(idir, 'ground_animal_images')
        os.makedirs(img_dir, exist_ok=True)
        ledger = os.path.join(idir, f'validated_images_{cell}.txt')
        vset = set()
        if os.path.exists(ledger):
            with open(ledger) as f:
                vset = {ln.strip() for ln in f if ln.strip()}
        lock = threading.Lock()
        tasks = []
        for row in animals.iter_rows(named=True):
            url = row.get('thumb_original_url')
            if not url:
                continue
            fp = os.path.join(img_dir, f"{row['image_id']}.jpg")
            if not bc.is_valid_image(fp, row['image_id'], vset, lock, ledger):
                tasks.append((row['image_id'], url, row.get('captured_at')))
        if not tasks:
            return
        cap = {t[0]: t[2] for t in tasks}
        with ThreadPoolExecutor(max_workers=args.download_workers) as ex:
            futs = [
                ex.submit(bc.download_single_image, iid, url, img_dir,
                          dl_session, vset, lock, ledger, None, None)
                for iid, url, _ in tasks
            ]
            for f in as_completed(futs):
                iid, fp, ok, _ = f.result()
                if ok and cap.get(iid):
                    bc.apply_exif_data(fp, cap[iid])

    with tqdm(total=end - start, initial=offset - start, desc=pkey,
              unit='img', unit_scale=True, smoothing=0.05, position=position,
              dynamic_ncols=True, mininterval=0.3) as pbar, \
            ThreadPoolExecutor(max_workers=args.entity_workers) as ex:
        while offset < end:
            if ca.SHUTDOWN.is_set():
                break
            n = min(args.batch, end - offset)
            rows = lf.slice(offset, n).collect().to_dicts()
            if not rows:
                break
            results = {}
            futs = [
                ex.submit(fetch_meta, r['image_id'], limiter, meta_session,
                          proxypool) for r in rows
            ]
            for f in as_completed(futs):
                iid, data, status = f.result()
                results[iid] = (data, status)
                pbar.update(1)
                if status == 'gone':
                    state['gone'] += 1
                elif status != 'ok':
                    state['err'] += 1
            for r in rows:
                cell = r['safe_region_id']
                if cell != state['cell']:
                    flush()
                    state['cell'] = cell
                    state['cdir'] = os.path.join(args.out_dir, cell)
                    os.makedirs(state['cdir'], exist_ok=True)
                    state['part'] = next_part(state['cdir'], cell, tag)
                data, status = results.get(r['image_id'], (None, 'err'))
                if status == 'ok' and data is not None:
                    rec = prep_record(r['image_id'], data)
                    state['buf'].append(rec)
                    if is_ground(rec):
                        state['animals'].add(rec['image_id'])
                    if len(state['buf']) >= PARQUET_CHUNK:
                        flush()
            offset += len(rows)
            pbar.set_postfix_str(
                f"GA {state['n_anim']:,} gone {state['gone']:,}",
                refresh=False)
            if ca.SHUTDOWN.is_set():
                break
            prog[pkey] = offset
            save_progress(prog_path, prog)
        flush()
        prog[pkey] = offset
        save_progress(prog_path, prog)

    tail = "interrupted" if ca.SHUTDOWN.is_set() else "done"
    tqdm.write(f"{pkey} {tail} · row {offset:,}/{end:,} · all_data "
               f"+{state['n_all']:,} · ground-animals +{state['n_anim']:,} · "
               f"gone {state['gone']:,} · err {state['err']:,}")


def download_only(args):
    """Continuously download ground-animal jpgs from the backfill parquets.

    Decoupled from the metadata scan: reads every
    ``<out-dir>/<cell>/ground_animals_*_backfill_*.parquet``, builds the set of
    not-yet-downloaded jpgs, and downloads them through one large pool so the
    link stays saturated (bandwidth, not worker count, is the cap). Resumable
    (skips files already on disk). With ``--watch`` it re-scans periodically to
    pick up parquets a concurrent metadata scan is still writing.

    Args:
        args: Parsed args (out_dir, image_dir, download_workers, watch, env).
    """
    pq_root = args.out_dir
    img_root = args.image_dir or args.out_dir
    dl_session = ca.build_session(pool=args.download_workers + 16)
    bc.MLY_KEY = ca.load_keys(args.env)[0]
    bc.PROXY_LIST = []
    io_lock = threading.Lock()
    cell_io = {}

    def get_io(cell, img_dir):
        """Lazily load a cell's validated-set + ledger (thread-safe)."""
        with io_lock:
            if cell not in cell_io:
                os.makedirs(img_dir, exist_ok=True)
                ledger = os.path.join(img_root, cell,
                                      f'validated_images_{cell}.txt')
                vset = set()
                if os.path.exists(ledger):
                    with open(ledger) as f:
                        vset = {ln.strip() for ln in f if ln.strip()}
                cell_io[cell] = (vset, threading.Lock(), ledger)
            return cell_io[cell]

    def do(task):
        """Download one jpg; return its byte size (0 on failure)."""
        cell, img_dir, iid, url, cap = task
        vset, lock, ledger = get_io(cell, img_dir)
        _, fp, ok, _ = bc.download_single_image(iid, url, img_dir, dl_session,
                                                vset, lock, ledger, None, None)
        if ok and cap:
            bc.apply_exif_data(fp, cap)
        try:
            return os.path.getsize(fp) if ok else 0
        except Exception:
            return 0

    while True:
        tasks = []
        for pq in sorted(
                glob.glob(
                    os.path.join(pq_root, '*',
                                 'ground_animals_*_backfill_*.parquet'))):
            cell = os.path.basename(os.path.dirname(pq))
            img_dir = os.path.join(img_root, cell, 'ground_animal_images')
            try:
                df = pl.read_parquet(
                    pq,
                    columns=['image_id', 'thumb_original_url', 'captured_at'])
            except Exception:
                continue
            for row in df.iter_rows(named=True):
                url = row.get('thumb_original_url')
                if not url:
                    continue
                fp = os.path.join(img_dir, f"{row['image_id']}.jpg")
                if os.path.exists(fp) and os.path.getsize(fp) > 100:
                    continue
                tasks.append((cell, img_dir, str(row['image_id']), url,
                              row.get('captured_at')))
        if not tasks:
            if args.watch and not ca.SHUTDOWN.is_set():
                tqdm.write("download · nothing pending; watching…")
                time.sleep(30)
                continue
            tqdm.write("✓ all ground-animal jpgs downloaded.")
            return
        tqdm.write(f"download · {len(tasks):,} jpgs · "
                   f"{args.download_workers} workers")
        nbytes = 0
        t0 = time.monotonic()
        with tqdm(total=len(tasks), desc='download', unit='img',
                  unit_scale=True, smoothing=0.1, dynamic_ncols=True,
                  mininterval=0.3) as pbar, \
                ThreadPoolExecutor(max_workers=args.download_workers) as ex:
            done = 0
            for fut in as_completed(ex.submit(do, t) for t in tasks):
                if ca.SHUTDOWN.is_set():
                    break
                nbytes += fut.result()
                done += 1
                pbar.update(1)
                if done % 64 == 0:
                    mibs = nbytes / (1024 * 1024) / max(
                        0.001,
                        time.monotonic() - t0)
                    pbar.set_postfix_str(f"{mibs:.1f} MiB/s", refresh=False)
        mibs = nbytes / (1024 * 1024) / max(0.001, time.monotonic() - t0)
        tqdm.write(
            f"   {nbytes / 1e9:.2f} GB · ~{mibs:.1f} MiB/s avg this pass")
        if not args.watch or ca.SHUTDOWN.is_set():
            return
        time.sleep(10)


def run_shard(args, i, n, position=0):
    """Set up tokens/sessions for shard i/n and backfill every parent region.

    Each shard uses a disjoint token slice (tokens[i::n]) and its own resume
    sidecar, so N shards can run as independent processes.

    Args:
        args: Parsed args.
        i: Shard index.
        n: Shard count.
        position: tqdm bar line (for stacked multi-process bars).
    """
    parents = resolve_files(args.inscope, args.region)
    if not parents:
        raise SystemExit(f"no inscope parquet under {args.inscope}")
    os.makedirs(args.out_dir, exist_ok=True)
    keys = ca.load_keys(args.env)
    if not keys:
        raise SystemExit("no MLY_KEY* tokens in env")
    if n > 1:
        keys = keys[i::n]
        if not keys:
            raise SystemExit(f"shard {i}/{n}: no tokens in this shard")
    args.shard = (i, n)
    limiter = TokenRateLimiter(keys, per_min=args.entity_rate)
    meta_session = ca.build_session(pool=args.entity_workers + 32)
    dl_session = ca.build_session(pool=args.download_workers + 16)
    proxypool = ca.ProxyPool(args.proxies) if args.proxies else None
    bc.MLY_KEY = keys[0]
    bc.PROXY_LIST = []
    ca._install_sigint()

    base = (os.path.join(args.inscope, '.backfill_progress.json')
            if os.path.isdir(args.inscope) else args.inscope +
            '.backfill_progress.json')
    prog_path = base if n == 1 else base.replace('.json', f'.s{i}of{n}.json')
    prog = load_progress(prog_path)

    tag = f"shard {i}/{n} · " if n > 1 else ""
    mode = "no-download" if args.no_download else (args.image_dir
                                                   or args.out_dir)
    tqdm.write(f"[{tag}{len(keys)} tokens · {args.entity_workers} workers · "
               f"images: {mode}]")

    for parent, f in parents:
        if ca.SHUTDOWN.is_set():
            break
        backfill_parent(parent,
                        f,
                        args,
                        limiter,
                        meta_session,
                        dl_session,
                        proxypool,
                        prog,
                        prog_path,
                        position=position)


def main():
    """Parse CLI; dispatch to download-only, a single shard, or N processes."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inscope', default='coverage_missing_inscope')
    ap.add_argument('--region',
                    default=None,
                    help='Parent region to backfill (default: all).')
    ap.add_argument('--out-dir',
                    required=True,
                    help='Parquet output root; cells written as '
                    '<out-dir>/<cell>/ (put on a drive with free space).')
    ap.add_argument('--image-dir',
                    default=None,
                    help='Separate root for downloaded jpgs '
                    '(<image-dir>/<cell>/ground_animal_images/). '
                    'Default: same as --out-dir.')
    ap.add_argument('--no-download',
                    action='store_true',
                    help='Write parquets only; skip all image downloads.')
    ap.add_argument('--entity-workers',
                    type=int,
                    default=520,
                    help='Concurrent metadata fetches (batch_chunks default).')
    ap.add_argument('--download-workers',
                    type=int,
                    default=10,
                    help='Concurrent jpg downloads (batch_chunks default).')
    ap.add_argument('--entity-rate',
                    type=int,
                    default=ENTITY_RATE_PER_MIN,
                    help='Per-token entity-API requests/min cap (default '
                    f'{ENTITY_RATE_PER_MIN}).')
    ap.add_argument('--batch',
                    type=int,
                    default=50000,
                    help='Image ids fetched per round (larger amortizes the '
                    'per-round straggler barrier; default 50000).')
    ap.add_argument(
        '--processes',
        type=int,
        default=1,
        help='Spawn this many shard PROCESSES from one command '
        '(like batch_chunks --outer-max-workers). Scales past one '
        "Python process's single-core JSON-parse limit. Default 1.")
    ap.add_argument('--shard',
                    default=None,
                    metavar='I/N',
                    help='Run a SINGLE shard I of N manually (e.g. across '
                    'machines): row-range I/N + tokens[I::N] + its own '
                    'backfill_s<I>_* files. Use --processes for one-box runs.')
    ap.add_argument('--proxies',
                    default=None,
                    help='Optional proxy file for METADATA calls only.')
    ap.add_argument('--download-only',
                    action='store_true',
                    help='Skip metadata; only download jpgs from existing '
                    'ground_animals_*_backfill_*.parquet (run alongside a '
                    'metadata scan to saturate bandwidth continuously).')
    ap.add_argument('--watch',
                    action='store_true',
                    help='In --download-only, keep re-scanning for new '
                    'parquets instead of exiting when caught up.')
    ap.add_argument('--env', default='.env')
    args = ap.parse_args()

    if args.download_only:
        ca._install_sigint()
        tqdm.write(f"download-only · {args.download_workers} workers · "
                   f"images → {args.image_dir or args.out_dir}")
        download_only(args)
        return

    if args.shard:
        i, n = parse_shard(args.shard)
        run_shard(args, i, n, position=0)
    elif args.processes > 1:
        n = args.processes
        ca._install_sigint()
        tqdm.write(f"backfill · spawning {n} shard processes")
        procs = []
        for i in range(n):
            p = mp.Process(target=run_shard,
                           args=(args, i, n, i),
                           daemon=False)
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        run_shard(args, 0, 1, position=0)

    if ca.SHUTDOWN.is_set():
        tqdm.write("⏸ interrupted; progress saved. Re-run to resume.")


if __name__ == '__main__':
    main()
