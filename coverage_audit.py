"""
Completeness audit: verify you have every Mapillary image (up to a cutoff date)
for your grid regions, and list what's missing so it can be re-requested.

ENUMERATION USES VECTOR TILES (not the /images?bbox API). The bbox API has two
fatal limits for a completeness audit, both verified live:
  * it hard-caps at ~2000 results per query with NO pagination cursor, so any
    tile with >2000 images is silently truncated (this affected BOTH the old
    coverage audit and batch_chunks_mp_api_v3.py); and
  * it returns HTTP 500 "reduce the amount of data" for hyper-dense city-center
    cells, which no amount of bbox-splitting, limit=, or date-windowing avoids.
Mapillary's own map tiles (mly1_public vector tiles, max zoom 14) have neither
limit: one z14 tile request returns EVERY image point in that tile, each
carrying its id, sequence_id, and captured_at. We tile each region at z14, fetch
each land tile once, and parse the `image` layer. Validated against the bbox API
on moderate tiles: the tile set is a strict superset (0 images only in bbox) and
sequence_ids agree 100%.

Outputs per region (in --data-dir, the crucial parquet/data dir -- NOT the image
dirs):
  coverage_checkpoint_<safe>.json.zst  {image_id: [sequence_id, captured_at_ms]}
  coverage_meta_<safe>.json            small sidecar: method/errors/failed_tiles/
                                       n_images -- so check/retry are instant and
                                       don't have to parse the big checkpoint.

Stages (subcommands):

  audit      GRID.csv --dirs ... [--region "Europe"]
      One command: enumerate -> retry (until clean / no progress / max rounds)
      -> diff -> datefilter. The normal entry point.

  enumerate  GRID.csv --dirs ... [--region "Europe"] [-w 64]
      Fetch every z14 land tile of each region as a vector tile; write the
      checkpoint + meta. Resumable: completed regions are skipped, and a region
      cut off mid-way by the daily limit resumes from its pending_tiles. Tiles
      that error are recorded in the meta's failed_tiles.

  DAILY LIMIT: tiles.mapillary.com allows ~50,000 requests/day PER TOKEN.
      Requests round-robin across every MLY_KEY* token, each with its own daily
      budget (so 16 tokens ~= 800k tiles/day). Usage is tracked per token in
      <data-dir>/.tile_request_budget.json; a token that returns 429 or a
      throttle page is logged + disabled for the rest of the run and its tile
      requeued on the next token (a re-run retries every token in case the
      throttle was transient). When all tokens are exhausted, enumerate/retry
      stop and resume automatically on the next run
      (the next day). A full Europe/global enumeration spans multiple days --
      just re-run the same command. --daily-tile-limit sets the per-token cap.

  check      [--region "Europe"] [--data-dir ...]
      Reads only the tiny meta sidecars (instant). Reports per-region image
      counts and how many tiles failed; points you at retry.

  retry      [--region "Europe"] [--data-dir ...] [-w 64]
      Re-fetch ONLY the failed_tiles in each region, merge their images in, and
      rewrite the checkpoint + meta with whatever still fails. Existing images
      are kept. Safe to run repeatedly (transient errors clear).

  diff       GRID.csv --dirs ... [--region "Europe"] --out coverage_missing.csv
      missing = coverage_ids - all_data_ids. Writes missing rows (with the
      captured_at carried from the checkpoint) + a per-region summary.

  datefilter --missing coverage_missing.csv --cutoff 2026-05-31 \
             --out coverage_missing_inscope.csv
      Keeps rows captured on/before the cutoff. LOCAL (no API) when captured_at
      is present in the missing CSV (it is, from MVT); falls back to the API
      only for any rows lacking it.

Tokens: MLY_KEY* from .env, rotated per call. Optional rotating proxies via
--proxies (per-IP tile throttle aware).
"""

import argparse
import compression.zstd as zstd
import csv
import datetime as dt
import gc
import glob
import itertools
import os
import re
import signal
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import mercantile
import orjson
import polars as pl
import requests
from dotenv import dotenv_values
from global_land_mask import globe
from requests.adapters import HTTPAdapter
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.table import Table
from urllib3.util.retry import Retry

csv.field_size_limit(sys.maxsize)
ZOOM = 14
ZSTD_OPTIONS = {zstd.CompressionParameter.nb_workers: 2}
API_CHUNK_SIZE = 5000
MVT_TILESET = 'https://tiles.mapillary.com/maps/vtp/mly1_public/2'
MVT_METHOD = 'mvt'
DAILY_TILE_LIMIT = 50000
BUDGET_FILE = '.tile_request_budget.json'
SAVE_INTERVAL = 60

console = Console()

SHUTDOWN = threading.Event()


def _install_sigint():
    """Install a graceful-shutdown SIGINT handler.

    The first Ctrl+C sets the module-level ``SHUTDOWN`` event; long-running
    loops poll it to finish the in-flight tile chunk, checkpoint, and exit so no
    work is lost. The handler then restores Python's default SIGINT behaviour,
    so a second Ctrl+C aborts immediately.
    """

    def handler(signum, frame):
        """Set ``SHUTDOWN`` and revert to the default handler for next time."""
        SHUTDOWN.set()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        console.print(
            "\n[yellow]⏸ Ctrl+C — finishing the current tile chunk and "
            "checkpointing; press Ctrl+C again to force-quit.[/]")

    signal.signal(signal.SIGINT, handler)


def make_progress():
    """Build the uniform rich progress display used across stages.

    An outer Regions task and a per-region tile task share one instance so bars
    nest cleanly and printed summaries interleave above them.

    Returns:
        A configured ``rich.progress.Progress`` instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def region_summary(safe, n_images, n_failed=0, n_pending=0, gained=None):
    """Print one aligned, colour-coded result line for a region.

    Args:
        safe: Sanitized region id (name only, no path).
        n_images: Number of images enumerated for the region.
        n_failed: Number of tiles that errored.
        n_pending: Number of tiles not yet attempted (budget cut).
        gained: If given, images newly recovered this run (shown as ``+N``).
    """
    icon = "[green]✓[/]" if not (n_failed or n_pending) else "[yellow]●[/]"
    line = f"{icon} [bold]{safe:<34}[/] [cyan]{n_images:>11,}[/] imgs"
    if gained is not None:
        line += f"  [green]+{gained:,}[/]"
    if n_failed:
        line += f"  failed=[red]{n_failed}[/]"
    if n_pending:
        line += f"  pending=[yellow]{n_pending}[/]"
    console.print(line)


def chunked_iterable(iterable, size):
    """Yield successive lists of up to ``size`` items from ``iterable``.

    Args:
        iterable: Any iterable.
        size: Maximum length of each yielded chunk.

    Yields:
        Lists of items, the last possibly shorter than ``size``.
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def sanitize_folder_name(name):
    """Return a filesystem-safe version of ``name``.

    Args:
        name: Arbitrary string (``&`` becomes ``and``).

    Returns:
        ``name`` with non-word characters replaced by underscores and trimmed.
    """
    return re.sub(r'[^\w\-_\.]', '_', name.replace('&', 'and')).strip('_')


def safe_of(r):
    """Return the sanitized region id for a grid row.

    Args:
        r: Grid row mapping with region/sw_lon/sw_lat/ne_lon/ne_lat.

    Returns:
        The ``safe_region_id`` string.
    """
    return sanitize_folder_name(f"{r['region']}_{r['sw_lon']}_{r['sw_lat']}_"
                                f"{r['ne_lon']}_{r['ne_lat']}")


def iso_to_ms(s):
    """Convert an ISO datetime string to epoch milliseconds.

    Args:
        s: ISO datetime (a trailing ``Z`` is accepted), or falsy.

    Returns:
        Epoch milliseconds (UTC assumed if naive), or None when ``s`` is falsy.
    """
    if not s:
        return None
    d = dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


# --------------------------------------------------------------------------- #
# checkpoint + meta files
# --------------------------------------------------------------------------- #
def coverage_ckpt_name(safe):
    """Return the checkpoint filename for a region.

    Args:
        safe: Sanitized region id.

    Returns:
        The ``coverage_checkpoint_<safe>.json.zst`` filename.
    """
    return f'coverage_checkpoint_{safe}.json.zst'


def coverage_meta_name(safe):
    """Return the meta-sidecar filename for a region.

    Args:
        safe: Sanitized region id.

    Returns:
        The ``coverage_meta_<safe>.json`` filename.
    """
    return f'coverage_meta_{safe}.json'


def find_coverage_ckpt(safe, dirs):
    """Find a region's checkpoint across candidate directories.

    Args:
        safe: Sanitized region id.
        dirs: Directories to search, in order.

    Returns:
        The first existing checkpoint path, or None.
    """
    for d in dirs:
        p = os.path.join(d, safe, coverage_ckpt_name(safe))
        if os.path.exists(p):
            return p
    return None


def safe_from_meta_path(path):
    """Recover the sanitized region id from a meta-sidecar path.

    Args:
        path: Path to a ``coverage_meta_<safe>.json`` file.

    Returns:
        The ``<safe>`` region id.
    """
    name = os.path.basename(path)
    return name[len('coverage_meta_'):-len('.json')]


def write_coverage(data_dir, safe, imgmap, failed_tiles, pending_tiles=None):
    """Atomically write a region's checkpoint and meta sidecar.

    Both files are written via a temp file + ``os.replace`` so an interrupted
    write can never corrupt them.

    Args:
        data_dir: Parquet/data dir for checkpoints (not the image dir).
        safe: Sanitized region id.
        imgmap: ``{image_id: [sequence_id, captured_at_ms]}``.
        failed_tiles: Tiles that errored, each ``[z, x, y]``.
        pending_tiles: Land tiles not yet fetched (daily budget ran out),
            resumed on the next run. Defaults to none.
    """
    rd = os.path.join(data_dir, safe)
    os.makedirs(rd, exist_ok=True)
    ckpt = os.path.join(rd, coverage_ckpt_name(safe))
    tmp = ckpt + '.tmp'
    with zstd.open(tmp, 'wb', options=ZSTD_OPTIONS) as f:
        f.write(orjson.dumps(imgmap))
    os.replace(tmp, ckpt)

    meta = os.path.join(rd, coverage_meta_name(safe))
    obj = {
        'method': MVT_METHOD,
        'errors': len(failed_tiles),
        'failed_tiles': failed_tiles,
        'pending_tiles': pending_tiles or [],
        'n_images': len(imgmap)
    }
    tmp = meta + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(orjson.dumps(obj))
    os.replace(tmp, meta)


def region_state(data_dir, safe):
    """Classify a region's on-disk enumeration state.

    Args:
        data_dir: Parquet/data dir for checkpoints.
        safe: Sanitized region id.

    Returns:
        A ``(state, meta_or_None)`` tuple where state is one of:
            'done'   - MVT checkpoint complete (no pending tiles);
            'resume' - MVT checkpoint exists but some tiles still pending;
            'fresh'  - nothing usable (missing or pre-MVT); enumerate anew.
    """
    mp = os.path.join(data_dir, safe, coverage_meta_name(safe))
    cp = os.path.join(data_dir, safe, coverage_ckpt_name(safe))
    if not (os.path.exists(mp) and os.path.exists(cp)):
        return 'fresh', None
    try:
        m = read_meta(mp)
    except Exception:
        return 'fresh', None
    if m.get('method') != MVT_METHOD:
        return 'fresh', None
    return ('resume' if m.get('pending_tiles') else 'done'), m


def load_imgmap(ckpt_path):
    """Load a region's checkpoint image map.

    JSON object keys are always strings, so orjson already returns string keys;
    the decoded dict is returned directly to avoid a full second copy (which
    would double peak memory for huge regions).

    Args:
        ckpt_path: Path to a ``coverage_checkpoint_*.json.zst`` file.

    Returns:
        ``{image_id: [sequence_id, captured_at_ms]}`` with string keys.
    """
    with zstd.open(ckpt_path, 'rb') as f:
        return orjson.loads(f.read())


def read_meta(meta_path):
    """Read and parse a region's meta sidecar.

    Args:
        meta_path: Path to a ``coverage_meta_*.json`` file.

    Returns:
        The decoded meta dict.
    """
    with open(meta_path, 'rb') as f:
        return orjson.loads(f.read())


def scan_meta(data_dir, region=None):
    """List all region meta sidecars, optionally filtered to a parent region.

    Cheap: meta files are tiny.

    Args:
        data_dir: Parquet/data dir for checkpoints.
        region: Optional parent-region name to restrict to.

    Returns:
        Sorted list of ``(safe, meta_path)`` tuples.
    """
    out = []
    pref = sanitize_folder_name(region) if region else None
    for p in sorted(
            glob.glob(os.path.join(data_dir, '*', coverage_meta_name('*')))):
        safe = safe_from_meta_path(p)
        if pref and not safe.startswith(pref):
            continue
        out.append((safe, p))
    return out


# --------------------------------------------------------------------------- #
# tokens / session / grid
# --------------------------------------------------------------------------- #
def load_keys(env_path):
    """Load distinct Mapillary tokens from an env file.

    Args:
        env_path: Path to a ``.env`` containing ``MLY_KEY`` / ``MLY_KEY_<n>``.

    Returns:
        List of unique token strings, in file order.
    """
    env = dotenv_values(env_path)
    keys, seen = [], set()
    for k, v in env.items():
        if (k == 'MLY_KEY' or re.fullmatch(r'MLY_KEY_\d+', k)) and v \
                and v not in seen:
            seen.add(v)
            keys.append(v)
    return keys


class KeyRotator:
    """Thread-safe round-robin over a list of tokens."""

    def __init__(self, keys):
        """Initialize the rotator.

        Args:
            keys: Tokens to cycle through.
        """
        self._c = itertools.cycle(keys)
        self._lock = threading.Lock()

    def next(self):
        """Return the next token in round-robin order (thread-safe)."""
        with self._lock:
            return next(self._c)


class TokenBudget:
    """Per-token daily request budget for tiles.mapillary.com.

    Mapillary rate-limits per application/token, and these are distinct apps, so
    each token gets its own ``limit``/day (default 50k); N tokens give ~N times
    the daily ceiling. The persisted file (``{date, counts:{token:n}}``) tracks
    per-token daily usage and is reset by date. ``acquire`` round-robins to the
    next token in a worker's slot that still has budget and isn't disabled.
    ``disable`` marks a token rate-limited for the current run only (in-memory,
    not persisted) so its tile is requeued onto another token; a later re-run
    retries every token in case the throttle was transient. Counting is one per
    tile fetch, so the reactive disable + requeue is the real safety net.
    """

    def __init__(self, path, keys, limit):
        """Initialize and load the persisted daily counts.

        Args:
            path: Path to the budget JSON file.
            keys: All tokens.
            limit: Per-token daily request cap.
        """
        self.path, self.keys, self.limit = path, list(keys), limit
        self._lock = threading.Lock()
        self._dirty = 0
        self._load()
        self.set_nslots(1)

    def set_nslots(self, n):
        """Partition tokens into n disjoint stride-slices, one per worker slot.

        Worker ``slot`` only ever uses ``keys[slot::n]`` so concurrent workers
        never draw the same token; ``n == 1`` is the whole pool (default).

        Args:
            n: Number of parallel region-worker slots.
        """
        self.nslots = n
        self._slot_keys = [self.keys[k::n] for k in range(n)]
        self._slot_i = [0] * n

    def n_tokens(self):
        """Return the total number of tokens."""
        return len(self.keys)

    @staticmethod
    def _today():
        """Return today's date as an ISO string."""
        return dt.date.today().isoformat()

    def _load(self):
        """Load persisted counts, resetting them when the date has rolled.

        The reactive disabled set is per-run (in-memory) and never persisted, so
        a re-run retries every token; only per-token success counts persist.
        """
        try:
            with open(self.path, 'rb') as f:
                d = orjson.loads(f.read())
        except Exception:
            d = {}
        self.disabled = set()
        if d.get('date') != self._today():
            self.date, self.counts = self._today(), {}
        else:
            self.date = d['date']
            self.counts = dict(d.get('counts', {}))

    def _save(self):
        """Atomically persist the date and per-token counts."""
        tmp = self.path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(orjson.dumps({'date': self.date, 'counts': self.counts}))
        os.replace(tmp, self.path)

    def _roll(self):
        """Reset counts and the disabled set if the date has changed."""
        if self._today() != self.date:
            self.date, self.counts, self.disabled = self._today(), {}, set()

    def acquire(self, slot=0):
        """Reserve one request on the next available token in a worker's slot.

        Round-robins within ``keys[slot::nslots]``.

        Args:
            slot: Worker slot index.

        Returns:
            A token string, or None if every token in the slot is
            exhausted/disabled for today.
        """
        with self._lock:
            self._roll()
            sk = self._slot_keys[slot]
            n = len(sk)
            for _ in range(n):
                tok = sk[self._slot_i[slot] % n]
                self._slot_i[slot] += 1
                if tok in self.disabled:
                    continue
                if self.counts.get(tok, 0) >= self.limit:
                    continue
                self.counts[tok] = self.counts.get(tok, 0) + 1
                self._dirty += 1
                if self._dirty >= 2000:
                    self._save()
                    self._dirty = 0
                return tok
            return None

    def disable(self, tok):
        """Mark a token rate-limited for the current run (in-memory).

        Args:
            tok: The token to disable.

        Returns:
            True if it was newly disabled (so the caller logs it once, not once
            per in-flight tile).
        """
        with self._lock:
            newly = tok not in self.disabled
            self.disabled.add(tok)
            return newly

    def token_label(self, tok):
        """Return a human label like ``#3/19`` for a token.

        Args:
            tok: The token to label.

        Returns:
            A ``#index/total`` string, or ``"?"`` if unknown.
        """
        try:
            return f"#{self.keys.index(tok) + 1}/{len(self.keys)}"
        except ValueError:
            return "?"

    def flush(self):
        """Persist any pending count increments."""
        with self._lock:
            if self._dirty:
                self._save()
                self._dirty = 0

    def n_active(self):
        """Return the number of tokens not disabled this run."""
        with self._lock:
            self._roll()
            return sum(1 for t in self.keys if t not in self.disabled)

    def remaining(self, slot=None):
        """Return remaining daily budget, globally or for one worker slot.

        Args:
            slot: Worker slot index, or None for the global total.

        Returns:
            Sum of remaining per-token budget over the relevant tokens.
        """
        with self._lock:
            self._roll()
            keys = self.keys if slot is None else self._slot_keys[slot]
            return sum(
                max(0, self.limit - self.counts.get(t, 0)) for t in keys
                if t not in self.disabled)


class ProxyPool:
    """Round-robin pool of HTTP proxies that spreads tile fetches across IPs.

    The tiles.mapillary.com abuse throttle is per-IP, so rotating proxies lifts
    the ceiling, orthogonal to ``TokenBudget`` (which guards the per-app daily
    cap). A proxy that throttles (the "Not Logged In" non-tile 200) or errors is
    put on a short cooldown and skipped until it expires; if every proxy is
    cooling, the soonest-to-free one is used so ``acquire`` never returns None.
    An in-use lease prevents two concurrent requests from sharing an IP while
    the ready pool is large enough.
    """

    def __init__(self, path, cooldown=180):
        """Load proxies and initialize lease/cooldown state.

        Args:
            path: File with one proxy per line (``host:port:user:pass`` or
                ``host:port``).
            cooldown: Seconds a throttled/erroring proxy is skipped.

        Raises:
            SystemExit: If no usable proxies are found.
        """
        self.proxies = self._load(path)
        if not self.proxies:
            raise SystemExit(f"coverage_audit: no usable proxies in {path}")
        self.cooldown = cooldown
        self._until = [0.0] * len(self.proxies)
        self._inuse = set()
        self._lock = threading.Lock()
        self.set_nslots(1)

    def set_nslots(self, n):
        """Partition proxies into n disjoint stride-slices, one per worker slot.

        Worker ``slot`` only ever uses ``proxies[slot::n]`` so concurrent
        workers never exit from the same IP; ``n == 1`` is the whole pool.

        Args:
            n: Number of parallel region-worker slots.
        """
        self.nslots = n
        idxs = list(range(len(self.proxies)))
        self._slot_idx = [idxs[k::n] for k in range(n)]
        self._slot_i = [0] * n

    @staticmethod
    def _load(path):
        """Parse a proxy file into requests-style proxy dicts.

        Args:
            path: File with one proxy per line; blank/``#`` lines are skipped.

        Returns:
            List of ``{'http': url, 'https': url}`` dicts.
        """
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(':')
                if len(parts) == 4:
                    host, port, user, pwd = parts
                    auth = f'{quote(user, safe="")}:{quote(pwd, safe="")}@'
                elif len(parts) == 2:
                    host, port = parts
                    auth = ''
                else:
                    continue
                url = f'http://{auth}{host}:{port}'
                out.append({'http': url, 'https': url})
        return out

    def acquire(self, slot=0):
        """Lease the next usable proxy in a worker's slot.

        Round-robins within ``proxies[slot::nslots]`` and prefers a proxy that
        is both ready (not cooling) and not currently in flight. Preference when
        saturated: ready & free, then ready but busy, then the soonest-to-free
        cooling proxy. Always leases (never None); the caller must
        ``release(idx)`` when the request finishes.

        Args:
            slot: Worker slot index.

        Returns:
            ``(idx, proxies_dict)``.
        """
        with self._lock:
            sub = self._slot_idx[slot]
            n = len(sub)
            now = time.monotonic()
            busy_ready = None
            soonest = None
            for _ in range(n):
                i = sub[self._slot_i[slot] % n]
                self._slot_i[slot] += 1
                cooling = self._until[i] > now
                if not cooling and i not in self._inuse:
                    self._inuse.add(i)
                    return i, self.proxies[i]
                if not cooling and busy_ready is None:
                    busy_ready = i
                elif cooling and (soonest is None
                                  or self._until[i] < self._until[soonest]):
                    soonest = i
            pick = busy_ready if busy_ready is not None else (
                soonest if soonest is not None else sub[0])
            self._inuse.add(pick)
            return pick, self.proxies[pick]

    def release(self, idx):
        """Free a proxy's lease once its request finishes.

        Args:
            idx: Proxy index returned by ``acquire`` (None is ignored).
        """
        if idx is None:
            return
        with self._lock:
            self._inuse.discard(idx)

    def penalize(self, idx):
        """Put a throttled/erroring proxy on cooldown so it is skipped.

        Args:
            idx: Proxy index to cool (None is ignored).
        """
        if idx is None:
            return
        with self._lock:
            self._until[idx] = time.monotonic() + self.cooldown

    def n_ready(self):
        """Return how many proxies are currently off cooldown."""
        with self._lock:
            now = time.monotonic()
            return sum(1 for u in self._until if u <= now)

    def __len__(self):
        """Return the total number of proxies."""
        return len(self.proxies)


def build_session(pool=64, retry_429=True):
    """Build a requests Session with a retrying HTTPS adapter.

    Mirrors batch_chunks_mp_api_v3.py's error-avoidance posture (total=5,
    backoff_factor=1) for transient codes. Tile fetching passes
    ``retry_429=False`` so a 429 surfaces immediately and the caller disables
    that token rather than blindly retrying the exhausted token five times.

    Args:
        pool: Connection pool size (connections and maxsize).
        retry_429: Whether to retry on HTTP 429 in addition to 5xx.

    Returns:
        A configured ``requests.Session``.
    """
    s = requests.Session()
    codes = [429, 500, 502, 503, 504] if retry_429 else [500, 502, 503, 504]
    retry = Retry(total=5,
                  backoff_factor=1,
                  status_forcelist=codes,
                  allowed_methods=['GET'])
    a = HTTPAdapter(max_retries=retry,
                    pool_connections=pool,
                    pool_maxsize=pool)
    s.mount('https://', a)
    return s


def load_proxies(args):
    """Build a ProxyPool from ``--proxies`` if given, else None (direct).

    Args:
        args: Parsed args; ``proxies`` is the optional proxy-file path.

    Returns:
        A ``ProxyPool`` or None.
    """
    path = getattr(args, 'proxies', None)
    if not path:
        return None
    pool = ProxyPool(path)
    console.print(f"[dim]proxies · {len(pool)} loaded from {path} "
                  f"(rotating per request, per-IP throttle aware)[/]")
    return pool


def region_rows(grid_csv, region):
    """Read grid rows from a CSV, optionally filtered to one parent region.

    Args:
        grid_csv: Path to the grid CSV.
        region: Optional parent-region name to keep.

    Returns:
        List of row dicts.
    """
    rows = list(pl.read_csv(grid_csv).iter_rows(named=True))
    if region:
        rows = [r for r in rows if r['region'] == region]
    return rows


def all_data_ids(safe, dirs):
    """Collect every downloaded image id for a region from its all_data parquets.

    Args:
        safe: Sanitized region id.
        dirs: Directories to scan for ``all_data_<safe>_*.parquet``.

    Returns:
        Set of image-id strings (empty on missing files or read error).
    """
    files = []
    for d in dirs:
        files += glob.glob(os.path.join(d, safe, f'all_data_{safe}_*.parquet'))
    if not files:
        return set()
    try:
        return set(
            map(
                str,
                pl.scan_parquet(list(set(files))).select(
                    pl.col('image_id').cast(
                        pl.Utf8)).unique().collect()['image_id'].to_list()))
    except Exception:
        return set()


# --------------------------------------------------------------------------- #
# Mapbox Vector Tile reader (feature properties only -- no geometry decode)
# --------------------------------------------------------------------------- #
def _mvt_varint(buf, pos):
    """Decode a protobuf varint.

    Args:
        buf: Bytes buffer.
        pos: Start offset.

    Returns:
        ``(value, new_pos)``.
    """
    shift = result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _mvt_skip(buf, pos, wire):
    """Skip one protobuf field of the given wire type.

    Args:
        buf: Bytes buffer.
        pos: Offset at the field's value.
        wire: Protobuf wire type.

    Returns:
        The offset just past the skipped value.
    """
    if wire == 0:
        _, pos = _mvt_varint(buf, pos)
    elif wire == 1:
        pos += 8
    elif wire == 2:
        ln, pos = _mvt_varint(buf, pos)
        pos += ln
    elif wire == 5:
        pos += 4
    return pos


def _mvt_value(buf):
    """Decode a single MVT ``Value`` message.

    Args:
        buf: Bytes of the Value sub-message.

    Returns:
        The decoded scalar (str/float/double/int/sint/bool), or None.
    """
    pos, val = 0, None
    while pos < len(buf):
        tag, pos = _mvt_varint(buf, pos)
        f, wire = tag >> 3, tag & 7
        if f == 1 and wire == 2:
            ln, pos = _mvt_varint(buf, pos)
            val = buf[pos:pos + ln].decode('utf-8', 'replace')
            pos += ln
        elif f == 2 and wire == 5:
            val = struct.unpack('<f', buf[pos:pos + 4])[0]
            pos += 4
        elif f == 3 and wire == 1:
            val = struct.unpack('<d', buf[pos:pos + 8])[0]
            pos += 8
        elif f in (4, 5) and wire == 0:
            val, pos = _mvt_varint(buf, pos)
        elif f == 6 and wire == 0:
            v, pos = _mvt_varint(buf, pos)
            val = (v >> 1) ^ -(v & 1)
        elif f == 7 and wire == 0:
            val, pos = _mvt_varint(buf, pos)
            val = bool(val)
        else:
            pos = _mvt_skip(buf, pos, wire)
    return val


def _mvt_layer(buf):
    """Decode an MVT ``Layer`` message's name, keys, values, and raw features.

    Args:
        buf: Bytes of the Layer sub-message.

    Returns:
        ``(name, keys, values, feats)`` where ``feats`` are raw feature bytes.
    """
    pos = 0
    name = None
    keys = []
    values = []
    feats = []
    while pos < len(buf):
        tag, pos = _mvt_varint(buf, pos)
        f, wire = tag >> 3, tag & 7
        if f == 1 and wire == 2:
            ln, pos = _mvt_varint(buf, pos)
            name = buf[pos:pos + ln].decode('utf-8', 'replace')
            pos += ln
        elif f == 2 and wire == 2:
            ln, pos = _mvt_varint(buf, pos)
            feats.append(buf[pos:pos + ln])
            pos += ln
        elif f == 3 and wire == 2:
            ln, pos = _mvt_varint(buf, pos)
            keys.append(buf[pos:pos + ln].decode('utf-8', 'replace'))
            pos += ln
        elif f == 4 and wire == 2:
            ln, pos = _mvt_varint(buf, pos)
            values.append(_mvt_value(buf[pos:pos + ln]))
            pos += ln
        else:
            pos = _mvt_skip(buf, pos, wire)
    return name, keys, values, feats


def _mvt_feature_props(buf, keys, values):
    """Decode one feature's properties from its packed tag list.

    Args:
        buf: Bytes of the feature sub-message.
        keys: Layer key strings.
        values: Layer decoded values.

    Returns:
        ``{key: value}`` properties dict for the feature.
    """
    pos = 0
    tags = []
    while pos < len(buf):
        tag, pos = _mvt_varint(buf, pos)
        f, wire = tag >> 3, tag & 7
        if f == 2 and wire == 2:
            ln, pos = _mvt_varint(buf, pos)
            end = pos + ln
            while pos < end:
                v, pos = _mvt_varint(buf, pos)
                tags.append(v)
        else:
            pos = _mvt_skip(buf, pos, wire)
    return {
        keys[tags[i]]: values[tags[i + 1]]
        for i in range(0,
                       len(tags) - 1, 2)
    }


def _mvt_iter_layers(content):
    """Iterate the layers of an MVT tile.

    Args:
        content: Raw tile protobuf bytes.

    Yields:
        ``(name, keys, values, feats)`` per layer (see ``_mvt_layer``).
    """
    pos = 0
    while pos < len(content):
        tag, pos = _mvt_varint(content, pos)
        f, wire = tag >> 3, tag & 7
        if f == 3 and wire == 2:
            ln, pos = _mvt_varint(content, pos)
            yield _mvt_layer(content[pos:pos + ln])
            pos += ln
        else:
            pos = _mvt_skip(content, pos, wire)


def images_from_tile(tile, key, session, end_ms, proxypool=None, slot=0):
    """Fetch and parse one z14 vector tile's ``image`` layer.

    Status handling reflects Mapillary's two distinct throttles: the documented
    per-app daily tile cap returns a 4xx (per token), while the per-IP abuse
    throttle returns a 200 with an HTML "Not Logged In" page (a real tile is
    ``application/x-protobuf``). 204/404 are treated as empty tiles.

    Args:
        tile: A ``mercantile.Tile`` at z14.
        key: Mapillary token used as ``access_token``.
        session: A requests Session.
        end_ms: Drop images captured after this epoch-ms cutoff, or None.
        proxypool: Optional ``ProxyPool`` for per-request proxy rotation.
        slot: Worker slot index for the proxy pool.

    Returns:
        ``(imgmap, error, status, proxy_id)`` where imgmap is
        ``{image_id: [sequence_id, captured_at_ms]}``, error is None on success
        else a string, proxy_id is the leased proxy index (None without a pool),
        and status is one of:
            'ok'          - got a tile (possibly empty, incl. 204/404);
            'token_limit' - a 429 or any 4xx; disable the token, requeue tile;
            'ip_throttle' - non-tile 200; with a pool cool the proxy (token ok),
                            else treated like token_limit;
            'proxy_err'   - connection/proxy failure; cool the proxy, requeue;
            'tile_err'    - other error; record the tile as failed.
    """
    proxy_id, proxies = (None, None)
    if proxypool is not None:
        proxy_id, proxies = proxypool.acquire(slot)
    url = f'{MVT_TILESET}/{tile.z}/{tile.x}/{tile.y}?access_token={key}'
    try:
        r = session.get(url, timeout=60, proxies=proxies)
        if r.status_code == 204:
            return {}, None, 'ok', proxy_id
        if r.status_code == 404:
            return {}, None, 'ok', proxy_id
        if 400 <= r.status_code < 500:
            return {}, f'tile {r.status_code}', 'token_limit', proxy_id
        r.raise_for_status()
        ctype = r.headers.get('content-type', '')
        if 'protobuf' not in ctype and 'mapbox' not in ctype:
            kind = 'ip_throttle' if proxypool is not None else 'token_limit'
            return {}, f'non-tile ({ctype}; {r.text[:60]!r})', kind, proxy_id
        content = r.content
    except Exception as e:
        if proxypool is not None and isinstance(
                e, (requests.exceptions.ProxyError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout)):
            return {}, str(e), 'proxy_err', proxy_id
        return {}, str(e), 'tile_err', proxy_id
    try:
        imgmap = {}
        for name, keys, values, feats in _mvt_iter_layers(content):
            if name != 'image':
                continue
            for raw in feats:
                props = _mvt_feature_props(raw, keys, values)
                iid = props.get('id')
                if iid is None:
                    continue
                cap = props.get('captured_at')
                if end_ms is not None and cap is not None and cap > end_ms:
                    continue
                imgmap[str(iid)] = [props.get('sequence_id'), cap]
        return imgmap, None, 'ok', proxy_id
    except Exception as e:
        return {}, f'parse:{e}', 'tile_err', proxy_id


def fetch_tiles(tiles,
                budget,
                workers,
                end_ms,
                session,
                progress,
                label,
                proxypool=None,
                slot=0,
                imgmap=None,
                failed=None,
                save=None,
                save_interval=SAVE_INTERVAL):
    """Fetch a list of z14 tiles via MVT for one region/worker slot.

    Spreads requests across the worker's slice of tokens/proxies. A tile that
    429s/4xx's is requeued on the next token; a per-IP throttle or proxy error
    cools the proxy and requeues. Memory is bounded by chunked submission with a
    per-chunk free + gc. A graceful Ctrl+C (``SHUTDOWN``) stops submitting and
    marks the rest pending. ``imgmap``/``failed`` may be pre-seeded (e.g. a
    resume region's prior state) so this owns the full accumulated state; if
    ``save`` is given it is called at most every ``save_interval`` seconds with
    the not-yet-fetched tiles as pending, an incremental crash-safe checkpoint.

    Args:
        tiles: Tiles to fetch (``mercantile.Tile``).
        budget: Shared ``TokenBudget``.
        workers: Tile-fetch thread count for this region.
        end_ms: Capture cutoff in epoch ms, or None.
        session: A requests Session.
        progress: Live ``rich.progress.Progress`` for the per-tile task.
        label: Per-region task label.
        proxypool: Optional ``ProxyPool``.
        slot: Worker slot index.
        imgmap: Pre-seeded image map to extend, or None for a fresh dict.
        failed: Pre-seeded failed-tile list to extend, or None for a fresh list.
        save: Optional ``save(imgmap, failed, pending)`` checkpoint callback.
        save_interval: Minimum seconds between incremental saves.

    Returns:
        ``(imgmap, failed_tiles, pending_tiles)`` where failed/pending tiles are
        ``[z, x, y]`` lists. failed = errored (retry later); pending = not
        attempted because budget ran out or Ctrl+C stopped the run.
    """
    imgmap = {} if imgmap is None else imgmap
    failed = [] if failed is None else failed
    pending = []
    work = list(tiles)
    task = progress.add_task(f"  [dim]{label}[/]", total=len(work))
    last_save = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            while work:
                if SHUTDOWN.is_set():
                    pending.extend([t.z, t.x, t.y] for t in work)
                    break
                batch, i, exhausted = [], 0, False
                while i < len(work) and len(batch) < API_CHUNK_SIZE:
                    tok = budget.acquire(slot)
                    if tok is None:
                        exhausted = True
                        break
                    batch.append((work[i], tok))
                    i += 1
                remaining_after = work[i:]
                futs = {
                    ex.submit(images_from_tile, t, tok, session, end_ms, proxypool, slot):
                    (t, tok)
                    for t, tok in batch
                }
                requeue = []
                for f in as_completed(futs):
                    t, tok = futs.pop(f)
                    m, err, status, pid = f.result()
                    if proxypool is not None:
                        proxypool.release(pid)
                    if status == 'token_limit':
                        if budget.disable(tok):
                            n = budget.n_active()
                            console.print(
                                f"[yellow]⚠ token {budget.token_label(tok)} "
                                f"rate-limited — disabled for today; "
                                f"{n} token(s) left[/]")
                        requeue.append(t)
                    elif status in ('ip_throttle', 'proxy_err'):
                        if proxypool is not None:
                            proxypool.penalize(pid)
                        requeue.append(t)
                    elif err is not None:
                        failed.append([t.z, t.x, t.y])
                        progress.advance(task)
                    else:
                        imgmap.update(m)
                        progress.advance(task)
                    del m
                budget.flush()
                gc.collect()
                if exhausted:
                    for t in remaining_after + requeue:
                        pending.append([t.z, t.x, t.y])
                    progress.advance(task, len(remaining_after) + len(requeue))
                    break
                work = remaining_after + requeue
                if save is not None and \
                        time.monotonic() - last_save >= save_interval:
                    save(imgmap, failed, [[t.z, t.x, t.y] for t in work])
                    last_save = time.monotonic()
    finally:
        progress.remove_task(task)
    return imgmap, failed, pending


def land_tiles(row):
    """Return the z14 tiles of a region whose centre is on land.

    Args:
        row: Grid row with sw_lon/sw_lat/ne_lon/ne_lat.

    Returns:
        List of ``mercantile.Tile`` over land.
    """
    out = []
    for t in mercantile.tiles(row['sw_lon'], row['sw_lat'], row['ne_lon'],
                              row['ne_lat'], ZOOM):
        b = mercantile.bounds(t)
        if globe.is_land((b.south + b.north) / 2, (b.west + b.east) / 2):
            out.append(t)
    return out


def plan_parallelism(args, budget, proxypool):
    """Derive concurrency from the request, token count, and proxy count.

    Each concurrent region (slot) gets a disjoint stride slice of tokens and
    proxies, so no two run simultaneously on the same token or IP; ``outer`` is
    therefore capped so every slot has at least one token (and one proxy when
    proxies are used). Also configures the budget/proxypool slot partitions and
    splits ``-w`` across the regions so total in-flight fetches stay near ``-w``.

    Args:
        args: Parsed args; ``outer_workers`` and ``workers`` are read.
        budget: Shared ``TokenBudget`` (partitioned in place).
        proxypool: Optional ``ProxyPool`` (partitioned in place).

    Returns:
        ``(outer, inner)`` - regions to run at once and per-region tile workers.
    """
    req = max(1, getattr(args, 'outer_workers', 1))
    caps = [budget.n_tokens()]
    if proxypool is not None:
        caps.append(len(proxypool))
    outer = max(1, min(req, *caps))
    budget.set_nslots(outer)
    if proxypool is not None:
        proxypool.set_nslots(outer)
    inner = max(1, args.workers // outer)
    if outer > 1:
        ptok = budget.n_tokens() // outer
        ppx = (len(proxypool) // outer) if proxypool is not None else 0
        note = (f"[dim]parallel · [cyan]{outer}[/] regions at once, "
                f"~{inner} tiles each · per region: ~{ptok} tokens" +
                (f", ~{ppx} proxies" if proxypool is not None else "") +
                " (disjoint slices)[/]")
        if outer < req:
            note += (f" [yellow](capped from {req}: only "
                     f"{budget.n_tokens()} tokens" +
                     (f"/{len(proxypool)} proxies"
                      if proxypool is not None else "") + ")[/]")
        console.print(note)
    return outer, inner


def run_regions(items, outer, progress, rtask, handle, should_continue):
    """Run ``handle(item, slot)`` over items across ``outer`` worker threads.

    Workers pull one region at a time; worker k is pinned to slot k so it always
    uses token/proxy slice k and concurrent regions never share a token or IP. A
    worker stops taking new regions on a hard stop (a region reported work left
    undone, i.e. handle returned truthy), on Ctrl+C, or when its own slot's
    budget is dry (a single dry slot does not stop the others). In-flight
    regions always finish. With ``outer == 1`` this is the sequential loop.
    ``handle`` runs concurrently and must guard shared state itself.

    Args:
        items: Region work items.
        outer: Number of worker threads / slots.
        progress: Live ``rich.progress.Progress``.
        rtask: The outer Regions task id to advance.
        handle: ``handle(item, slot) -> truthy`` to signal a hard stop.
        should_continue: ``should_continue(slot) -> bool`` budget gate.

    Returns:
        ``(stopped, leftover)`` where leftover is the items never started and
        stopped is True iff leftover is non-empty.
    """
    items = list(items)
    stop = threading.Event()
    idx = [0]
    lock = threading.Lock()

    def worker(slot):
        """Pull and process regions for one fixed slot until done/stopped."""
        while True:
            with lock:
                if stop.is_set() or SHUTDOWN.is_set():
                    return
                if idx[0] >= len(items):
                    return
                if not should_continue(slot):
                    return
                item = items[idx[0]]
                idx[0] += 1
            if handle(item, slot):
                stop.set()
            progress.advance(rtask)

    with ThreadPoolExecutor(max_workers=outer) as ex:
        for f in as_completed([ex.submit(worker, k) for k in range(outer)]):
            f.result()
    with lock:
        leftover = items[idx[0]:]
    return bool(leftover), leftover


# --------------------------------------------------------------------------- #
# enumerate
# --------------------------------------------------------------------------- #
def cmd_enumerate(args, budget=None, proxypool=None):
    """Enumerate every z14 land tile of each region and write checkpoints.

    Resumable: completed regions are skipped and a region cut off by the daily
    budget resumes from its pending tiles. Within ``audit`` the shared budget
    and proxy pool are passed in; standalone they are built here.

    Args:
        args: Parsed args (grid_csv, data_dir, region, workers, etc.).
        budget: Shared ``TokenBudget``, or None to build one.
        proxypool: Shared ``ProxyPool``, or None to build from ``--proxies``.

    Returns:
        ``(failures, stopped)`` where failures is ``{safe: failed_tiles}``
        found this run (so audit can retry without rescanning disk) and stopped
        is True if the run ended early (budget exhausted or Ctrl+C).
    """
    rows = region_rows(args.grid_csv, args.region)
    if proxypool is None:
        proxypool = load_proxies(args)
    session = build_session(pool=args.workers + 32, retry_429=False)
    end_ms = iso_to_ms(args.end_captured_at)
    if budget is None:
        budget = TokenBudget(
            os.path.join(args.data_dir, BUDGET_FILE), load_keys(args.env),
            getattr(args, 'daily_tile_limit', DAILY_TILE_LIMIT))

    todo = [(r, *region_state(args.data_dir, safe_of(r))) for r in rows]
    todo = [(r, st, m) for (r, st, m) in todo if st != 'done']
    n_resume = sum(1 for _, st, _ in todo if st == 'resume')
    outer, inner = plan_parallelism(args, budget, proxypool)
    console.print(
        f"[bold]enumerate[/] · {len(rows)} regions, [cyan]{len(todo)}[/] to do "
        f"({n_resume} resuming) · budget [cyan]{budget.remaining():,}[/] across "
        f"{budget.n_active()} tokens (~{budget.limit:,}/token)")

    failures = {}
    flock = threading.Lock()

    def handle(item, slot):
        """Enumerate one region (fresh or resumed) and checkpoint it.

        Args:
            item: ``(row, state, meta)`` for the region.
            slot: Worker slot index.

        Returns:
            The count of pending tiles (truthy signals a budget mid-region cut).
        """
        r, st, m = item
        safe = safe_of(r)
        if st == 'resume':
            base = load_imgmap(
                os.path.join(args.data_dir, safe, coverage_ckpt_name(safe)))
            base_failed = list(m.get('failed_tiles', []))
            todo_tiles = [
                mercantile.Tile(x=x, y=y, z=z)
                for z, x, y in m.get('pending_tiles', [])
            ]
        else:
            base, base_failed = {}, []
            todo_tiles = land_tiles(r)

        def save(im, fl, pend):
            """Incremental crash-safe checkpoint callback for this region."""
            write_coverage(args.data_dir, safe, im, fl, pend)

        imgmap, failed, pending = fetch_tiles(todo_tiles,
                                              budget,
                                              inner,
                                              end_ms,
                                              session,
                                              progress,
                                              f"{safe} tiles",
                                              proxypool=proxypool,
                                              slot=slot,
                                              imgmap=base,
                                              failed=base_failed,
                                              save=save,
                                              save_interval=getattr(
                                                  args, 'save_interval',
                                                  SAVE_INTERVAL))
        write_coverage(args.data_dir, safe, imgmap, failed, pending)
        if failed:
            with flock:
                failures[safe] = failed
        region_summary(safe, len(imgmap), len(failed), len(pending))
        if pending:
            console.print(
                f"[yellow]⏸ budget reached mid-region ({safe}); "
                f"{len(pending):,} tiles pending. Re-run to continue.[/]")
        del imgmap
        gc.collect()
        return len(pending)

    with make_progress() as progress:
        rtask = progress.add_task("[bold]Regions", total=len(todo))
        stopped, _ = run_regions(todo, outer, progress, rtask, handle,
                                 lambda slot: budget.remaining(slot) > 0)
    if stopped:
        if SHUTDOWN.is_set():
            console.print("[yellow]⏸ interrupted (Ctrl+C); checkpointed. "
                          "Re-run to resume.[/]")
        else:
            console.print("[yellow]⏸ daily tile budget exhausted; stopping. "
                          "Re-run to resume tomorrow.[/]")
    return failures, stopped


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def cmd_check(args):
    """Report per-region coverage status from the tiny meta sidecars.

    Reads only the meta files (instant) and prints counts of clean / failed /
    incomplete / legacy regions plus what to run next.

    Args:
        args: Parsed args (data_dir, region).
    """
    metas = scan_meta(args.data_dir, args.region)
    if not metas:
        console.print("[yellow]No coverage checkpoints found.[/]")
        return
    bad, good, legacy, incomplete = [], [], [], []
    for safe, mp in metas:
        try:
            m = read_meta(mp)
        except Exception:
            legacy.append(safe)
            continue
        if m.get('method') != MVT_METHOD:
            legacy.append(safe)
            continue
        nft = len(m.get('failed_tiles', []))
        npd = len(m.get('pending_tiles', []))
        n = m.get('n_images', 0)
        if npd:
            incomplete.append((safe, npd, nft, n))
        elif nft:
            bad.append((safe, nft, n))
        else:
            good.append((safe, nft, n))

    total_imgs = (sum(n for *_, n in good) + sum(n for _, n in bad) +
                  sum(n for *_, n in incomplete))
    console.print(
        f"[bold]check[/] · {len(metas)} regions · "
        f"[green]{len(good)} clean[/] · [red]{len(bad)} failed[/] · "
        f"[yellow]{len(incomplete)} incomplete[/] · [dim]{len(legacy)} legacy[/]"
        f" · [cyan]{total_imgs:,}[/] images")

    if incomplete or bad or legacy:
        t = Table(box=None, pad_edge=False, header_style="bold")
        t.add_column("region")
        t.add_column("images", justify="right")
        t.add_column("failed", justify="right")
        t.add_column("pending", justify="right")
        t.add_column("status")
        for safe, npd, nft, n in incomplete:
            t.add_row(safe, f"{n:,}", str(nft), str(npd),
                      "[yellow]incomplete[/]")
        for safe, nft, n in bad:
            t.add_row(safe, f"{n:,}", str(nft), "0", "[red]failed[/]")
        for safe in legacy:
            t.add_row(safe, "-", "-", "-", "[dim]legacy/non-MVT[/]")
        console.print(t)
        hint = []
        if bad:
            hint.append("[bold]retry[/] redoes failed tiles")
        if incomplete:
            hint.append("[bold]enumerate[/] resumes pending tiles")
        if legacy:
            hint.append("[bold]enumerate[/] rebuilds legacy regions")
        rgn = f" --region \"{args.region}\"" if args.region else ""
        console.print(f"[dim]→ {'; '.join(hint)} (e.g. "
                      f"coverage_audit.py retry{rgn})[/]")
    else:
        console.print("[green]✓ All MVT regions are complete and clean.[/]")


# --------------------------------------------------------------------------- #
# retry
# --------------------------------------------------------------------------- #
def retry_core(args, failures, session, end_ms, budget, proxypool=None):
    """Re-fetch failed tiles per region, merge recovered images, rewrite state.

    Each region's pending_tiles (from a budget-truncated enumerate) are
    preserved untouched and existing images are kept. Regions never started
    because budget ran out are carried forward so the next run retries them.

    Args:
        args: Parsed args (data_dir, workers, etc.).
        failures: ``{safe: failed_tiles}`` to retry.
        session: A requests Session.
        end_ms: Capture cutoff in epoch ms, or None.
        budget: Shared ``TokenBudget``.
        proxypool: Optional ``ProxyPool``.

    Returns:
        ``(still_failures, total)`` - regions still failing and the total count
        of tiles still failing.
    """
    if not failures:
        console.print("[dim]retry · nothing to retry (no failed tiles).[/]")
        return {}, 0
    outer, inner = plan_parallelism(args, budget, proxypool)
    still_failures, total = {}, 0
    slock = threading.Lock()
    console.print(
        f"[bold]retry[/] · {len(failures)} region(s) with failed tiles · "
        f"budget [cyan]{budget.remaining():,}[/] across {budget.n_active()} "
        f"tokens")

    def handle(item, slot):
        """Retry one region's failed tiles and rewrite its checkpoint.

        Args:
            item: ``(safe, failed_tiles)`` for the region.
            slot: Worker slot index.

        Returns:
            The count of pending tiles (truthy signals a budget mid-region cut).
        """
        safe, failed = item
        tiles = [mercantile.Tile(x=x, y=y, z=z) for z, x, y in failed]
        recovered, new_failed, pending = fetch_tiles(tiles,
                                                     budget,
                                                     inner,
                                                     end_ms,
                                                     session,
                                                     progress,
                                                     f"{safe} retry-tiles",
                                                     proxypool=proxypool,
                                                     slot=slot)
        still = new_failed + pending
        _, m = region_state(args.data_dir, safe)
        region_pending = m.get('pending_tiles', []) if m else []
        ckpt = os.path.join(args.data_dir, safe, coverage_ckpt_name(safe))
        imgmap = load_imgmap(ckpt)
        before = len(imgmap)
        imgmap.update(recovered)
        write_coverage(args.data_dir, safe, imgmap, still, region_pending)
        gained = len(imgmap) - before
        with slock:
            if still:
                still_failures[safe] = still
            nonlocal total
            total += len(still)
        region_summary(safe, len(imgmap), len(still), 0, gained=gained)
        del imgmap, recovered, tiles
        gc.collect()
        return len(pending)

    items = sorted(failures.items())
    with make_progress() as progress:
        rtask = progress.add_task("[bold]Regions", total=len(failures))
        _, leftover = run_regions(items, outer, progress, rtask, handle,
                                  lambda slot: budget.remaining(slot) > 0)
    for safe, failed in leftover:
        still_failures[safe] = failed
        total += len(failed)
    tail = "[green]all clean[/]" if total == 0 else f"[red]{total:,}[/] still failing"
    console.print(
        f"[bold]retry[/] done · {tail} across {len(failures)} region(s)")
    return still_failures, total


def cmd_retry(args):
    """Standalone retry: discover failed tiles from meta sidecars and re-fetch.

    Within ``audit`` the failed-tile set is threaded straight from enumerate
    (no rescan); this entry point rescans disk instead.

    Args:
        args: Parsed args (data_dir, region, workers, etc.).

    Returns:
        Total count of tiles still failing after the retry.
    """
    session = build_session(pool=args.workers + 32, retry_429=False)
    proxypool = load_proxies(args)
    end_ms = iso_to_ms(args.end_captured_at)
    budget = TokenBudget(os.path.join(args.data_dir, BUDGET_FILE),
                         load_keys(args.env),
                         getattr(args, 'daily_tile_limit', DAILY_TILE_LIMIT))

    failures, legacy = {}, []
    for safe, mp in scan_meta(args.data_dir, args.region):
        try:
            m = read_meta(mp)
        except Exception:
            continue
        if m.get('method') != MVT_METHOD:
            legacy.append(safe)
        elif m.get('failed_tiles'):
            failures[safe] = m['failed_tiles']
    if legacy:
        console.print(
            f"[dim]retry · {len(legacy)} legacy/non-MVT region(s) -- "
            f"re-run enumerate to rebuild those.[/]")
    _, total = retry_core(args, failures, session, end_ms, budget, proxypool)
    return total


# --------------------------------------------------------------------------- #
# audit  (enumerate -> retry-until-clean -> diff -> datefilter)
# --------------------------------------------------------------------------- #
def cmd_audit(args):
    """Run the full pipeline: enumerate, retry-until-clean, diff, datefilter.

    One shared budget and session are reused across enumerate and every retry
    round so per-token daily caps are honoured across the whole pipeline. Stops
    early (and skips diff/datefilter) on budget exhaustion or Ctrl+C.

    Args:
        args: Parsed args for all four stages.
    """
    from types import SimpleNamespace

    session = build_session(pool=args.workers + 32, retry_429=False)
    proxypool = load_proxies(args)
    end_ms = iso_to_ms(args.end_captured_at)
    budget = TokenBudget(os.path.join(args.data_dir, BUDGET_FILE),
                         load_keys(args.env), args.daily_tile_limit)

    console.rule("[bold]audit 1/4 · enumerate")
    failures, stopped = cmd_enumerate(SimpleNamespace(
        grid_csv=args.grid_csv,
        data_dir=args.data_dir,
        region=args.region,
        end_captured_at=args.end_captured_at,
        workers=args.workers,
        outer_workers=args.outer_workers,
        save_interval=args.save_interval,
        env=args.env,
        daily_tile_limit=args.daily_tile_limit),
                                      budget=budget,
                                      proxypool=proxypool)
    if stopped:
        why = ("interrupted (Ctrl+C)"
               if SHUTDOWN.is_set() else "daily tile budget reached")
        console.print(
            f"\n[yellow]⏸ Enumeration incomplete -- {why}. "
            "Re-run this same command to continue (resumes automatically); "
            "diff/datefilter run once enumeration completes.[/]")
        return

    console.rule("[bold]audit 2/4 · retry until clean")
    prev = None
    for rnd in range(1, args.max_retry_rounds + 1):
        console.print(f"[dim]retry round {rnd}/{args.max_retry_rounds}[/]")
        failures, total = retry_core(args, failures, session, end_ms, budget,
                                     proxypool)
        if SHUTDOWN.is_set():
            console.print("\n[yellow]⏸ Interrupted (Ctrl+C) during retry. "
                          "Re-run to continue.[/]")
            return
        if total == 0:
            console.print("[green]✓ all regions clean.[/]")
            break
        if budget.remaining() <= 0:
            console.print(
                "\n[yellow]⏸ daily tile budget exhausted during retry. Re-run "
                "to finish the remaining tiles (resumes automatically).[/]")
            return
        if prev is not None and total >= prev:
            console.print(f"[yellow]no progress since last round ({prev:,} -> "
                          f"{total:,}); remaining tiles look permanently "
                          f"unavailable. Moving on.[/]")
            break
        prev = total
    else:
        console.print(f"[yellow]hit --max-retry-rounds "
                      f"({args.max_retry_rounds}); moving on with tiles still "
                      f"failing.[/]")

    console.rule("[bold]audit 3/4 · diff")
    cmd_diff(
        SimpleNamespace(grid_csv=args.grid_csv,
                        dirs=args.dirs,
                        data_dir=args.data_dir,
                        region=args.region,
                        out=args.missing_out))
    if SHUTDOWN.is_set():
        console.print("\n[yellow]⏸ Interrupted (Ctrl+C) during diff. "
                      "Re-run to resume; datefilter runs once diff finishes.[/]")
        return

    console.rule("[bold]audit 4/4 · datefilter")
    cmd_datefilter(
        SimpleNamespace(missing=args.missing_out,
                        cutoff=args.cutoff,
                        out=args.inscope_out,
                        workers=args.entity_workers,
                        env=args.env))

    console.print(f"\n[bold green]✓ audit DONE[/] · backfill targets "
                  f"(captured ≤ {args.cutoff}) → [cyan]{args.inscope_out}[/]")


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #
def _save_diff_progress(path, done, offset, stats):
    """Atomically persist diff resume state.

    Args:
        path: Progress sidecar path.
        done: Set of safe region ids already written.
        offset: Byte offset in the output CSV up to the last fully-written
            region (any bytes past this are a partial region, dropped on resume).
        stats: Cumulative summary counters.
    """
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(
            orjson.dumps({
                'done': sorted(done),
                'offset': offset,
                'stats': stats
            }))
    os.replace(tmp, path)


def cmd_diff(args):
    """Compute missing images per region and write them to a CSV.

    missing = coverage_ids - all_data_ids. Only MVT checkpoints are processed
    (pre-MVT ones are skipped). The captured_at from the checkpoint is carried
    into each output row, and a per-region/overall summary is printed.

    Resumable per region: a ``<out>.progress.json`` sidecar records the regions
    already written, the CSV byte offset up to the last fully-written region,
    and the running summary. On a re-run that sidecar (if present) makes diff
    truncate any partial region, skip the done ones, and continue; it is deleted
    on full completion, so a clean run always starts fresh. Ctrl+C / SHUTDOWN
    stops after the current region with progress saved. Delete the sidecar to
    force a full re-diff.

    Memory: missing rows are streamed straight to the CSV (never accumulated
    across regions) and each region does a single pass over its image map with
    no duplicate id sets, so peak stays at roughly one region's map plus its
    downloaded-id set.

    Args:
        args: Parsed args (grid_csv, dirs, data_dir, region, out).
    """
    rows = region_rows(args.grid_csv, args.region)
    cols = [
        'image_id', 'safe_region_id', 'parent_region', 'sequence',
        'captured_at'
    ]
    prog_path = args.out + '.progress.json'
    done = set()
    stats = {
        'tot_cov': 0,
        'tot_have': 0,
        'tot_missing': 0,
        'full_seq': 0,
        'part_seq': 0,
        'n_written': 0
    }
    resume = os.path.exists(prog_path) and os.path.exists(args.out)
    offset = 0
    if resume:
        try:
            pj = read_meta(prog_path)
            done = set(pj.get('done', []))
            stats.update(pj.get('stats', {}))
            offset = min(pj.get('offset', 0), os.path.getsize(args.out))
        except Exception:
            resume = False

    fout = open(args.out, 'r+', newline='') if resume \
        else open(args.out, 'w', newline='')
    interrupted = False
    try:
        if resume:
            fout.seek(offset)
            fout.truncate()
            console.print(f"[dim]diff · resuming — {len(done)} region(s) "
                          f"already done, continuing[/]")
        w = csv.writer(fout)
        if not resume:
            w.writerow(cols)
            fout.flush()
            offset = fout.tell()
        with make_progress() as progress:
            dtask = progress.add_task("[bold]diff regions", total=len(rows))
            for r in rows:
                safe = safe_of(r)
                if safe in done:
                    progress.advance(dtask)
                    continue
                if SHUTDOWN.is_set():
                    interrupted = True
                    break
                cov_f = find_coverage_ckpt(safe, [args.data_dir])
                meta_p = os.path.join(args.data_dir, safe,
                                      coverage_meta_name(safe))
                ok = bool(cov_f) and os.path.exists(meta_p)
                if ok:
                    try:
                        ok = read_meta(meta_p).get('method') == MVT_METHOD
                    except Exception:
                        ok = False
                if not cov_f:
                    pass
                elif not ok:
                    console.print(f"  [yellow][skip][/] {safe}: non-MVT "
                                  f"checkpoint -- re-run enumerate")
                if cov_f and ok:
                    imgmap = load_imgmap(cov_f)
                    have = all_data_ids(safe, args.dirs)
                    region = r['region']
                    region_cov = len(imgmap)
                    region_missing = 0
                    seq_total, seq_missing = {}, {}
                    for iid, val in imgmap.items():
                        seq, cap = val[0], val[1]
                        seq_total[seq] = seq_total.get(seq, 0) + 1
                        if iid in have:
                            continue
                        region_missing += 1
                        seq_missing[seq] = seq_missing.get(seq, 0) + 1
                        w.writerow((iid, safe, region, seq or '',
                                    cap if cap is not None else ''))
                    for s, mc in seq_missing.items():
                        if mc == seq_total[s]:
                            stats['full_seq'] += 1
                        else:
                            stats['part_seq'] += 1
                    stats['tot_cov'] += region_cov
                    stats['tot_missing'] += region_missing
                    stats['tot_have'] += region_cov - region_missing
                    stats['n_written'] += region_missing
                    if region_missing:
                        console.print(
                            f"  [bold]{safe:<34}[/] "
                            f"coverage=[cyan]{region_cov:>11,}[/]"
                            f"  have=[green]{region_cov - region_missing:>11,}[/]"
                            f"  missing=[red]{region_missing:>10,}[/]")
                    del imgmap, have, seq_total, seq_missing
                    gc.collect()
                fout.flush()
                offset = fout.tell()
                done.add(safe)
                _save_diff_progress(prog_path, done, offset, stats)
                progress.advance(dtask)
    finally:
        fout.close()

    if interrupted:
        console.print("[yellow]⏸ diff interrupted; progress saved "
                      f"({len(done)}/{len(rows)} regions). Re-run to resume.[/]")
        return

    if os.path.exists(prog_path):
        os.remove(prog_path)
    console.print(f"[bold]diff[/] · coverage [cyan]{stats['tot_cov']:,}[/] · "
                  f"have [green]{stats['tot_have']:,}[/] · "
                  f"[red]MISSING {stats['tot_missing']:,}[/]")
    console.print(f"[dim]   sequences fully missing {stats['full_seq']:,} · "
                  f"partially missing {stats['part_seq']:,}[/]")
    console.print(f"[dim]   wrote {stats['n_written']:,} rows → {args.out} · "
                  f"next: datefilter[/]")


# --------------------------------------------------------------------------- #
# datefilter  (local when captured_at present; API fallback otherwise)
# --------------------------------------------------------------------------- #
def cmd_datefilter(args):
    """Keep missing rows captured on/before the cutoff date.

    Local (no API) when ``captured_at`` is present in the missing CSV (it is,
    from MVT); falls back to the Graph API only for rows lacking it.

    Args:
        args: Parsed args (missing, cutoff, out, workers, env, proxies).
    """
    cutoff = dt.datetime.strptime(args.cutoff, '%Y-%m-%d')
    cutoff_ms = int((cutoff + dt.timedelta(days=1)).timestamp() * 1000)
    rows = list(csv.DictReader(open(args.missing)))
    cols = [
        'image_id', 'safe_region_id', 'parent_region', 'sequence',
        'captured_at'
    ]

    keep, newer, gone, unk = [], 0, 0, 0
    need_api = []
    for r in rows:
        c = r.get('captured_at')
        if c in (None, ''):
            need_api.append(r)
            continue
        try:
            c = int(c)
        except (TypeError, ValueError):
            need_api.append(r)
            continue
        if c <= cutoff_ms:
            keep.append(r)
        else:
            newer += 1
    n_local = len(keep) + newer

    if need_api:
        rot = KeyRotator(load_keys(args.env))
        session = build_session(pool=args.workers + 32)
        proxypool = load_proxies(args)

        def cap(row):
            """Fetch one image's captured_at via the Graph API.

            Args:
                row: The missing-CSV row (mutated/returned).

            Returns:
                ``(row, captured_at_or_None, dead)`` where dead marks a 400/404.
            """
            iid = row['image_id']
            url = (f'https://graph.mapillary.com/{iid}'
                   f'?access_token={rot.next()}&fields=captured_at')
            pid, proxies = (None, None)
            if proxypool is not None:
                pid, proxies = proxypool.acquire()
            try:
                r = session.get(url, timeout=15, proxies=proxies)
                if r.status_code in (400, 404):
                    return row, None, True
                return row, r.json().get('captured_at'), False
            except Exception:
                return row, None, False
            finally:
                if proxypool is not None:
                    proxypool.release(pid)

        with make_progress() as progress, \
                ThreadPoolExecutor(max_workers=args.workers) as ex:
            task = progress.add_task("[bold]captured_at (API fallback)",
                                     total=len(need_api))
            for row, c, dead in (f.result() for f in as_completed(
                    ex.submit(cap, r) for r in need_api)):
                if dead:
                    gone += 1
                elif c is None:
                    unk += 1
                elif int(c) <= cutoff_ms:
                    row['captured_at'] = c
                    keep.append(row)
                else:
                    newer += 1
                progress.advance(task)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in keep:
            w.writerow({k: r.get(k, '') for k in cols})
    console.print(
        f"[bold]datefilter[/] · in-scope (≤ {args.cutoff}) "
        f"[green]{len(keep):,}[/] · newer (skip) [yellow]{newer:,}[/] · "
        f"local [cyan]{n_local:,}[/] · via API {len(need_api):,}")
    if need_api:
        console.print(f"[dim]   gone (404/400) {gone:,} · "
                      f"unknown/transient {unk:,}[/]")
    console.print(f"[dim]   wrote → {args.out}[/]")


def main():
    """Parse the CLI, install the SIGINT handler, and dispatch the subcommand."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)

    def add_data_dir(sp):
        """Add the shared ``--data-dir`` option to a subparser."""
        sp.add_argument('--data-dir',
                        required=True,
                        help='Dir for coverage checkpoints (parquet/data dir, '
                        'NOT image dirs). Default: crucial grid_runs.')

    def add_proxies(sp):
        """Add the shared ``--proxies`` option to a subparser."""
        sp.add_argument(
            '--proxies',
            default=None,
            metavar='FILE',
            help='Opt-in: file of HTTP proxies (one per line, '
            '`host:port:user:pass` or `host:port`), rotated per '
            'request to spread tile fetches across many IPs. The '
            'tiles.mapillary.com throttle is per-IP, so this lifts '
            'the ceiling. Off by default (direct).')

    def add_outer(sp):
        """Add the shared ``--outer-workers`` / ``--save-interval`` options."""
        sp.add_argument(
            '--outer-workers',
            type=int,
            default=1,
            help='Process this many regions concurrently (like '
            "batch_chunks' --outer-max-workers). Each concurrent "
            'region is auto-assigned a DISJOINT slice of tokens and '
            'proxies (capped so every slice has >=1 of each), so no '
            'two regions run on the same token or IP at once. The '
            '-w tile concurrency is split across them (total '
            'in-flight stays ~ -w; raise -w for more). Default 1.')
        sp.add_argument('--save-interval',
                        type=int,
                        default=SAVE_INTERVAL,
                        metavar='SECONDS',
                        help='Seconds between incremental within-region '
                        'checkpoints (crash/Ctrl+C resume from the last flush '
                        f'instead of restarting the region). Default '
                        f'{SAVE_INTERVAL}.')

    e = sub.add_parser('enumerate')
    e.add_argument('grid_csv')
    e.add_argument('--dirs',
                   nargs='+',
                   required=False,
                   help='(accepted for symmetry; enumeration only writes to '
                   '--data-dir)')
    add_data_dir(e)
    e.add_argument('--region', default=None)
    e.add_argument('--end-captured-at',
                   default=None,
                   help='ISO date w/ tz, e.g. 2026-06-01T00:00:00Z; images '
                   'captured after this are skipped (optional).')
    e.add_argument('-w',
                   '--workers',
                   type=int,
                   default=64,
                   help='Concurrent tile fetches. Lower than the bbox-era 150 '
                   'because MVT tiles are much larger.')
    e.add_argument(
        '--daily-tile-limit',
        type=int,
        default=DAILY_TILE_LIMIT,
        help='tiles.mapillary.com requests/day cap (default 50000). '
        'Stops at the cap; resumes next run.')
    add_outer(e)
    add_proxies(e)
    e.add_argument('--env', default='.env')

    c = sub.add_parser('check')
    add_data_dir(c)
    c.add_argument('--region',
                   default=None,
                   help='Limit to regions whose id starts with this.')

    rt = sub.add_parser('retry')
    add_data_dir(rt)
    rt.add_argument('--region',
                    default=None,
                    help='Limit to regions whose id starts with this.')
    rt.add_argument('--end-captured-at',
                    default=None,
                    help='Same value used in the original enumerate, if any.')
    rt.add_argument('-w', '--workers', type=int, default=64)
    rt.add_argument(
        '--daily-tile-limit',
        type=int,
        default=DAILY_TILE_LIMIT,
        help='tiles.mapillary.com requests/day cap (default 50000).')
    add_outer(rt)
    add_proxies(rt)
    rt.add_argument('--env', default='.env')

    d = sub.add_parser('diff')
    d.add_argument('grid_csv')
    d.add_argument('--dirs', nargs='+', required=True)
    add_data_dir(d)
    d.add_argument('--region', default=None)
    d.add_argument('--out', default='coverage_missing.csv')

    f = sub.add_parser('datefilter')
    f.add_argument('--missing', default='coverage_missing.csv')
    f.add_argument('--cutoff', default='2026-05-31')
    f.add_argument('--out', default='coverage_missing_inscope.csv')
    f.add_argument('-w',
                   '--workers',
                   type=int,
                   default=520,
                   help='Concurrency for the API fallback only (mirrors the '
                   'pipeline ENTITY_MAX_WORKERS=520).')
    add_proxies(f)
    f.add_argument('--env', default='.env')

    a = sub.add_parser(
        'audit', help='enumerate -> retry-until-clean -> diff -> datefilter.')
    a.add_argument('grid_csv')
    a.add_argument('--dirs', nargs='+', required=True)
    add_data_dir(a)
    a.add_argument('--region', default=None)
    a.add_argument(
        '--end-captured-at',
        default=None,
        help='ISO date w/ tz; images captured after this are skipped.')
    a.add_argument('-w',
                   '--workers',
                   type=int,
                   default=64,
                   help='Tile-fetch concurrency for enumerate/retry.')
    a.add_argument('--entity-workers',
                   type=int,
                   default=520,
                   help='datefilter API-fallback concurrency.')
    a.add_argument('--max-retry-rounds',
                   type=int,
                   default=5,
                   help='Cap on retry passes (default 5). Stops early when '
                   'clean or no progress is made.')
    a.add_argument(
        '--daily-tile-limit',
        type=int,
        default=DAILY_TILE_LIMIT,
        help='tiles.mapillary.com requests/day cap (default 50000). '
        'Audit stops at the cap and resumes on the next run.')
    a.add_argument('--cutoff', default='2026-05-31')
    a.add_argument('--missing-out', default='coverage_missing.csv')
    a.add_argument('--inscope-out', default='coverage_missing_inscope.csv')
    add_outer(a)
    add_proxies(a)
    a.add_argument('--env', default='.env')

    args = p.parse_args()
    if args.mode in ('enumerate', 'retry', 'audit', 'diff'):
        _install_sigint()
    {
        'enumerate': cmd_enumerate,
        'check': cmd_check,
        'retry': cmd_retry,
        'diff': cmd_diff,
        'datefilter': cmd_datefilter,
        'audit': cmd_audit
    }[args.mode](args)


if __name__ == '__main__':
    main()
