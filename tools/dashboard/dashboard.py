"""
A self-refreshing progress dashboard for the street_dogs_mp collection.

Reads the DuckDB catalog (tools/catalog/catalog.py) and renders a beautiful
static dark dashboard showing, overall and per region: images scanned
(all_data), ground animals found, downloaded jpgs, and download progress
(downloaded / ground_animals). Each refresh appends an hourly snapshot to a
local history DB so the trend charts grow over time.

    # build once from the existing catalog (no drive scan):
    python tools/dashboard/dashboard.py build --no-refresh

    # serve + auto-refresh every hour:
    python tools/dashboard/dashboard.py serve --host <bind-addr> --port 8050

There is no authentication. Bind it to a private interface (a Tailscale or LAN
address), never to a public one.

Read-only on the data. Writes only data/dashboard/ (index.html + history.duckdb)
and refreshes the catalog.

Machine-specific settings -- drive paths, the sweep's interpreter, the crop
dataset -- are read from the environment or from an optional gitignored
tools/dashboard/dashboard.config.json; see dashboard.config.example.json and
the "configuration" block below. Nothing here is hardcoded to one host, and
tools/detect/tests/adv_no_hardcoded_paths.py fails the build if that changes.
"""

import argparse
import functools
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import signal
import subprocess
import threading
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import duckdb

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOG = os.path.join(REPO, 'tools', 'catalog', 'catalog.py')
OUT = os.path.join(REPO, 'data', 'dashboard')
HIST = os.path.join(OUT, 'history.duckdb')
# recent positive-detection crops, written by tools/detect's PreviewWriter.
# Inside OUT on purpose: the static handler already serves it at /recent_crops/.
CROPS = os.path.join(OUT, 'recent_crops')
ECHARTS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'echarts.min.js')

# ── configuration ───────────────────────────────────────────────────────────
# Everything that is true of THIS machine and not of the repo lives here, so
# the module itself carries no absolute paths. Same shape as sweep.py's
# load_cfg(): an OPTIONAL gitignored JSON beside the script, with defaults
# that either derive from REPO or are empty.
#
# Precedence, identical for every key:  environment  >  config file  >  default
#
# An empty value is not an error. Each consumer degrades on its own: the
# command generator prints a placeholder, the dataset panel says which key to
# set, the sweep launcher falls back to the running interpreter. A fresh clone
# with no config and no environment builds and serves without a traceback.
CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'dashboard.config.json')


_cfg_cache = {'mtime': None, 'data': {}}


def load_cfg():
    """Parsed dashboard.config.json, or {} if absent.

    Re-read whenever the file changes, NOT cached for the process lifetime.
    The server runs for days; when this was an lru_cache, repointing
    dogbin_dataset from v3 to v4 had no effect on the live page, and the
    review panel went on counting every flag made since v3's build as "not
    built yet" -- 641 of them, all of which were in fact in v4. A stale config
    that looks like live data is worse than no config.

    A malformed config is reported and then ignored rather than being allowed
    to take the dashboard down -- it is local preference, not data.
    """
    try:
        mtime = os.path.getmtime(CFG_PATH)
    except OSError:
        _cfg_cache.update(mtime=None, data={})
        return {}
    if _cfg_cache['mtime'] == mtime:
        return _cfg_cache['data']
    try:
        with open(CFG_PATH) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        sys.stderr.write(f'warning: ignoring {CFG_PATH}: {e}\n')
        data = {}
    if not isinstance(data, dict):
        data = {}
    _cfg_cache.update(mtime=mtime, data=data)
    return data


def cfg(key, default='', env=None):
    """One config key. ``env`` names the environment variable, and defaults to
    ``DASHBOARD_<KEY>``; a few keys reuse a variable name that other tools in
    this repo already honour (SWEEP_PYTHON, DOGBIN_DATASET)."""
    v = os.environ.get(env or ('DASHBOARD_' + key.upper()))
    if v:
        return v
    v = load_cfg().get(key)
    if v is not None and not isinstance(v, str):
        # A key that IS set but is not a string used to fall through to the
        # default with nothing on screen -- a config the user had written and
        # the dashboard silently ignored. Lists have their own reader below.
        sys.stderr.write(
            f'warning: {CFG_PATH}: "{key}" is {type(v).__name__}, not a '
            f'string -- ignored. Use cfg_list() for a list-valued key.\n')
    return v if isinstance(v, str) and v else default


def cfg_int(key, default=0, env=None):
    """An integer config key, from a JSON number or a numeric string.

    cfg() returns strings only, so a key written the natural way -- 48, not
    "48" -- fell through to the default. It warns now, which is how this one
    was caught, but a caller that wants a number should ask for a number.
    """
    raw = os.environ.get(env or ('DASHBOARD_' + key.upper()))
    v = raw if raw else load_cfg().get(key)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def cfg_list(key, env=None):
    """A list-valued config key, from JSON or a comma-separated string."""
    raw = os.environ.get(env or ('DASHBOARD_' + key.upper()))
    if raw:
        return [x.strip() for x in raw.split(',') if x.strip()]
    v = load_cfg().get(key)
    if isinstance(v, str):
        return [x.strip() for x in v.split(',') if x.strip()]
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return []

REGION_SQL = """
WITH f AS (
  SELECT region,
         sum(n_rows) FILTER (WHERE kind='all_data')       AS all_data,
         sum(n_rows) FILTER (WHERE kind='ground_animals') AS dogs
  FROM files GROUP BY region),
im AS ({im_src})
SELECT f.region, coalesce(all_data,0), coalesce(dogs,0), coalesce(downloaded,0)
FROM f LEFT JOIN im USING(region) ORDER BY dogs DESC NULLS LAST
"""
# deduped per-cell image counts when available (corrects cross-drive
# double-counting); falls back to the raw per-drive sum otherwise.
DEDUP_SRC = "SELECT region, sum(n_unique) AS downloaded FROM cell_images GROUP BY region"
RAW_SRC = "SELECT region, sum(n_images) AS downloaded FROM images GROUP BY region"

WORKLIST = os.path.join(REPO, 'data', 'detect', 'worklist')
DISTINCT_CACHE = os.path.join(OUT, 'distinct_counts.json')


def latest_worklist():
    """Newest frozen worklist generation directory, or None."""
    gens = sorted(glob.glob(os.path.join(WORKLIST, 'gen=*')))
    return gens[-1] if gens else None


def distinct_counts(db, force=False):
    """DISTINCT image_id counts for both sides of the completeness figure.

    The catalog stores ROW counts, never id counts, and the two sides of
    ``dogs - downloaded`` are inflated in opposite directions:

      * ground_animals manifests re-list ids across backfill parquets --
        33,179,477 rows carry only 32,049,453 distinct ids (3.41% repeats);
      * ``cell_images.n_unique`` dedupes across drives but not across cells,
        so an image that falls in two cells is counted twice.

    Subtracting one from the other reported 637K images still to download
    that were already on disk. Both sides are counted here in the same unit.

    The jpg side reads the frozen worklist's ``.ids.npy`` arrays rather than
    walking the drives -- same ids, no I/O against the six disks. Returns
    None if either side is unavailable (no worklist, no numpy), in which case
    the caller must not present a completeness figure at all.

    ~25s to compute, so it is cached against a signature of its inputs.
    """
    gen = latest_worklist()
    if not gen:
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    dirs_json = os.path.join(gen, '_dirs.json')
    con = duckdb.connect(db, read_only=True)
    try:
        files = con.execute(
            "SELECT region, path, n_rows FROM files WHERE kind='ground_animals'"
        ).fetchall()
    finally:
        con.close()
    if not files:
        return None
    try:
        sig = [len(files), sum(r[2] or 0 for r in files), gen,
               int(os.path.getmtime(dirs_json))]
    except OSError:
        return None
    if not force:
        try:
            with open(DISTINCT_CACHE) as fh:
                cached = json.load(fh)
            if cached.get('sig') == sig:
                return cached
        except (OSError, ValueError):
            pass

    def q(con, paths, group):
        src = 'read_parquet([' + ','.join(
            "'" + p.replace("'", "''") + "'" for p in paths) + '])'
        return con.execute(
            f'SELECT count(DISTINCT image_id) FROM {src}').fetchone()[0]

    by_region = {}
    for reg, path, _ in files:
        by_region.setdefault(reg, []).append(path)
    con = duckdb.connect(db, read_only=True)
    try:
        manifest = {r: q(con, p, r) for r, p in by_region.items()}
        manifest_all = q(con, [f[1] for f in files], None)
    finally:
        con.close()

    with open(dirs_json) as fh:
        pairs = json.load(fh)
    arrays = {}
    for pr in pairs:
        f = os.path.join(gen, pr['cell'], pr['drive'] + '.ids.npy')
        if os.path.exists(f):
            arrays.setdefault(pr['region'], []).append(np.load(f))
    if not arrays:
        return None
    jpg = {r: int(np.unique(np.concatenate(a)).size)
           for r, a in arrays.items()}
    jpg_all = int(np.unique(np.concatenate(
        [np.concatenate(a) for a in arrays.values()])).size)

    out = {'sig': sig, 'gen': os.path.basename(gen),
           'manifest': manifest, 'manifest_all': manifest_all,
           'jpg': jpg, 'jpg_all': jpg_all}
    os.makedirs(OUT, exist_ok=True)
    tmp = DISTINCT_CACHE + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, DISTINCT_CACHE)
    return out


def human(n):
    """Format a count compactly, always 2 decimals (2.28B, 24.70M, 9.96K).

    Trailing zeros are KEPT: 4.9M and 4.90M read as different precisions, and
    a column that alternates between them is hard to scan. The JS fmt() in the
    page mirrors this exactly -- they used to disagree (4.78M vs 4.8M for the
    same number) depending on whether a figure was rendered server- or
    client-side.
    """
    n = n or 0
    for div, suf in ((1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
        if abs(n) >= div:
            return f"{n / div:.2f}{suf}"
    return f"{n:,}"


def hbytes(n):
    """Format a byte count compactly (e.g. 1.66 TB, 51 GB)."""
    n = n or 0
    for div, suf in ((1e12, 'TB'), (1e9, 'GB'), (1e6, 'MB'), (1e3, 'KB')):
        if n >= div:
            return f"{n / div:.2f}".rstrip('0').rstrip('.') + ' ' + suf
    return f"{n} B"


class CatalogMissing(Exception):
    """The DuckDB catalog has not been built yet.

    Deliberately NOT a SystemExit: serve() already treats a failed build as
    survivable (the catalog takes an exclusive lock, so a maintenance job
    holding it must not take the server down) and catches Exception. Raising
    SystemExit here would sail past that handler and reintroduce the outage.
    main() turns it into a one-line exit for the CLI.
    """


def require_catalog(db):
    """Fail with an instruction rather than a duckdb traceback.

    data/catalog.duckdb is built by scanning the drives, so it can never be
    committed -- every fresh clone starts without it. duckdb's own message
    ("Cannot open database ... in read-only mode") does not tell a new
    operator which command produces the file, and this is the first thing a
    fresh checkout hits.
    """
    if not os.path.exists(db):
        raise CatalogMissing(
            f'no catalog at {db}\n'
            '  build one:  python tools/catalog/catalog.py refresh\n'
            '  it scans the data roots in data/catalog_dirs.txt; see '
            'tools/dashboard/dashboard.config.example.json for the paths '
            'the dashboard itself needs.')


def query_metrics(db):
    """Return ``(overall, per_region)`` metrics dicts from the catalog."""
    require_catalog(db)
    con = duckdb.connect(db, read_only=True)
    try:
        has_dedup = con.execute(
            "SELECT count(*) FROM cell_images").fetchone()[0]
    except duckdb.Error:
        has_dedup = 0
    rows = con.execute(
        REGION_SQL.format(
            im_src=DEDUP_SRC if has_dedup else RAW_SRC)).fetchall()
    files = con.execute("SELECT count(*) FROM files").fetchone()[0]
    drives = con.execute(
        "SELECT count(DISTINCT d) FROM (SELECT drive d FROM "
        "files UNION SELECT drive d FROM images)").fetchone()[0]
    con.close()
    # dogs/downloaded arrive as row counts; swap in distinct-id counts where
    # we have them, so the two are subtractable. See distinct_counts().
    dc = distinct_counts(db) or {}
    man, jpg = dc.get('manifest') or {}, dc.get('jpg') or {}
    per = [{
        'region': r[0].replace('_', ' '),
        'key': r[0],
        'all_data': r[1],
        'dogs': man.get(r[0], r[2]),
        'downloaded': jpg.get(r[0], r[3]),
    } for r in rows]
    for p in per:
        p['pct'] = (p['downloaded'] / p['dogs'] * 100) if p['dogs'] else 0
    per.sort(key=lambda p: p['dogs'], reverse=True)
    ov = {
        'all_data': sum(p['all_data'] for p in per),
        # global distinct, not the sum of per-region: 22.7K ids sit in cells
        # straddling two regions and would otherwise be counted twice.
        'dogs': dc.get('manifest_all') or sum(p['dogs'] for p in per),
        'downloaded': dc.get('jpg_all') or sum(p['downloaded'] for p in per),
        'exact': bool(dc),
        'regions': len(per),
        'files': files,
        'drives': drives
    }
    ov['pct'] = (ov['downloaded'] / ov['dogs'] * 100) if ov['dogs'] else 0
    return ov, per


def record_history(per, ts):
    """Append a per-region snapshot to the local history DB."""
    os.makedirs(OUT, exist_ok=True)
    h = duckdb.connect(HIST)
    h.execute("CREATE TABLE IF NOT EXISTS hist(ts TIMESTAMP, region VARCHAR, "
              "all_data BIGINT, dogs BIGINT, downloaded BIGINT)")
    h.executemany("INSERT INTO hist VALUES (?,?,?,?,?)",
                  [(ts, p['key'], p['all_data'], p['dogs'], p['downloaded'])
                   for p in per])
    h.close()


def trend():
    """Return the overall time series (one point per snapshot)."""
    if not os.path.exists(HIST):
        return []
    h = duckdb.connect(HIST, read_only=True)
    rows = h.execute(
        "SELECT ts, sum(all_data), sum(dogs), sum(downloaded) FROM hist "
        "GROUP BY ts ORDER BY ts").fetchall()
    h.close()
    return [{
        'ts': r[0].strftime('%Y-%m-%d %H:%M'),
        'all_data': r[1],
        'dogs': r[2],
        'downloaded': r[3]
    } for r in rows]


def bar_color(pct):
    """Pick a progress-bar color by completion percentage."""
    if pct >= 99:
        return '#3fb27f'
    if pct >= 70:
        return '#e8a645'
    return '#d8743a'


# ── pipeline status board ───────────────────────────────────────────────────
STATUS_FILE = os.path.join(OUT, 'regions_status.json')
STATS_FILE = os.path.join(OUT, 'board_stats.json')
MAP_FILE = os.path.join(OUT, 'map_points.json')
STAGES = [
    'pending', 'extract', 'coverage', 'backfill', 'complete', 'downloading',
    'downloaded'
]
STAGE_LABEL = {
    'pending': 'Queued',
    'extract': 'Extract',
    'coverage': 'Coverage',
    'backfill': 'Backfill missing',
    'complete': 'Awaiting download',
    'downloading': 'Downloading',
    'downloaded': 'Complete'
}
_status_lock = threading.Lock()


def derive_stage(m):
    """Best-guess pipeline stage for a region from its catalog metrics."""
    if m['all_data'] == 0 and m['dogs'] == 0:
        return 'pending'
    if m['dogs'] == 0:
        return 'downloaded'  # scanned, no ground animals -> nothing to do
    pct = m['downloaded'] / m['dogs'] * 100
    if pct >= 99:
        return 'downloaded'
    if pct >= 1:
        return 'downloading'
    return 'complete'


def load_status():
    """Read the persisted ``{region_key: stage}`` map (empty on any error)."""
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_status(d):
    """Atomically persist the status map."""
    os.makedirs(OUT, exist_ok=True)
    tmp = STATUS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(d, f, indent=2, sort_keys=True)
    os.replace(tmp, STATUS_FILE)


def write_board_stats(per, db):
    """Snapshot per-region stats (incl. which drives hold each region's data)
    for the board + command APIs, and seed stages for any new regions.

    Seeding only *adds* stages for regions not yet in the status file -- it
    never overwrites a stage the user set, so manual board edits always win.
    """
    rd = region_drives(db)
    stats = [{
        'key': p['key'],
        'name': p['region'],
        'pct': round(p['pct'], 1),
        'downloaded': p['downloaded'],
        'dogs': p['dogs'],
        'all_data': p['all_data'],
        'suggested': derive_stage(p),
        'src_dirs': rd.get(p['key'], {}).get('src', []),
        'img_dirs': rd.get(p['key'], {}).get('img', [])
    } for p in per]
    os.makedirs(OUT, exist_ok=True)
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)
    with _status_lock:
        cur = load_status()
        new = {p['key']: derive_stage(p) for p in per if p['key'] not in cur}
        if new:
            cur.update(new)
            save_status(cur)


def board_payload():
    """Merge the live stats snapshot with the user's saved stages."""
    try:
        with open(STATS_FILE) as f:
            stats = json.load(f)
    except (OSError, ValueError):
        stats = []
    status = load_status()
    for r in stats:
        r['stage'] = status.get(r['key'], r.get('suggested', 'pending'))
    return {'stages': STAGES, 'labels': STAGE_LABEL, 'regions': stats}


def set_stage(region, stage):
    """Persist one region's stage; returns True if valid and saved."""
    if stage not in STAGES or not region:
        return False
    with _status_lock:
        cur = load_status()
        cur[region] = stage
        save_status(cur)
    return True


def build_map_points(res_list=(0.5, 0.15)):
    """Bin every ground-animal point into density grids at several resolutions.

    Reads ``computed_geometry`` (GeoJSON Point) from the ground-animal parquets
    and aggregates each resolution to weighted cells; the browser renders them
    as geo-anchored raster rects and swaps to the finer grid on zoom. Paths
    come from the lock-free catalog snapshot, so this never contends with the
    live catalog DB.
    """
    snap = os.path.join(REPO, 'data', 'catalog.parquet')
    if not os.path.exists(snap):
        return
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")  # stay polite to the running jobs
    con.execute("INSTALL json; LOAD json;")
    # Per-point land filter: a few Mapillary sequences carry bad/interpolated
    # GPS that strings their images across open ocean (with spurious animal
    # detections), drawing fake lines on the map. Drop anything not on land --
    # tested per point, so coastal cities (whose 0.5° cell center may sit just
    # offshore) are kept.
    land_filter = ""
    try:
        import numpy as np
        import pyarrow as pa
        from global_land_mask import globe

        def _is_land(lon, lat):
            lo, la = np.asarray(lon), np.asarray(lat)
            out = np.zeros(len(lo), dtype=bool)
            ok = (np.isfinite(lo) & np.isfinite(la)
                  & (np.abs(lo) <= 180) & (np.abs(la) <= 90))
            if ok.any():
                out[ok] = globe.is_land(la[ok], lo[ok])
            return pa.array(out)

        con.create_function('is_land',
                            _is_land, ['DOUBLE', 'DOUBLE'],
                            'BOOLEAN',
                            type='arrow')
        land_filter = "AND is_land(lon, lat)"
    except Exception as e:
        print('map: land mask unavailable, keeping all points:', e)
    paths = [
        r[0] for r in con.execute(
            "SELECT path FROM read_parquet(?) WHERE kind='ground_animals'",
            [snap]).fetchall()
    ]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        con.close()
        return
    con.execute(
        f"""
      CREATE TEMP TABLE pts AS
      SELECT lon, lat FROM (
        SELECT TRY_CAST(json_extract(computed_geometry,'$.coordinates[0]') AS DOUBLE) lon,
               TRY_CAST(json_extract(computed_geometry,'$.coordinates[1]') AS DOUBLE) lat
        FROM read_parquet(?) WHERE computed_geometry IS NOT NULL)
      WHERE lon BETWEEN -180 AND 180 AND lat BETWEEN -90 AND 90 {land_filter}""",
        [paths])
    total = con.execute("SELECT count(*) FROM pts").fetchone()[0]
    levels = {}
    for res in res_list:
        rows = con.execute(f"""
          SELECT round(floor(lon/{res})*{res}+{res / 2}, 4) x,
                 round(floor(lat/{res})*{res}+{res / 2}, 4) y, count(*) n
          FROM pts GROUP BY 1, 2""").fetchall()
        pts = [[r[0], r[1], r[2]] for r in rows]
        levels[str(res)] = {
            'res': res,
            'max': max((p[2] for p in pts), default=0),
            'points': pts
        }
    con.close()
    out = {'total': total, 'levels': levels}
    os.makedirs(OUT, exist_ok=True)
    tmp = MAP_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, MAP_FILE)


# ── per-region command generator ────────────────────────────────────────────
# These name drives, so they are configuration. Unset, the generator still
# renders -- with the placeholder in place of the path, which is honest about
# what the operator has to supply and is copy-paste-obvious when it is wrong.
CMD_GRID_CSV = cfg('grid_csv', 'original_global_grid_5deg.csv')
CMD_WORK = cfg('work_dir', '<work-dir>')  # outputs
CMD_IMAGE_DIR = cfg('image_dir', '<image-dir>')  # dl target
CMD_PROXIES = cfg('proxies', 'proxies.txt')
# written by consolidate_data.py; repo-relative, so not configuration
CMD_DATA_CATALOGUE = os.path.join(REPO, 'data', 'data_root.txt')


def data_root():
    """The single consolidated DATA grid_runs dir, read live from the data-root
    catalogue written by tools/catalog/consolidate_data.py. Falls back to the
    work drive's grid_runs if no catalogue exists yet."""
    try:
        with open(CMD_DATA_CATALOGUE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return line.rstrip('/')
    except OSError:
        pass
    return CMD_WORK + '/grid_runs'


def _drive_of(path):
    """Mount-point basename for a path (matches the catalog's drive labels)."""
    try:
        p = os.path.realpath(path)
        dev = os.stat(p).st_dev
        while True:
            par = os.path.dirname(p)
            if par == p or os.stat(par).st_dev != dev:
                return os.path.basename(p) or None
            p = par
    except OSError:
        return None


def _root_by_drive():
    """{drive label: grid_runs root} from the gitignored catalog dirs file."""
    cfg = os.path.join(REPO, 'data', 'catalog_dirs.txt')
    out = {}
    try:
        with open(cfg) as f:
            roots = [
                ln.strip() for ln in f if ln.strip() and not ln.startswith('#')
            ]
    except OSError:
        return out
    for r in roots:
        d = _drive_of(r)
        if d:
            out[d] = r
    return out


def region_data_root(region):
    """The grid_runs root where THIS region's parquet data actually lives, found
    by scanning the filesystem (so it stays accurate even right after a move,
    when the catalog/data-root catalogue may be stale). Picks the drive holding
    the most of the region's all_data; falls back to data_root() if none found."""
    best, best_n = None, 0
    for root in _root_by_drive().values():
        n = len(
            glob.glob(
                os.path.join(root, f'{region}_*',
                             f'all_data_{region}_*.parquet')))
        if n > best_n:
            best, best_n = root, n
    return best or data_root()


def region_drives(db):
    """{region: {'src': [roots with data], 'img': [roots with images]}}."""
    rbd = _root_by_drive()
    try:
        con = duckdb.connect(db, read_only=True)
    except duckdb.Error:
        return {}
    data, img = {}, {}
    try:
        for reg, drv in con.execute(
                "SELECT DISTINCT region, drive FROM files").fetchall():
            data.setdefault(reg, set()).add(drv)
        for reg, drv in con.execute(
                "SELECT DISTINCT region, drive FROM images").fetchall():
            img.setdefault(reg, set()).add(drv)
    except duckdb.Error:
        pass
    con.close()
    out = {}
    for reg in set(data) | set(img):
        allds = data.get(reg, set()) | img.get(reg, set())
        out[reg] = {
            'src': sorted({rbd[d]
                           for d in allds if d in rbd}),
            'img': sorted({rbd[d]
                           for d in img.get(reg, set()) if d in rbd}),
        }
    return out


def region_locations(db):
    """Per region: which drives hold its parquet data, and which hold its jpgs.

    Rows are sorted by region name and each region's drives biggest-first, so
    "where does Europe actually live?" is answerable at a glance -- including
    the regions whose data and images sit on different drives.
    """
    try:
        con = duckdb.connect(db, read_only=True)
    except duckdb.Error:
        return []
    data, img = {}, {}
    try:
        for reg, drv, n, b in con.execute(
                "SELECT region, drive, count(*), coalesce(sum(size_bytes),0) "
                "FROM files GROUP BY 1,2 ORDER BY 4 DESC").fetchall():
            data.setdefault(reg, []).append((drv, n, b))
        for reg, drv, n, b in con.execute(
                "SELECT region, drive, coalesce(sum(n_images),0), "
                "coalesce(sum(bytes),0) FROM images GROUP BY 1,2 "
                "ORDER BY 3 DESC").fetchall():
            img.setdefault(reg, []).append((drv, n, b))
    except duckdb.Error:
        pass
    con.close()
    return [{
        'region': r.replace('_', ' '),
        'data': data.get(r, []),
        'img': img.get(r, [])
    } for r in sorted(set(data) | set(img))]


def render_locations(locs):
    """The 'where everything lives' table: data drives vs image drives."""

    def chips(items, unit, kind):
        """``kind`` ('data'/'img') tints the chip, so a drive's role is legible
        without reading the column header -- which matters because the same
        drive name can appear on both sides of a row."""
        if not items:
            return '<span class="lnone">none</span>'
        out = []
        for drv, n, b in items:
            size = f' · {hbytes(b)}' if b else ''
            lab = unit[:-1] if n == 1 else unit
            out.append(f'<span class="chip {kind}"><b>{drv}</b> '
                       f'{human(n)} {lab}{size}</span>')
        return ''.join(out)

    head = ('<div class="loc lh"><div>Region</div>'
            '<div><span class="swatch data"></span>Data — parquet</div>'
            '<div><span class="swatch img"></span>Images — jpgs</div></div>')
    return head + ''.join(
        f'<div class="loc"><div class="lname">{l["region"]}</div>'
        f'<div class="chips">{chips(l["data"], "files", "data")}</div>'
        f'<div class="chips">{chips(l["img"], "jpgs", "img")}</div></div>'
        for l in locs)


def build_commands(region, src_dirs, img_dirs):
    """The 5 ready-to-run pipeline commands for one region, drives filled in."""
    work_grid = region_data_root(region)  # per-region: where THIS region's
    work_root = os.path.dirname(work_grid)  # data actually lives (filesystem)
    # The audit's diff builds its "already have" set by scanning --dirs for
    # all_data_*.parquet, so the data drive ITSELF must be in --dirs. Leaving it
    # out makes the diff treat every already-extracted image on it as missing.
    src = [work_grid
           ] + [d for d in src_dirs if d not in (work_grid, CMD_IMAGE_DIR)]
    img = [d for d in img_dirs if d != CMD_IMAGE_DIR]
    dirs = ' \\\n         '.join(src)
    have = (' \\\n  --have-dir ' + ' '.join(img)) if img else ''
    extract = (
        f"python batch_chunks_mp_api.py {CMD_GRID_CSV} \\\n"
        f"  --parent-dir {work_grid} \\\n"
        f"  --image-dir {CMD_IMAGE_DIR}{have} \\\n"
        f"  --region {region} \\\n"
        f"  --token 1 --outer-max-workers 5 --search-max-workers 150 \\\n"
        f"  --entity-max-workers 520 --api-chunk-size 5000 \\\n"
        f"  --parquet-chunk-size 100000 --no-download-images --visualize")
    audit = (f"python coverage_audit.py audit {CMD_GRID_CSV} \\\n"
             f"  --dirs {dirs} \\\n"
             f'  --data-dir "{work_grid}" \\\n'
             f'  --missing-out "{work_root}/coverage_missing" \\\n'
             f'  --inscope-out "{work_root}/coverage_missing_inscope" \\\n'
             f"  --proxies {CMD_PROXIES} \\\n"
             f"  --wait \\\n"
             f"  --outer-workers 5 \\\n"
             f"  --region {region}")
    meta = (f"python backfill_missing.py \\\n"
            f"  --inscope {work_root}/coverage_missing_inscope \\\n"
            f"  --out-dir {work_grid} \\\n"
            f"  --no-download --entity-workers 256 --processes 2 \\\n"
            f"  --region {region}")
    dl = (f"python backfill_missing.py --download-only \\\n"
          f"  --out-dir {work_grid} \\\n"
          f"  --image-dir {CMD_IMAGE_DIR} \\\n"
          f"  --download-workers 10{have} \\\n"
          f"  --region {region}")
    consolidate = (f"python tools/catalog/consolidate_data.py \\\n"
                   f"  --dest {work_grid} \\\n"
                   f"  --region {region} \\\n"
                   f"  --on-conflict merge")
    return [extract, audit, meta, dl, consolidate]


def commands_payload(region, db):
    """Build a region's commands with drives read LIVE from the catalog.

    The region name is validated against the stats snapshot (which lists every
    region), then its drives are pulled straight from the catalog so data newly
    added to any drive is reflected on the next Generate. Falls back to the
    snapshot's stored drives only if the catalog is momentarily locked
    (mid-refresh).
    """
    try:
        with open(STATS_FILE) as f:
            stats = json.load(f)
    except (OSError, ValueError):
        stats = []
    by = {s['key']: s for s in stats}
    key = next((k for k in by if k.lower() == region.lower()), None)
    if not key:
        return None
    live = region_drives(db).get(key)  # live from the catalog
    if live is None:  # catalog busy / no data -> snapshot
        s = by[key]
        live = {'src': s.get('src_dirs', []), 'img': s.get('img_dirs', [])}
    return {
        'region': key,
        'commands': build_commands(key, live['src'], live['img'])
    }


# ── detection sweep status (§7.2) ───────────────────────────────────────────
# The sweep engine publishes $DETECT_ROOT/status.json atomically; we only ever
# read that file — never DuckDB — from a handler thread (§7.2: the catalog
# writer lock must not be contended by a status poll).
try:
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    from status import read_status as _read_detect_status
except Exception:  # detect tooling absent -> off
    _read_detect_status = None

_detect_lock = threading.Lock()
_detect_memo = {'t': 0.0, 'body': None}


def detect_payload():
    """Sweep status for /api/detect, memoized 2 s under a lock so N open
    tabs collapse to <=0.5 file reads/s (§7.2). Absent / stale (>120 s) /
    unparsable all degrade to {'running': False} — the client's single
    "sweep not running" state."""
    with _detect_lock:
        now = time.monotonic()
        if _detect_memo['body'] is not None and now - _detect_memo['t'] < 2.0:
            return _detect_memo['body']
        if _read_detect_status is None:
            body = {'running': False}
        else:
            try:
                body = _read_detect_status(stale_after=120.0)
            except Exception:
                body = {'running': False}
        _detect_memo.update(t=now, body=body)
        return body


# PreviewWriter filenames: <epoch_ms>_<image_id>_<conf*100, 3 digits>.jpg
RECENT_N = 100  # size of the pool the grid samples from
_CROP_RE = re.compile(r'^(\d{10,})_([A-Za-z0-9_-]{1,64})_(\d{3})\.jpg$')
CROP_WINDOW_S = 60.0  # keep the grid strictly "the last minute"
# The client trims this to a whole number of grid rows, so it must ask for
# MORE than it shows (a 8-wide grid wants 16) and slice off the remainder.
CROP_CAP = 24  # tiles offered; a random sample when more are eligible

# ── hard negatives (false-positive flags) ───────────────────────────────────
# recent_crops/ is a rolling ~200-file window the sweep prunes within minutes,
# so a flag CANNOT be a reference into it: we copy the pixels out at flag time
# and the jsonl record is fsync'd before the request returns. The image_id is
# recorded too, so even a flag that lost the race with the pruner can be
# re-cut from the original full-res image later.
HN_DIR = os.path.join(REPO, 'data', 'hard_negatives')
# Interpreter that runs the sweep -- a different env from the dashboard's
# (it needs torch/TensorRT/ultralytics). Falls back to the interpreter running
# this process, which is right for a single-env checkout and wrong-but-loud
# for a split one: the sweep fails on import rather than silently not starting.
SWEEP_PYTHON = cfg('sweep_python', sys.executable, env='SWEEP_PYTHON')
HN_CROPS = os.path.join(HN_DIR, 'crops')
HN_FULL = os.path.join(HN_DIR, 'full')
HN_LABELS = os.path.join(HN_DIR, 'labels.jsonl')
FLAG_LABEL = 'false_positive'
# Reviewers mine BOTH error directions. A false positive is a hard negative;
# a low-confidence detection that really is a dog is a hard POSITIVE, and for
# a gate tuned on recall those are the expensive ones to get wrong. They live
# in a parallel tree so harvest_flagged.py -- which reads hard_negatives and
# filters on label -- keeps working untouched.
HP_DIR = os.path.join(REPO, 'data', 'hard_positives')
POS_LABEL = 'true_positive'
FLAG_LABELS = (FLAG_LABEL, POS_LABEL)

_flag_lock = threading.Lock()
_flagged = {}  # label -> set of crop names; built once, then kept in memory


def _store_for(label):
    """Paths for one label, resolved from the CURRENT module globals.

    Deliberately not a constant dict built at import: HN_DIR/HN_LABELS are
    module-level knobs that tests (and any future config) rebind, and a
    snapshot taken at import silently keeps writing to the old location.
    """
    if label == POS_LABEL:
        return {'dir': HP_DIR,
                'crops': os.path.join(HP_DIR, 'crops'),
                'full': os.path.join(HP_DIR, 'full'),
                'labels': os.path.join(HP_DIR, 'labels.jsonl')}
    return {'dir': HN_DIR, 'crops': HN_CROPS, 'full': HN_FULL,
            'labels': HN_LABELS}


def _load_flags(path=None):
    """Crop names already in a ledger. Call under ``_flag_lock``.

    A torn final line (crash mid-append) is skipped rather than fatal — the
    rest of the ledger is still authoritative.
    """
    names = set()
    try:
        with open(path or HN_LABELS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get('crop'):
                    names.add(rec['crop'])
    except OSError:  # never flagged anything yet
        pass
    return names


def _flag_names(label=FLAG_LABEL):
    """The in-memory flag set for one label, loaded once. Under ``_flag_lock``.

    ``_flagged = None`` still means "forget everything and re-read", which is
    how tests force a reload from disk; it used to be the single set's own
    uninitialised sentinel.
    """
    global _flagged
    if not isinstance(_flagged, dict):
        _flagged = {}
    if label not in _flagged:
        _flagged[label] = _load_flags(_store_for(label)['labels'])
    return _flagged[label]


def _all_flagged_ids():
    """image_ids judged under ANY label. Under ``_flag_lock``.

    Either verdict removes a crop from the queue: the reviewer has decided,
    and showing it again is the exact waste this whole ledger exists to stop.
    """
    out = set()
    for lb in FLAG_LABELS:
        for nm in _flag_names(lb):
            m = _CROP_RE.match(nm)
            if m:
                out.add(m.group(2))
    return out


def _copy_out(src, dst):
    """Copy src->dst through a tmp file, so a partial copy is never visible.

    Returns False (not an exception) when src is already gone — that is the
    expected race with the writer's pruner, not an error.
    """
    tmp = dst + '.part'
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, 'rb') as r, open(tmp, 'wb') as w:
            shutil.copyfileobj(r, w)
            w.flush()
            os.fsync(w.fileno())
        os.replace(tmp, dst)
        return True
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def _rewrite_labels(drop, label=FLAG_LABEL):
    """Rewrite one ledger without ``drop``'s record, atomically."""
    st = _store_for(label)
    keep = []
    try:
        with open(st['labels']) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get('crop') == drop:
                    continue
                keep.append(s)
    except OSError:
        keep = []
    os.makedirs(st['dir'], exist_ok=True)
    tmp = st['labels'] + '.part'
    with open(tmp, 'w') as w:
        for s in keep:
            w.write(s + '\n')
        w.flush()
        os.fsync(w.fileno())
    os.replace(tmp, st['labels'])


def flag_crop(name, label=FLAG_LABEL, undo=False, now=None):
    """Record (or undo) one crop's verdict. Returns (body, code).

    ``label`` picks the store: false_positive -> hard_negatives,
    true_positive -> hard_positives. Idempotent in both directions: a second
    flag neither duplicates the jsonl line nor re-copies, and undoing
    something never flagged is a no-op success. Only a malformed name or an
    unknown label is a 4xx; every filesystem failure comes back 200 with
    ``ok:false`` so a cosmetic button never 500s.
    """
    name = name or ''
    m = _CROP_RE.match(name)
    if not m:
        return {'ok': False, 'error': 'malformed crop name'}, 400
    if label not in FLAG_LABELS:
        return {'ok': False, 'error': 'unknown label %r' % (label,)}, 400
    st = _store_for(label)
    try:
        with _flag_lock:
            names = _flag_names(label)
            if undo:
                if name in names:
                    _rewrite_labels(name, label)
                    names.discard(name)
                    for p in (os.path.join(st['crops'], name),
                              os.path.join(st['full'], name)):
                        try:
                            os.remove(p)
                        except OSError:  # never copied, or already gone
                            pass
                return {'ok': True, 'undone': True, 'label': label,
                        'total': len(names),
                        'flagged_total': len(_flag_names(FLAG_LABEL)),
                        'positive_total': len(_flag_names(POS_LABEL))}, 200
            if name in names:  # already flagged -> no second line, no re-copy
                return {'ok': True, 'copied': False, 'duplicate': True,
                        'label': label, 'total': len(names),
                        'flagged_total': len(_flag_names(FLAG_LABEL)),
                        'positive_total': len(_flag_names(POS_LABEL))}, 200
            # A crop cannot be both. Re-deciding replaces the old verdict
            # rather than filing the same image under both labels, which
            # would put one image in a dataset twice with opposite classes.
            other = POS_LABEL if label == FLAG_LABEL else FLAG_LABEL
            if name in _flag_names(other):
                _rewrite_labels(name, other)
                _flag_names(other).discard(name)
                ost = _store_for(other)
                for p in (os.path.join(ost['crops'], name),
                          os.path.join(ost['full'], name)):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            # the two copies are independent: the full frame can survive the
            # prune a beat longer than the crop, or vice versa
            got_crop = _copy_out(os.path.join(CROPS, name),
                                 os.path.join(st['crops'], name))
            got_full = _copy_out(os.path.join(CROPS, 'full', name),
                                 os.path.join(st['full'], name))
            rec = {'image_id': m.group(2),
                   'conf': round(int(m.group(3)) / 100.0, 2),
                   'ts': int(m.group(1)), 'crop': name, 'label': label,
                   'copied': bool(got_crop or got_full),
                   'flagged_at': int(time.time() if now is None else now)}
            os.makedirs(st['dir'], exist_ok=True)
            with open(st['labels'], 'a') as w:
                w.write(json.dumps(rec) + '\n')
                w.flush()
                os.fsync(w.fileno())  # a flag must survive a crash
            names.add(name)
            return {'ok': True, 'copied': bool(got_crop or got_full),
                    'label': label, 'total': len(names),
                    'flagged_total': len(_flag_names(FLAG_LABEL)),
                    'positive_total': len(_flag_names(POS_LABEL))}, 200
    except OSError as e:
        sys.stderr.write('flag_crop(%s,%s): %s\n' % (name, label, e))
        return {'ok': False, 'error': str(e)}, 200


def crops_payload(window_s=CROP_WINDOW_S, cap=CROP_CAP, now_ms=None):
    """Recent detection crops for /api/detect/crops.

    One ``os.listdir`` and no stat calls — the age comes out of the filename —
    so this stays cheap enough to serve from a handler thread. The writer keeps
    ~120 s of files; we show only the last ``window_s``, and when more than
    ``cap`` qualify we return a *random* sample so repeated calls (and the
    Shuffle button) surface different detections rather than the same newest N.

    A missing/unreadable directory is the normal pre-sweep state, not an error:
    it degrades to the empty payload the client renders as "no detections".

    ``has_full`` says whether the writer also left a full frame (with the box
    already drawn) in the ``full/`` subdir, i.e. whether the tile is worth
    making clickable. It comes from ONE extra listdir folded into a set — never
    a stat per crop — and an absent subdir just means "no lightboxes yet".
    """
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    try:
        names = os.listdir(CROPS)
    except OSError:  # absent, unreadable, not a dir
        return {'crops': [], 'total_last_min': 0}
    try:
        full = set(os.listdir(os.path.join(CROPS, 'full')))
    except OSError:  # writer predates full frames, or none written yet
        full = frozenset()
    # Pool = the newest RECENT_N detections, not a wall-clock window. On the
    # slow drives a 60 s window can legitimately contain nothing (the sweep
    # samples ~4 crops/s only from positives), so the grid sat empty; "last
    # N detections" always has something to show once the sweep has started.
    elig = []
    for name in names:
        m = _CROP_RE.match(name)
        if not m:
            continue
        ts = int(m.group(1))
        if ts > now_ms + 5000:  # clock skew guard
            continue
        elig.append({
            'name': name,
            'image_id': m.group(2),
            'ts': ts,
            'conf': round(int(m.group(3)) / 100.0, 2),
            'age_s': max(0, int((now_ms - ts) / 1000)),
            'has_full': name in full
        })
    # newest RECENT_N form the pool, then a random sample of `cap` from it
    elig.sort(key=lambda c: -c['ts'])
    pool = elig[:RECENT_N]
    # `flagged` seeds the client's Set so a flag survives the 60 s refresh and
    # Shuffle: it covers the whole pool, not just this sample, because the next
    # shuffle draws from the pool.
    with _flag_lock:
        fl = _flag_names(FLAG_LABEL)
        total = len(fl)
        seen = sorted({c['name'] for c in pool} & fl)
    return {
        'crops': random.sample(pool, cap) if len(pool) > cap else pool,
        'total_last_min': len(pool),
        'pool_n': RECENT_N,
        'flagged': seen,
        'flagged_total': total
    }


_build_lock = threading.Lock()
_refresh = {'running': False, 'last': None, 'error': None}


def _do_build(args):
    """Run one build under the global lock (shared by the timer and button)."""
    with _build_lock:
        try:
            build(args)
            _refresh.update(last=datetime.now().strftime('%H:%M:%S'),
                            error=None)
        except Exception as e:
            _refresh['error'] = str(e)
            print('build error:', e)


def trigger_refresh(db):
    """Kick off a full refresh (catalog + images) in the background.

    Returns False if a refresh is already in progress.
    """
    if _refresh['running']:
        return False
    _refresh['running'] = True

    def work():
        try:
            _do_build(argparse.Namespace(db=db, no_refresh=False, images=True))
        finally:
            _refresh['running'] = False

    threading.Thread(target=work, daemon=True).start()
    return True




REVIEW_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Not a dog — detection review</title><style>
:root{--bg:#13151a;--panel:#1b2027;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;--red:#d8743a;
/* the negative VERDICT only. --red stays rust for everything else; at rust
   this one button read as orange and did not register as the "no". */
--no:#ef5350}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,
/* bottom padding clears the fixed undo toast (~80px tall incl. its offset) so
   the pager can always be scrolled above it, at any width */
"Segoe UI",Roboto,sans-serif;line-height:1.5;padding:0 24px 130px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1560px;margin:0 auto}

/* ── header: identity left, the two numbers that matter right ── */
header{margin:0 -24px 16px;padding:14px 24px 0;position:sticky;top:0;z-index:20;
background:rgba(19,21,26,.93);backdrop-filter:saturate(140%) blur(12px);
border-bottom:1px solid var(--bd)}
.hrow{display:flex;flex-wrap:wrap;align-items:center;gap:16px 20px}
h1{font-size:17px;font-weight:640;letter-spacing:-.2px;display:flex;align-items:center;gap:9px}
h1 .fl{color:var(--red)}
.back{color:var(--mut);text-decoration:none;font-size:12.5px;border:1px solid var(--bd);
border-radius:8px;padding:4px 10px;transition:color .12s,border-color .12s}
.back:hover{color:var(--acc);border-color:rgba(232,166,69,.35)}
.score{margin-left:auto;display:flex;align-items:baseline;gap:8px}
.score b{font-size:26px;font-weight:660;letter-spacing:-.6px;font-variant-numeric:tabular-nums}
.score b.sec{font-size:18px;color:var(--mut);font-weight:620}
.score b.pos{color:var(--green)}
.score b.dup{color:var(--dim)}
.score span{font-size:12px;color:var(--dim)}
.score .sep{width:1px;height:20px;background:var(--bd);margin:0 4px}

/* ── toolbar ── */
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:12px 0 13px}
.rbtn{background:rgba(232,166,69,.13);border:1px solid rgba(232,166,69,.34);
color:var(--acc);border-radius:8px;padding:5px 12px;font-size:12.5px;font-weight:600;
cursor:pointer;font-family:inherit;font-variant-numeric:tabular-nums;transition:background .12s}
.rbtn:hover{background:rgba(232,166,69,.23)}
.rbtn[disabled]{opacity:.35;cursor:default;background:rgba(130,140,150,.08);
border-color:var(--bd);color:var(--dim)}
.rbtn:focus-visible,select:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.quiet{background:rgba(130,140,150,.09);border-color:var(--bd);color:var(--mut)}
.quiet:hover{background:rgba(130,140,150,.16);color:var(--tx)}
/* recessive until hover: it is used rarely and undoes a lot of work */
.danger{background:transparent;border-color:var(--bd);color:var(--dim)}
.danger:hover{background:rgba(216,116,58,.16);border-color:rgba(216,116,58,.5);
color:#e8894f}
.danger:focus-visible{outline-color:var(--red)}
select{background:var(--panel2);border:1px solid var(--bd);color:var(--tx);
border-radius:8px;padding:5px 9px;font-size:12.5px;font-family:inherit;cursor:pointer}
.sp{flex:1}
.cnt{color:var(--mut);font-size:12.5px;font-variant-numeric:tabular-nums}
.hint{color:var(--dim);font-size:11.5px;padding-bottom:14px;display:flex;
flex-wrap:wrap;gap:6px 14px;align-items:center}
/* ── balance strip: one bar, two lines, no card chrome ── */
.bal{display:flex;align-items:center;gap:13px;padding:11px 0 13px;
border-top:1px solid var(--bd)}
.balbar{position:relative;flex:1;min-width:120px;height:6px;border-radius:4px;
background:rgba(130,140,150,.16);overflow:hidden}
/* negatives already in the dataset */
.balbar i{display:block;height:100%;width:0;background:var(--red);
transition:width .45s ease}
/* negatives earned by flags not yet folded into a rebuild -- lighter, so
   "banked but not built" never reads as "already in the training set" */
.balbar b{position:absolute;top:0;height:100%;width:0;
background:repeating-linear-gradient(90deg,rgba(216,116,58,.55) 0 3px,
transparent 3px 6px);transition:width .45s ease,left .45s ease}
.baltx{font-size:12px;color:var(--mut);white-space:nowrap;
font-variant-numeric:tabular-nums}
.baltx b{color:var(--tx);font-weight:650}
.balsub{display:block;font-size:11px;color:var(--dim)}
.ballg{display:flex;flex-wrap:wrap;gap:3px 13px;font-size:11px;color:var(--dim);
margin-top:4px}
.ballg span{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.ballg i{width:10px;height:8px;border-radius:2px;flex:none}
.ballg .s1{background:var(--red)}
/* identical gradient to .balbar b, or the legend would be a lie */
.ballg .s2{background:repeating-linear-gradient(90deg,rgba(216,116,58,.55) 0 3px,
transparent 3px 6px);box-shadow:inset 0 0 0 1px rgba(216,116,58,.3)}
.ballg .s3{background:rgba(130,140,150,.16)}
/* s4 draws no bar segment -- the reserved crops are outside the training set
   entirely, so an outline says "accounted for, not counted here" */
.ballg .s4{background:transparent;box-shadow:inset 0 0 0 1px rgba(130,140,150,.5)}
.bal.ok .balbar i{background:var(--green)}
@media(max-width:700px){.bal{flex-wrap:wrap}.baltx{white-space:normal}}
kbd{background:var(--panel2);border:1px solid var(--bd);border-bottom-width:2px;
border-radius:5px;padding:0 5px;font:600 10.5px/17px ui-monospace,SFMono-Regular,Menlo,monospace;
color:var(--mut);display:inline-block;min-width:17px;text-align:center}

/* ── the grid: tiles are evidence, so nothing is cropped ── */
/* overflow-anchor:none -- flagging removes one tile and appends another, which
   leaves the document height identical, but Chrome's scroll anchoring still
   compensated by a full row (scrollY 1800 -> 1511, measured) and the grid
   appeared to jump down every single flag. Nothing above the viewport actually
   moves, so there is nothing to anchor to. */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(186px,1fr));gap:13px;
overflow-anchor:none}
.card{overflow-anchor:none}
.card{position:relative;background:var(--panel);border:1px solid var(--bd);
border-radius:11px;overflow:hidden;transition:opacity .16s ease,transform .16s ease,
border-color .12s}
.card:hover{border-color:rgba(130,140,150,.3)}
.card.sel{border-color:var(--acc)}
.card.go{opacity:0;transform:scale(.93)}
/* confidence meter: how sure the detector was, as a bar under the crop.
   Horizontal, because length reads as magnitude at a glance -- a vertical rail
   on the card edge reads as a border and collides with the selected state.
   A confident wrong call is a better hard negative than a hesitant one, so
   this plus the sort control is how the valuable mistakes get found. */
.rail{height:2px;background:rgba(130,140,150,.14)}
.rail i{display:block;height:100%;background:var(--acc);opacity:.75}
.thumb{width:100%;aspect-ratio:1;object-fit:contain;display:block;background:#0c0e11}
.thumb.zoom{cursor:zoom-in}
.meta{display:flex;justify-content:space-between;gap:6px;padding:6px 9px;
font-size:11px;color:var(--dim);font-variant-numeric:tabular-nums}
.meta .id{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.meta .cf{color:var(--mut);font-weight:600;flex:none}
/* two verdicts, side by side and equal weight -- neither is the default, and
   a hairline keeps them from reading as one wide button */
.acts{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--bd)}
.fbtn{border:0;background:rgba(130,140,150,.05);color:var(--mut);padding:8px 4px;
font-size:11.5px;cursor:pointer;font-family:inherit;font-weight:600;
transition:background .12s,color .12s;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.fbtn+.fbtn{border-left:1px solid var(--bd)}
/* only hover arms a button. Tinting one for the merely-selected tile read as
   "this one is already flagged". */
.fbtn.no:hover{background:rgba(239,83,80,.2);color:#f0736a}
.fbtn.yes:hover{background:rgba(67,181,129,.2);color:#5ec89a}
.fbtn.no:focus-visible{outline:2px solid var(--no);outline-offset:-2px}
.fbtn.yes:focus-visible{outline:2px solid var(--green);outline-offset:-2px}
@media(max-width:420px){.fbtn{font-size:11px;padding:8px 2px}}

/* ── states ── */
.state{padding:56px 0;text-align:center;color:var(--dim);font-size:13.5px}
.state b{display:block;color:var(--tx);font-size:15px;font-weight:620;margin-bottom:5px}
.state .rbtn{margin-top:14px}
.sk{background:var(--panel);border:1px solid var(--bd);border-radius:11px;
aspect-ratio:.78;animation:bre 1.5s ease-in-out infinite}
@keyframes bre{0%,100%{opacity:.5}50%{opacity:.85}}
.foot{display:flex;justify-content:center;align-items:center;gap:10px;padding:22px 0 0}

/* ── undo: shows WHAT you discarded, and how long you have ──
   Anchored bottom-RIGHT, not bottom-centre: the pager is centred, and a
   centred toast lands straight on top of Prev/Next when you are scrolled to
   the end of a page -- exactly when you are about to use them. */
.toast{position:fixed;right:24px;bottom:24px;
background:#20252c;border:1px solid rgba(130,140,150,.24);border-radius:12px;
padding:9px 12px;font-size:12.5px;display:flex;gap:11px;align-items:center;z-index:90;
box-shadow:0 10px 34px rgba(0,0,0,.55);overflow:hidden}
.toast img{width:34px;height:34px;object-fit:cover;border-radius:7px;background:#000;flex:none}
.toast .tt{color:var(--mut);white-space:nowrap}
.toast .tt b{display:block;color:var(--tx);font-weight:620}
.tbar{position:absolute;left:0;bottom:0;height:2px;background:var(--red);width:100%}
.tbar.run{width:0;transition:width 5s linear}

/* ── lightbox ── */
.lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:flex;
align-items:center;justify-content:center;flex-direction:column;gap:13px;z-index:100;padding:24px}
.lbw{position:relative;overflow:auto;max-width:93vw;max-height:72vh;
border-radius:10px;border:1px solid var(--bd);background:#000;line-height:0;
overscroll-behavior:contain}
.lbw img{display:block;max-width:none;max-height:none;border:0;border-radius:0;
image-rendering:auto}
/* the box lives in the same scaled space as the image */
.lbw .bx{cursor:move}
.bx{position:absolute;border:1px solid var(--acc);cursor:move;
box-shadow:0 0 0 1px rgba(0,0,0,.8),inset 0 0 0 1px rgba(0,0,0,.55)}
.bx[hidden]{display:none}
/* Handles are HOLLOW and sit OUTSIDE the border, so on a 30px object they
   frame the thing instead of covering it. Their hit area is padded well
   beyond the visible dot -- easy to grab, nothing hidden. */
.hd{position:absolute;width:9px;height:9px;background:rgba(19,21,26,.55);
border:2px solid var(--acc);border-radius:2px;box-shadow:0 0 0 1px rgba(0,0,0,.8)}
.hd::after{content:'';position:absolute;inset:-7px}
.h-nw{left:-6px;top:-6px;cursor:nwse-resize}
.h-n {left:50%;top:-6px;margin-left:-6px;cursor:ns-resize}
.h-ne{right:-6px;top:-6px;cursor:nesw-resize}
.h-e {right:-6px;top:50%;margin-top:-6px;cursor:ew-resize}
.h-se{right:-6px;bottom:-6px;cursor:nwse-resize}
.h-s {left:50%;bottom:-6px;margin-left:-6px;cursor:ns-resize}
.h-sw{left:-6px;bottom:-6px;cursor:nesw-resize}
.h-w {left:-6px;top:50%;margin-top:-6px;cursor:ew-resize}
/* a tight box has no room for edge handles between the corners */
.bx.tiny .h-n,.bx.tiny .h-s,.bx.tiny .h-e,.bx.tiny .h-w{display:none}
/* crosshair through the centre: on a small object the box edges alone give
   the eye nothing to align against */
.bx::before,.bx::after{content:'';position:absolute;background:rgba(232,166,69,.35);
pointer-events:none}
.bx::before{left:50%;top:-9px;bottom:-9px;width:1px}
.bx::after{top:50%;left:-9px;right:-9px;height:1px}
/* autosave status: a word, not a button -- there is nothing to press */
.bsave{font-size:11.5px;color:var(--dim);min-width:74px}
.bsave.on{color:var(--green)}
.bsave.err{color:var(--red)}
.zoomb{display:flex;align-items:center;gap:6px}
.zoomb .rbtn{padding:3px 9px;font-size:11.5px}
.zlv{font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums;
min-width:44px;text-align:center}
.bxy{font-variant-numeric:tabular-nums;color:var(--dim)}
.lbf{display:flex;gap:13px;align-items:center;flex-wrap:wrap;justify-content:center}
.lbx{position:absolute;top:16px;right:18px}
.lbyes{background:rgba(67,181,129,.14);border-color:rgba(67,181,129,.4);color:var(--green)}
.lbyes:hover{background:rgba(67,181,129,.24)}
@media(max-width:560px){.score{margin-left:0;width:100%}.grid{grid-template-columns:repeat(auto-fill,minmax(140px,1fr))}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style></head><body><div class="wrap">

<header>
  <div class="hrow">
    <h1><span class="fl">&#9873;</span> Not a dog <span class="cnt">&middot; detection review</span></h1>
    <a class="back" href="/">&larr; dashboard</a>
    <!-- Two counts, no ratio between them: "left" is the live retained pool
         (crops age out after 24 h / 3000), "flagged" is cumulative all-time.
         A bar dividing one by the other would be a made-up percentage. -->
    <div class="score">
      <b id="left">&mdash;</b><span id="leftlab">left to review</span>
      <span class="sep"></span>
      <b class="sec" id="done">&mdash;</b><span>flagged</span>
      <b class="sec pos" id="pos">&mdash;</b><span>marked dog</span>
      <b class="sec" id="seen">&mdash;</b><span>kept</span>
      <b class="sec dup" id="dups">&mdash;</b><span>repeats hidden</span>
    </div>
  </div>
  <div class="bar">
    <select id="sort" title="which crops to surface first">
      <option value="low" selected>Least confident first</option>
      <option value="conf">Most confident first</option>
      <option value="new">Newest first</option>
    </select>
    <!-- Populated from /api/review, which lists only countries the sweep has
         actually produced crops for, with counts. Rebuilt hourly alongside the
         dashboard refresh, so newly swept ground appears on its own. -->
    <select id="country" title="only review crops from one country">
      <option value="">All countries</option>
    </select>
    <select id="size"><option value="50">50 per page</option>
      <option value="100">100 per page</option></select>
    <button class="rbtn quiet" id="reload" title="pull in detections found since this page loaded">&#8635; Refresh pool</button>
    <!-- destructive, and deliberately NOT beside Prev/Next: a mis-click there
         would throw away every keep decision made so far -->
    <button class="rbtn danger" id="unkeep" title="put every crop you already judged a dog back into the queue">&#8630; Restore kept</button>
    <span class="sp"></span>
    <span class="cnt" id="pg">&mdash;</span>
    <button class="rbtn quiet" id="next" title="bank this screen and bring up the next unjudged crops">Next &rsaquo;</button>
  </div>
</header>

<!-- Training-set balance. The queue tells you what is left to look at; this
     tells you what the looking is FOR, and when to stop. -->
<div class="bal" id="bal" hidden>
  <div class="balbar"><i id="balFill"></i><b id="balPend"></b></div>
  <div class="baltx">
    <span id="balMain">&mdash;</span>
    <!-- the bar has three fills and nothing said which was which; swatches
         here carry the exact same backgrounds so the mapping is readable
         without hovering or guessing -->
    <span class="ballg" id="balLg"></span>
  </div>
</div>

<div class="hint">
  <span>Flag what is <b>not</b> a dog, and mark the low-confidence ones that <b>are</b>. Moving to another page passes on the rest, so nothing you have judged comes back.</span>
  <span><kbd>&larr;</kbd><kbd>&rarr;</kbd><kbd>&uarr;</kbd><kbd>&darr;</kbd> move</span>
  <span><kbd>F</kbd> not a dog</span>
  <span><kbd>D</kbd> is a dog</span>
  <span><kbd>&#9166;</kbd> full frame &amp; edit box</span>
  <span><kbd>&#8679;</kbd>+arrows nudge box &middot; saves itself</span>
  <span><kbd>U</kbd> undo</span>
  <span>The bar under each crop is detector confidence.</span>
  <span>One crop per camera pass &mdash; repeat frames of the same animal are hidden.</span>
</div>

<div class="grid" id="grid"></div>
<div id="state"></div>
<div class="foot" id="foot" hidden>
  <span class="cnt" id="pg2"></span>
  <button class="rbtn quiet" id="next2" title="bank this screen and bring up the next unjudged crops">Next &rsaquo;</button>
</div>
</div>
<script>
/* sel = -1 means NOTHING is selected. The page opens that way on purpose:
   a pre-selected first tile looks like a choice the user did not make. The
   first arrow press picks tile 0 and keyboard flow takes over from there. */
var page=0,size=50,sort='low',country='',countryName='',items=[],reserve=[],pages=1,sel=-1,
    smallN=0,minPx=0,
    todoN=0,flaggedN=0,posN=0,seenN=0,dupN=0,session=0,lastUndo=null,toastT=null,lb=null,busy={};
var SOFT=!window.matchMedia||
         !window.matchMedia('(prefers-reduced-motion:reduce)').matches;
function $(i){return document.getElementById(i)}
function esc(t){var d=document.createElement('div');d.textContent=t;return d.innerHTML}
/* esc() leaves quotes alone; anything landing in a title="" needs att() */
function att(s){return esc(s).replace(/"/g,'&quot;')}
function n(v){return (v||0).toLocaleString()}

/* ── load ───────────────────────────────────────────────────────────────── */
function skeleton(){
  var g=$('grid');g.innerHTML='';
  for(var i=0;i<Math.min(size,18);i++){
    var d=document.createElement('div');d.className='sk';g.appendChild(d);
  }
  $('state').innerHTML='';
}
function toTop(){
  try{
    window.scrollTo({top:0,left:0,behavior:SOFT?'smooth':'auto'});
  }catch(_){
    window.scrollTo(0,0);          /* older engines: no options object */
  }
}
function load(){
  skeleton();
  /* returns the promise: callers (and the test harness) can await a settled
     grid instead of guessing at microtask depth */
  return fetch('/api/review?page='+page+'&size='+size+'&sort='+sort+
               '&country='+encodeURIComponent(country))
  .then(function(r){if(!r.ok)throw 0;return r.json()})
  .then(function(j){
    if(j.error)throw 0;
    items=j.items||[];reserve=j.reserve||[];page=j.page||0;pages=j.pages||1;
    todoN=j.total_unflagged||0;flaggedN=j.flagged_total||0;
    smallN=j.too_small||0;minPx=j.min_px||0;
    if(j.seen_total!=null)seenN=j.seen_total;
    if(j.positive_total!=null)posN=j.positive_total;
    if(j.collapsed!=null)dupN=j.collapsed;
    paintCountries(j.countries,j.country);
    score();
    /* "Page 3 of 47" described an offset that no longer moves. What the
       reader actually needs is how much is left after this screen. */
    var more=Math.max(0,todoN-items.length);
    var lab=items.length?(n(items.length)+' shown \u00b7 '+n(more)+' left'):
      'nothing left to review';
    /* Held-back crops are stated, not silently dropped -- and the threshold
       is named so the number can be argued with. */
    if(smallN)lab+=' \u00b7 '+n(smallN)+' too small to judge (under '+minPx+'px)';
    $('pg').textContent=lab;$('pg2').textContent=lab;
    $('next').disabled=$('next2').disabled=!more;
    $('foot').hidden=!items.length;
    if(sel>=items.length)sel=items.length-1;   /* -1 when the page is empty */
    render();
    /* New page, new content: start at the top. Every caller of load() is a
       "show me different crops" event (page step, sort, size, refresh), and
       landing mid-grid means the first rows are judged without being seen.
       mark() cannot fight this -- a fresh page has no selection. */
    toTop();
  })
  .catch(function(){
    $('grid').innerHTML='';$('foot').hidden=true;
    $('state').innerHTML='<div class="state"><b>Could not reach the dashboard</b>'+
      'The review queue is served by the dashboard process. Check that it is '+
      'still running, then try again.'+
      '<div><button class="rbtn" id="retry">Try again</button></div></div>';
    $('retry').onclick=load;
  });
}
/* the two header numbers, from state -- never re-parsed back out of the DOM,
   so a flag and a reload can never disagree */
function score(){
  $('left').textContent=n(todoN);
  /* 'left to review' and 'repeats hidden' are scoped to the active country
     filter; 'flagged', 'marked dog' and 'kept' are all-time global. Sitting
     side by side with no marker, 198 next to 1,166 reads as "198 left in
     total". Say which country the 198 belongs to. */
  var lb=$('leftlab');
  if(lb)lb.textContent=countryName?('left in '+countryName):'left to review';
  $('done').textContent=n(flaggedN);
  var s=$('seen');if(s)s.textContent=n(seenN);
  var pz=$('pos');if(pz)pz.textContent=n(posN);
  var dz=$('dups');if(dz){dz.textContent=n(dupN);
    dz.parentNode.title=n(dupN)+' crops hidden because another frame from the '
      +'same Mapillary sequence is already in the queue -- same animal, same '
      +'pass, one decision';}
}

/* ── training-set balance ─────────────────────────────────────────────────
   Flagging only ever produces NEGATIVES, so while not_dog < dog the answer is
   always "keep flagging" -- annotating more dogs would move the target away.
   Counts come from the built dataset; flags made since that build are shown
   separately as "banked", because they only become training data at the next
   rebuild. Updated locally on every flag, re-fetched on load and page change. */
var BAL=null;
function loadBal(){
  return fetch('/api/dataset').then(function(r){return r.json()})
    /* an error payload must still be PAINTED -- bailing here left the
       "Dataset not found" branch unreachable and the strip showing stale
       numbers for a dataset that no longer exists */
    .then(function(j){if(!j)return;BAL=j;paintBal()})
    .catch(function(){});
}
function paintBal(){
  var b=BAL;if(!b)return;
  var el=$('bal');el.hidden=false;
  if(b.ok===false){
    el.className='bal';
    $('balFill').style.width='0%';$('balPend').style.width='0%';
    $('balMain').innerHTML='<b>Dataset not found</b>';
    $('balSub')&&($('balSub').textContent='');
    $('balLg').textContent=b.error||('missing '+(b.dataset||'dataset'));
    return;
  }
  /* the server's measured value, never a copy: this line used to carry its
     own 0.822 and drifted 1.8x out of date when the acceptance reservation
     started withholding 30% of every harvest. 0 is the honest fallback -- it
     paints "nothing banked yet" instead of inventing progress. */
  var y=(typeof b.yield_per_flag==='number')?b.yield_per_flag:0;
  var pend=Math.round((b.new_flags||0)*y);
  var pendPos=Math.round((b.new_positive_flags||0)*y);
  var have=b.not_dog||0, want=(b.dog||0)+pendPos;   /* positives raise the bar */
  var got=Math.min(want,have+pend);
  var pctHave=want?Math.min(100,100*have/want):0;
  var pctGot =want?Math.min(100,100*got /want):0;
  $('balFill').style.width=pctHave.toFixed(1)+'%';
  var pb=$('balPend');
  pb.style.left=pctHave.toFixed(1)+'%';
  pb.style.width=Math.max(0,pctGot-pctHave).toFixed(1)+'%';
  var short=Math.max(0,want-have-pend);
  var need=short?Math.ceil(short/y):0;
  el.className='bal'+(need?'':' ok');
  var ds=b.dataset||'the dataset';
  if(!need){
    $('balMain').innerHTML='<b>Balanced.</b> '+n(have+pend)+' not-dog vs '+
      n(want)+' dog';
  }else{
    $('balMain').innerHTML='<b>'+n(need)+'</b> more to flag for a balanced set';
  }
  /* one legend entry per fill, each naming the number it draws */
  var togo=Math.max(0,want-have-pend);
  var L=[['s1',n(have),'in '+ds],
         ['s2',n(pend),'banked from '+n(b.new_flags||0)+' new flag'+
                       ((b.new_flags||0)===1?'':'s')+', not built yet']];
  if(togo)L.push(['s3',n(togo),'still to find'+
    (pendPos?' (target +'+n(pendPos)+' from crops you marked as dogs)':'')]);
  else L.push(['s3','0','still to find']);
  /* Name the reservation. Roughly a third of what you flag is withheld as the
     acceptance set and never trains, so the target is far higher than "one
     flag, one crop" -- without saying so the panel just looks pessimistic. */
  if(b.reserved_ids)L.push(['s4',n(b.reserved_ids),
    'reserved to test the gate, never trained on']);
  $('balLg').innerHTML=L.map(function(x){
    return '<span><i class="'+x[0]+'"></i><b>'+x[1]+'</b> '+esc(x[2])+'</span>';
  }).join('');
  $('balFill').title=n(have)+' not-dog crops already in '+ds;
  $('balPend').title=n(pend)+' negatives earned from flags since that build';
}
/* a verdict immediately banks yield_per_flag of a crop; reflect it without a
   round trip. Negatives close the gap, positives widen it. */
function bumpBal(dNeg,dPos){
  if(!BAL)return;
  BAL.new_flags=Math.max(0,(BAL.new_flags||0)+(dNeg||0));
  BAL.new_positive_flags=Math.max(0,(BAL.new_positive_flags||0)+(dPos||0));
  paintBal();
}

/* ── render ─────────────────────────────────────────────────────────────── */
function tile(c){
  var d=document.createElement('div');
  d.className='card';d.dataset.name=c.name;
  var pc=Math.round(Math.max(0,Math.min(1,+c.conf||0))*100);
  /* HQ cut from the ORIGINAL. The preview thumbnails are cut from the 1280
     letterbox and capped at 160px, which throws away 3.6-5.3x the pixels
     actually available -- at that size a distant dog and a distant goat are
     the same picture. Fall back to the preview if the HQ cut 404s (the crop's
     shard may not be committed yet, or the cache is still warming).
     Every crop is clickable: has_full only says a burned-in preview frame was
     saved, and the editor reads the original either way. */
  d.innerHTML='<img class="thumb zoom" loading="lazy" alt="detection crop" '+
      'src="/hq?name='+encodeURIComponent(c.name)+'" '+
      "onerror=\"this.onerror=null;this.src='/recent_crops/"+
      encodeURIComponent(c.name)+"'\">"+
    '<div class="rail"><i style="width:'+pc+'%"></i></div>'+
    '<div class="meta"><span class="id" title="'+att(c.image_id)+'">'+esc(c.image_id)+
      '</span><span class="cf">'+(+c.conf||0).toFixed(2)+'</span></div>'+
    '<div class="acts">'+
      '<button class="fbtn no" type="button" title="false positive (F)">'+
        '&#9873; Not a dog</button>'+
      '<button class="fbtn yes" type="button" title="a real dog the detector '+
        'was unsure about (D)">&#10003; Is a dog</button>'+
    '</div>';
  var im=d.querySelector('.thumb');
  im.onclick=function(){openLb(idx(c.name))};
  d.querySelector('.fbtn.no').onclick=function(e){
    e.stopPropagation();flag(idx(c.name),false,'false_positive')};
  d.querySelector('.fbtn.yes').onclick=function(e){
    e.stopPropagation();flag(idx(c.name),false,'true_positive')};
  /* pressing the flag button must NOT select the tile: the tile is about to
     be removed, and selecting it means the highlight lands on whatever slides
     into that index -- an auto-advance nobody asked for */
  d.onmousedown=function(e){
    if(e&&e.target&&e.target.closest&&e.target.closest('.acts'))return;
    sel=idx(c.name);mark();
  };
  return d;
}
function render(){
  var g=$('grid');g.innerHTML='';
  if(!items.length){
    $('state').innerHTML='<div class="state"><b>Queue is clear</b>'+
      'Every detection in the pool has been judged. New crops appear here as the '+
      'sweep finds them.<div><button class="rbtn" id="rl2">&#8635; Check for more</button></div></div>';
    $('rl2').onclick=load;return;
  }
  $('state').innerHTML='';
  var f=document.createDocumentFragment();
  items.forEach(function(c){f.appendChild(tile(c))});
  g.appendChild(f);
  mark();
  prefetchNext();
}
/* Warm the NEXT page's crops once THIS page has finished loading -- never
   before, or the two compete for the same connections and the page you are
   looking at gets slower to make the one you are not faster.
   `reserve` is already in the payload (it is what backfills the grid as you
   flag), and /hq answers with max-age=86400, so a warmed crop is served from
   cache when the page turns. */
var prefetchGen=0,prefetched={};
function prefetchNext(){
  /* Wrapped whole: this is an optimisation, and an optimisation that can
     throw takes down the render it was meant to speed up. It already did --
     with no Image/navigator in the environment, render() aborted after
     building the grid and the page showed nothing. */
  try{ prefetchNext_(); }catch(e){}
}
function prefetchNext_(){
  if(typeof Image==='undefined'||typeof document==='undefined') return;
  var gen=++prefetchGen;
  var conn=(typeof navigator!=='undefined'&&navigator.connection)||{};
  if(conn.saveData) return;              /* the user asked not to spend data */
  function warm(){
    if(gen!==prefetchGen||document.hidden) return;
    var queue=(reserve||[]).map(function(c){return c.name})
      .filter(function(n){return n&&!prefetched[n]});
    if(!queue.length) return;
    var at=0;
    /* four workers, each pulling the next name when its own image settles */
    function pump(){
      if(gen!==prefetchGen||at>=queue.length) return;
      var name=queue[at++];
      prefetched[name]=1;
      var im=new Image();
      /* onerror counts as settled: a crop with no full-res file must not
         stall the queue behind it */
      im.onload=im.onerror=pump;
      im.src='/hq?name='+encodeURIComponent(name);
    }
    for(var k=0;k<4;k++) pump();
  }
  var imgs=[].slice.call(document.querySelectorAll('#grid img'));
  var pending=imgs.filter(function(im){return !im.complete});
  if(!pending.length){ warm(); return; }
  var left=pending.length;
  pending.forEach(function(im){
    var fin=function(){ if(--left<=0) warm(); };
    im.addEventListener('load',fin,{once:true});
    im.addEventListener('error',fin,{once:true});
  });
}
function idx(name){
  for(var i=0;i<items.length;i++)if(items[i].name===name)return i;
  return -1;
}
function cardAt(i){
  var c=items[i];if(!c)return null;
  return document.querySelector('.card[data-name="'+CSS.escape(c.name)+'"]');
}
function mark(){
  var cards=document.querySelectorAll('.card');
  for(var i=0;i<cards.length;i++)cards[i].classList.toggle('sel',i===sel);
  if(sel<0)return;                 /* nothing selected: mark nothing, scroll nowhere */
  var e=cards[sel];
  if(e&&e.scrollIntoView)
    e.scrollIntoView({block:'nearest',behavior:SOFT?'smooth':'auto'});
}
function cols(){
  var g=$('grid');
  var t=getComputedStyle(g).gridTemplateColumns;
  return Math.max(1,(t||'').split(' ').filter(Boolean).length);
}

/* ── flag ───────────────────────────────────────────────────────────────── */
/* viaKey: the flag came from the F key, so the user is working through the
   grid by keyboard and the selection must stay put (the next crop slides into
   this index, which is exactly what they want to judge next). A MOUSE flag
   clears the selection instead -- advancing a highlight the user never asked
   for is just the grid moving on its own. */
function flag(i,viaKey,label){
  var c=items[i];if(!c||busy[c.name])return;
  label=label||'false_positive';
  busy[c.name]=1;
  var card=cardAt(i);
  if(card)card.classList.add('go');
  return fetch('/api/detect/flag',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:c.name,label:label})})
   .then(function(r){return r.json()}).then(function(j){
      delete busy[c.name];
      if(!j||j.ok===false){if(card)card.classList.remove('go');return}
      /* Surgical removal + backfill: the rest of the grid does not re-render,
         so nothing reflows under the cursor and no image reloads. */
      items.splice(i,1);
      if(card&&card.parentNode)card.parentNode.removeChild(card);
      var nx=reserve.shift();
      if(nx){items.push(nx);$('grid').appendChild(tile(nx))}
      session++;
      todoN=Math.max(0,todoN-1);
      if(label==='true_positive')posN++;else flaggedN++;
      score();bumpBal(label==='true_positive'?0:1,label==='true_positive'?1:0);
      if(!viaKey)sel=-1;                         /* mouse: no auto-advance */
      if(sel>=items.length)sel=items.length-1;   /* stays -1 if unset */
      if(!items.length)render();else mark();
      showUndo(c,i,!!nx,label);
   }).catch(function(){delete busy[c.name];if(card)card.classList.remove('go')});
}

/* ── undo ───────────────────────────────────────────────────────────────── */
function showUndo(c,at,pulled,label){
  /* `pulled` records whether the flag consumed a crop from `reserve` to keep
     the grid full; undo has to hand that one back or the page grows on every
     flag/undo cycle */
  lastUndo={crop:c,at:at,pulled:pulled,label:label||'false_positive'};
  var t=$('tbox');
  if(!t){t=document.createElement('div');t.className='toast';t.id='tbox';
    document.body.appendChild(t)}
  t.innerHTML='<img src="/recent_crops/'+encodeURIComponent(c.name)+'" alt="">'+
    '<span class="tt"><b>'+(label==='true_positive'?'Marked as a dog'
      :'Flagged as not a dog')+'</b>'+n(session)+
    (session===1?' crop':' crops')+' this session</span>'+
    '<button class="rbtn quiet" id="undoB">Undo</button><i class="tbar"></i>';
  $('undoB').onclick=undo;
  var bar=t.querySelector('.tbar');
  /* next frame, so the transition actually runs from 100% -> 0 */
  requestAnimationFrame(function(){requestAnimationFrame(function(){
    bar.className='tbar run'})});
  clearTimeout(toastT);
  /* 5 s of no interaction and the flag stands (it is already saved on disk) */
  toastT=setTimeout(function(){lastUndo=null;hideToast()},5000);
}
function undo(){
  var u=lastUndo;if(!u)return;
  lastUndo=null;clearTimeout(toastT);hideToast();
  return fetch('/api/detect/flag',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:u.crop.name,label:u.label,undo:true})})
   .then(function(r){return r.json()}).then(function(j){
      if(!j||j.ok===false)return;
      var g=$('grid');
      /* give the backfill back FIRST, then re-insert -- doing it in this
         order there is no edge case where the popped tile is the one we are
         about to restore */
      if(u.pulled&&items.length){
        reserve.unshift(items.pop());
        var lastEl=g.children[g.children.length-1];
        if(lastEl)g.removeChild(lastEl);
      }
      /* back where it was, not at the front: the eye is still on that spot */
      var at=Math.min(u.at,items.length);
      if(!items.length)$('state').innerHTML='';
      items.splice(at,0,u.crop);
      g.insertBefore(tile(u.crop),g.children[at]||null);
      session=Math.max(0,session-1);
      if(sel>=0)sel=at;   /* don't invent a selection for a mouse user */
      todoN++;
      if(u.label==='true_positive')posN=Math.max(0,posN-1);
      else flaggedN=Math.max(0,flaggedN-1);
      score();bumpBal(u.label==='true_positive'?0:-1,
                      u.label==='true_positive'?-1:0);
      mark();
   });
}
function hideToast(){var t=$('tbox');if(t)t.remove()}

/* ── lightbox ───────────────────────────────────────────────────────────── */
function openLb(i){
  var c=items[i];if(!c)return;
  sel=i;mark();
  if(!lb){
    lb=document.createElement('div');lb.className='lb';
    lb.innerHTML='<button class="rbtn quiet lbx" id="lbx" aria-label="close">&#10005;</button>'+
      '<div class="lbw" id="lbw">'+
        '<img id="lbi" alt="source image with the detection box">'+
        '<div class="bx" id="lbbox" hidden>'+
          ['nw','n','ne','e','se','s','sw','w'].map(function(h){
            return '<i class="hd h-'+h+'" data-h="'+h+'"></i>'}).join('')+
        '</div>'+
      '</div>'+
      '<div class="lbf">'+
        '<span class="cnt" id="lbc"></span>'+
        '<span class="cnt bxy" id="lbxy"></span>'+
        '<span class="zoomb">'+
          '<button class="rbtn quiet" id="zout" title="zoom out">&minus;</button>'+
          '<span class="zlv" id="zlv">100%</span>'+
          '<button class="rbtn quiet" id="zin" title="zoom in">+</button>'+
          '<button class="rbtn quiet" id="zbox" title="fill the view with the box">Fit box</button>'+
          '<button class="rbtn quiet" id="zfit" title="show the whole image">Fit image</button>'+
        '</span>'+
        '<span class="bsave" id="lbstat"></span>'+
        '<button class="rbtn quiet" id="lbrst">Reset box</button>'+
        '<button class="rbtn" id="lbf">&#9873; Not a dog</button>'+
        '<button class="rbtn lbyes" id="lby">&#10003; Is a dog</button></div>';
    lb.onclick=function(e){if(e.target===lb)closeLb()};
    document.body.appendChild(lb);
    document.body.style.overflow='hidden';
    $('lbx').onclick=closeLb;
    /* the edit must be durable before the verdict is filed, or the crop is
       classified against a box that never reached disk */
    $('lbf').onclick=function(){var k=sel;
      flushSave().then(function(){closeLb();if(k>=0)flag(k,false,'false_positive')})};
    $('lby').onclick=function(){var k=sel;
      flushSave().then(function(){closeLb();if(k>=0)flag(k,false,'true_positive')})};
    $('lbrst').onclick=function(){
      if(origBox){curBox=origBox.slice();paintBox();dirty(true)}};
    $('zin').onclick=function(){zoomBy(1.5)};
    $('zout').onclick=function(){zoomBy(1/1.5)};
    $('zbox').onclick=function(){fitBox()};
    $('zfit').onclick=function(){fitImage()};
    /* A fixed step per wheel EVENT is a mouse-wheel assumption. One notch is
       one event with deltaY ~100, so 1.15x felt right; a trackpad sends a
       stream of events with deltaY ~3, so the same step compounded to 1.15^30
       and the image shot to 60x from one flick. Scale the step by how far the
       wheel actually turned, normalised across deltaMode, so both devices
       cover the same zoom for the same physical gesture. */
    $('lbw').addEventListener('wheel',function(e){
      /* not gated on curBox: ~9.5% of the live pool has no stored geometry
         yet (shard uncommitted), and those are exactly the crops worth
         zooming into. A box is needed to FIT to, never to zoom. */
      if(!$('lbi').naturalWidth)return;
      e.preventDefault();                       /* the view zooms, not the page */
      var d=e.deltaY;
      if(e.deltaMode===1)d*=16;                 /* lines -> px */
      else if(e.deltaMode===2)d*=($('lbw').clientHeight||600);   /* pages */
      /* a trackpad PINCH arrives as a ctrl-wheel with small deltas, and wants
         to move faster per unit than a two-finger scroll */
      var f=Math.exp(-d*(e.ctrlKey?0.010:0.0015));
      zoomBy(Math.max(0.5,Math.min(2,f)),e);    /* never more than 2x per event */
    },{passive:false});

    $('lbi').onload=function(){if(curBox)fitBox();else fitImage()};
    window.addEventListener('resize',function(){applyZoom()});
    initDrag();
    initPan();
    $('lbx').focus();
  }
  curBox=origBox=null;boxMeta=null;zoom=1;
  $('lbbox').hidden=true;$('lbxy').textContent='';
  saveT=null;savingP=null;setStat('');
  $('lbc').textContent=c.image_id+' · confidence '+(+c.conf||0).toFixed(2);
  /* the burned-in full frame cannot be edited: it is cut from the 1280
     letterbox with the box already drawn on it. Editing needs the ORIGINAL
     plus the store's coordinates, so fall back to the old frame only when
     the original cannot be resolved. */
  $('lbi').src='/orig?name='+encodeURIComponent(c.name);
  fetch('/api/review/box?name='+encodeURIComponent(c.name))
   .then(function(r){return r.json()}).then(function(j){
      if(!lb||sel!==i)return;                    /* stepped away meanwhile */
      if(!j||!j.ok||!j.boxes||!j.boxes.length||!j.has_file){
        $('lbi').src='/recent_crops/full/'+encodeURIComponent(c.name);
        $('lbxy').textContent=(j&&j.error)?j.error:'box not editable';
        return;
      }
      /* pick the detection this crop came from: the filename carries its
         confidence, which is what harvest_flagged matches on too */
      var want=+c.conf||0,pick=j.boxes[0];
      for(var k=0;k<j.boxes.length;k++)
        if(Math.abs(j.boxes[k].conf-want)<=0.006){pick=j.boxes[k];break}
      boxMeta={name:c.name,det_idx:pick.det_idx,w:j.w,h:j.h};
      origBox=[pick.x1,pick.y1,pick.x2,pick.y2];
      var sv=j.saved;
      curBox=sv?[sv.x1,sv.y1,sv.x2,sv.y2]:origBox.slice();
      $('lbbox').hidden=false;
      fitBox();                       /* start where the work is */
      if(sv)setStat('Saved ✓','on');
   }).catch(function(){});
}

/* ── box editing ─────────────────────────────────────────────────────────
   curBox is always ORIGINAL-image pixels; the overlay is derived from it via
   the rendered scale. Storing display coords instead would silently rescale
   the saved box every time the window changed size. */
var curBox=null,origBox=null,boxMeta=null,drag=null,zoom=1;
function imgScale(){return zoom}
/* The image is rendered at an EXPLICIT width (natural * zoom) rather than
   being fitted by CSS. That keeps one number as the whole coordinate story,
   and it means the handles stay a constant SCREEN size however far you zoom
   in -- which is the entire point when the object is 30px wide. */
function applyZoom(){
  var im=$('lbi');
  if(!im||!im.naturalWidth)return;
  zoom=Math.max(0.02,Math.min(zoom,40));
  im.style.width=Math.round(im.naturalWidth*zoom)+'px';
  im.style.height=Math.round(im.naturalHeight*zoom)+'px';
  var z=$('zlv');if(z)z.textContent=Math.round(zoom*100)+'%';
  var w=$('lbw');if(w&&w.__cursor)w.__cursor();
  paintBox();
}
function centreOn(cx,cy){
  var w=$('lbw');if(!w)return;
  w.scrollLeft=cx*zoom-w.clientWidth/2;
  w.scrollTop =cy*zoom-w.clientHeight/2;
}
function zoomBy(f,ev){
  var w=$('lbw'),im=$('lbi');
  if(!w||!im||!im.naturalWidth)return;
  /* keep the point under the cursor (or the view centre) pinned */
  var r=w.getBoundingClientRect();
  var px=ev?ev.clientX-r.left:w.clientWidth/2;
  var py=ev?ev.clientY-r.top :w.clientHeight/2;
  var ix=(w.scrollLeft+px)/zoom, iy=(w.scrollTop+py)/zoom;
  /* bounded: unclamped, a fast gesture could leave the image at 6000% with
     nothing on screen to say where you were */
  zoom=Math.max(0.02,Math.min(16,zoom*f));
  applyZoom();
  w.scrollLeft=ix*zoom-px;w.scrollTop=iy*zoom-py;
}
function fitImage(){
  var w=$('lbw'),im=$('lbi');
  if(!w||!im||!im.naturalWidth)return;
  zoom=Math.min(w.clientWidth/im.naturalWidth,w.clientHeight/im.naturalHeight);
  applyZoom();w.scrollLeft=0;w.scrollTop=0;
}
/* Open zoomed so the BOX fills ~45% of the view. A 30px dog fitted to a 4000px
   frame is four screen pixels; no amount of handle tuning makes that editable,
   so the fix is to start where the work actually is. */
function fitBox(){
  var w=$('lbw'),im=$('lbi');
  if(!w||!im||!im.naturalWidth||!curBox)return;
  var bw=Math.max(4,curBox[2]-curBox[0]),bh=Math.max(4,curBox[3]-curBox[1]);
  zoom=Math.min(w.clientWidth*0.45/bw,w.clientHeight*0.45/bh);
  zoom=Math.max(zoom,Math.min(w.clientWidth/im.naturalWidth,
                              w.clientHeight/im.naturalHeight));
  applyZoom();
  centreOn((curBox[0]+curBox[2])/2,(curBox[1]+curBox[3])/2);
}
function paintBox(){
  var im=$('lbi'),bx=$('lbbox');
  if(!im||!bx||!curBox)return;
  bx.style.left=(curBox[0]*zoom)+'px';
  bx.style.top=(curBox[1]*zoom)+'px';
  var w=Math.max(1,(curBox[2]-curBox[0])*zoom),h=Math.max(1,(curBox[3]-curBox[1])*zoom);
  bx.style.width=w+'px';bx.style.height=h+'px';
  /* under ~46px on screen the edge handles would overlap the corners */
  bx.classList.toggle('tiny',w<46||h<46);
  $('lbxy').textContent=Math.round(curBox[0])+','+Math.round(curBox[1])+
    ' → '+Math.round(curBox[2])+','+Math.round(curBox[3])+
    '  ('+Math.round(curBox[2]-curBox[0])+'×'+
    Math.round(curBox[3]-curBox[1])+' px)';
}
function setStat(t,cls){
  var e=$('lbstat');if(!e)return;
  e.textContent=t;e.className='bsave'+(cls?' '+cls:'');
}
/* Autosave. Every edit is a correction the user meant to make; a Save button
   only adds a step they can forget, and a forgotten click files the verdict
   against the detector's box instead of theirs. Debounced, because a drag
   fires a stream of changes and a nudge repeats. */
var saveT=null,savingP=null;
function dirty(on){
  if(!on)return;
  setStat('Saving…');
  if(saveT)clearTimeout(saveT);
  saveT=setTimeout(function(){saveT=null;doSave()},400);
}
/* Commit anything still pending and resolve when it is durable. Called before
   a verdict, before stepping away and before closing: the corrected box must
   reach disk before the crop leaves the screen. */
function flushSave(){
  if(saveT){clearTimeout(saveT);saveT=null;return doSave()}
  return savingP||Promise.resolve();
}
function doSave(){
  if(!curBox||!boxMeta)return Promise.resolve();
  savingP=fetch('/api/review/box',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:boxMeta.name,det_idx:boxMeta.det_idx,
                         box:curBox.slice()})})
   .then(function(r){return r.json()}).then(function(j){
      setStat(j&&j.ok?'Saved ✓':'Not saved',j&&j.ok?'on':'err');
   }).catch(function(){setStat('Not saved','err')});
  return savingP;
}
function clampBox(){
  var w=$('lbi').naturalWidth||boxMeta.w,h=$('lbi').naturalHeight||boxMeta.h;
  curBox[0]=Math.max(0,Math.min(curBox[0],w));
  curBox[2]=Math.max(0,Math.min(curBox[2],w));
  curBox[1]=Math.max(0,Math.min(curBox[1],h));
  curBox[3]=Math.max(0,Math.min(curBox[3],h));
  if(curBox[2]<curBox[0])curBox=[curBox[2],curBox[1],curBox[0],curBox[3]];
  if(curBox[3]<curBox[1])curBox=[curBox[0],curBox[3],curBox[2],curBox[1]];
}
/* Drag anywhere on the image to pan it. Zoom already resized the <img>
   inside a scrolling wrapper, so panning is just scrollLeft/scrollTop -- but
   reaching for a scrollbar to inspect a 40-pixel animal is the wrong gesture
   for the job. The box overlay stops propagation on mousedown, so editing a
   box still wins over panning wherever the two overlap. */
function initPan(){
  var w=$('lbw'), im=$('lbi');
  if(!w||w.__pan)return;
  w.__pan=1;
  im.draggable=false;
  im.addEventListener('dragstart',function(e){e.preventDefault()});
  var p=null;
  function canPan(){
    return w.scrollWidth>w.clientWidth+1||w.scrollHeight>w.clientHeight+1;
  }
  function cursor(){ w.style.cursor=canPan()?(p?'grabbing':'grab'):''; }
  w.addEventListener('pointerdown',function(e){
    /* left button only, and never when there is nothing to pan to */
    if(e.button!==0||!canPan())return;
    /* The box overlay stops propagation on MOUSEDOWN, which does nothing to
       a pointerdown -- they are separate streams and pointer fires first. So
       this has to opt out by target, or setPointerCapture steals the drag
       and a box resize silently pans the image instead. */
    if(e.target&&e.target.closest&&e.target.closest('#lbbox'))return;
    p={x:e.clientX,y:e.clientY,l:w.scrollLeft,t:w.scrollTop,moved:0};
    try{w.setPointerCapture(e.pointerId)}catch(_){}
    cursor(); e.preventDefault();
  });
  w.addEventListener('pointermove',function(e){
    if(!p)return;
    var dx=e.clientX-p.x, dy=e.clientY-p.y;
    p.moved=Math.max(p.moved,Math.abs(dx)+Math.abs(dy));
    w.scrollLeft=p.l-dx; w.scrollTop=p.t-dy;
  });
  function end(e){
    if(!p)return;
    /* a pan that ends over the backdrop must not read as a click and close
       the lightbox; swallow the click this gesture is about to produce */
    if(p.moved>4)w.__swallow=1;
    p=null;
    try{w.releasePointerCapture(e.pointerId)}catch(_){}
    cursor();
  }
  w.addEventListener('pointerup',end);
  w.addEventListener('pointercancel',end);
  w.addEventListener('click',function(e){
    if(w.__swallow){w.__swallow=0;e.stopPropagation();e.preventDefault();}
  },true);
  w.addEventListener('mouseenter',cursor);
  w.__cursor=cursor;
}
function initDrag(){
  var bx=$('lbbox');
  bx.addEventListener('mousedown',function(e){
    if(!curBox)return;
    e.preventDefault();e.stopPropagation();
    var h=e.target&&e.target.getAttribute&&e.target.getAttribute('data-h');
    drag={h:h||'move',x:e.clientX,y:e.clientY,box:curBox.slice()};
  });
  document.addEventListener('mousemove',function(e){
    if(!drag||!curBox)return;
    var s=imgScale();if(!s)return;
    var dx=(e.clientX-drag.x)/s,dy=(e.clientY-drag.y)/s,b=drag.box;
    if(drag.h==='move')curBox=[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy];
    else{
      curBox=b.slice();
      if(drag.h.indexOf('w')>=0)curBox[0]=b[0]+dx;
      if(drag.h.indexOf('e')>=0)curBox[2]=b[2]+dx;
      if(drag.h.indexOf('n')>=0)curBox[1]=b[1]+dy;
      if(drag.h.indexOf('s')>=0)curBox[3]=b[3]+dy;
    }
    clampBox();paintBox();dirty(true);
  });
  document.addEventListener('mouseup',function(){drag=null});
}
/* named entry point kept for the tests and for flushSave() */
function saveBox(){return flushSave()}
/* every crop opens now -- the editor reads the ORIGINAL jpg, which exists
   regardless of whether the writer also saved a burned-in preview frame */
function stepLb(d){
  var k=sel+d;
  if(k<0||k>=items.length)return Promise.resolve();
  return flushSave().then(function(){openLb(k)});
}
function closeLb(){
  if(!lb)return;
  flushSave();                 /* fire-and-forget: the POST outlives the DOM */
  lb.remove();lb=null;document.body.style.overflow='';
}

/* ── keyboard ───────────────────────────────────────────────────────────── */
document.addEventListener('keydown',function(e){
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  if(e.key==='Escape'){if(lb)closeLb();else hideToast();return}
  var tag=(e.target&&e.target.tagName)||'';
  if(tag==='SELECT'||tag==='INPUT')return;
  if(lb){
    /* Shift+Arrow = one-pixel nudge. Sub-pixel accuracy by mouse is not a
       thing on a 30px object, so the keyboard has to be able to finish the
       job the drag started. */
    if(e.shiftKey&&curBox&&e.key.indexOf('Arrow')===0){
      var dx=(e.key==='ArrowRight')-(e.key==='ArrowLeft');
      var dy=(e.key==='ArrowDown')-(e.key==='ArrowUp');
      curBox=[curBox[0]+dx,curBox[1]+dy,curBox[2]+dx,curBox[3]+dy];
      clampBox();paintBox();dirty(true);e.preventDefault();return;
    }
    if(e.key==='ArrowRight'){stepLb(1);e.preventDefault()}
    else if(e.key==='ArrowLeft'){stepLb(-1);e.preventDefault()}
    else if(e.key==='f'||e.key==='F'){var k=sel;
      flushSave().then(function(){closeLb();if(k>=0)flag(k,true,'false_positive')});
      e.preventDefault()}
    else if(e.key==='d'||e.key==='D'){var k2=sel;
      flushSave().then(function(){closeLb();if(k2>=0)flag(k2,true,'true_positive')});
      e.preventDefault()}
    return;
  }
  var c=cols(),moved=true;
  /* from "nothing selected", ANY arrow lands on the first tile */
  if(e.key==='ArrowRight'||e.key==='ArrowLeft'||
     e.key==='ArrowDown'||e.key==='ArrowUp'){
    if(sel<0){sel=items.length?0:-1;mark();e.preventDefault();return}
  }
  if(e.key==='ArrowRight')sel=Math.min(items.length-1,sel+1);
  else if(e.key==='ArrowLeft')sel=Math.max(0,sel-1);
  else if(e.key==='ArrowDown')sel=Math.min(items.length-1,sel+c);
  else if(e.key==='ArrowUp')sel=Math.max(0,sel-c);
  else moved=false;
  if(moved){mark();e.preventDefault();return}
  if((e.key==='f'||e.key==='F')&&sel>=0){
    flag(sel,true,'false_positive');e.preventDefault()}
  else if((e.key==='d'||e.key==='D')&&sel>=0){
    flag(sel,true,'true_positive');e.preventDefault()}
  else if(e.key==='u'||e.key==='U'){undo();e.preventDefault()}
  else if(e.key==='Enter'&&sel>=0){openLb(sel);e.preventDefault()}
});

/* ── "I looked at these" ──────────────────────────────────────────────────
   Flagging records only the NEGATIVE decision, so without this every crop
   judged "yes, a dog" stays in the pool and the same dogs reopen on every
   visit. Paging away IS the positive decision: whatever is still on screen
   when you leave was looked at and kept. Recorded by image_id, so a cell twin
   of the same photo cannot reappear later either. */
function markSeen(){
  var names=items.map(function(c){return c.name});
  if(!names.length)return Promise.resolve();
  return fetch('/api/review/seen',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({names:names})})
   .then(function(r){return r.json()})
   .then(function(j){if(j&&j.seen_total!=null)seenN=j.seen_total})
   .catch(function(){});
}
/* leaving the tab counts too -- sendBeacon survives unload where fetch does not */
window.addEventListener('pagehide',function(){
  if(!items.length||!navigator.sendBeacon)return;
  try{
    navigator.sendBeacon('/api/review/seen',
      new Blob([JSON.stringify({names:items.map(function(c){return c.name})})],
               {type:'application/json'}));
  }catch(_){}
});

/* ── controls ───────────────────────────────────────────────────────────── */
/* every navigation banks the current screen first, then loads the next one */
function nav(fn){return function(){markSeen().then(fn)}}
/* THE QUEUE IS CONSUMED FROM THE HEAD, NOT PAGED THROUGH.
   nav() banks every crop on screen first (markSeen removes them from the
   queue), so the queue has ALREADY advanced by a screenful by the time the
   next load runs. Adding an offset on top of that skipped a second screenful
   every turn: measured, page 2 shared 4 of 50 crops with the page-1 reserve,
   and 43 of the other 46 were still sitting at the head of the queue
   afterwards, never reviewed. Staying at offset 0 shows exactly the crops
   that banking uncovered -- which is also the block the prefetcher warmed. */
function go(d){return nav(function(){page=0;sel=-1;load()})}
$('next').onclick=$('next2').onclick=go(1);
$('reload').onclick=nav(function(){page=0;sel=-1;load()});
/* Restore kept: undoes every "this is a dog" decision. Names the real count
   in the prompt -- "are you sure?" over an unknown quantity is not consent.
   Flags are a separate ledger and are explicitly left alone. */
$('unkeep').onclick=function(){
  if(!seenN){window.alert('Nothing to restore \u2014 no crops have been kept yet.');return}
  if(!window.confirm('Put '+n(seenN)+' kept crop'+(seenN===1?'':'s')+
      ' back into the review queue?\n\nThese are the ones you already judged '+
      'to be dogs. You will be shown them again.\n\nYour '+n(flaggedN)+
      ' flagged false positives are NOT affected.'))return;
  var b=$('unkeep');b.disabled=true;b.textContent='Restoring\u2026';
  fetch('/api/review/seen',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({reset:true})})
   .then(function(r){return r.json()}).then(function(j){
      b.disabled=false;b.innerHTML='\u21ba Restore kept';
      if(!j||j.ok===false){window.alert('Could not restore: '+
        ((j&&j.error)||'unknown error'));return}
      seenN=0;page=0;sel=-1;load();loadBal();
   }).catch(function(){b.disabled=false;b.innerHTML='\u21ba Restore kept';});
};
$('sort').onchange=function(){var v=this.value;markSeen().then(function(){
  sort=v;page=0;sel=-1;load()})};
$('size').onchange=function(){var v=parseInt(this.value,10)||50;
  markSeen().then(function(){size=v;page=0;sel=-1;load()})};
/* Rebuilt from every response so the hourly refresh reaches an open tab, but
   only when the option set actually CHANGED -- rewriting the <select> on each
   page turn would drop the open dropdown and reset the caret mid-click. */
var countrySig='';
function paintCountries(list,cur){
  if(!list)return;
  var sig=list.map(function(c){return c.iso+':'+c.n}).join(',');
  countryName='';
  for(var q=0;q<list.length;q++)if(list[q].iso===cur)countryName=list[q].name;
  if(sig!==countrySig){
    countrySig=sig;
    var el=$('country');
    var html='<option value="">All countries</option>';
    for(var i=0;i<list.length;i++){
      var c=list[i];
      html+='<option value="'+att(c.iso)+'">'+esc(c.name)+' ('+n(c.n)+')</option>';
    }
    el.innerHTML=html;
  }
  if(cur!=null)$('country').value=cur;
}
$('country').onchange=function(){var v=this.value;
  markSeen().then(function(){country=v;page=0;sel=-1;load()})};
load();loadBal();
</script></body></html>"""

# ── bulk review page (/review) + sweep control ──────────────────────────────

REVIEW_PAGE = 50

# ── near-duplicate collapse ─────────────────────────────────────────────────
# Mapillary ships images in SEQUENCES: consecutive frames from one camera pass,
# seconds apart, same animal, same street. Measured on a live 3,012-crop pool:
# 884 crops (29%) were a repeat of a sequence already in the queue, one
# sequence contributing 54 frames. Judging those one at a time is the same
# decision over and over.
#
# Perceptual hashing is not enough on its own -- at Hamming <= 6 it caught 59
# of those 884, because the camera moves between frames. The sequence id is
# the ground truth for "same pass", so it leads and dHash is the safety net
# for crops whose sequence cannot be resolved.
#
# The lookup costs ~41 s over the ground_animals parquets, far too slow for a
# request, so it is cached to disk and refreshed on a background thread.
SEQ_CACHE = os.path.join(OUT, 'sequence_cache.json')
# A Mapillary "sequence" is a whole recording session, not one animal. Measured
# on the live pool: 41% of the sequences being collapsed spanned more than a
# minute and one spanned 87 minutes, so collapsing a whole sequence was hiding
# genuinely different dogs -- 692 crops hidden where only 451 are repeats.
# Frames of the SAME animal on one pass are seconds apart, so only collapse
# within this window.
SEQ_WINDOW_S = 30.0
_seq_lock = threading.Lock()
_seq = None            # image_id -> sequence
_seq_busy = False
_dhash_cache = {}      # crop name -> 64-bit dHash
DHASH_MAX = 8000       # bounded: the sweep writes ~4 crops/s for days


def _epoch(v):
    """captured_at -> unix seconds. Mapillary ships ms ints or ISO strings."""
    if v is None:
        return None
    try:
        return float(v) / 1000.0
    except (TypeError, ValueError):
        pass
    try:
        import datetime as _dt
        return _dt.datetime.fromisoformat(
            str(v).replace('Z', '+00:00')).timestamp()
    except Exception:
        return None


def _seq_of(entry):
    """(sequence, captured_at) from a cache entry, tolerating the old format
    where the value was a bare sequence string."""
    if isinstance(entry, list) and entry:
        return entry[0], (entry[1] if len(entry) > 1 else None)
    if isinstance(entry, str) and entry:
        return entry, None
    return '', None


def _seq_map():
    """image_id -> sequence, loaded once from disk. Under ``_seq_lock``."""
    global _seq
    if _seq is None:
        try:
            with open(SEQ_CACHE) as f:
                _seq = json.load(f)
        except (OSError, ValueError):
            _seq = {}
    return _seq


def _resolve_sequences(ids, db):
    """Fill the cache for ``ids`` from the ground_animals parquets."""
    if not ids:
        return 0
    con = duckdb.connect(db, read_only=True)
    try:
        paths = [r[0] for r in con.execute(
            "SELECT path FROM files WHERE path LIKE '%ground_animals%'"
        ).fetchall()]
        if not paths:
            return 0
        con.execute('CREATE TEMP TABLE want(image_id VARCHAR)')
        con.executemany('INSERT INTO want VALUES (?)', [(i,) for i in ids])
        src = ('read_parquet([' +
               ','.join("'" + p.replace("'", "''") + "'" for p in paths) + '])')
        rows = con.execute(
            'SELECT CAST(p.image_id AS VARCHAR), CAST(p.sequence AS VARCHAR), '
            'CAST(p.captured_at AS VARCHAR) '
            f'FROM {src} p JOIN want w ON CAST(p.image_id AS VARCHAR)='
            'w.image_id').fetchall()
    finally:
        con.close()
    with _seq_lock:
        m = _seq_map()
        for i, sq, cap in rows:
            if sq:
                m[str(i)] = [str(sq), _epoch(cap)]
        # ids with no row are recorded as '' so they are not re-queried forever
        for i in ids:
            m.setdefault(str(i), '')
        # Prune. The sweep writes ~4 crops/s, so an unbounded map would reach
        # ~2.4M entries (~145 MB of JSON, re-serialised on every resolve and
        # held in RAM) over the rest of the run. Only ids still in the rolling
        # crop pool, or already judged, can ever be consulted again.
        try:
            live = {mm.group(2) for mm in
                    (_CROP_RE.match(n) for n in os.listdir(CROPS)) if mm}
        except OSError:
            live = set()
        with _flag_lock:
            live |= _all_flagged_ids()
        with _seen_lock:
            live |= set(_seen_ids())
        if live:
            for k in [k for k in m if k not in live]:
                del m[k]
        try:
            os.makedirs(os.path.dirname(SEQ_CACHE), exist_ok=True)
            tmp = SEQ_CACHE + '.part'
            with open(tmp, 'w') as f:
                json.dump(m, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, SEQ_CACHE)
        except OSError:
            pass
    return len(rows)


def warm_sequences(db, ids):
    """Kick off a background resolve for ids we have never looked up.

    Never blocks the request: until it lands those crops simply stay visible,
    which is the safe direction to fail -- showing a crop twice costs a click,
    hiding one loses a judgement.
    """
    global _seq_busy
    with _seq_lock:
        known = set(_seq_map())
        missing = [i for i in ids if i not in known]
        if not missing or _seq_busy:
            return 0
        _seq_busy = True

    def work():
        global _seq_busy
        try:
            _resolve_sequences(missing, db)
        except Exception as e:
            sys.stderr.write('sequence resolve failed: %s\n' % e)
        finally:
            _seq_busy = False

    threading.Thread(target=work, daemon=True).start()
    return len(missing)


def _dhash(path, name):
    """64-bit difference hash, cached by crop name (crops are immutable)."""
    if name in _dhash_cache:
        return _dhash_cache[name]
    try:
        from PIL import Image
        im = Image.open(path).convert('L').resize((9, 8), Image.BILINEAR)
        px = list(im.getdata())
        bits = 0
        for r in range(8):
            row = px[r * 9:(r + 1) * 9]
            for c in range(8):
                bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    except Exception:
        bits = None
    if len(_dhash_cache) >= DHASH_MAX:
        _dhash_cache.clear()      # cheap to recompute; never grow without end
    _dhash_cache[name] = bits
    return bits

# ── best model per Comet project ────────────────────────────────────────────
BEST_MODELS = os.path.join(REPO, 'data', 'best_models.json')


def best_models():
    """The hand-curated best-model-per-project file, or {} if absent."""
    try:
        with open(BEST_MODELS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# What each metric KEY actually claims, in the operator's language.
# The keys are precise and unreadable -- acceptance_rejected_at_full_dog_recall
# is the number a promotion turns on, and nothing about the name says so. The
# first field is the plain-English claim used as the headline label; the second
# is the hover text, which says what the number measures AND on what population,
# because this project has twice promoted a model on a figure that turned out to
# be measured on training data.
METRIC_GLOSSARY = {
    # The HARVEST collects ground animals; the DETECTOR is one class -- dog.
    # train-30 and dogdet_v2 both train single_cls on a set whose only name is
    # "target", so a detector metric is a statement about dogs, and calling it
    # "ground animals" credited the model with a job it was never given.
    'recall': (
        'dogs found',
        'Share of real dogs the detector finds. The expensive error: the '
        'sweep runs once, so anything missed here is gone for good.', 'tuning split'),
    'precision': (
        'detections that were dogs',
        'Share of the detector\'s boxes that are really dogs. Cheap to get '
        'wrong -- the gate downstream and the review page both filter these.', 'tuning split'),
    'mAP50': (
        'box quality, loose overlap',
        'Mean average precision at 50% box overlap. Rewards finding the dog; '
        'forgiving about exactly where the box sits.', 'tuning split'),
    'mAP50-95': (
        'box quality, strict overlap',
        'Mean average precision averaged over overlap thresholds 50-95%. '
        'Punishes sloppy boxes, so it tracks crop quality for the classifiers.', 'tuning split'),
    'accuracy_top1': (
        'plain accuracy',
        'Share of crops classified correctly at threshold 0.5. NOT what a gate '
        'is promoted on: on a dog-heavy split, always answering "dog" already '
        'scores high.', 'tuning split'),
    'roc_auc_val': (
        'separation on the tuning split',
        'Ranking quality on the val split. That split drives early stopping, so '
        'it is a tuning set, not an independent one.', 'tuning split'),
    'acceptance_roc_auc': (
        'separation, unseen data',
        'Ranking quality on crops reserved before the dataset was split -- never '
        'trained on, never early-stopped on.', 'reserved crops'),
    'acceptance_rejected_at_full_dog_recall': (
        'false positives removed, at zero dog loss',
        'The promotion number. Share of real detector mistakes the gate discards '
        'at the strictest threshold that still keeps every dog in the val set. '
        'Measured on reserved crops the model has never seen.', 'reserved crops'),
    'acceptance_rejected_at_t0.5': (
        'false positives removed, at 0.5',
        'Same reserved crops, but at the default threshold -- which costs a few '
        'real dogs. Higher than the headline for that reason.', 'reserved crops'),
    'balanced_accuracy_acceptance': (
        'accuracy, both classes weighted equally',
        'Mean of the two class recalls on reserved crops. The headline here '
        'because leashed and unleashed cost the same to get wrong, so plain '
        'accuracy would just reward leaning on the bigger class.', 'reserved crops'),
    'roc_auc_acceptance': (
        'separation, unseen data',
        'Ranking quality on the 312 reserved image_ids, held out before the '
        'split.', 'reserved crops'),
    'recall_leashed_acceptance': (
        'leashed dogs caught',
        'Share of leashed dogs called leashed, on reserved crops.', 'reserved crops'),
    'recall_unleashed_acceptance': (
        'unleashed dogs caught',
        'Share of unleashed dogs called unleashed, on reserved crops.', 'reserved crops'),
    'balanced_accuracy_val': (
        'balanced accuracy, tuning split',
        'Same measure on the val split, which drove early stopping -- expect it '
        'to flatter the model relative to the reserved number.', 'tuning split'),
    'accuracy_top1_val': (
        'plain accuracy, tuning split',
        'Top-1 on the val split. Listed for continuity with older runs; not '
        'what this model was promoted on.', 'tuning split'),
    'roc_auc_as_split': (
        'separation, as originally split',
        'Ranking quality on the split as shipped -- which leaked, so this '
        'number is inflated.', 'leaked split'),
    'roc_auc_sequence_clean': (
        'separation, leak removed',
        'Ranking quality after dropping val images that share a Mapillary '
        'sequence with training. The honest version of the number above it.', 'leak removed'),
    'balanced_accuracy_sequence_clean': (
        'balanced accuracy, leak removed',
        'Both class recalls averaged, after removing val images that share a '
        'sequence with training.', 'leak removed'),
    'sweep_fp_rejected_at_t0.5_heldout': (
        'false positives removed, at 0.5',
        'Share of real detector mistakes discarded at threshold 0.5, on flagged '
        'crops this model did not train on.', 'held-out flags'),
    'sweep_fp_rejected_at_full_dog_recall': (
        'false positives removed, at zero dog loss',
        'Share discarded at the strictest threshold that keeps every val dog.', 'held-out flags'),
}


def metric_meaning(key):
    """(label, hover text, population). Unknown keys degrade to the raw name
    rather than inventing an explanation."""
    hit = METRIC_GLOSSARY.get(key)
    if hit:
        return hit
    return (key.replace('_', ' '), '', '')


# ── training tracker ────────────────────────────────────────────────────────
# The runs live in the OTHER repo (dogs_detection), so the root is
# configuration with no sensible default -- unset, the section says so instead
# of guessing.
def training_root():
    """Resolved per call, so repointing the config reaches a running server."""
    return cfg('training_root', '', env='TRAINING_ROOT')


# A training writes at most one epoch every few minutes, so a 20s window is
# fresh enough and stops a page full of assets re-walking every run directory.
# Deliberately NOT lru_cache: that is exactly what froze the reserved-crop
# count and the dataset path against a config the user had already changed.
_TRK = {'at': 0.0, 'root': None, 'runs': [], 'error': None,
        'hidden': 0}
TRK_TTL = 20


def hidden_projects():
    """Project names the panel should not list, from `training_hide_projects`.

    A folder that once held work and no longer represents a live project is
    still on disk, and nothing in the run files marks it dead -- so which ones
    to retire is a judgement, and judgements belong in config rather than in a
    constant shipped to everyone who clones this.
    """
    return {x.lower() for x in
            cfg_list('training_hide_projects', env='TRAINING_HIDE_PROJECTS')}


def training_runs():
    root = training_root()
    now = time.time()
    if _TRK['root'] == root and now - _TRK['at'] < TRK_TTL:
        return _TRK['runs']
    runs, err, n_hidden = [], None, 0
    if root:
        try:
            import training_tracker
            runs = training_tracker.collect(root, registry=best_models())
            hide = hidden_projects()
            if hide:
                keep = [r for r in runs if r['project'].lower() not in hide]
                n_hidden = len(runs) - len(keep)
                runs = keep
        except Exception as e:
            # swallowing this printed "no runs found", which is a different
            # fact and sends the reader to check the wrong thing
            err = f'{type(e).__name__}: {e}'
    _TRK.update(at=now, root=root, runs=runs, error=err,
                hidden=n_hidden)
    return runs


# Validated on the dark panels (#1b2027 and #21262d): lightness band, chroma
# floor, CVD adjacent separation (dE 21.8 protan / 22.6 tritan), normal-vision
# floor (23.3) and contrast all pass. The dashboard's own --acc (#e8a645) sits
# at OKLCH L 0.77, outside the dark band, so it stays an INK accent and never
# becomes a chart mark.
TRK_A, TRK_B = '#c2872e', '#5b93cf'
TRK_STATUS = {
    'running': ('running', 'live', '&#9679;'),
    'early_stopped': ('early-stopped', 'ok', '&#10003;'),
    'completed': ('ran to last epoch', 'ok', '&#10003;'),
    'interrupted': ('interrupted', 'halt', '&#9632;'),
    'never_started': ('no epoch finished', 'idle', '&#8212;'),
}


def _t(hint):
    return f' title="{esc_html(hint)}"' if hint else ''


def _int(v):
    """args.yaml numbers arrive as floats; "imgsz 1280.0" reads as a typo."""
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return '?'


def _hms(sec):
    if not sec or sec < 0:
        return '--'
    sec = int(sec)
    if sec < 60:
        # a 35s epoch printed as "0m", which reads as a broken clock
        return f'{sec}s'
    if sec < 3600:
        return f'{sec // 60}m'
    if sec < 86400:
        return f'{sec // 3600}h {(sec % 3600) // 60:02d}m'
    return f'{sec // 86400}d {(sec % 86400) // 3600}h'


def _pts(vals, x0, y0, w, h, lo, hi):
    n = len(vals)
    span = (hi - lo) or 1.0
    out = []
    for i, v in enumerate(vals):
        if v is None or v != v or v in (float('inf'), float('-inf')):
            continue
        x = x0 + (w * (i / (n - 1)) if n > 1 else w / 2)
        out.append((x, y0 + h - h * ((v - lo) / span), i, v))
    return out


def _path(pts):
    return ' '.join(('M' if k == 0 else 'L') + f'{x:.1f} {y:.1f}'
                    for k, (x, y, _, _) in enumerate(pts))


def _nice(lo, hi):
    """A padded range that never collapses to zero height on a flat series.

    The padding is not allowed to push a non-negative quantity below zero: a
    loss axis labelled "-0.00" invites the reader to look for a negative loss.
    """
    if hi - lo < 1e-9:
        out = (lo - max(abs(lo) * 0.05, 0.01), hi + max(abs(hi) * 0.05, 0.01))
    else:
        pad = (hi - lo) * 0.08
        out = (lo - pad, hi + pad)
    return (max(0.0, out[0]) if lo >= 0 else out[0], out[1])


def _chart(cid, run, metric, series, marks=(), fmt='.3f'):
    """One line chart as inline SVG plus the JSON its hover layer reads.

    series: [{name, values, color, dim}] -- dim renders as context gray, which
    is the emphasis form: the run being read is in colour, its predecessors
    are the grey it is being compared against.
    """
    title = f'{run} — {metric}'
    W, H = 780, 250
    L, R, T, B = 46, 16, 14, 26
    w, h = W - L - R, H - T - B
    live = [v for s in series for v in s['values'] if v is not None]
    if not live:
        return '<div class="tempty">no epochs recorded yet</div>'
    lo, hi = _nice(min(live), max(live))
    n = max((len(s['values']) for s in series), default=1)

    grid, ticks = [], []
    for k in range(4):
        y = T + h - h * (k / 3)
        grid.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + w}" y2="{y:.1f}"/>')
        ticks.append(f'<text x="{L - 7}" y="{y + 3.5:.1f}" text-anchor="end">'
                     f'{lo + (hi - lo) * k / 3:{fmt}}</text>')
    for k, e in ((0, 1), (1, n)):
        x = L + (w * ((e - 1) / (n - 1)) if n > 1 else w / 2)
        ticks.append(f'<text x="{x:.1f}" y="{T + h + 16}" '
                     f'text-anchor="{"start" if k == 0 else "end"}">{e}</text>')

    body, keyed = [], []
    for s in series:
        p = _pts(s['values'], L, T, w, h, lo, hi)
        if not p:
            continue
        if s.get('dim'):
            body.append(f'<path class="ctx" d="{_path(p)}"/>')
        elif len(p) == 1:
            # a one-point path paints nothing; a successful 1-epoch run would
            # render as an empty plot, which is what "no data" looks like
            body.append(f'<circle class="mk" cx="{p[0][0]:.1f}" '
                        f'cy="{p[0][1]:.1f}" r="4" '
                        f'style="fill:{s["color"]}"/>')
        else:
            body.append(f'<path class="ln" d="{_path(p)}" '
                        f'style="stroke:{s["color"]}"/>')
            keyed.append({'name': s['name'], 'color': s['color'],
                          'values': [None if v is None or v != v
                                     or v in (float('inf'), float('-inf'))
                                     else round(v, 6) for v in s['values']]})
    for m in marks:
        p = _pts([m['v'] if i == m['i'] else None for i in range(n)],
                 L, T, w, h, lo, hi)
        if not p:
            continue
        x, y = p[0][0], p[0][1]
        end = x > L + w * 0.7
        body.append(
            f'<circle class="mk" cx="{x:.1f}" cy="{y:.1f}" r="5"/>'
            f'<text class="mklab" x="{x + (-11 if end else 11):.1f}" '
            f'y="{y + 4:.1f}" text-anchor="{"end" if end else "start"}">'
            f'{esc_html(m["label"])}</text>')

    legend = ''
    if len(keyed) > 1:
        legend = '<span class="tleg">' + ''.join(
            f'<span><i style="background:{s["color"]}"></i>'
            f'{esc_html(s["name"])}</span>' for s in keyed) + '</span>'
    # No rotated y-axis label: the caption already names the quantity, and
    # the rotated text landed on top of the tick values.
    axis = ''
    # allow_nan=False would raise; the point is to emit valid JSON, so
    # non-finite values become null -- the same thing a missing epoch is.
    data = json.dumps({'x0': L, 'w': w, 'n': n, 'top': T, 'h': h,
                       'series': keyed}, allow_nan=False)
    return (f'<figure class="tfig" data-chart="{esc_html(data)}">'
            f'<figcaption><b class="tmetric">{esc_html(metric)}</b>'
            f'<span class="trun">{esc_html(run)}</span>{legend}</figcaption>'
            f'<svg class="tsvg" viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="{esc_html(title)}">'
            f'<g class="grid">{"".join(grid)}</g>'
            f'<g class="tick">{"".join(ticks)}</g>{axis}{"".join(body)}'
            f'<line class="cross" x1="0" y1="{T}" x2="0" y2="{T + h}"/>'
            f'<rect class="hit" x="{L}" y="{T}" width="{w}" height="{h}"/>'
            f'</svg><div class="ttip" hidden></div></figure>')


TRK_LATEST, TRK_PEAK = '#c2872e', '#5b93cf'


def _metric_card(m, label, hover):
    """One metric: what it just did, against the best it has ever done.

    The peak is drawn as a RULE ACROSS THE TRACE at its own height, and the
    latest point sits below it by exactly the shortfall -- so the gap is a
    distance rather than a second number to subtract. When the run is at its
    peak the point lands on the rule and the card resolves to one value, which
    is the state worth recognising at a glance.
    """
    vals = [v for v in m['series'] if v is not None]
    if not vals:
        return ''
    lo, hi = min(vals), m['peak']
    span = (hi - lo) or max(abs(hi) * 0.08, 1e-6)
    # RIGHT is inset by the marker radius: the current point is the whole
    # subject of the card and it was being clipped in half by the viewBox edge.
    W, H, PAD, RIGHT = 232.0, 44.0, 6.0, 6.0
    n = len(m['series'])

    def xy(i, v):
        x = ((W - RIGHT) * (i / (n - 1))) if n > 1 else (W - RIGHT) / 2
        return (x, PAD + (H - 2 * PAD) * (1 - (v - lo) / span))

    pts = [xy(i, v) for i, v in enumerate(m['series']) if v is not None]
    trace = ' '.join(('M' if k == 0 else 'L') + f'{x:.1f} {y:.1f}'
                     for k, (x, y) in enumerate(pts))
    py = xy(m['peak_index'], m['peak'])[1]
    lx, ly = xy(len(m['series']) - 1, m['latest'])
    at_peak = (m['peak'] - m['latest']) <= 1e-9
    spark = (
        f'<svg class="mspk" viewBox="0 0 {W:.0f} {H:.0f}" '
        f'preserveAspectRatio="none" aria-hidden="true">'
        # the rule: the benchmark, drawn where the benchmark actually is
        f'<line class="pk" x1="0" y1="{py:.1f}" x2="{W:.0f}" y2="{py:.1f}"/>'
        + (f'<line class="gap" x1="{lx:.1f}" y1="{py:.1f}" '
           f'x2="{lx:.1f}" y2="{ly:.1f}"/>' if not at_peak else '')
        + (f'<path class="tr" d="{trace}"/>' if len(pts) > 1 else '')
        + f'<circle class="pkd" cx="{xy(m["peak_index"], m["peak"])[0]:.1f}" '
          f'cy="{py:.1f}" r="2.6"/>'
          f'<circle class="now" cx="{lx:.1f}" cy="{ly:.1f}" r="3.4"/>'
          f'</svg>')
    # The two numbers separate by POSITION as well as by colour: the peak sits
    # on the label line, where a reference belongs, and the working value gets
    # the card to itself.
    return (
        f'<div class="mcard"{_t(hover)}>'
        f'<div class="mhead"><span class="mlab">{esc_html(label)}</span>'
        f'<span class="mpv">{m["peak"]:.4f}'
        f'<em>peak @{m["peak_epoch"]}</em></span></div>'
        f'<div class="mnow">{m["latest"]:.4f}</div>'
        f'<div class="mgap">'
        + ('at its peak' if at_peak else
           f'{m["peak"] - m["latest"]:.4f} below peak')
        + f'</div>{spark}</div>')


def _live_card(r):
    """The card that answers "how long until this stops?".

    Early stopping fires on epochs since the BEST epoch, never on epochs
    elapsed -- so that ratio is the meter and everything else is context.
    """
    ep, tot = r['epochs_done'], r['epochs_planned']
    pat, since, sec = r['patience'], r['since_best'], r['secs_per_epoch']
    tiles = []

    def tile(v, k, hint=''):
        tiles.append(f'<div class="ttile"{_t(hint)}><b>{v}</b>'
                     f'<span>{k}</span></div>')

    if not ep:
        tile('&mdash;', 'no epoch finished yet',
             'results.csv gets a row only when an epoch completes. Until then '
             'the run is loading its dataset or is inside epoch 1.')
        if r.get('started'):
            tile(_hms(time.time() - r['started']), 'running for')
        if tot:
            tile(str(tot), 'epochs planned')
        if pat:
            tile(str(pat), 'patience',
                 'Epochs without an improvement that end the run.')
    else:
        tile(f'{ep}<em>/{tot or "?"}</em>', 'epoch',
             'Epochs finished, out of the epochs= this run was started with.')
        if r['best_epoch']:
            tile(f'@{r["best_epoch"]}', 'best epoch',
                 'The epoch early stopping is measuring from -- the peak of '
                 'the fitness this run is scored on, which is not necessarily '
                 'the peak of any single metric beside it.')
        if sec:
            tile(_hms(sec), 'per epoch',
                 'Mean over the last 10 epochs, so a slowdown shows instead '
                 'of being averaged away.')

    # ── what the LATEST epoch reported ─────────────────────────────────
    # The tiles above answer "when does this stop". These answer "is it any
    # good yet", which is the other question asked mid-run. Every value is
    # from the newest epoch; the peak is shown beside it because on a metric
    # that is still climbing the two are the same number and on one that has
    # turned over they are not.
    # Two limits end this run and they are not the same race. The epoch
    # budget is fixed and visible from the start; patience moves -- it resets
    # to zero on every improvement -- and on these runs it is almost always
    # the one that fires. Showing only one of them answered half the question.
    def _meter(label, val, limit, foot, hint):
        return (f'<div class="tmeter"{_t(hint)}>'
                f'<div class="tmhead"><span>{label}</span>'
                f'<b>{val}<em>&thinsp;/&thinsp;{limit}</em></b></div>'
                f'<div class="tmtrack"><i style="width:'
                f'{min(1.0, val / limit) * 100:.1f}%"></i></div>'
                f'<div class="tmfoot">{foot}</div></div>')

    meters = []
    if tot and ep:
        left_ep = max(0, tot - ep)
        meters.append(_meter(
            'epochs run', ep, tot,
            f'{left_ep} left in the budget'
            + (f' &middot; about {_hms(left_ep * sec)} away' if sec else ''),
            'The epochs= this run was started with. Reaching it is one of the '
            'two ways the run can end, and usually the slower one.'))
    if pat and since is not None:
        left_pat = max(0, pat - since)
        meters.append(_meter(
            'epochs since the best', since, pat,
            f'{left_pat} more without an improvement and the run stops'
            + (f' &middot; about {_hms(left_pat * sec)} away' if sec else ''),
            'Ultralytics stops the run when this reaches patience. It resets '
            'to zero every time the metric improves, so this bar can go '
            'backwards while the one beside it only ever fills.'))

    meter = ''
    if meters:
        # which limit is actually nearer, said once rather than left to be
        # worked out from two bars
        note = ''
        if tot and ep and pat and since is not None:
            a_left, b_left = max(0, tot - ep), max(0, pat - since)
            note = (f'<div class="tmnote">'
                    + ('patience is nearer -- expect the run to early-stop'
                       if b_left < a_left else
                       'the epoch budget is nearer -- expect the run to use '
                       'all of it' if a_left < b_left else
                       'both limits are the same distance away')
                    + '</div>')
        meter = f'<div class="tmeters">{"".join(meters)}</div>{note}'

    return (f'<div class="tlive">'
            f'<div class="tlhead"><span class="bdg live">running</span>'
            f'<b>{esc_html(r["project"])}/{esc_html(r["name"])}</b>'
            f'<span class="tsub">{esc_html(r.get("model") or "")} &middot; '
            f'imgsz {_int(r.get("imgsz"))} &middot; '
            f'batch {_int(r.get("batch"))}'
            + (' &middot; one class' if r.get('single_cls') else '')
            + f' &middot; pid {r["pid"]}</span></div>'
            f'<div class="ttiles">{"".join(tiles)}</div>'
            + _metric_row(r)
            + f'{meter}</div>')


# A run that finished, on its own terms. early_stopped IS a completion --
# patience firing is how these runs are meant to end. 'interrupted' (killed
# before either patience or the epoch budget) and 'never_started' are not
# results, and putting them in the history invites comparing a killed run's
# best epoch against a finished one's.
REAL = ('running', 'early_stopped', 'completed')
TRK_PAGE = 8


def _history(runs):
    """Finished runs as a table -- also the non-hover route to every value."""
    body = []
    for r in runs:
        lab, cls, gly = TRK_STATUS.get(r['status'], (r['status'], 'idle', ''))
        best = ('&mdash;' if r['best_headline'] is None
                else f'{r["best_headline"]:.4f}')
        prom = ''
        if r.get('promoted'):
            p = r['promoted']
            prom = ('<span class="bdg live">in production</span>'
                    if p.get('deployed') else
                    '<span class="bdg ok">promoted</span>'
                    if not p.get('candidate') else
                    '<span class="tcand">candidate</span>')
        # results.csv already carries cumulative seconds; multiplying a
        # 10-epoch rate by the epoch count invents a number that disagrees
        # with it whenever the rate changed.
        dur = _hms(r.get('wall_clock_s')) if r.get('wall_clock_s') else '&mdash;'
        body.append(
            f'<tr data-proj="{esc_html(r["project"])}" '
            f'data-key="{esc_html(run_key(r))}" tabindex="0" '
            f'role="button" aria-label="Show metrics for '
            f'{esc_html(run_key(r))}"'
            + (' class="onair"' if r['live'] else '') + '>'
            f'<td class="tn"><b>{esc_html(r["name"])}</b>'
            f'<span>{esc_html(r["project"])}</span></td>'
            f'<td><span class="tst {cls}">{gly} {lab}</span></td>'
            f'<td class="num">{r["epochs_done"] or "&mdash;"}</td>'
            f'<td class="num">{r["best_epoch"] or "&mdash;"}</td>'
            f'<td class="num">{best}</td>'
            f'<td class="num">{dur}</td>'
            f'<td>{prom}</td></tr>')
    return (
        '<div class="tscroll"><table class="thist"><thead><tr>'
        '<th>run</th><th>status</th>'
        f'<th class="num"{_t("Epochs written to results.csv.")}>epochs</th>'
        f'<th class="num"{_t("The epoch early stopping measured from.")}>'
        'best@</th>'
        f'<th class="num"{_t("Best value of the deciding metric on this "
                             "run own validation split.")}>best metric</th>'
        f'<th class="num"{_t("Epochs times the mean seconds per epoch.")}>'
        'wall clock</th>'
        '<th>registry</th></tr></thead><tbody>'
        + ''.join(body) + '</tbody></table></div>'
        '<div class="tpage" hidden><button type="button" class="tpb" '
        'data-d="-1" aria-label="previous page">&#8592;</button>'
        '<span class="tpn"></span>'
        '<button type="button" class="tpb" data-d="1" '
        'aria-label="next page">&#8594;</button></div>')


def run_key(r):
    """The identity the client sends back. Never a path: a directory from the
    client is a traversal waiting to happen, and this resolves by exact match
    against the runs already discovered."""
    return f'{r["project"]}/{r["name"]}'


def _metric_row(r):
    """The metric cards. Every run has them -- for a finished run the "latest"
    epoch is simply its last one, which is the number you want."""
    cards = []
    for m in (r.get('latest') or ()):
        label, hover, _ = metric_meaning(m['key'])
        cards.append(_metric_card(m, label, hover))
    if not cards:
        return ''
    return (f'<div class="tlatest"><span class="tlab">epoch '
            f'{r.get("last_epoch") or "?"} '
            f'<i class="k now"></i>latest'
            f'<i class="k pk"></i>its own peak</span>'
            f'<div class="mcards">{"".join(cards)}</div></div>')


def _charts(r):
    """The deciding metric and the losses, for one run."""
    if not r.get('epochs_done'):
        return ''
    marks = []
    if r['best_epoch'] and r['best_headline'] is not None:
        # best_epoch is the number in the epoch COLUMN; the chart is indexed
        # by row. They differ on a resumed run, which put the marker at the
        # right height and the wrong x.
        try:
            bi = r['curve'].index(r['best_headline'])
        except ValueError:
            bi = min(r['best_epoch'] - 1, len(r['curve']) - 1)
        marks = [{'i': bi, 'v': r['best_headline'],
                  'label': f'best @{r["best_epoch"]}'}]
    who = f'{r["project"]}/{r["name"]}'
    return ('<div class="tgrid">'
            + _chart('trk-metric', who,
                     r['headline_key'] or r['headline_label'],
                     [{'name': r['headline_label'], 'values': r['curve'],
                       'color': TRK_A}], marks, fmt='.3f')
            + _chart('trk-loss', who, r.get('loss_label') or 'loss',
                     [{'name': 'train', 'values': r['train_loss'],
                       'color': TRK_A},
                      {'name': 'validation', 'values': r['val_loss'],
                       'color': TRK_B}], fmt='.2f')
            + '</div>')


def _past_head(r):
    """A finished run's identity line -- the live card's shape without the
    clocks, because none of them are still running."""
    lab, cls, gly = TRK_STATUS.get(r['status'], (r['status'], 'idle', ''))
    bits = [f'{_int(r.get("imgsz"))} px', f'batch {_int(r.get("batch"))}']
    if r.get('single_cls'):
        bits.append('one class')
    if r.get('wall_clock_s'):
        bits.append(f'ran {_hms(r["wall_clock_s"])}')
    if r.get('ultralytics'):
        bits.append(f'ultralytics {esc_html(r["ultralytics"])}')
    tiles = [f'<div class="ttile"><b>{r["epochs_done"]}'
             f'<em>/{r.get("epochs_planned") or "?"}</em></b>'
             f'<span>epochs</span></div>']
    if r.get('best_epoch'):
        tiles.append(f'<div class="ttile"><b>@{r["best_epoch"]}</b>'
                     f'<span>best epoch</span></div>')
    if r.get('secs_per_epoch'):
        tiles.append(f'<div class="ttile"><b>{_hms(r["secs_per_epoch"])}</b>'
                     f'<span>per epoch</span></div>')
    return (f'<div class="tlive past">'
            f'<div class="tlhead"><span class="tst {cls}">{gly} {lab}</span>'
            f'<b>{esc_html(r["project"])}/{esc_html(r["name"])}</b>'
            f'<span class="tsub">{esc_html(r.get("model") or "")} &middot; '
            f'{" &middot; ".join(bits)}</span>'
            f'<button type="button" class="rbtn quiet tback" id="trkBack">'
            f'&larr; back to the live run</button></div>'
            f'<div class="ttiles">{"".join(tiles)}</div>'
            f'{_metric_row(r)}</div>')


def render_run_detail(key):
    """One run's detail region, resolved by key against what was discovered."""
    for r in training_runs():
        if run_key(r) == key:
            return _past_head(r) + _charts(r)
    return '<div class="mnone">That run is no longer on disk.</div>'


def sweep_db_path():
    """The unified DuckDB file, resolved at render time. Never a constant:
    this repo is public and the store lives on a drive specific to one box."""
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
        import build_sqldb
        return build_sqldb.default_db(build_sqldb.detect_root())
    except Exception:
        return None


def _db_facts(path):
    """(built_at, images) from the database's own _meta, or (None, None).

    Read-only and best-effort: a build in progress holds the write lock, and
    the panel should still show the path rather than fail because someone is
    refreshing it.
    """
    if not path or not os.path.exists(path):
        return (None, None)
    try:
        import duckdb
        con = duckdb.connect(path, read_only=True)
        try:
            got = dict(con.execute(
                "SELECT key, value FROM _meta WHERE key IN "
                "('built_at', 'images_at_build')").fetchall())
        finally:
            con.close()
        return (got.get('built_at'), got.get('images_at_build'))
    except Exception:
        return (None, None)


def render_store_path():
    """The one path worth copying: what you paste into duckdb or a notebook.

    It is a DERIVED file, so the line says so. A path handed over without that
    is an invitation to read a stale copy as the live one -- and this database
    is stale by design, the moment the sweep writes another part.
    """
    db = sweep_db_path()
    if not db:
        return ''
    built, imgs = _db_facts(db)
    cmd = 'python tools/detect/build_sqldb.py build'
    if not os.path.exists(db):
        hint = (f'not built yet &mdash; run <code>{esc_html(cmd)}</code>')
    else:
        n = f'{int(imgs):,} images' if imgs and str(imgs).isdigit() else ''
        hint = ('derived from the parquet store beside it'
                + (f' &middot; built {esc_html(built)}' if built else '')
                + (f' &middot; {n}' if n else '')
                + f' &middot; refresh with <code>{esc_html(cmd)}</code>')
    return (f'<div class="spath">'
            f'<span class="splab">sweep database</span>'
            f'<code id="storePath">{esc_html(db)}</code>'
            f'<button type="button" class="cp" id="storeCp" '
            f'title="Copy the database path" '
            f'aria-label="Copy the database path">'
            f'<svg viewBox="0 0 24 24" width="13" height="13" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round"><rect x="9" y="9" width="12" '
            f'height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 '
            f'2-2h9a2 2 0 0 1 2 2v1"/></svg></button>'
            f'<span class="sphint">{hint}</span></div>')


def render_training():
    """Runs on disk: what is training now, how it compares, what came before.

    The live run leads because it is the only part that changes while the page
    is open, and the only question with a deadline attached.
    """
    root = training_root()
    if not root:
        return ('<div class="mnone">No training root configured. Set '
                '<code>training_root</code> in the dashboard config (or '
                '<code>$TRAINING_ROOT</code>) to the directory holding the '
                'ultralytics project folders.</div>')
    if not os.path.isdir(root):
        what = ('is a file, not a directory' if os.path.exists(root)
                else 'does not exist')
        return (f'<div class="mnone">Training root <code>{esc_html(root)}'
                f'</code> {what}. Fix <code>training_root</code> in the '
                f'dashboard config.</div>')
    if not os.access(root, os.R_OK | os.X_OK):
        return (f'<div class="mnone">Training root <code>{esc_html(root)}'
                f'</code> is not readable by the dashboard process.</div>')
    runs = training_runs()
    if _TRK.get('error'):
        return (f'<div class="mnone">Could not read training runs under '
                f'<code>{esc_html(root)}</code>: '
                f'<code>{esc_html(_TRK["error"])}</code></div>')
    if not runs:
        return (f'<div class="mnone">No ultralytics runs found under '
                f'<code>{esc_html(root)}</code>. A run is any directory with '
                f'an <code>args.yaml</code> in it.</div>')

    out = ['<div id="trkdet">']
    for r in runs:
        if r['live']:
            out.append(_live_card(r))

    focus = next((r for r in runs if r['live'] and r['epochs_done']), None) \
        or next((r for r in runs if r['epochs_done']), None)
    if focus:
        if not focus['live']:
            any_live = any(r['live'] for r in runs)
            out.append('<div class="tlead">'
                       + ('The live run has not written an epoch yet. Charts '
                          'below are the most recent run that did.'
                          if any_live else
                          'Nothing is training. Charts below are the most '
                          'recent run that recorded epochs.')
                       + '</div>')
        out.append(_charts(focus))
    out.append('</div>')

    out.append('<div class="tbar"><label for="tproj">project</label>'
               '<select id="tproj" title="scope the table below to one '
               'project"><option value="">all projects</option>'
               + ''.join(f'<option value="{esc_html(p)}">{esc_html(p)}'
                         f'</option>' for p in
                         sorted({r['project'] for r in runs
                                 if r['status'] in REAL}))
               + '</select><span class="tnote">Numbers here are each run\'s '
                 'own validation split &mdash; the split that drove its early '
                 'stopping. What a model is <em>accepted</em> on is the '
                 'reserved set, in Best models above.</span></div>')
    shown = [r for r in runs if r['status'] in REAL]
    hidden = len(runs) - len(shown)
    out.append(_history(shown))
    # Never silently, and never with one reason standing in for two: a run
    # left out for being unfinished and a project retired in the config are
    # different facts, and one sentence covering both said something untrue
    # about each.
    why = []
    if hidden:
        why.append(f'{hidden} interrupted before patience or the epoch '
                   f'budget, or never finished an epoch')
    if _TRK.get('hidden'):
        why.append(f'{_TRK["hidden"]} in projects hidden by '
                   f'<code>training_hide_projects</code>')
    if why:
        out.append('<div class="thid">Not listed: ' + '; '.join(why)
                   + '.</div>')
    return ''.join(out)


def render_models():
    """The three projects as the PIPELINE they are, not three cards.

    A crop must clear stage 1 to reach stage 2. So a stage with no accepted
    model does not just lack a model -- it stops everything downstream, and
    that is the fact this view exists to show. The rail is solid where work
    flows and broken below the first stage that has nothing.

    Metrics render as tags rather than a run-on string, with the one that
    decided the promotion (key_metric) carrying the accent. Values are set in
    a monospace face so they read as instrument readings, not prose.
    """
    st = best_models()
    projs = (st.get('projects') or {})
    if not projs:
        return '<div class="mnone">no data/best_models.json yet</div>'
    tmpl = st.get('url_template',
                  'https://www.comet.com/{workspace}/{project}/{key}')
    ws = st.get('workspace', '')

    def link(project, rec):
        k = (rec or {}).get('key')
        if not k:
            return None
        try:
            return tmpl.format(workspace=ws, project=project, key=k)
        except (KeyError, IndexError):
            return None

    def readout(metrics, key):
        """The deciding metric, large, labelled with the claim it makes.

        The key names are precise and unreadable. Leading with
        'acceptance_rejected_at_full_dog_recall = 0.5234' asks the reader to
        already know which of six numbers decided the promotion and what it
        was measured on; leading with the claim does not.
        """
        if not metrics or not key or key not in metrics:
            return ''
        label, hover, pop = metric_meaning(key)
        v = metrics[key]
        try:
            shown = (f'{float(v) * 100:.1f}%' if 0 <= float(v) <= 1
                     else f'{float(v):.4g}')
        except (TypeError, ValueError):
            shown = str(v)
        # The population, always. Without it the leash headline read
        # "accuracy, both classes weighted equally" directly above a chip
        # reading "balanced accuracy, tuning split" -- the same measure on
        # different data, with only the chip saying which.
        eb = (f'<em class="pop">{esc_html(pop)}</em>' if pop else '')
        return (f'<div class="hero" title="{esc_html(hover)}">'
                f'{eb}<b>{esc_html(shown)}</b>'
                f'<span>{esc_html(label)}</span></div>')

    def tags(metrics, key=None, small=False):
        """Metric chips, with the deciding one accented.

        A key_metric naming a metric the model does not carry accents NOTHING
        and looks identical to a project that simply has no headline -- which
        is how leash-models sat unhighlighted after its metrics were renamed,
        while dog-bin quietly accented accuracy_top1, the one metric its own
        entry says it was NOT promoted on. So an unmatched key is called out
        rather than rendered as silence.
        """
        if not metrics:
            return ''
        out = []
        for k, v in metrics.items():
            if k == key:
                continue          # already the headline; twice is noise
            label, hover, pop = metric_meaning(k)
            tail = (f'<em class="pop">{esc_html(pop)}</em>' if pop else '')
            out.append(f'<span class="tag" title="{esc_html(hover or label)}">'
                       f'<i>{esc_html(label)}</i><b>{esc_html(v)}</b>'
                       f'{tail}</span>')
        if key and key not in metrics:
            out.append('<span class="tag warn" title="key_metric names a '
                       'metric this model does not report, so nothing is '
                       'accented -- fix data/best_models.json">'
                       f'<i>key_metric</i><b>{esc_html(key)}?</b></span>')
        return f'<div class="tags{" sm" if small else ""}">{"".join(out)}</div>'

    order = sorted(projs.items(), key=lambda kv: kv[1].get('stage', 99))
    # the first stage with nothing accepted is where the pipeline stops
    broken_at = next((d.get('stage') for _, d in order if not d.get('best')),
                     None)
    rows = []
    for name, d in order:
        b = d.get('best')
        stage = d.get('stage', 0)
        live = bool(b)
        blocked = broken_at is not None and stage > broken_at
        cls = 'live' if live else ('halt' if stage == broken_at else 'idle')
        # A promoted model is not a deployed one. Only the detector runs
        # inside the sweep today; the gate and the leash classifier are
        # accepted but not wired in, and printing "in production" for all
        # three told the operator the pipeline was doing work it is not.
        deployed = bool(b and b.get('deployed'))
        badge = (('<span class="bdg live">in production</span>' if deployed
                  else '<span class="bdg ok">promoted &middot; not wired in'
                       '</span>') if live else
                 '<span class="bdg halt">pipeline stops here</span>'
                 if stage == broken_at else
                 '<span class="bdg idle">waiting on stage %d</span>' % broken_at)
        head = (f'<div class="sh"><span class="sname">'
                f'{esc_html(d.get("short") or name)}</span>{badge}'
                f'<span class="semit">&rarr; {esc_html(d.get("emits") or "")}</span>'
                f'<span class="sproj">{esc_html(name)}</span></div>')
        if live:
            u = link(name, b)
            run = (f'<a class="srun" href="{esc_html(u)}" target="_blank" '
                   f'rel="noopener">{esc_html(b.get("run"))}<span class="ext">'
                   f'&#8599;</span></a>' if u else
                   f'<span class="srun">{esc_html(b.get("run"))}</span>')
            body = (run + readout(b.get('metrics'), d.get('key_metric'))
                    + tags(b.get('metrics'), d.get('key_metric')))
            w = b.get('weights')
            if w:
                body += f'<div class="sfile">{esc_html(w)}</div>'
            note = b.get('why') or ''
            cav = b.get('caveat')
            if cav:
                note = note + '\n\n' + cav
        else:
            body = '<span class="srun none">no model accepted</span>'
            note = d.get('why_blank') or ''
            cands = d.get('candidates') or []
            if cands:
                cl = []
                for c in cands:
                    u = link(name, c)
                    # no key_metric here: the accent means "this is why the
                    # model was promoted", and nothing in this list was.
                    # Amber on a rejected run's score reads as endorsement.
                    inner = (f'<span class="cname">{esc_html(c.get("run"))}</span>'
                             + tags(c.get('metrics'), None, True))
                    cl.append(f'<a class="cand" href="{esc_html(u)}" '
                              f'target="_blank" rel="noopener">{inner}</a>'
                              if u else f'<span class="cand">{inner}</span>')
                body += ('<div class="cwrap"><div class="clab">tried</div>'
                         + ''.join(cl) + '</div>')
        # The justification is 200-400 words of measurement history -- the
        # reason to trust the number, which matters exactly once and then
        # never again until someone questions it. Collapsed by default: at
        # full length it buried the number it was defending.
        why = ''
        if note:
            paras = ''.join(f'<p>{esc_html(x)}</p>'
                            for x in note.split('\n\n') if x.strip())
            why = ('<details class="swhy"><summary>Why this model'
                   '</summary>' + paras + '</details>')
        rows.append(f'<div class="stg {cls}"><div class="dot"></div>'
                    f'{head}<div class="sbody">{body}{why}</div></div>')
    upd = esc_html(st.get('updated_at') or '')
    # The words under every number, defined once. Which data a figure came
    # from is the whole difference between a result and a train-set score
    # here, and "reserved crops" carries that difference in two words -- so
    # the two words have to be defined somewhere the reader can reach them.
    terms = [
        ('reserved crops',
         'Images set aside BEFORE the dataset was split, listed in a '
         'dogbin_/leash_acceptance_set.json. They appear in no training and no '
         'validation split, so a model has never seen them in any form. This '
         'is the only population a promotion is written from.'),
        ('tuning split',
         'The val folder. It decides when training stops, which makes the '
         'model indirectly fitted to it -- so its numbers run optimistic and '
         'are shown here for context, not for deciding.'),
        ('leak removed',
         'A number recomputed after dropping validation images that share a '
         'Mapillary sequence with training. Consecutive frames of one animal '
         'seconds apart are effectively the same photo; counting them as '
         'held-out inflated two models in this project by 4-6 points.'),
        ('at zero dog loss',
         'The strictest threshold that still keeps every dog in the '
         'validation set. The sweep runs once over 32.5M images, so a discarded '
         'dog is gone for good, while a surviving false positive costs one '
         'click in the review page.'),
        ('separation',
         'ROC AUC: the chance the model scores a random positive above a '
         'random negative. Independent of where the threshold sits, so it says '
         'whether the two classes are distinguishable at all.'),
        ('promoted, not wired in',
         'Accepted as the best model for its stage, but nothing runs it yet. '
         'Only the detector currently executes inside the sweep.'),
    ]
    gl = ''.join(
        f'<div class="gt"><dt>{esc_html(t)}</dt><dd>{esc_html(v)}</dd></div>'
        for t, v in terms)
    return (f'<div class="pipe">{"".join(rows)}</div>'
            f'<div class="mfoot">data/best_models.json &middot; updated {upd} '
            f'&middot; refresh with <code>tools/detect/best_models.py --update</code>'
            f'</div>'
            f'<details class="gloss"><summary>'
            f'<span class="gk">Glossary</span>'
            f'How to read these numbers'
            f'</summary>'
            f'<dl class="gwrap">{gl}</dl></details>')


def esc_html(v):
    return (str(v if v is not None else '')
            .replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            .replace('"', '&quot;'))


# ── box correction ──────────────────────────────────────────────────────────
# The saved full frames have the box BURNED IN and are cut from the 1280
# letterboxed tensor, so neither the pixels nor the geometry can be edited
# from them. Editing works off the ORIGINAL jpg plus the store's own
# x1,y1,x2,y2 (already in original-image pixels, spec 5.3).
BOX_DIR = os.path.join(REPO, 'data', 'box_corrections')
BOX_LABELS = os.path.join(BOX_DIR, 'boxes.jsonl')
_box_lock = threading.Lock()
_ID_RE = re.compile(r'^[0-9]{6,24}$')
_CELL_RE = re.compile(r'^[A-Za-z_]+(?:_-?\d+){4}$')


@functools.lru_cache(maxsize=1)
def _grid_roots():
    """{drive label: grid_runs root} from the gitignored roots file."""
    out = {}
    try:
        for ln in open(os.path.join(REPO, 'data', 'catalog_dirs.txt')):
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            # Ask the filesystem where the mount actually is. The old parser
            # split on '/media/<user>/<drive>' | '/home/<user>/<drive>', which
            # is one distro's automount convention -- on '/mnt/<drive>' it
            # matched nothing and every box correction silently lost its root.
            label = _drive_of(ln)
            if not label:
                parts = os.path.abspath(ln).split(os.sep)
                for i, p in enumerate(parts):
                    if p in ('media', 'home') and i + 2 < len(parts):
                        label = parts[i + 2]
                        break
            if label:
                out.setdefault(label, ln)
    except OSError:
        pass
    return out


def _detect_sql():
    """(img_src, det_src) duckdb sources for the predictions store."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import store as _store
    root = _store.get_detect_root()
    return (_store._sql_src(_store._store_globs(root, 'img')),
            _store._sql_src(_store._store_globs(root, 'det')))


def box_for(name):
    """Original-pixel boxes + source jpg for one crop, or an error dict.

    Read-only. The image_id and cell come back from the store and are both
    re-validated against strict patterns before touching the filesystem --
    they decide a path, so treating them as trusted would be a traversal.
    """
    m = _CROP_RE.match(name or '')
    if not m:
        return {'ok': False, 'error': 'malformed crop name'}
    iid = m.group(2)
    if not _ID_RE.match(iid):
        return {'ok': False, 'error': 'unexpected image_id'}
    try:
        img, det = _detect_sql()
        con = duckdb.connect()
        meta = con.execute(
            f"SELECT orig_w, orig_h, cell, drive FROM {img} "
            f"WHERE CAST(image_id AS VARCHAR)=? LIMIT 1", [iid]).fetchall()
        rows = con.execute(
            f"SELECT det_idx, x1, y1, x2, y2, conf FROM {det} "
            f"WHERE CAST(image_id AS VARCHAR)=? ORDER BY det_idx",
            [iid]).fetchall()
        con.close()
    except Exception as e:
        return {'ok': False, 'error': str(e)}
    if not meta:
        # ~9.5% of the live pool at any moment: the crop was written by the
        # preview sampler but its shard has not been committed yet, so the
        # store has no geometry for it. It becomes editable on its own.
        return {'ok': False, 'pending': True,
                'error': 'not committed yet — this detection\'s shard is '
                         'still in flight; it becomes editable once the '
                         'sweep commits it'}
    w, h, cell, drive = meta[0]
    if not _CELL_RE.match(str(cell or '')):
        return {'ok': False, 'error': 'unexpected cell'}
    root = _grid_roots().get(str(drive))
    if not root:
        return {'ok': False, 'error': 'unknown drive %r' % (drive,)}
    path = os.path.join(root, str(cell), 'ground_animal_images', iid + '.jpg')
    if not os.path.realpath(path).startswith(os.path.realpath(root) + os.sep):
        return {'ok': False, 'error': 'path escapes the drive root'}
    saved = _saved_box(name)
    return {'ok': True, 'image_id': iid, 'w': int(w or 0), 'h': int(h or 0),
            'has_file': os.path.exists(path), 'path': path,
            'boxes': [{'det_idx': int(d), 'x1': float(a), 'y1': float(b),
                       'x2': float(c), 'y2': float(e), 'conf': round(float(f), 3)}
                      for d, a, b, c, e, f in rows],
            'saved': saved}


def _orig_path(name):
    """Filesystem path of the original jpg behind a crop, or None.

    box_for() already resolved (cell, drive) and validated the path; re-running
    the whole lookup and then querying duckdb a SECOND time doubled the work on
    every image load for nothing.
    """
    info = box_for(name)
    p = info.get('path') if info.get('ok') else None
    return p if p and os.path.exists(p) else None


HQ_DIR = os.path.join(OUT, 'hq_crops')
HQ_MAX = 512          # long side; boxes are rarely larger, so no upscaling
HQ_PAD = 0.12         # same context padding the preview writer uses


def hq_crop(name):
    """Path to a high-quality crop cut from the ORIGINAL, generating it once.

    The grid used the preview thumbnails, which are cut from the 1280
    letterboxed tensor and then capped at 160 px -- measured on live crops,
    that discards 3.6-5.3x the pixels actually available (a 309x223 box shown
    at 128x101). At that size a distant dog and a distant goat look the same,
    which is exactly the judgement this page asks for.

    Cut from the source jpg at native box resolution instead. Cached to disk
    because decoding a 12 MP original per tile is far too slow to repeat, and
    returns None on any failure so the client can fall back to the preview.
    """
    m = _CROP_RE.match(name or '')
    if not m:
        return None
    out = os.path.join(HQ_DIR, name)
    if os.path.exists(out):
        return out
    info = box_for(name)
    if not info.get('ok') or not info.get('boxes') or not info.get('path'):
        return None
    # the crop's own confidence picks its detection, as everywhere else
    want = round(int(m.group(3)) / 100.0, 2)
    box = info['boxes'][0]
    for b in info['boxes']:
        if abs(b['conf'] - want) <= 0.006:
            box = b
            break
    try:
        from PIL import Image
        im = Image.open(info['path'])
        im.draft('RGB', im.size)          # no-op scale, but primes the decoder
        w, h = im.size
        bw, bh = box['x2'] - box['x1'], box['y2'] - box['y1']
        pad = HQ_PAD * max(bw, bh)
        r = im.convert('RGB').crop((
            int(max(0, box['x1'] - pad)), int(max(0, box['y1'] - pad)),
            int(min(w, box['x2'] + pad)), int(min(h, box['y2'] + pad))))
        if max(r.size) > HQ_MAX:
            r.thumbnail((HQ_MAX, HQ_MAX), Image.LANCZOS)
        os.makedirs(HQ_DIR, exist_ok=True)
        tmp = out + '.part'
        r.save(tmp, 'JPEG', quality=92, optimize=True)
        os.replace(tmp, out)
        return out
    except Exception as e:
        sys.stderr.write('hq_crop(%s): %s\n' % (name, e))
        return None


_hq_lock = threading.Lock()
_hq_busy = False


def warm_hq(names):
    """Pre-generate HQ crops for a page in the background.

    A cold tile costs ~0.45 s (decode a 12 MP original, crop, re-encode). The
    browser fetches six at a time, so a cold page is a few seconds -- fine
    once, irritating every page. Warming the page AND its reserve means only
    the very first page is ever slow. One worker at a time: this competes with
    the sweep for the same drives.
    """
    global _hq_busy
    with _hq_lock:
        if _hq_busy:
            return
        todo = [n for n in names
                if not os.path.exists(os.path.join(HQ_DIR, n))][:120]
        if not todo:
            return
        _hq_busy = True

    def work():
        global _hq_busy
        try:
            for n in todo:
                try:
                    hq_crop(n)
                except Exception:
                    pass
            # drop cached cuts whose crop has aged out of the rolling pool --
            # the preview writer prunes recent_crops, nothing pruned this
            try:
                pool = set(os.listdir(CROPS))
                for f in os.listdir(HQ_DIR):
                    if f not in pool:
                        try:
                            os.remove(os.path.join(HQ_DIR, f))
                        except OSError:
                            pass
            except OSError:
                pass
        finally:
            _hq_busy = False

    threading.Thread(target=work, daemon=True).start()


def _saved_box(name):
    """The most recent correction for a crop, if any."""
    out = None
    try:
        with open(BOX_LABELS) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get('crop') == name:
                    out = r          # last write wins
    except OSError:
        pass
    return out


def save_box(name, det_idx, box, now=None):
    """Record a corrected box in ORIGINAL pixels. Append-only, last wins."""
    m = _CROP_RE.match(name or '')
    if not m:
        return {'ok': False, 'error': 'malformed crop name'}
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return {'ok': False, 'error': 'box must be four numbers'}
    if not all(v == v and abs(v) < 1e7 for v in (x1, y1, x2, y2)):
        return {'ok': False, 'error': 'box out of range'}   # NaN/inf guard
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return {'ok': False, 'error': 'box is degenerate'}
    rec = {'crop': name, 'image_id': m.group(2),
           'det_idx': int(det_idx or 0),
           'x1': round(x1, 2), 'y1': round(y1, 2),
           'x2': round(x2, 2), 'y2': round(y2, 2),
           'saved_at': int(time.time() if now is None else now)}
    try:
        with _box_lock:
            os.makedirs(BOX_DIR, exist_ok=True)
            with open(BOX_LABELS, 'a') as f:
                f.write(json.dumps(rec) + '\n')
                f.flush()
                os.fsync(f.fileno())
    except OSError as e:
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'saved': rec}

# ── "already looked at" ledger ──────────────────────────────────────────────
# Flagging records the NEGATIVE decision. Without recording the positive one
# too, every crop judged "yes that's a dog" stays in the pool forever, so each
# visit to the review page reopens the same dogs -- the queue never advances.
# Paging past a screen IS the decision: everything on it that was not flagged
# was judged a dog.
SEEN_FILE = os.path.join(HN_DIR, 'reviewed.jsonl')
_seen_lock = threading.Lock()
_seen = None  # set of image_ids judged and kept


def _load_seen():
    """image_ids already reviewed-and-kept. Call under ``_seen_lock``."""
    out = set()
    try:
        with open(SEEN_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # torn final line from a crash mid-append
                if isinstance(rec, dict) and rec.get('image_id'):
                    out.add(str(rec['image_id']))
    except OSError:
        pass
    return out


def _seen_ids():
    global _seen
    if _seen is None:
        _seen = _load_seen()
    return _seen


def reset_seen(now=None):
    """Put every kept crop back in the review queue.

    The ledger is RENAMED, not unlinked: this discards review work that cannot
    be recreated by re-deriving anything, so a mis-click should cost a rename
    to undo rather than the decisions themselves. Flags live in a separate
    file and are never touched -- they are training data.
    """
    now = int(time.time() if now is None else now)
    with _seen_lock:
        n = len(_seen_ids())
        bak = None
        if os.path.exists(SEEN_FILE):
            bak = '%s.%d.bak' % (SEEN_FILE, now)
            try:
                os.replace(SEEN_FILE, bak)
            except OSError as e:
                return {'ok': False, 'error': str(e), 'seen_total': n}
        globals()['_seen'] = set()
        return {'ok': True, 'restored': n, 'seen_total': 0,
                'backup': os.path.basename(bak) if bak else None}


def mark_seen(names, now=None):
    """Record crops as reviewed-and-kept. Keyed by image_id, not crop name, so
    a cell twin of the same photo cannot come back on a later page."""
    now = int(time.time() if now is None else now)
    added = 0
    with _seen_lock:
        cur = _seen_ids()
        new = []
        for nm in (names or []):
            m = _CROP_RE.match(str(nm) or '')
            if not m:
                continue
            iid = m.group(2)
            if iid in cur:
                continue
            cur.add(iid)
            new.append({'image_id': iid, 'crop': str(nm), 'seen_at': now})
        if new:
            try:
                os.makedirs(HN_DIR, exist_ok=True)
                with open(SEEN_FILE, 'a') as f:
                    for r in new:
                        f.write(json.dumps(r) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                added = len(new)
            except OSError as e:
                for r in new:      # keep memory honest if the write failed
                    cur.discard(r['image_id'])
                return {'ok': False, 'error': str(e), 'seen_total': len(cur)}
        return {'ok': True, 'added': added, 'seen_total': len(cur)}

# ── dataset balance: how much more flagging buys a balanced gate dataset ────
# The crop dataset lives outside the repo (it is tens of thousands of jpgs),
# so its location is configuration with no sensible default. Unset, the panel
# says so instead of guessing. $DOGBIN_DATASET is the same variable
# tools/detect/eval_dogbin.py and rebuild_crop_dataset.py already read.
def dataset_dir():
    """Resolved per call, not frozen at import.

    A module constant here is what made repointing the config invisible to a
    server that had already started: load_cfg() could reload all it liked and
    this name still held the value read at import.
    """
    return cfg('dogbin_dataset', '', env='DOGBIN_DATASET')
DATASET_CLASSES = ('dog', 'not_dog')
# Crops that survive per FLAG, measured -- not guessed. Across the dogbin_v4
# build: 1,075 flags -> 714 harvested at full res -> 494 reached the dataset.
#
# The v3-era value was 0.829 and is now wrong by 1.8x, because a term was
# added between those two numbers: 30% of every harvest is reserved into
# data/dogbin_acceptance_set.json and NEVER enters training. Those crops are
# not wasted -- they are the only honest way to accept the gate -- but a
# tracker that ignores them tells the reviewer to flag ~900 more when the real
# figure is ~1,600, and the panel exists precisely to answer "how much longer".
#
# The rest of the shortfall is unchanged: a flag is matched to its own box
# (ambiguous ones dropped), crops under the 64px floor go, and the
# near-duplicate and per-sequence caps trim what is left.
FLAG_YIELD = 0.460


def dataset_balance():
    """Class counts in the built dataset + how much flagging is still needed.

    Answers the only two questions the review page cannot: how far from
    balanced the training data is, and whether the work left is flagging
    (negatives) or annotating (positives). Flagging only ever produces
    negatives, so while not_dog < dog the answer is always "keep flagging" --
    adding dogs would move the target further away.
    """
    DATASET_DIR = dataset_dir()
    if not DATASET_DIR:
        # Unconfigured is a distinct state from missing, and the fix differs.
        # Without this guard os.path.join('', 'train', 'dog') is RELATIVE and
        # would quietly count whatever happens to sit under the cwd.
        return {'ok': False, 'dataset': None, 'dog': 0, 'not_dog': 0,
                'error': 'set DOGBIN_DATASET (or "dogbin_dataset" in '
                         'tools/dashboard/dashboard.config.json) to the crop '
                         'dataset directory'}
    counts = {}
    for cls in DATASET_CLASSES:
        n = 0
        for split in ('train', 'val'):
            d = os.path.join(DATASET_DIR, split, cls)
            try:
                n += sum(1 for f in os.listdir(d) if f.endswith('.jpg'))
            except OSError:
                pass
        counts[cls] = n
    pos, neg = counts.get('dog', 0), counts.get('not_dog', 0)
    if not pos and not neg:
        # A missing or empty dataset dir used to fall through the arithmetic
        # below and report "Balanced" -- deficit is 0 when there is nothing to
        # balance. Absence of data must never render as success.
        # The name, never the path: this JSON is served to the browser over a
        # no-auth LAN endpoint, and an absolute path discloses the host layout.
        return {'ok': False, 'dataset': os.path.basename(DATASET_DIR),
                'error': 'dataset directory not found',
                'dog': 0, 'not_dog': 0}
    deficit = max(0, pos - neg)

    # flags recorded AFTER the dataset was built are not in it yet
    built = 0.0
    try:
        built = os.path.getmtime(
            os.path.join(DATASET_DIR, 'rebuild_manifest.json'))
    except OSError:
        pass
    def _fresh(path, want):
        k = 0
        try:
            with open(path) as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    if r.get('label') == want and \
                            float(r.get('flagged_at') or 0) > built:
                        k += 1
        except OSError:
            pass
        return k

    fresh = _fresh(HN_LABELS, FLAG_LABEL)
    fresh_pos = _fresh(_store_for(POS_LABEL)['labels'], POS_LABEL)

    pending = int(round(fresh * FLAG_YIELD))       # negatives already earned
    # hard positives join the DOG class, so they raise the bar rather than
    # closing it -- worth collecting, but the estimate must say so
    pending_pos = int(round(fresh_pos * FLAG_YIELD))
    still = max(0, (pos + pending_pos) - (neg + pending))
    return {
        'dataset': os.path.basename(DATASET_DIR),
        'dog': pos,
        'not_dog': neg,
        'ratio': round(pos / neg, 2) if neg else None,
        'deficit': deficit,
        'new_flags': fresh,
        'new_positive_flags': fresh_pos,
        'pending_negatives': pending,
        'pending_positives': pending_pos,
        'flags_needed': int(-(-still // FLAG_YIELD)) if still else 0,
        'yield_per_flag': FLAG_YIELD,
        # so the panel can explain why the target is what it is, rather than
        # the number just doubling one day with no visible reason
        'reserved_ids': _reserved_count(),
        'balanced': still == 0,
    }


_reserved_cache = {'mtime': None, 'n': 0}


def _reserved_count():
    """How many flagged ids are withheld as the acceptance set.

    mtime-keyed, NOT lru_cache. reserve_acceptance_set.py --force rewrites this
    file, and a process-lifetime cache would leave a server that has been up
    for days quoting the old reservation forever -- the same staleness that
    made the review panel count 641 already-built flags as pending.
    """
    p = os.path.join(REPO, 'data', 'dogbin_acceptance_set.json')
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        _reserved_cache.update(mtime=None, n=0)
        return 0
    if _reserved_cache['mtime'] != mtime:
        try:
            with open(p) as fh:
                n = len(json.load(fh).get('image_ids') or [])
        except (OSError, ValueError):
            n = 0
        _reserved_cache.update(mtime=mtime, n=n)
    return _reserved_cache['n']


# ── country filter ──────────────────────────────────────────────────────────
COUNTRY_INDEX = os.path.join(OUT, 'countries.json')
# seconds between incremental rebuilds. Well under the ~4 minute pool
# turnover, and an incremental pass is ~1.7s, so the duty cycle is ~1%.
COUNTRY_REFRESH = 120
_country_cache = {'mtime': None, 'doc': {'by_image': {}, 'counts': {},
                                         'names': {}}}


def country_index():
    """Reloaded on mtime, like load_cfg -- the hourly rebuild has to reach a
    server that has been up for days, which an lru_cache would prevent."""
    try:
        mtime = os.path.getmtime(COUNTRY_INDEX)
    except OSError:
        return _country_cache['doc']
    if _country_cache['mtime'] != mtime:
        try:
            with open(COUNTRY_INDEX) as fh:
                doc = json.load(fh)
            _country_cache.update(mtime=mtime, doc=doc)
        except (OSError, ValueError) as e:
            sys.stderr.write(f'warning: bad {COUNTRY_INDEX}: {e}\n')
    return _country_cache['doc']


def refresh_countries():
    """Rebuild the index in-process. Failure is never fatal: the filter is a
    convenience and the queue must stay reviewable without it."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import country_index as ci
        ci.build(REPO, COUNTRY_INDEX)
    except Exception as e:                       # geopandas missing, lock, ...
        sys.stderr.write(f'country index refresh failed: {e}\n')


# One name for the default, because it was written in three places and the
# request handler's copy shadowed the one inside review_payload -- the page
# said "least confident" while the API still answered newest-first.
REVIEW_SORT_DEFAULT = 'low'
REVIEW_SORTS = {
    'new': lambda c: -c['ts'],
    'conf': lambda c: (-c['conf'], -c['ts']),
    'low': lambda c: (c['conf'], -c['ts']),
}


# ── crops too small for anyone to judge ─────────────────────────────────────
# /hq cuts each tile from the ORIGINAL at native box resolution, so what the
# reviewer sees is the box's true pixel size. A 17x15 box in a 1920x1080 frame
# is ~255 pixels of animal blown up to 640 -- not a rendering problem, and not
# a judgement any person or model can make. Asking for one is worse than
# skipping it: a coin-flip "not a dog" can turn the whole frame into a
# detector NEGATIVE, teaching it to miss a dog it did find.
#
# Off by default (0). A fresh clone behaves exactly as before.
_SZ = {'at': 0.0, 'by_key': {}}
SZ_TTL = 240          # the pool turns over every ~4 minutes


def min_review_px():
    return max(0, cfg_int('review_min_px', 0, env='REVIEW_MIN_PX'))


def box_short_sides(keys):
    """{(image_id, conf2): shorter side in ORIGINAL pixels}.

    One query for the whole page, not two per crop like box_for. Keyed on
    (image_id, conf x100) because that is how a crop filename names its own
    detection everywhere else in this file -- an image with a big box and a
    tiny one must not have the tiny crop judged by the big box's size.
    """
    now = time.time()
    if now - _SZ['at'] > SZ_TTL:
        _SZ['by_key'].clear()
        _SZ['at'] = now
    want = [k for k in keys if k not in _SZ['by_key']]
    if want:
        ids = sorted({i for i, _ in want if _ID_RE.match(i)})
        if ids:
            try:
                _, det = _detect_sql()
                lst = ','.join("'" + i + "'" for i in ids)
                con = duckdb.connect()
                rows = con.execute(
                    f"SELECT CAST(image_id AS VARCHAR), "
                    f"CAST(round(conf * 100) AS INT), "
                    f"min(least(x2 - x1, y2 - y1)) "
                    f"FROM {det} WHERE CAST(image_id AS VARCHAR) IN ({lst}) "
                    f"GROUP BY 1, 2").fetchall()
                con.close()
                for iid, c2, side in rows:
                    _SZ['by_key'][(iid, int(c2))] = float(side)
            except Exception:
                # the store is the source; if it cannot be read the floor
                # simply does not apply. Never drop a crop on a failed lookup.
                pass
    return _SZ['by_key']


def review_payload(page=0, size=REVIEW_PAGE, sort=None, country=''):
    """Unflagged crops for the bulk-review page, paginated (§ bulk flagging).

    Flagged names are excluded server-side so a reload, a restart or a second
    browser can never resurface something already judged. One listdir over the
    pool (retention is 3000), no per-file stat.

    ``sort`` matters for what this page is actually FOR, and the default is
    'low' -- least confident first. The detector's low-confidence boxes are
    where it is least sure and where a human adds the most information: they
    are the crops most likely to be wrong in either direction, so judging them
    both cleans the pipeline and teaches the gate the most per click.

    'conf' (highest first) is still one option away, and it answers a
    different question -- the confident mistakes are the ones worth mining as
    hard negatives. It was the default until the queue grew a size floor;
    with unjudgeable crops now held back, the low end is worth a person's
    time again.

    ``reserve`` is the next slice after the page. The client consumes it to
    backfill a tile it just flagged, so the grid keeps a constant length
    without a round trip and without renumbering everything on screen.
    """
    size = 100 if int(size) >= 100 else REVIEW_PAGE
    page = max(0, int(page))
    sort = sort if sort in REVIEW_SORTS else REVIEW_SORT_DEFAULT
    key = REVIEW_SORTS[sort]
    # _flag_names(), never the raw global: _flagged is None until something
    # lazily loads the ledger, so reading it directly made /review 500 on a
    # freshly started server that had not yet served /api/detect/crops.
    with _flag_lock:
        flagged = set(_flag_names(FLAG_LABEL))
        positives = set(_flag_names(POS_LABEL))
        judged_ids = _all_flagged_ids()
        n_pos = len(positives)
    # the empty payload carries the dropdown too, or a reviewer who filters
    # down to zero crops loses the control that would let them filter back out
    _cd = country_index()
    empty = {'items': [], 'reserve': [], 'page': 0, 'size': size, 'sort': sort,
             'total_unflagged': 0, 'flagged_total': len(flagged), 'pages': 0,
             'positive_total': n_pos, 'seen_total': 0,
             'country': (country or '').upper(),
             'countries': [{'iso': i, 'name': _cd.get('names', {}).get(i, i),
                            'n': n}
                           for i, n in sorted((_cd.get('counts') or {}).items(),
                                              key=lambda kv: (-kv[1], kv[0]))],
             'countries_generated': _cd.get('generated')}
    try:
        names = os.listdir(CROPS)
    except OSError:
        return empty
    try:
        full = set(os.listdir(os.path.join(CROPS, 'full')))
    except OSError:
        full = set()
    # An image_id can legitimately produce SEVERAL crops: the harvest wrote the
    # same Mapillary image into more than one 5-degree cell (boundary overlap),
    # and the sweep enumerates per cell, so it detects on the same jpg once per
    # cell. Showing all of them wastes the reviewer's time on identical tiles.
    # Judge each image ONCE: keep one crop per image_id, and treat the image as
    # already judged if ANY of its crops was flagged -- otherwise flagging one
    # copy just promotes its twin into the queue.
    flagged_ids = {m.group(2) for m in
                   (_CROP_RE.match(nm) for nm in flagged) if m}
    with _seen_lock:
        seen_ids = set(_seen_ids())
    # either verdict, or a pass, takes the image out of the queue
    judged = judged_ids | seen_ids
    # The country of each crop, by point-in-polygon on the image's real
    # lat/lon (see country_index.py). '' means the lookup found no coordinates
    # or the point fell outside every polygon -- at sea, usually. Those crops
    # stay in the unfiltered queue; hiding them would silently shrink the
    # reviewable pool.
    cdoc = country_index()
    by_country = cdoc.get('by_image') or {}
    want = (country or '').upper()
    cands = []
    for name in names:
        m = _CROP_RE.match(name)
        if not m or name in flagged or name in positives:
            continue
        iid = m.group(2)
        if iid in judged:      # flagged, or already looked at and kept
            continue
        iso = by_country.get(iid, '')
        cands.append({'name': name, 'image_id': iid,
                      'ts': int(m.group(1)),
                      'conf': round(int(m.group(3)) / 100.0, 2),
                      'country': iso,
                      'has_full': name in full})
    # Hold back what cannot be judged. A crop whose geometry the store does
    # not have yet (~9.5% of the live pool, shard not committed) is NOT
    # dropped: unknown size is not the same fact as too small.
    too_small = 0
    floor = min_review_px()
    if floor > 0 and cands:
        sides = box_short_sides([(c['image_id'], int(round(c['conf'] * 100)))
                                 for c in cands])
        big = []
        for c in cands:
            side = sides.get((c['image_id'], int(round(c['conf'] * 100))))
            if side is not None and side < floor:
                too_small += 1
                continue
            big.append(c)
        cands = big
    cands.sort(key=key)          # sort first: the survivor is the best copy
    # NB: not `seen_ids` -- that name holds the reviewed-and-kept ledger above,
    # and reusing it here reported the page count as the reviewed total
    items, emitted = [], set()
    for c in cands:
        if c['image_id'] in emitted:
            continue
        emitted.add(c['image_id'])
        items.append(c)

    # ── collapse near-duplicates ───────────────────────────────────────────
    # One row per Mapillary sequence: consecutive frames of the same animal on
    # the same pass are one judgement, not eight. A crop whose sequence is not
    # cached yet stays visible (failing toward showing it twice, never toward
    # hiding a judgement) and gets queued for a background lookup.
    with _seq_lock:
        smap = dict(_seq_map())
    warm_sequences(getattr(BoardHandler, 'db', 'data/catalog.duckdb'),
                   [c['image_id'] for c in items])
    # A verdict covers the whole pass. Without this the collapse would be
    # cosmetic: judge the one crop on screen and its seven siblings simply
    # promote themselves into the next page.
    # a verdict covers the frames around it, not the whole recording session
    judged_seq = {}
    for i in judged:
        sq, ts = _seq_of(smap.get(i))
        if sq and ts is not None:
            judged_seq.setdefault(sq, []).append(ts)
    kept, seen_seq, hash_reps, collapsed = [], {}, [], 0

    def near(bucket, sq, ts):
        """Another frame of this pass already accounted for, within the window."""
        if ts is None:
            return sq in bucket          # no timestamp: fall back to per-session
        return any(abs(ts - t) <= SEQ_WINDOW_S for t in bucket.get(sq, []))

    for c in items:
        sq, ts = _seq_of(smap.get(c['image_id']))
        if sq and near(judged_seq, sq, ts):
            collapsed += 1
            continue
        if sq:
            if near(seen_seq, sq, ts):
                collapsed += 1
                continue
            seen_seq.setdefault(sq, []).append(ts if ts is not None else 0.0)
            c['seq'] = sq
        else:
            # no sequence known -- fall back to a perceptual hash so an
            # unresolved crop is still not shown twice
            h = _dhash(os.path.join(CROPS, c['name']), c['name'])
            if h is not None:
                if any(bin(h ^ r).count('1') <= 6 for r in hash_reps):
                    collapsed += 1
                    continue
                hash_reps.append(h)
        kept.append(c)

    # Tally, then filter -- in that order, and both AFTER the collapse.
    # Filtering at candidate time made the counts describe a larger population
    # than the queue: the per-image dedup and the sequence collapse run later
    # and remove more, so every option overstated (BRA advertised 1,460 and
    # returned 1,079). Counting what actually survives makes the number the
    # option shows identical to the number it delivers, by construction.
    offer, tallied = {}, set()
    for c in kept:
        if c['country'] and c['image_id'] not in tallied:
            tallied.add(c['image_id'])
            offer[c['country']] = offer.get(c['country'], 0) + 1
    coverage = (round(len(tallied) / len(kept), 3)) if kept else 0
    items = [c for c in kept if c['country'] == want] if want else kept
    total = len(items)
    pages = max(1, -(-total // size))
    page = min(page, pages - 1)
    lo = page * size
    warm_hq([c['name'] for c in items[lo:lo + 2 * size]])
    return {'items': items[lo:lo + size], 'reserve': items[lo + size:lo + 2 * size],
            'page': page, 'size': size, 'sort': sort, 'total_unflagged': total,
            # images judged, not crop FILES judged -- so it is the same unit as
            # total_unflagged and the two sum to a meaningful denominator
            'pages': pages, 'flagged_total': len(flagged_ids),
            # never a silent cap: the page says how many it is holding back
            'too_small': too_small, 'min_px': floor,
            'positive_total': n_pos, 'seen_total': len(seen_ids),
            'collapsed': collapsed,
            # Options tallied from the live queue, NOT from the index's
            # counts. The index spans the rolling pool plus both flag ledgers,
            # while this queue excludes everything already judged, kept, or
            # collapsed -- so index counts advertised crops that could never be
            # returned. Measured on the running server: 60 of 60 options were
            # dead, together promising 4,090 crops that did not exist. Counting
            # what actually survives makes an empty option impossible rather
            # than merely unintended.
            'country': want,
            'countries': [{'iso': i,
                           'name': cdoc.get('names', {}).get(i, i), 'n': n}
                          for i, n in sorted(offer.items(),
                                             key=lambda kv: (-kv[1], kv[0]))],
            'countries_generated': cdoc.get('generated'),
            # how much of the queue the index can currently place. The pool
            # turns over every ~4 minutes at sweep rate, so this is the number
            # that says whether the filter is usable at all.
            'country_coverage': coverage}


def sweep_pids():
    """PIDs of running sweep processes (never matches this server)."""
    out = []
    try:
        for d in os.listdir('/proc'):
            if not d.isdigit():
                continue
            try:
                with open(f'/proc/{d}/cmdline', 'rb') as f:
                    cl = f.read().decode('utf-8', 'replace')
            except OSError:
                continue
            if 'sweep.py' in cl and ' run' in cl.replace('\x00', ' '):
                if 'python' in cl:
                    out.append(int(d))
    except OSError:
        pass
    return out


def sweep_control(action):
    """stop = SIGTERM (graceful: commits its contiguous prefix, loses nothing);
    resume = relaunch detached. Resume is safe because the store's tiling
    resume replays only uncommitted ranges."""
    pids = sweep_pids()
    if action == 'stop':
        if not pids:
            return {'ok': True, 'running': False, 'msg': 'already stopped'}
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        return {'ok': True, 'running': False,
                'msg': 'stopping — the sweep commits what it has, then exits'}
    if action == 'resume':
        if pids:
            return {'ok': True, 'running': True, 'msg': 'already running'}
        py = os.environ.get('SWEEP_PYTHON', SWEEP_PYTHON)
        log = open(os.path.join(REPO, 'data', 'sweep_resume.log'), 'a')
        try:
            subprocess.Popen(
                [py, os.path.join(REPO, 'tools', 'detect', 'sweep.py'),
                 'run', '--gen', '1'],
                cwd=REPO, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                start_new_session=True)
        except Exception as e:
            return {'ok': False, 'running': False, 'msg': str(e)}
        return {'ok': True, 'running': True, 'msg': 'resuming'}
    return {'ok': False, 'msg': 'unknown action'}


class BoardHandler(SimpleHTTPRequestHandler):
    """Serve the static dashboard plus a tiny JSON board API."""

    db = 'data/catalog.duckdb'

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')  # live data (§7.2)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/board':
            try:
                self._json(board_payload())
            except Exception as e:
                self._json({'error': str(e)}, 500)
            return
        if self.path == '/api/refresh':
            self._json(_refresh)
            return
        # split('?') so cache-busting query strings still match (§7.2 —
        # /api/board's == match 404s on ?t=1 and that bit us before)
        if self.path.split('?', 1)[0] == '/api/detect':
            try:
                self._json(detect_payload())
            except Exception:
                self._json({'running': False})  # 404-safe by construction
            return
        if self.path.split('?', 1)[0] == '/api/review':
            try:
                q = {}
                if '?' in self.path:
                    # parse_qs is imported at module scope; a local import
                    # here made the NAME function-local, so every earlier use
                    # in do_GET raised UnboundLocalError
                    q = {k: v[0] for k, v in
                         parse_qs(self.path.split('?', 1)[1]).items()}
                self._json(review_payload(int(q.get('page', 0)),
                                          int(q.get('size', REVIEW_PAGE)),
                                          str(q.get('sort', REVIEW_SORT_DEFAULT)),
                                          str(q.get('country', ''))))
            except Exception as e:
                self._json({'items': [], 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/review':
            body = REVIEW_HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split('?', 1)[0] == '/api/review/count':
            # Just the queue depth. review_payload does one listdir over a
            # pool capped at 3000 plus the ledgers, so asking for size=0 is
            # the whole computation without serialising any crop.
            try:
                self._json({'left': review_payload(0, 0)['total_unflagged']})
            except Exception as e:
                self._json({'left': None, 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/training/run':
            try:
                q = parse_qs(urlparse(self.path).query)
                self._json({'html': render_run_detail(
                    q.get('key', [''])[0])})
            except Exception as e:
                self._json({'html': '', 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/training':
            # The page itself is a build artefact, so the live card, its
            # clock and the patience countdown were frozen until the next
            # hourly rebuild -- the one part of the page whose whole purpose
            # is to be current. Rendered here, in Python, so there is still
            # exactly one implementation of the section.
            try:
                self._json({'html': render_training()})
            except Exception as e:
                self._json({'html': '', 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/sweep':
            self._json({'running': bool(sweep_pids())})
            return
        if self.path.split('?', 1)[0] == '/api/review/box':
            q = parse_qs(urlparse(self.path).query)
            info = box_for((q.get('name', [''])[0]))
            # 'path' is for _orig_path only -- the browser has no use for a
            # server filesystem path and every reason not to be handed one
            info.pop('path', None)
            self._json(info)
            return
        if self.path.split('?', 1)[0] == '/hq':
            q = parse_qs(urlparse(self.path).query)
            p = hq_crop((q.get('name', [''])[0]))
            if not p:
                self.send_error(404)   # client falls back to the preview
                return
            try:
                with open(p, 'rb') as f:
                    body = f.read()
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'private, max-age=86400')
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split('?', 1)[0] == '/orig':
            q = parse_qs(urlparse(self.path).query)
            p = _orig_path((q.get('name', [''])[0]))
            if not p:
                self.send_error(404)
                return
            try:
                with open(p, 'rb') as f:
                    body = f.read()
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'private, max-age=300')
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split('?', 1)[0] == '/api/dataset':
            try:
                self._json(dataset_balance())
            except Exception as e:
                self._json({'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/detect/crops':
            try:
                self._json(crops_payload())
            except Exception:  # never 500 on a cosmetic grid
                self._json({'crops': [], 'total_last_min': 0})
            return
        if self.path.startswith('/api/commands'):
            q = parse_qs(urlparse(self.path).query)
            region = (q.get('region', [''])[0]).strip()
            payload = commands_payload(region, self.db)
            if payload:
                self._json(payload)
            else:
                self._json({'error': 'unknown region'}, 404)
            return
        super().do_GET()

    def do_POST(self):
        if self.path == '/api/board':
            try:
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                ok = set_stage(data.get('region', ''), data.get('stage', ''))
                self._json({'ok': ok}, 200 if ok else 400)
            except Exception as e:
                self._json({'error': str(e)}, 500)
            return
        if self.path.split('?', 1)[0] == '/api/review/count':
            # Just the queue depth. review_payload does one listdir over a
            # pool capped at 3000 plus the ledgers, so asking for size=0 is
            # the whole computation without serialising any crop.
            try:
                self._json({'left': review_payload(0, 0)['total_unflagged']})
            except Exception as e:
                self._json({'left': None, 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/training/run':
            try:
                q = parse_qs(urlparse(self.path).query)
                self._json({'html': render_run_detail(
                    q.get('key', [''])[0])})
            except Exception as e:
                self._json({'html': '', 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/training':
            # The page itself is a build artefact, so the live card, its
            # clock and the patience countdown were frozen until the next
            # hourly rebuild -- the one part of the page whose whole purpose
            # is to be current. Rendered here, in Python, so there is still
            # exactly one implementation of the section.
            try:
                self._json({'html': render_training()})
            except Exception as e:
                self._json({'html': '', 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/sweep':
            try:
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                self._json(sweep_control(str(data.get('action') or '')))
            except Exception as e:
                self._json({'ok': False, 'msg': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/review/box':
            try:
                n = int(self.headers.get('Content-Length', 0) or 0)
                d = json.loads(self.rfile.read(n) or b'{}')
                self._json(save_box(str(d.get('name') or ''),
                                    d.get('det_idx'),
                                    d.get('box') or []))
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/review/seen':
            try:
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                if isinstance(data, dict) and data.get('reset'):
                    self._json(reset_seen())
                    return
                names = data.get('names') if isinstance(data, dict) else None
                self._json(mark_seen(names or []))
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/detect/flag':
            try:
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                if not isinstance(data, dict):
                    raise ValueError('body is not an object')
                body, code = flag_crop(str(data.get('name') or ''),
                                       str(data.get('label') or FLAG_LABEL),
                                       bool(data.get('undo')))
            except Exception as e:  # a cosmetic button never 500s
                body, code = {'ok': False, 'error': str(e)}, 200
            self._json(body, code)
            return
        if self.path == '/api/refresh':
            started = trigger_refresh(self.db)
            self._json({'started': started, 'running': _refresh['running']})
            return
        self.send_error(404)

    def log_message(self, *a):
        pass


def render(ov, per, tr, now, locs=()):
    """Render the full dashboard HTML."""
    # 'hot' marks the number that implies work left to do -- at 97% complete the
    # interesting figure is the remainder, not the headline total.
    # Both operands are distinct image_ids (see distinct_counts). When they are
    # not, there is no honest subtraction to do and the cell shows nothing
    # rather than the row-count-minus-id-count figure it used to.
    remaining = ov['dogs'] - ov['downloaded']
    rem = (human(remaining) if remaining > 0 else '0') if ov.get('exact') else '--'
    kpis = [
        ('Images scanned', human(ov['all_data']), ''),
        ('Ground animals', human(ov['dogs']), ''),
        ('Downloaded', human(ov['downloaded']), ' ok'),
        ('Remaining', rem, ' hot' if remaining > 0 and ov.get('exact') else ' ok'),
        ('Regions', str(ov['regions']), ''),
        ('Drives', str(ov['drives']), ''),
    ]
    kpi_html = ''.join(
        f'<div class="kpi{cls}"><div class="kpi-label">{lab}</div>'
        f'<div class="kpi-val">{val}</div></div>' for lab, val, cls in kpis)

    cards = ''.join(f'''<div class="rcard">
  <div class="rtop"><span class="rname">{p['region']}</span>
    <span class="rpct" style="color:{bar_color(p['pct'])}">{p['pct']:.0f}%</span></div>
  <div class="bar"><div class="fill" style="width:{min(p['pct'],100):.1f}%;background:{bar_color(p['pct'])}"></div></div>
  <div class="rmeta"><span>{human(p['downloaded'])} / {human(p['dogs'])} ground animals</span>
    <span class="b">{human(p['all_data'])} imgs</span></div>
</div>''' for p in per)

    data = {
        'regions': [p['region'] for p in per][::-1],
        'dogs': [p['dogs'] for p in per][::-1],
    }
    page = (TEMPLATE
            .replace('__KPIS__', kpi_html)
            .replace('__CARDS__', cards)
            .replace('__LOCS__', render_locations(locs))
            .replace('__DATA__', json.dumps(data))
            .replace('__NOW__', now.strftime('%Y-%m-%d %H:%M'))
            .replace('__PROG__', f"{ov['pct']:.1f}")
            .replace('__LB_CSS__', LB_CSS)
            .replace('__LB_HTML__', LB_HTML)
            .replace('__LB_JS__', LB_JS)
            .replace('__MODELS__', render_models())
            .replace('__TRAINING__', render_training())
            .replace('__STOREPATH__', render_store_path()))
    # A placeholder that survives is not cosmetic: __LB_JS__ left inside the
    # <script> is a syntax error that kills EVERY handler on the page (charts,
    # polling, folds, the sweep control) in one shot. Fail the build instead --
    # build() then never writes index.html, so the last good page stays served.
    missed = sorted(set(re.findall(r'__[A-Z][A-Z0-9_]*__', page)))
    if missed:
        raise RuntimeError('render(): unsubstituted template placeholder(s) '
                           + ', '.join(missed))
    return page


def build(args):
    """Refresh (optionally) the catalog, snapshot history, write index.html."""
    if not getattr(args, 'no_refresh', False):
        subprocess.run([sys.executable, CATALOG, 'refresh'],
                       cwd=REPO,
                       check=False)
        if getattr(args, 'images', False):
            subprocess.run([sys.executable, CATALOG, 'images'],
                           cwd=REPO,
                           check=False)
    now = datetime.now()
    ov, per = query_metrics(args.db)
    record_history(per, now)
    write_board_stats(per, args.db)
    if (getattr(args, 'images', False) and not getattr(
            args, 'no_refresh', False)) or not os.path.exists(MAP_FILE):
        try:
            build_map_points()
        except Exception as e:
            print('map build error:', e)
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, 'echarts.min.js')
    if os.path.exists(ECHARTS_SRC) and not os.path.exists(dst):
        shutil.copy(ECHARTS_SRC, dst)
    with open(os.path.join(OUT, 'index.html'), 'w') as f:
        f.write(render(ov, per, trend(), now, region_locations(args.db)))
    print(f"[{now:%H:%M:%S}] dashboard built · {human(ov['downloaded'])}/"
          f"{human(ov['dogs'])} downloaded ({ov['pct']:.1f}%) · "
          f"{os.path.join(OUT, 'index.html')}")


def serve(args):
    """Build once, serve the dashboard dir, and refresh on an interval."""
    os.makedirs(OUT, exist_ok=True)
    try:
        build(
            argparse.Namespace(db=args.db,
                               no_refresh=args.no_initial_refresh,
                               images=True))
    except Exception as e:
        # A failed FIRST build must not stop the server. The catalog takes an
        # exclusive lock, so any maintenance job holding it (a refresh, an
        # --with-size image scan) made the dashboard refuse to start at all --
        # turning a transient lock into an outage. Serve the last good
        # index.html and let the interval loop rebuild when the lock clears.
        print(f'initial build failed ({e}); serving the existing page',
              file=sys.stderr)
        if not os.path.exists(os.path.join(OUT, 'index.html')):
            raise

    def loop():
        cyc = 1
        while True:
            time.sleep(args.interval)
            _do_build(
                argparse.Namespace(db=args.db,
                                   no_refresh=False,
                                   images=(cyc % args.images_every == 0)))
            cyc += 1

    # The country index gets its OWN cadence, not the hourly build's. At sweep
    # rate the 3000-crop pool turns over about every 4 minutes, so an hourly
    # index described a queue that no longer existed: measured on the running
    # server, every one of 50 returned crops had no country at all. An
    # incremental rebuild costs ~1.7s, so this was never a cost trade -- just
    # the wrong number.
    def country_loop():
        while True:
            refresh_countries()
            time.sleep(COUNTRY_REFRESH)

    threading.Thread(target=country_loop, daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()
    BoardHandler.db = args.db
    handler = functools.partial(BoardHandler, directory=OUT)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving dashboard on http://{args.host}:{args.port}/ "
          f"(refresh every {args.interval // 60} min)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


def main():
    """Parse the CLI and dispatch to build / serve."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db',
                   default='data/catalog.duckdb',
                   help='Catalog DuckDB file (default: data/catalog.duckdb).')
    sub = p.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build',
                       help='Refresh + regenerate the dashboard once.')
    b.add_argument('--no-refresh',
                   action='store_true',
                   help='Use the existing catalog (skip the drive scan).')
    b.add_argument('--images',
                   action='store_true',
                   help='Also re-scan downloaded jpg counts (slow).')
    b.set_defaults(func=build)

    s = sub.add_parser('serve', help='Serve + auto-refresh on an interval.')
    s.add_argument('--host',
                   default='0.0.0.0',
                   help='Bind address (e.g. a Tailscale IP). Default 0.0.0.0.')
    s.add_argument('--port', type=int, default=8050)
    s.add_argument('--interval',
                   type=int,
                   default=3600,
                   help='Seconds between refreshes (default 3600 = hourly).')
    s.add_argument('--images-every',
                   type=int,
                   default=1,
                   help='Re-scan jpg counts every N cycles (default 1; the '
                   'scan is incremental, so hourly is cheap).')
    s.add_argument('--no-initial-refresh',
                   action='store_true',
                   help='Skip the catalog scan on the very first build.')
    s.set_defaults(func=serve)

    args = p.parse_args()
    try:
        args.func(args)
    except CatalogMissing as e:
        sys.exit(str(e))


# ── shared lightbox ─────────────────────────────────────────────────────────
# The full frame with the detection box ALREADY BAKED IN by PreviewWriter —
# nothing is drawn client-side, there is no canvas here. Both the dashboard's
# live grid and the /review page mount the same markup, CSS and component, so
# the overlay can only ever look and behave one way.
LB_CSS = """
/* above the sticky header (20) and the toast (9) -- not tied with either */
.lb{position:fixed;inset:0;z-index:60;background:rgba(0,0,0,.85);
  display:flex;align-items:center;justify-content:center;flex-direction:column;gap:12px;padding:24px}
.lb[hidden]{display:none}
.lb img{max-width:92vw;max-height:88vh;object-fit:contain;border:1px solid var(--bd);
  border-radius:12px;background:var(--bg);display:block}
.lbcap{color:var(--mut);font-size:12.5px;font-variant-numeric:tabular-nums;text-align:center}
.lbfoot{display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap}
/* the flag button sits in the footer row, so scope the absolute nav buttons
   to DIRECT children of .lb only */
.lb>.rbtn{position:absolute;padding:3px 11px;font-size:15px;line-height:1.35;background:rgba(33,38,45,.9)}
.lb>.rbtn:hover{background:rgba(232,166,69,.24)}
.lbflag{padding:3px 11px;font-size:11.5px}
.lbflag.on{background:#d8743a;border-color:#d8743a;color:#15100a}
.lbflag.on:hover{background:#c9682f}
.lbx{top:16px;right:18px}
.lbprev{left:14px;top:50%;transform:translateY(-50%)}
.lbnext{right:14px;top:50%;transform:translateY(-50%)}"""

# Lives at body level on both pages: it is a fixed overlay, and nesting it in
# a panel would let that panel's own hide/refresh yank it away mid-view.
LB_HTML = """
<div class="lb" id="cropLb" hidden role="dialog" aria-modal="true" aria-label="full frame with detection box">
  <button class="rbtn lbx" id="cropLbClose" title="close (Esc)" aria-label="close">✕</button>
  <button class="rbtn lbprev" id="cropLbPrev" title="previous detection (←)" aria-label="previous">‹</button>
  <button class="rbtn lbnext" id="cropLbNext" title="next detection (→)" aria-label="next">›</button>
  <img id="cropLbImg" alt="full frame with the detection box drawn">
  <div class="lbfoot">
    <div class="lbcap" id="cropLbCap"></div>
    <button class="rbtn lbflag" id="cropLbFlag" title="flag as false positive">⚑ not a dog</button>
  </div>
</div>"""

# cfg.flagged(name) -> bool and cfg.toggle(name) let the host page own the flag
# state; the component only reflects and requests it.
LB_JS = r"""
function makeLightbox(cfg){
  cfg=cfg||{};
  var lb=document.getElementById('cropLb'),
      img=document.getElementById('cropLbImg'),
      cap=document.getElementById('cropLbCap'),
      bx=document.getElementById('cropLbClose'),
      bp=document.getElementById('cropLbPrev'),
      bn=document.getElementById('cropLbNext'),
      bf=document.getElementById('cropLbFlag');
  var list=[],at=-1,prevOv='';
  var ON='flagged as false positive — click to undo',
      OFF='flag as false positive (not a dog)';
  function sync(){
    if(!bf)return;
    var c=(at>=0&&list[at])||null;
    if(!c){bf.style.display='none';return}
    bf.style.display='';
    var on=!!(cfg.flagged&&cfg.flagged(c.name));
    bf.className='rbtn lbflag'+(on?' on':'');
    bf.textContent=on?'⚑ flagged — undo':'⚑ not a dog';
    bf.title=on?ON:OFF;
  }
  function show(i){
    if(!list.length)return;
    at=(i%list.length+list.length)%list.length;   /* wraps both ways */
    var c=list[at];
    if(img)img.src='/recent_crops/full/'+encodeURIComponent(''+c.name);
    if(cap)cap.textContent='image_id '+c.image_id+' · conf '+
      (+c.conf||0).toFixed(2)+' · '+Math.max(0,Math.round(+c.age_s||0))+'s ago';
    var many=list.length>1?'':'none';   /* one crop: nothing to step to */
    if(bp)bp.style.display=many;
    if(bn)bn.style.display=many;
    sync();
  }
  /* The caller hands over a SNAPSHOT: a refresh on the host page can neither
     close the overlay nor swap the picture out from under the arrow keys. */
  function open(items,i){
    if(!lb||!items||!items.length)return;
    list=items.slice();
    show(i);
    lb.hidden=false;
    prevOv=document.body.style.overflow||'';
    document.body.style.overflow='hidden';        /* freeze the page behind */
    if(bx&&bx.focus)bx.focus();
  }
  function close(){
    if(!lb||lb.hidden)return;
    lb.hidden=true;
    if(img&&img.removeAttribute)img.removeAttribute('src');   /* stop load */
    document.body.style.overflow=prevOv;          /* restore, don't assume '' */
    list=[];at=-1;
  }
  function step(d){if(lb&&!lb.hidden&&list.length)show(at+d)}
  if(lb)lb.addEventListener('click',function(e){if(e.target===lb)close()});
  if(bx)bx.addEventListener('click',function(){close()});
  if(bp)bp.addEventListener('click',function(){step(-1)});
  if(bn)bn.addEventListener('click',function(){step(1)});
  if(bf)bf.addEventListener('click',function(e){
    if(e.stopPropagation)e.stopPropagation();
    var c=(at>=0&&list[at])||null;
    if(c&&cfg.toggle)cfg.toggle(c.name);
  });
  document.addEventListener('keydown',function(e){
    if(!lb||lb.hidden)return;              /* keys are ours only when open */
    if(e.key==='Escape')close();
    else if(e.key==='ArrowLeft')step(-1);
    else if(e.key==='ArrowRight')step(1);
    else return;
    e.preventDefault();
  });
  return {open:open,close:close,step:step,sync:sync,
          list:function(){return list},at:function(){return at}};
}"""


TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="3600">
<title>Street Dogs — Collection Progress</title>
<script src="echarts.min.js"></script>
<style>
:root{--bg:#13151a;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;
--gap:22px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:radial-gradient(1100px 560px at 72% -12%,#1d222b 0%,#13151a 56%) fixed,#13151a;
color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
/* no top padding: the sticky header supplies its own, and any gap above it
   shows as a dead band once the header has a background of its own */
line-height:1.5;padding:0 24px 56px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1300px;margin:0 auto}
/* one spacing scale instead of per-section inline margins */
.sec{margin-bottom:var(--gap)}
/* the page is ~7 sections tall; keep identity + the refresh control reachable */
header{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:12px;
  margin:0 -24px 20px;padding:18px 24px 15px;position:sticky;top:0;z-index:20;
  background:rgba(19,21,26,.82);backdrop-filter:saturate(140%) blur(12px);
  border-bottom:1px solid var(--bd)}
h1{font-size:clamp(21px,3vw,29px);font-weight:650;letter-spacing:-.4px}
h1 .o{color:var(--acc)}
.sub{color:var(--dim);font-size:13px;margin-top:3px}
.upd{display:flex;align-items:center;gap:7px;color:var(--mut);font-size:12.5px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2.4s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(67,181,129,.5)}70%{box-shadow:0 0 0 7px rgba(67,181,129,0)}100%{box-shadow:0 0 0 0 rgba(67,181,129,0)}}
/* The one solid-filled control on the page. Everything else is a tint or an
   outline, so filling exactly one thing makes it unambiguously THE action --
   and the action this dashboard exists to send someone to is the queue.
   #13151a on #e8a645 is the page background on the page accent: 8.9:1. */
.hact{display:flex;align-items:center;flex-wrap:wrap;gap:10px 16px;
justify-content:flex-end}
.revbtn{display:inline-flex;align-items:center;gap:10px;text-decoration:none;
background:var(--acc);color:#13151a;border:1px solid var(--acc);
border-radius:10px;padding:7px 12px 7px 13px;line-height:1.1;
transition:transform .12s ease,box-shadow .12s ease,background .12s ease}
.revbtn:hover{background:#f0b45c;transform:translateY(-1px);
box-shadow:0 6px 16px rgba(232,166,69,.18)}
.revbtn:focus-visible{outline:2px solid var(--acc);outline-offset:3px}
.revbtn .rvf{font-size:15px;opacity:.7}
.revbtn .rvn{display:flex;flex-direction:column}
.revbtn b{font-size:17px;font-weight:680;letter-spacing:-.35px;
font-variant-numeric:tabular-nums}
.revbtn em{font-style:normal;font-size:10px;font-weight:600;opacity:.68;
letter-spacing:.04em;text-transform:uppercase}
/* An empty queue is not an achievement to celebrate in the accent colour; it
   is a page with nothing to do, and it should read that way. */
.revbtn.quiet{background:transparent;color:var(--mut);
border-color:var(--bd);box-shadow:none}
.revbtn.quiet:hover{background:rgba(130,140,150,.08);transform:none;
box-shadow:none}
.revbtn.quiet b{font-size:13px;font-weight:600}
@media (prefers-reduced-motion:reduce){
  .revbtn,.revbtn:hover{transition:none;transform:none}
}
.rbtn{background:rgba(232,166,69,.14);border:1px solid rgba(232,166,69,.35);color:var(--acc);border-radius:8px;padding:4px 11px;font-size:12px;font-weight:600;cursor:pointer;transition:background .12s;font-variant-numeric:tabular-nums}
.rbtn:hover{background:rgba(232,166,69,.24)}
.rbtn:disabled{cursor:default;opacity:.85}
.rbtn:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.rbtn.nav{text-decoration:none;display:inline-flex;align-items:center}
/* ── sweep control: lives in the Detection sweep header, beside what it acts
   on, not in the page-meta line. State is spelled out in the pill so the
   button label is never the only thing carrying it. ── */
.swctl{margin-left:auto;align-self:center;display:flex;align-items:center;gap:9px}
.swpill{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;
  letter-spacing:.04em;text-transform:uppercase;color:var(--dim);
  border:1px solid var(--bd);border-radius:999px;padding:3px 10px}
.swpill::before{content:'';width:6px;height:6px;border-radius:50%;background:currentColor;flex:none}
.swpill.on{color:var(--green);border-color:rgba(67,181,129,.32);background:rgba(67,181,129,.09)}
.swpill.on::before{animation:pulse 2.4s infinite}
.swpill.off{color:var(--mut);background:rgba(130,140,150,.08)}
/* stopping a multi-day GPU job is not a primary action: neutral until hover */
.rbtn.sw{background:rgba(130,140,150,.1);border-color:rgba(130,140,150,.28);color:var(--mut)}
.rbtn.sw.stop:hover{background:rgba(216,116,58,.18);border-color:rgba(216,116,58,.5);color:#e0864f}
.rbtn.sw.go{background:rgba(67,181,129,.13);border-color:rgba(67,181,129,.4);color:var(--green)}
.rbtn.sw.go:hover{background:rgba(67,181,129,.22)}
.spin{display:inline-block;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.ring-row{display:flex;flex-wrap:wrap;gap:26px;align-items:center;margin-bottom:18px;
background:linear-gradient(180deg,#1d222a,#181c22);border:1px solid var(--bd);border-radius:18px;padding:24px 28px}
.ring{position:relative;width:140px;height:140px;flex:none}
.ring svg{transform:rotate(-90deg)}
.ring .ctr{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.ring .ctr b{font-size:27px;font-weight:680;color:var(--acc);letter-spacing:-.5px}
.ring .ctr span{font-size:10.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em;margin-top:2px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(116px,1fr));gap:12px;flex:1;min-width:280px}
.kpi{background:var(--panel2);border:1px solid rgba(130,140,150,.07);border-radius:13px;padding:14px 16px}
.kpi-label{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim)}
.kpi-val{font-size:23px;font-weight:650;margin-top:5px;letter-spacing:-.5px;font-variant-numeric:tabular-nums}
/* semantic accent: what is left to do reads differently from what is done */
.kpi.hot{border-color:rgba(232,166,69,.3);background:linear-gradient(180deg,rgba(232,166,69,.09),transparent),var(--panel2)}
.kpi.hot .kpi-val{color:var(--acc)}
.kpi.ok .kpi-val{color:var(--green)}
.grid2{display:grid;grid-template-columns:1.4fr 1fr;gap:18px;margin-bottom:22px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.panel{background:linear-gradient(180deg,#1c2128,#181c22);border:1px solid var(--bd);border-radius:18px;padding:20px 22px}
.phead{display:flex;align-items:center;gap:9px;margin-bottom:12px}
.phead b{font-size:13.5px;font-weight:620;color:var(--tx)}
.phead i{width:9px;height:9px;border-radius:3px;background:var(--acc)}
.phint{font-size:11px;color:var(--dim);font-weight:400}
.chart{width:100%;height:430px}
.mapwrap{position:relative}
.mapgate{position:absolute;inset:0;z-index:2;display:flex;align-items:center;
justify-content:center;cursor:pointer;border-radius:12px;
background:rgba(19,21,26,.32);transition:background .15s,opacity .15s}
.mapgate:hover{background:rgba(19,21,26,.2)}
.mapgate[hidden]{display:none}
.mapgb{background:rgba(19,21,26,.9);border:1px solid var(--bd);border-radius:10px;
padding:8px 14px;font-size:12.5px;color:var(--mut);pointer-events:none}
.mapgate:hover .mapgb{color:var(--tx);border-color:rgba(232,166,69,.4)}
.maplock{position:absolute;right:12px;top:12px;z-index:3}
.maplock[hidden]{display:none}
.sect{display:flex;align-items:baseline;gap:10px;font-size:15px;font-weight:620;margin:8px 0 14px}
.sect span{font-size:12.5px;font-weight:400;color:var(--dim)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(292px,1fr));gap:13px}
.rcard{background:linear-gradient(180deg,#1c2128,#181c22);border:1px solid var(--bd);border-radius:14px;padding:15px 17px;transition:border-color .15s,transform .15s}
.rcard:hover{border-color:rgba(232,166,69,.4);transform:translateY(-2px)}
.rtop{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;gap:8px}
.rname{font-size:14.5px;font-weight:620}
.rpct{font-size:14px;font-weight:680;font-variant-numeric:tabular-nums}
.bar{height:8px;border-radius:6px;background:rgba(130,140,150,.16);overflow:hidden}
.fill{height:100%;border-radius:6px;transition:width .5s ease}
.rmeta{display:flex;justify-content:space-between;margin-top:10px;font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums;gap:8px}
.rmeta .b{color:var(--dim)}
/* ── best models: the three projects ARE a pipeline, so draw the pipe ──
   The rail runs down the left. It is solid while work flows and goes dashed
   below the first stage with no accepted model, because that stage stops
   everything after it. Metric values sit in a monospace face so they read as
   readings rather than prose; the metric that decided the promotion is the
   only thing here allowed to wear the accent. */
.pipe{position:relative;padding-left:30px}
.stg{position:relative;padding:0 0 26px 0}
.stg:last-child{padding-bottom:2px}
/* the rail segment BELOW each node */
.stg::before{content:'';position:absolute;left:-22px;top:16px;bottom:-4px;
width:2px;background:var(--green);opacity:.55}
.stg:last-child::before{display:none}
/* below the break, flow is only potential -- dash it and drain the colour */
.stg.halt::before,.stg.idle::before{background:none;opacity:1;
border-left:2px dashed rgba(130,140,150,.38)}
.dot{position:absolute;left:-27px;top:5px;width:12px;height:12px;border-radius:50%;
background:var(--bg);border:2px solid var(--green)}
.stg.live .dot{background:var(--green);animation:pulse 2.6s infinite}
.stg.halt .dot{border-color:var(--mut);border-style:dashed}
.stg.idle .dot{border-color:rgba(130,140,150,.4)}
.sh{display:flex;align-items:baseline;flex-wrap:wrap;gap:8px 10px}
.sname{font-size:14.5px;font-weight:650;letter-spacing:-.2px;color:var(--tx)}
.stg.live .sname{color:var(--green)}
.semit{font-size:11.5px;color:var(--dim)}
.sproj{font-size:10.5px;color:var(--dim);font-family:ui-monospace,SFMono-Regular,
Menlo,monospace;margin-left:auto;opacity:.7}
.bdg{font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
border-radius:999px;padding:2px 8px;border:1px solid}
.bdg.live{color:var(--green);border-color:rgba(67,181,129,.35);
background:rgba(67,181,129,.1)}
/* accepted, but nothing downstream runs it yet -- neutral, not green: green
   here would claim the pipeline is doing work it is not */
.bdg.ok{color:var(--mut);border-color:rgba(130,140,150,.28);
background:rgba(130,140,150,.07)}
.bdg.halt{color:var(--mut);border-color:rgba(130,140,150,.3);
background:rgba(130,140,150,.08)}
.bdg.idle{color:var(--dim);border-color:transparent;background:transparent;
padding-left:0}
.sbody{margin-top:9px}
.srun{display:inline-block;font-size:13px;font-weight:650;color:var(--tx);
text-decoration:none;border-bottom:1px solid rgba(130,140,150,.35);padding-bottom:1px}
a.srun:hover{color:var(--acc);border-bottom-color:var(--acc)}
.srun .ext{font-size:10px;margin-left:3px;opacity:.6}
.srun.none{color:var(--dim);font-weight:500;border:0}
/* ── the ask: metrics as tags ── */
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.tag{display:inline-flex;align-items:baseline;gap:6px;background:var(--panel2);
cursor:help;
border:1px solid var(--bd);border-radius:7px;padding:3px 9px}
/* sentence case, not caps: these were metric KEYS (MAP50) and are now short
   phrases ("separation on the tuning split"), which caps turns into shouting */
.tag i{font-style:normal;font-size:10.5px;letter-spacing:.005em;
color:var(--dim)}
.tag b{font-size:12.5px;font-weight:640;color:var(--tx);
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-variant-numeric:tabular-nums}
/* the metric the promotion actually turned on */
.tag.hot{border-color:rgba(232,166,69,.45);background:rgba(232,166,69,.09)}
.tag.hot i{color:var(--acc)}
.tag.hot b{color:var(--acc)}
/* a key_metric that matches nothing: red, not amber -- amber is the accent
   that means "this decided the promotion", and reusing it here would make a
   config error look like a result */
.tag.warn{border-color:rgba(219,84,84,.5);background:rgba(219,84,84,.10)}
.tag.warn i,.tag.warn b{color:var(--red)}
.tags.sm{margin-top:5px;gap:4px}
.tags.sm .tag{padding:0;border:0;background:none}
.tags.sm .tag b{color:var(--mut)}
a.cand:hover .tags.sm .tag b{color:var(--tx)}
.tags.sm .tag i{font-size:9px}
.tags.sm .tag b{font-size:11px}
/* ── the deciding metric, as an instrument readout ────────────────────────
   Everything else on the card is a supporting reading. The value is set in
   the mono face at display size with the CLAIM underneath, because the metric
   key alone ("acceptance_rejected_at_full_dog_recall") tells a reader nothing
   about which of six numbers the promotion rests on. */
.hero{margin:12px 0 2px;display:flex;flex-direction:column;gap:1px;
cursor:help;width:max-content}
.hero b{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-size:34px;font-weight:600;line-height:1;letter-spacing:-.02em;
color:var(--acc);font-variant-numeric:tabular-nums}
.hero span{font-size:11.5px;color:var(--mut);letter-spacing:.01em;
max-width:34ch;line-height:1.35}
.stg.live .hero b{color:var(--acc)}

/* ── the audit trail, closed ──────────────────────────────────────────────
   Kept on the card because "why should I believe this" is a real question,
   collapsed because it is asked once. The word count sets the expectation
   before the click. */
.swhy{margin-top:14px;max-width:74ch}
.swhy summary{font-size:11.5px;color:var(--mut);cursor:pointer;
list-style:none;display:inline-flex;align-items:center;gap:8px;
padding:3px 9px 3px 0;border-radius:5px;user-select:none}
.swhy summary::-webkit-details-marker{display:none}
.swhy summary::before{content:"";width:0;height:0;
border-left:4.5px solid currentColor;border-top:3.5px solid transparent;
border-bottom:3.5px solid transparent;margin-right:1px;
transition:transform .15s ease}
.swhy[open] summary::before{transform:rotate(90deg)}
.swhy summary:hover{color:var(--tx)}
.swhy summary:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.swhy p{font-size:11.5px;color:var(--dim);line-height:1.65;margin-top:9px}
.swhy p+p{margin-top:7px}

/* ── the population marker ────────────────────────────────────────────────
   Which data a number came from, on the number itself. Deliberately quiet:
   it is a qualifier, not a reading. */
.pop{font-style:normal;font-size:9.5px;letter-spacing:.06em;
text-transform:uppercase;color:var(--dim);white-space:nowrap}
.hero .pop{margin-bottom:5px}
.tag .pop{margin-left:2px;padding-left:7px;
border-left:1px solid rgba(130,140,150,.22)}

/* ── the shared vocabulary, closed ─────────────────────────────────────── */
/* A block of its own, not a third disclosure on the last card. Panel fill and
   full width put it at section scope; the per-card "Why this one" stays a bare
   caret inside the rail. */
.gloss{margin:14px 0 0;background:var(--panel2);border:1px solid var(--bd);
border-radius:9px;padding:11px 14px}
.gloss[open]{padding-bottom:15px}
.gloss summary{font-size:11.5px;color:var(--mut);cursor:pointer;
list-style:none;display:flex;align-items:center;gap:9px;user-select:none}
.gloss .gk{font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;
color:var(--acc);font-weight:600}
.gloss summary{font-size:12.5px;color:var(--tx)}
.gloss summary::-webkit-details-marker{display:none}
.gloss summary::before{content:"";width:0;height:0;
border-left:4.5px solid currentColor;border-top:3.5px solid transparent;
border-bottom:3.5px solid transparent;transition:transform .15s ease}
.gloss[open] summary::before{transform:rotate(90deg)}
.gloss summary:hover{color:var(--tx)}
.gloss summary:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.gloss .wc{font-size:10px;color:var(--dim);
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.gwrap{margin-top:14px;display:grid;gap:12px;
grid-template-columns:repeat(auto-fit,minmax(280px,1fr));max-width:none}
.gt dt{font-size:10px;letter-spacing:.07em;text-transform:uppercase;
color:var(--acc);margin-bottom:3px}
.gt dd{font-size:11.5px;color:var(--dim);line-height:1.6}
@media(prefers-reduced-motion:reduce){.gloss summary::before{transition:none}}

@media(prefers-reduced-motion:reduce){.swhy summary::before{transition:none}}
.sfile{font-size:10.5px;color:var(--dim);margin-top:8px;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.cwrap{margin-top:12px;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:flex-start}
.clab{font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;
color:var(--dim);padding-top:6px;margin-right:2px}
.cand{display:block;padding:5px 10px 6px 0;text-decoration:none;
border-left:2px solid var(--bd);padding-left:10px;transition:border-color .12s}
a.cand:hover{border-left-color:var(--acc)}
.cname{font-size:11.5px;color:var(--mut);font-weight:600}
a.cand:hover .cname{color:var(--tx)}
.mfoot{font-size:11px;color:var(--dim);margin-top:18px;padding-top:12px;
border-top:1px solid var(--bd)}
.mfoot code{background:var(--panel2);padding:1px 6px;border-radius:5px}
.mnone{font-size:12px;color:var(--dim)}
@media(max-width:760px){.pipe{padding-left:24px}.sproj{margin-left:0;width:100%}}
@media(prefers-reduced-motion:reduce){.stg.live .dot{animation:none}}
.board{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:12px;margin-bottom:22px;align-items:start}
@media(max-width:1600px){.board{grid-template-columns:repeat(4,minmax(0,1fr))}}
@media(max-width:1040px){.board{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:560px){.board{grid-template-columns:1fr}}
.col{background:linear-gradient(180deg,#1a1f25,#161a20);border:1px solid var(--bd);border-radius:14px;padding:11px 11px 5px;min-height:88px;transition:background .15s,border-color .15s}
.col.over{border-color:var(--acc);background:#1e242c}
.col h3{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--dim);margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;gap:6px}
.col h3 .lab{display:flex;align-items:center;gap:6px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.col h3 .cnt{background:rgba(130,140,150,.16);color:var(--mut);border-radius:9px;padding:1px 7px;font-size:11px;flex:none}
.colbody{max-height:332px;overflow-y:auto;overscroll-behavior:contain;padding-right:3px;margin-right:-3px}
.colbody::-webkit-scrollbar{width:6px}
.colbody::-webkit-scrollbar-thumb{background:rgba(130,140,150,.3);border-radius:3px}
.colbody::-webkit-scrollbar-thumb:hover{background:rgba(232,166,69,.5)}
.colbody{scrollbar-width:thin;scrollbar-color:rgba(130,140,150,.3) transparent}
.more{font-size:10.5px;color:var(--dim);text-align:center;padding:2px 0 5px}
.dotc{width:8px;height:8px;border-radius:50%;flex:none}
.rc{background:#232830;border:1px solid rgba(130,140,150,.12);border-radius:9px;padding:9px 11px;margin-bottom:7px;cursor:grab}
.rc:active{cursor:grabbing}.rc.drag{opacity:.35}
.rc:hover{border-color:rgba(232,166,69,.4)}
.rc .rn{display:flex;align-items:center;justify-content:space-between;gap:6px;font-size:12.5px;font-weight:600;letter-spacing:-.2px}
.rc .rn .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cp{background:none;border:0;color:var(--dim);cursor:pointer;padding:2px;border-radius:5px;display:flex;align-items:center;flex:none;opacity:.5;transition:opacity .12s,color .12s,background .12s}
.rc:hover .cp{opacity:1}
.cp:hover{color:var(--acc);background:rgba(232,166,69,.12)}
.cp:active{transform:scale(.9)}
.rc .rs{display:flex;justify-content:space-between;align-items:center;margin-top:5px;font-size:11px;color:var(--mut);font-variant-numeric:tabular-nums}
.rc .mini{height:4px;border-radius:3px;background:rgba(130,140,150,.18);margin-top:6px;overflow:hidden}
.rc .mini i{display:block;height:100%;border-radius:3px}
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%) translateY(16px);background:#21262d;border:1px solid var(--bd);color:var(--tx);padding:9px 16px;border-radius:10px;font-size:13px;opacity:0;transition:.25s;pointer-events:none;z-index:9}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.cmdbar{display:flex;gap:8px;align-items:center;margin-bottom:13px;flex-wrap:wrap}
.cmdbar input{background:var(--panel2);border:1px solid var(--bd);border-radius:9px;color:var(--tx);padding:7px 12px;font-size:13px;min-width:210px;font-family:inherit}
.cmdbar input:focus{outline:none;border-color:var(--acc)}
.genbtn{background:var(--acc);border:0;color:#15100a;border-radius:9px;padding:7px 16px;font-size:13px;font-weight:680;cursor:pointer;transition:filter .12s}
.genbtn:hover{filter:brightness(1.08)}
.cmdblock{margin-bottom:11px;border:1px solid var(--bd);border-radius:12px;overflow:hidden;background:#15191e}
.cmdhead{display:flex;justify-content:space-between;align-items:center;padding:8px 13px;font-size:11.5px;font-weight:600;color:var(--mut);background:#1c2128;border-bottom:1px solid var(--bd)}
.cmdblock pre{margin:0;padding:13px 15px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#cdd6df;overflow-x:auto;white-space:pre}
.fold>summary{cursor:pointer;list-style:none;user-select:none}
.fold>summary::-webkit-details-marker{display:none}
.fold>summary::before{content:'';display:inline-block;width:0;height:0;flex:none;
  border-left:5px solid var(--dim);border-top:4px solid transparent;border-bottom:4px solid transparent;
  margin-right:8px;transition:transform .15s;transform-origin:2px 50%}
.fold[open]>summary::before{transform:rotate(90deg)}
.fold>summary:hover::before{border-left-color:var(--acc)}
.fold>summary.sect{display:flex;align-items:baseline}
.fold>summary.phead{display:flex;align-items:center}
/* collapsed: drop the summary's own bottom margin, but keep a gap between
   this header and the next section (they touched otherwise) */
.fold:not([open])>summary.phead{margin-bottom:0}
.fold:not([open])>summary.sect{margin-bottom:0}
.fold:not([open]){margin-bottom:var(--gap)}
.fold.panel:not([open]){padding-bottom:14px}
.loc{display:grid;grid-template-columns:minmax(140px,.9fr) 1.5fr 1.5fr;gap:12px;align-items:center;padding:10px 4px;border-top:1px solid var(--bd)}
.loc.lh{border-top:0;padding-top:0;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--dim)}
.lname{font-size:13px;font-weight:620;letter-spacing:-.2px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{background:#232830;border:1px solid rgba(130,140,150,.14);border-radius:8px;padding:3px 9px;font-size:11.5px;color:var(--mut);font-variant-numeric:tabular-nums;white-space:nowrap}
.chip b{color:var(--tx);font-weight:620;margin-right:2px}
/* Colour encodes ROLE (parquet vs jpgs), not size: the same drive can hold both,
   so the tint is what tells you which side of the split you are looking at. */
.chip.data{border-color:rgba(91,143,214,.4);background:rgba(91,143,214,.09)}
.chip.data b{color:#7fb0ea}
.chip.img{border-color:rgba(79,182,196,.4);background:rgba(79,182,196,.09)}
.chip.img b{color:#5fcbd8}
.swatch{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;vertical-align:middle}
.swatch.data{background:#7fb0ea}
.swatch.img{background:#5fcbd8}
.lnone{color:var(--dim);font-size:12px}
@media(max-width:760px){.loc{grid-template-columns:1fr;gap:6px;padding:12px 4px}
  .loc.lh{display:none}}
footer{margin-top:32px;color:var(--dim);font-size:11.5px;text-align:center;line-height:1.7}
/* ── detection sweep panel (§7.4) ── */
.dnone{color:var(--dim);font-size:12.5px;padding:2px}
.dsub{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);margin:16px 0 8px}
/* headline: compact KPI chips (the page's existing .kpi look); the img/s
   sparkline draws as the subtle BACKGROUND of the img/s (now) chip */
.kpi.spk{position:relative;overflow:hidden}
.kpi.spk .kpi-label,.kpi.spk .kpi-val{position:relative;z-index:1}
.dspark{position:absolute;inset:0;z-index:0;opacity:.5;pointer-events:none}
.dmain{height:10px}
.dcount{font-size:12.5px;color:var(--mut);font-variant-numeric:tabular-nums;margin:7px 0 0}
/* per-PROCESS tally, deliberately quieter than the all-time headline */
.drun{font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums;margin:2px 0 4px}
/* "sweep idle" reads as a status line above the cards, not as their stand-in */
.dstat{margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid var(--bd)}
.dgrid{display:grid;grid-template-columns:1fr 1fr;gap:0 26px}
@media(max-width:900px){.dgrid{grid-template-columns:1fr}}
.drow{display:flex;align-items:center;gap:10px;padding:4px 0;font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}
.drow .dn{width:86px;font-weight:620;color:var(--tx);flex:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}
.drow .bar{flex:1}
.drow .dv{flex:none;text-align:right;white-space:nowrap}
.drow.dmut{opacity:.55}
.dbadge{background:rgba(216,116,58,.13);border:1px solid rgba(216,116,58,.45);color:#d8743a;border-radius:7px;padding:0 7px;font-size:10.5px;font-weight:620;flex:none}
/* not_a_dog gauge: the 7–16% labelled prior renders as a shaded healthy zone
   (Addendum A.5) so in/out of band is legible without reading numbers */
.dband{position:relative;height:10px;border-radius:6px;background:rgba(130,140,150,.16);flex:1}
.dband .zone{position:absolute;top:0;bottom:0;background:rgba(67,181,129,.3);border-radius:2px}
.dband .cur{position:absolute;top:-3px;bottom:-3px;width:3px;border-radius:2px;background:var(--acc)}
.dband .cur.bad{background:#d8743a}
.dchips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
/* live detection crops: a random sample of the last minute's positives.
   Fixed square tiles so a row of wildly different crop aspect ratios still
   reads as a grid; the confidence badge sits over the image, not beside it,
   because the thumbnails are too small to spare a caption row. */
.dcrophead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.dcrophead .dsub{margin:0}
.dcropsub{font-size:11px;color:var(--dim);font-variant-numeric:tabular-nums}
.dcrophead .rbtn{padding:2px 9px;font-size:11px}
/* Shuffle + Review sit together at the right end, both acting on this grid */
.dcrophead #dcropShuffle{margin-left:auto}
/* Review all is the one control here that leaves the page for a task, so it
   does not wear the amber every other button wears. Rust ties it to the flag
   it opens. The pulse is a ring, not a scale/opacity bounce -- next to live
   numbers a moving button is noise, a breathing outline is a nudge -- and it
   stops on hover and focus so it never animates under an active pointer. */
.dcrophead .rev{text-decoration:none;background:rgba(216,116,58,.16);
  border-color:rgba(216,116,58,.5);color:#e8894f;
  animation:revpulse 2.8s ease-out infinite}
@keyframes revpulse{
  0%{box-shadow:0 0 0 0 rgba(216,116,58,.5)}
  65%{box-shadow:0 0 0 7px rgba(216,116,58,0)}
  100%{box-shadow:0 0 0 0 rgba(216,116,58,0)}}
.dcrophead .rev:hover,.dcrophead .rev:focus-visible{
  animation:none;background:rgba(216,116,58,.3);color:#f2a473;
  border-color:rgba(216,116,58,.75)}
@media(prefers-reduced-motion:reduce){.dcrophead .rev{animation:none}}
.dcrops{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;margin-top:8px}
.dcrop{position:relative;aspect-ratio:1;border:1px solid var(--bd);border-radius:8px;overflow:hidden;background:#15191e}
.dcrop img{width:100%;height:100%;object-fit:cover;display:block}
.dcrop .cf{position:absolute;right:4px;bottom:4px;background:rgba(19,21,26,.78);border:1px solid rgba(130,140,150,.28);
  color:var(--tx);border-radius:6px;padding:0 5px;font-size:10.5px;font-weight:620;font-variant-numeric:tabular-nums}
/* only tiles whose full frame actually exists on disk advertise the click */
.dcrop.cx{cursor:pointer;transition:border-color .12s,transform .12s}
.dcrop.cx:hover{border-color:var(--acc);transform:translateY(-1px)}
.dcrop.cx:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
/* false-positive flag: TOP-LEFT, opposite the confidence badge bottom-right.
   Hidden until hover/focus on a pointer device; always shown where there is
   no hover to reveal it (touch). */
.dcrop .fx{position:absolute;left:4px;top:4px;z-index:1;width:19px;height:19px;padding:0;
  display:flex;align-items:center;justify-content:center;cursor:pointer;
  background:rgba(19,21,26,.78);border:1px solid rgba(130,140,150,.28);color:var(--mut);
  border-radius:6px;font-size:11px;font-weight:700;line-height:1;font-family:inherit;
  opacity:0;transition:opacity .12s,background .12s,color .12s,border-color .12s}
.dcrop:hover .fx,.dcrop .fx:focus-visible{opacity:1}
.dcrop .fx:hover{background:rgba(216,116,58,.85);border-color:#d8743a;color:#fff}
@media(hover:none){.dcrop .fx{opacity:1}}
/* flagged: persistent red-family state, kept through refresh and Shuffle */
.dcrop.fl{border-color:#d8743a}
.dcrop.fl img{opacity:.45}
.dcrop.fl .fx{opacity:1;background:#d8743a;border-color:#d8743a;color:#15100a}
.dflag{font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums;margin-top:8px}
__LB_CSS__
.derr{font-size:12px;color:var(--mut);margin-top:14px}
.derr.ok{color:var(--green)}
.derr .dt{cursor:pointer;user-select:none}
.derr .dt:hover{color:var(--tx)}
.dmeta{font-size:11px;color:var(--dim);margin-top:12px}
.spath{display:flex;align-items:center;gap:9px;flex-wrap:wrap;
margin:0 0 13px;padding:8px 11px;background:var(--panel2);
border:1px solid var(--bd);border-radius:9px}
.splab{font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;
color:var(--dim)}
.spath > code{font-size:12px;color:var(--tx);font-family:ui-monospace,
SFMono-Regular,Menlo,Consolas,monospace;word-break:break-all;min-width:0}
.spath .cp{flex:none}
/* The hint carries the staleness caveat and the rebuild command, so it wraps
   to its own line rather than being hidden on a narrow screen -- a path handed
   over without "this is derived" is the whole risk of showing it at all. */
.sphint{font-size:11px;color:var(--dim);margin-left:auto}
.sphint code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-size:10.5px;color:var(--mut)}
@media (max-width:820px){
  .sphint{margin-left:0;flex-basis:100%}
}
/* ── training tracker ─────────────────────────────────────────────────── */
/* a history row is a control: it opens that run's metrics above */
.thist tbody tr{cursor:pointer;transition:background .1s}
.thist tbody tr:hover{background:rgba(130,140,150,.06)}
.thist tbody tr:focus-visible{outline:2px solid var(--acc);outline-offset:-2px}
.thist tbody tr.sel{background:rgba(232,166,69,.09);
box-shadow:inset 2px 0 0 var(--acc)}
.thist tbody tr.onair .tn b{color:var(--green)}
.tlive.past{border-color:var(--bd)}
.tlive.past .tlhead b{font-size:15px}
.tback{margin-left:auto;flex:none}
.rbtn.quiet{background:transparent;border-color:var(--bd);color:var(--mut)}
.rbtn.quiet:hover{background:rgba(130,140,150,.1);color:var(--tx)}
@media (prefers-reduced-motion:reduce){.thist tbody tr{transition:none}}

.tlive{background:var(--panel2);border:1px solid rgba(67,181,129,.22);
border-radius:11px;padding:15px 16px 16px;margin-bottom:14px}
.tlhead{display:flex;align-items:center;flex-wrap:wrap;gap:8px 11px;
margin-bottom:13px}
.tlhead b{font-size:15px;font-weight:640;letter-spacing:-.15px}
.tsub{color:var(--dim);font-size:11.5px}
.ttiles{display:flex;flex-wrap:wrap;gap:10px}
.ttile{flex:0 1 190px;background:var(--panel);border:1px solid var(--bd);
border-radius:9px;padding:10px 12px 9px}
.ttile b{display:block;font-size:21px;font-weight:600;letter-spacing:-.4px;
font-variant-numeric:tabular-nums;line-height:1.15}
.ttile b em{font-style:normal;font-size:13px;color:var(--dim);font-weight:500}
.ttile span{display:block;margin-top:2px;font-size:11px;color:var(--mut)}
/* the latest-epoch row is context for the row above it, so it reads quieter:
   smaller numbers, its own rule, and a label that says what it is */
.tlatest{margin-top:13px;padding-top:12px;border-top:1px solid var(--bd)}
.tlab{display:block;font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;
color:var(--dim);margin-bottom:8px}
/* two roles, held to across the card and the charts:
   amber = the working value, what the run just produced
   blue  = the benchmark it is measured against (its own peak; validation loss)
   validated on both panel surfaces -- lightness band, chroma, CVD dE 21.8
   protan / 22.6 tritan, normal-vision 23.3, contrast all pass */
.tlab .k{display:inline-block;width:12px;height:2px;border-radius:1px;
vertical-align:middle;margin:0 5px 0 14px}
.tlab .k.now{background:#c2872e}
.tlab .k.pk{background:#5b93cf}
/* auto-fit with a 1fr max let a single card span the whole row, which
   stretched its sparkline into a flat smear. Cap the track. */
.mcards{display:grid;gap:12px;justify-content:start;
grid-template-columns:repeat(auto-fit,minmax(238px,340px))}
.mcard{background:var(--panel);border:1px solid var(--bd);border-radius:10px;
padding:11px 13px 9px;min-width:0}
.mcard:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
@media (max-width:430px){
  .mcards{grid-template-columns:1fr}
  .mhead{flex-wrap:wrap;gap:2px 10px}
}
.mhead{display:flex;align-items:baseline;gap:10px;margin-bottom:4px}
.mlab{font-size:11px;color:var(--mut);white-space:nowrap;overflow:hidden;
text-overflow:ellipsis;flex:1 1 auto;min-width:0}
.mnow{font-size:27px;font-weight:600;letter-spacing:-.7px;line-height:1.05;
color:#c2872e;font-variant-numeric:tabular-nums}
.mpv{flex:none;font-size:12.5px;color:#5b93cf;font-variant-numeric:tabular-nums;
font-weight:560}
.mpv em{font-style:normal;font-size:10px;color:var(--dim);margin-left:5px;
font-weight:500}
.mgap{margin-top:2px;font-size:11.5px;color:var(--mut);
font-variant-numeric:tabular-nums}
/* NOT .spk: the detection-sweep KPI chip is .kpi.ok.spk, and a bare
   .spk{height:44px} here clipped its value away with overflow:hidden --
   the chip rendered its label and nothing else. */
.mspk{display:block;width:100%;height:44px;margin-top:8px;
overflow:visible}
/* the signature: the benchmark is a rule at its own height, and the drop from
   it to the current point IS the shortfall -- not a badge describing one */
.mspk .pk{stroke:#5b93cf;stroke-width:1;stroke-dasharray:2 4;opacity:.6}
.mspk .gap{stroke:#5b93cf;stroke-width:1.5;opacity:.8;stroke-linecap:round}
.mspk .tr{fill:none;stroke:rgba(150,160,172,.42);stroke-width:1.5;
stroke-linejoin:round;stroke-linecap:round}
.mspk .pkd{fill:#5b93cf}
.mspk .now{fill:#c2872e;stroke:var(--panel);stroke-width:1.5}
.tmeters{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
gap:14px 22px;margin-top:13px}
.tmnote{margin-top:7px;font-size:11px;color:var(--mut)}
.tmeter{margin-top:0}
.tmhead{display:flex;justify-content:space-between;align-items:baseline;
font-size:11.5px;color:var(--mut)}
.tmhead b{color:var(--tx);font-size:13.5px;font-variant-numeric:tabular-nums}
.tmhead b em{font-style:normal;color:var(--dim);font-weight:500}
/* a meter, not a chart mark: one hue on its own track */
.tmtrack{height:7px;border-radius:4px;background:rgba(130,140,150,.16);
margin:6px 0 5px;overflow:hidden}
.tmtrack i{display:block;height:100%;border-radius:4px;background:#c2872e;
transition:width .4s ease}
.tmfoot{font-size:11px;color:var(--dim)}

.tgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
gap:12px;margin-bottom:14px}
/* below ~370px the 330px track is wider than the viewport and the whole page
   scrolls sideways; one column is the right answer, not a scrollbar */
@media (max-width:420px){.tgrid{grid-template-columns:1fr}}
.tfig{position:relative;background:var(--panel2);border:1px solid var(--bd);
border-radius:10px;padding:11px 12px 6px;min-width:0}
.tfig figcaption{font-size:11.5px;color:var(--mut);margin-bottom:6px;
display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.tleg{display:flex;gap:11px;margin-left:auto}
.tleg span{display:inline-flex;align-items:center;gap:5px;font-size:11px;
color:var(--mut)}
/* lines get a line key, never a filled box */
.tleg i{width:13px;height:2px;border-radius:1px}
.tsvg{width:100%;height:auto;display:block;overflow:visible}
.tsvg .grid line{stroke:rgba(130,140,150,.13);stroke-width:1}
.tsvg .tick{fill:var(--dim);font-size:9.5px;
font-variant-numeric:tabular-nums}
.tsvg .ylab{fill:var(--dim);font-size:9.5px;letter-spacing:.03em}
.tsvg .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.tsvg .ctx{fill:none;stroke:rgba(130,140,150,.34);stroke-width:1.5}
.tsvg .mk{fill:#c2872e;stroke:var(--panel2);stroke-width:2}
.tsvg .mklab{fill:var(--mut);font-size:10px}
.tsvg .cross{stroke:rgba(150,160,172,.5);stroke-width:1;stroke-dasharray:3 3;
display:none;pointer-events:none}
.tsvg .hit{fill:transparent;cursor:crosshair}
.tsvg .dot{fill:var(--panel2);stroke-width:2;display:none}
.ttip{position:absolute;pointer-events:none;z-index:6;background:#0f1116;
border:1px solid var(--bd);border-radius:8px;padding:8px 10px;font-size:11.5px;
box-shadow:0 8px 22px rgba(0,0,0,.5);min-width:118px}
.ttip .tth{color:var(--dim);font-size:10px;letter-spacing:.05em;
text-transform:uppercase;margin-bottom:5px}
.ttip .ttr{display:flex;align-items:center;gap:7px;margin-top:3px}
.ttip .ttr i{width:11px;height:2px;border-radius:1px;flex:none}
.ttip .ttr b{font-variant-numeric:tabular-nums;font-size:12.5px}
.ttip .ttr span{color:var(--mut);font-size:11px}
.tempty{color:var(--dim);font-size:12px;padding:22px 0;text-align:center}

.tbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:2px 0 9px}
.tbar label{font-size:11px;color:var(--dim);text-transform:uppercase;
letter-spacing:.06em}
.tbar select{background:var(--panel2);color:var(--tx);border:1px solid var(--bd);
border-radius:7px;padding:5px 9px;font-size:12px;font-family:inherit}
.tnote{font-size:11px;color:var(--dim);flex:1 1 260px;min-width:0}
.tnote em{color:var(--mut);font-style:normal}

.thist{width:100%;border-collapse:collapse;font-size:12.5px}
.thist th{text-align:left;font-weight:600;font-size:10.5px;color:var(--dim);
text-transform:uppercase;letter-spacing:.06em;padding:0 10px 7px;
border-bottom:1px solid var(--bd)}
.thist td{padding:9px 10px;border-bottom:1px solid rgba(130,140,150,.07)}
.thist tr:last-child td{border-bottom:0}
.thist .num{text-align:right;font-variant-numeric:tabular-nums}
.thist .tn b{display:block;font-weight:600}
.thist .tn span{color:var(--dim);font-size:10.5px}
.tst{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;
color:var(--mut);white-space:nowrap}
.tst.live{color:var(--green)}
.tst.halt{color:var(--red)}
.tst.idle{color:var(--dim)}
.tcand{font-size:10.5px;color:var(--dim)}
.tscroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.thist{min-width:640px}
.tpage{display:flex;align-items:center;gap:10px;justify-content:flex-end;
margin-top:11px}
/* a bare display:flex beats the UA's [hidden]{display:none}, so pager.hidden
   had no visual effect and an unpaginated table still showed "1-3 of 3" */
.tpage[hidden]{display:none}
.tpage .tpn{font-size:11.5px;color:var(--dim);
font-variant-numeric:tabular-nums}
.tpb{background:var(--panel2);color:var(--tx);border:1px solid var(--bd);
border-radius:7px;width:28px;height:26px;font-size:13px;cursor:pointer;
line-height:1}
.tpb:disabled{color:var(--dim);cursor:default;opacity:.5}
.tpb:focus-visible,.tfig:focus-visible{outline:2px solid var(--acc);
outline-offset:2px}
.thid{margin-top:10px;font-size:11px;color:var(--dim)}
.tlead{font-size:11.5px;color:var(--mut);margin:0 0 9px}
.tmetric{font-size:12.5px;font-weight:620;color:var(--tx);letter-spacing:-.1px}
.trun{font-size:11px;color:var(--dim)}
@media (max-width:700px){
  .thist th:nth-child(6),.thist td:nth-child(6){display:none}
}
@media (prefers-reduced-motion:reduce){
  .tmtrack i{transition:none}
}
</style></head><body><div class="wrap">

<header>
  <div><h1>Street Dogs · <span class="o">Collection Progress</span></h1>
    <div class="sub">global Mapillary ground-animal harvest</div></div>
  <div class="hact">
    <!-- The count is the point: a queue depth is what makes someone open the
         page, and an empty queue should say so quietly rather than shout a
         zero in the accent colour. -->
    <a class="revbtn" id="revBtn" href="/review"
       title="Judge detections one by one — dog or not a dog">
      <span class="rvf">&#9873;</span>
      <span class="rvn"><b id="revN">&mdash;</b><em id="revL">to review</em>
      </span></a>
    <div class="upd"><span class="dot"></span>updated __NOW__ · auto-refreshes hourly<button id="refreshBtn" class="rbtn" title="Re-scan the catalog + image counts now">↻ Refresh now</button></div>
  </div>
</header>

<div class="ring-row">
  <div class="ring">
    <svg width="140" height="140" viewBox="0 0 140 140">
      <defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#e8a645"/><stop offset="1" stop-color="#f5c570"/></linearGradient></defs>
      <circle cx="70" cy="70" r="60" fill="none" stroke="rgba(130,140,150,.16)" stroke-width="12"/>
      <circle id="ringfill" cx="70" cy="70" r="60" fill="none" stroke="url(#rg)" stroke-width="12"
        stroke-linecap="round" stroke-dasharray="377" stroke-dashoffset="377"/>
    </svg>
    <div class="ctr"><b>__PROG__%</b><span>downloaded</span></div>
  </div>
  <div class="kpis">__KPIS__</div>
</div>

<details class="fold sec" id="f-detect" open>
<summary class="sect">Detection sweep <span>yolo26x @1280 — live, updates every 5 s while open</span>
  <span class="swctl"><span class="swpill" id="sweepState">checking</span><button id="sweepBtn" class="rbtn sw" disabled>Checking&hellip;</button></span></summary>
<div class="panel">
  <!-- status line ABOVE the cards, never instead of them: the layout below is
       always present and goes to em-dashes when idle, so nothing jumps when
       the sweep starts. #detOn is kept (and never hidden) as the cards' box. -->
  __STOREPATH__
  <div id="detOff" class="dnone dstat">sweep idle</div>
  <div id="detOn">
    <div class="kpis" style="margin-bottom:12px">
      <div class="kpi"><div class="kpi-label">Complete</div><div class="kpi-val" id="dhPct" style="font-size:19px">—</div></div>
      <div class="kpi"><div class="kpi-label" title="images the detector has covered all-time, across every restart">Processed</div><div class="kpi-val" id="dhDone" style="font-size:19px">—</div></div>
      <div class="kpi"><div class="kpi-label">ETA</div><div class="kpi-val" id="dhEta" style="font-size:19px">—</div></div>
      <div class="kpi"><div class="kpi-label" title="images with at least one detection, all-time across every restart">Positives</div><div class="kpi-val" id="dhPos" style="font-size:19px">—</div></div>
      <div class="kpi ok spk"><div id="detSpark" class="dspark"></div><div class="kpi-label" title="throughput over the last 60 seconds">img/s (now)</div><div class="kpi-val" id="dhNow" style="font-size:19px">—</div></div>
      <div class="kpi"><div class="kpi-label" title="throughput over the last 15 minutes — the ETA is computed from this">img/s (sustained)</div><div class="kpi-val" id="dhSus" style="font-size:19px">—</div></div>
    </div>
    <div class="bar dmain"><div class="fill" id="dhFill" style="background:var(--acc)"></div></div>
    <div class="dcount" id="dhCount">—</div>
    <!-- secondary: this PROCESS's tally. The headline above is all-time. -->
    <div class="drun" id="dhRun">—</div>
    <div class="dgrid">
      <div><div class="dsub">Per drive</div><div id="detDrives"></div></div>
      <div><div class="dsub" id="detRegHead">Per region</div>
        <div id="detRegions" class="colbody" style="max-height:300px"></div></div>
    </div>
    <div class="dsub">Classifier</div>
    <div id="detHealth"></div>
    <div id="detErrs"></div>
    <div class="dmeta" id="detMeta"></div>
    <!-- the one exception to "always visible": an empty grid is just dead
         space, so no crops keeps the muted line instead of blank tiles -->
    <div class="dcrophead" style="margin-top:16px">
      <div class="dsub">Live detections</div>
      <span class="dcropsub" id="dcropSub">random sample from the last minute</span>
      <button id="dcropShuffle" class="rbtn" title="draw a different random sample from the last minute">↻ Shuffle</button>
      <a href="/review" class="rbtn nav rev" title="flag false positives in bulk — 50 or 100 per page">⚑ Review all</a>
    </div>
    <div class="dcrops" id="dcropGrid"></div>
    <div class="dflag" id="dcropFlagged"></div>
  </div>
</div>
</details>

<details class="fold sec" id="f-training" open>
<summary class="sect">Training <span>what is training now, its curves, and the runs behind it</span></summary>
<div class="panel" id="trk">__TRAINING__</div>
</details>

<details class="fold sec" id="f-models" open>
<summary class="sect">Best models <span>one per Comet project — click a run to open it</span></summary>
<div class="panel">__MODELS__</div>
</details>

<details class="fold sec" id="f-board" open>
<summary class="sect">Pipeline tracker <span>drag a region between stages — saved automatically</span></summary>
<div class="board" id="board"></div>
</details>
<div class="toast" id="toast"></div>

__LB_HTML__

<details class="fold panel sec" id="f-cmd" open>
  <summary class="phead"><i></i><b>Generate commands</b><span class="phint">enter a region — drives auto-filled from where its data &amp; images live</span></summary>
  <div class="cmdbar">
    <input id="cmdRegion" list="cmdRegions" placeholder="region, e.g. Greenland" autocomplete="off">
    <datalist id="cmdRegions"></datalist>
    <button class="genbtn" id="cmdGen">Generate</button>
  </div>
  <div id="cmdOut"></div>
</details>

<details class="fold panel sec" id="f-map" open>
  <summary class="phead"><i></i><b>Ground-animal density</b><span class="phint">every point binned to a raster · zoom in for finer 0.15° detail</span></summary>
  <!-- The map roams on wheel, so hovering it while scrolling the page used to
       zoom the map instead. A scrim swallows wheel/drag until you opt in;
       echarts keeps roam:true and never has to be reconfigured. -->
  <div class="mapwrap">
    <div id="map" style="width:100%;height:520px"></div>
    <div class="mapgate" id="mapGate">
      <span class="mapgb">Click to interact &mdash; scroll to zoom, drag to pan</span>
    </div>
    <button class="rbtn maplock" id="mapLock" hidden>&#128274; Lock map</button>
  </div></details>

<details class="fold panel sec" id="f-bars" open>
  <summary class="phead"><i></i><b>Ground animals by region</b></summary>
  <div id="bars" class="chart"></div></details>

<details class="fold sec" id="f-cards" open>
<summary class="sect">By region — download progress <span>downloaded ÷ ground-animal manifest</span></summary>
<div class="cards">__CARDS__</div>
</details>

<details class="fold sec" id="f-locs" open>
<summary class="sect">Where everything lives <span>which drive holds each region's parquet data vs its jpgs — colour marks the role, drives ordered biggest first</span></summary>
<div class="panel">__LOCS__</div>
</details>

<footer>Source: DuckDB catalog · downloaded jpgs ÷ ground-animal manifest rows<br>generated by tools/dashboard/dashboard.py</footer>
</div>

<script>
var D=__DATA__;
/* Progress ring. A round linecap paints half the stroke width PAST each end of
   the arc, so a naive dashoffset overstates the fill by a full stroke width
   (12px here) -- at 97.5% the 9.4px gap is swallowed and the two caps overlap,
   drawing a closed circle. Shorten the arc by that overhang so the ink matches
   the number, and drop to butt caps when the arc is too short to shorten. */
(function(){
  var C=2*Math.PI*60, cap=6;                    /* r=60, stroke-width 12 */
  var el=document.getElementById('ringfill');
  var pct=Math.max(0,Math.min(parseFloat("__PROG__")||0,100));
  var arc=C*pct/100;
  if(pct>=100){ el.setAttribute('stroke-linecap','butt'); el.setAttribute('stroke-dashoffset',0); }
  else if(arc>3*cap){ el.setAttribute('stroke-dashoffset',C-(arc-2*cap)); }
  else { el.setAttribute('stroke-linecap','butt'); el.setAttribute('stroke-dashoffset',C-arc); }
})();
/* mirrors human() in dashboard.py exactly -- see the note there */
function fmt(v){v=+v;if(v>=1e9)return (v/1e9).toFixed(2)+'B';if(v>=1e6)return (v/1e6).toFixed(2)+'M';if(v>=1e3)return (v/1e3).toFixed(2)+'K';return ''+v}
/* ── pipeline tracker board ── */
var STAGE_COLOR={pending:'#7d8893',extract:'#8b7fd6',coverage:'#5b8fd6',backfill:'#4fb6c4',complete:'#b083d6',downloading:'#e8a645',downloaded:'#43b581'};
/* cards past this many stay in the column's scroll area (see .colbody max-height) */
var BOARD_VISIBLE=4;
function pctColor(p){return p>=99?'#43b581':p>=70?'#e8a645':'#d8743a'}
var COPY_SVG='<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
function execCopy(t){var ta=document.createElement('textarea');ta.value=t;ta.setAttribute('readonly','');ta.style.position='fixed';ta.style.top='-1000px';ta.style.opacity='0';document.body.appendChild(ta);ta.select();ta.setSelectionRange(0,t.length);var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(ta);return ok;}
function copyText(t){var done=function(){toast('copied '+t)},fail=function(){toast('copy failed')};if(window.isSecureContext&&navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(t).then(done,function(){execCopy(t)?done():fail()});}else{execCopy(t)?done():fail();}}
var boardEl=document.getElementById('board'),toastEl=document.getElementById('toast'),dragKey=null,toastT;
function toast(t){toastEl.textContent=t;toastEl.classList.add('show');clearTimeout(toastT);toastT=setTimeout(function(){toastEl.classList.remove('show')},1700)}
function bcard(r){
  var d=document.createElement('div');d.className='rc';d.draggable=true;d.dataset.key=r.key;
  d.innerHTML='<div class="rn"><span class="nm">'+r.name+'</span><button type="button" class="cp" title="Copy &quot;'+r.key+'&quot;" aria-label="Copy region name">'+COPY_SVG+'</button></div><div class="rs"><span>'+fmt(r.downloaded)+' / '+fmt(r.dogs)+'</span><span style="color:'+pctColor(r.pct)+'">'+r.pct+'%</span></div><div class="mini"><i style="width:'+Math.min(r.pct,100)+'%;background:'+pctColor(r.pct)+'"></i></div>';
  var cp=d.querySelector('.cp');cp.draggable=false;
  cp.addEventListener('mousedown',function(e){e.stopPropagation()});
  cp.addEventListener('click',function(e){e.stopPropagation();e.preventDefault();copyText(r.key)});
  d.addEventListener('dragstart',function(e){dragKey=r.key;d.classList.add('drag');e.dataTransfer.effectAllowed='move';try{e.dataTransfer.setData('text/plain',r.key)}catch(_){}});
  d.addEventListener('dragend',function(){d.classList.remove('drag')});
  return d;
}
function brender(data){
  boardEl.innerHTML='';var by={};data.stages.forEach(function(s){by[s]=[]});
  data.regions.forEach(function(r){(by[r.stage]||by.pending).push(r)});
  data.stages.forEach(function(s){
    var col=document.createElement('div');col.className='col s-'+s;col.dataset.stage=s;
    var items=(by[s]||[]).sort(function(a,b){return b.dogs-a.dogs});
    col.innerHTML='<h3><span class="lab"><span class="dotc" style="background:'+STAGE_COLOR[s]+'"></span>'+data.labels[s]+'</span><span class="cnt">'+items.length+'</span></h3>';
    var body=document.createElement('div');body.className='colbody';
    items.forEach(function(r){body.appendChild(bcard(r))});
    col.appendChild(body);
    if(items.length>BOARD_VISIBLE){
      var m=document.createElement('div');m.className='more';
      m.textContent='scroll for '+(items.length-BOARD_VISIBLE)+' more';
      col.appendChild(m);
    }
    col.addEventListener('dragover',function(e){e.preventDefault();col.classList.add('over')});
    col.addEventListener('dragleave',function(){col.classList.remove('over')});
    col.addEventListener('drop',function(e){e.preventDefault();col.classList.remove('over');if(dragKey)bmove(dragKey,s)});
    boardEl.appendChild(col);
  });
  var cl=document.getElementById('cmdRegions');
  if(cl)cl.innerHTML=data.regions.map(function(r){return '<option value="'+r.key+'">'+(r.name||r.key)+'</option>'}).join('');
}
function bmove(key,stage){
  fetch('/api/board',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({region:key,stage:stage})}).then(function(r){return r.json()}).then(function(j){if(j&&j.ok){toast(key.replace(/_/g,' ')+' → '+stage.replace(/_/g,' '));bload()}else toast('save failed')}).catch(function(){toast('save failed (offline?)')});
}
function bload(){fetch('/api/board').then(function(r){return r.json()}).then(brender).catch(function(){boardEl.innerHTML='<div style="color:#69727d;padding:18px">board API unavailable — is the server running?</div>'})}
if(boardEl)bload();
/* ── force refresh ── */
var rbtn=document.getElementById('refreshBtn');
function refreshNow(){
  if(!rbtn||rbtn.disabled)return;
  var t0=Date.now(),tick;
  rbtn.disabled=true;rbtn.innerHTML='<span class="spin">↻</span> Refreshing… 0s';
  tick=setInterval(function(){rbtn.innerHTML='<span class="spin">↻</span> Refreshing… '+Math.round((Date.now()-t0)/1000)+'s';},1000);
  function stop(msg){clearInterval(tick);rbtn.disabled=false;rbtn.innerHTML='↻ Refresh now';if(msg)toast(msg);}
  function poll(){fetch('/api/refresh').then(function(r){return r.json()}).then(function(s){
    if(s.running){setTimeout(poll,2500);}
    else if(s.error){stop('refresh failed: '+s.error);}
    else{clearInterval(tick);toast('refreshed ✓');setTimeout(function(){location.reload()},400);}
  }).catch(function(){setTimeout(poll,3000)});}
  fetch('/api/refresh',{method:'POST'}).then(function(r){return r.json()}).then(function(j){if(j&&j.error){stop('refresh failed');return;}poll();}).catch(function(){stop('refresh failed (offline?)')});
}
if(rbtn)rbtn.addEventListener('click',refreshNow);
/* ── copy the store path ── */
(function(){
  var b=document.getElementById('storeCp'),p=document.getElementById('storePath');
  if(b&&p) b.addEventListener('click',function(e){
    e.preventDefault(); copyText(p.textContent.trim());
  });
})();
/* ── command generator ── */
var cmdRegion=document.getElementById('cmdRegion'),cmdOut=document.getElementById('cmdOut'),cmdGen=document.getElementById('cmdGen');
function esc(s){return (''+s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
__LB_JS__
/* ── training tracker: crosshair + tooltip, one readout for every series ── */
/* Split deliberately. Opening a past run replaces only the detail region, so
   it re-binds CHARTS alone; calling the whole initialiser there re-entered the
   selection-restore at its tail and openRun called itself forever. */
function bindCharts(){
  var figs=document.querySelectorAll('.tfig[data-chart]');
  Array.prototype.forEach.call(figs,function(fig){
    var spec;
    try{ spec=JSON.parse(fig.getAttribute('data-chart')); }catch(e){ return; }
    if(!spec||!spec.series||!spec.series.length) return;
    var svg=fig.querySelector('svg'), cross=fig.querySelector('.cross'),
        tip=fig.querySelector('.ttip'), hit=fig.querySelector('.hit');
    if(!svg||!hit) return;
    /* one dot per series, so the reader sees which point is being read */
    var dots=spec.series.map(function(s){
      var c=document.createElementNS('http://www.w3.org/2000/svg','circle');
      c.setAttribute('class','dot'); c.setAttribute('r','4');
      c.setAttribute('stroke',s.color); svg.appendChild(c); return c;
    });

    function idxFor(clientX){
      var box=svg.getBoundingClientRect();
      /* viewBox units, not pixels: the svg scales with the column */
      var vb=svg.viewBox.baseVal.width||780;
      var ux=(clientX-box.left)*(vb/box.width);
      var f=(ux-spec.x0)/spec.w;
      return Math.max(0,Math.min(spec.n-1,Math.round(f*(spec.n-1))));
    }
    function xOf(i){
      return spec.x0+(spec.n>1?spec.w*(i/(spec.n-1)):spec.w/2);
    }
    function show(clientX){
      var i=idxFor(clientX), x=xOf(i);
      cross.setAttribute('x1',x); cross.setAttribute('x2',x);
      /* NOT '' -- clearing the inline style falls back to the stylesheet,
         where these start at display:none, so the crosshair never appeared */
      cross.style.display='block';
      /* values ARE reachable without this -- the history table below holds
         every run's numbers, and the best epoch is direct-labelled */
      while(tip.firstChild) tip.removeChild(tip.firstChild);
      var head=document.createElement('div');
      head.className='tth';
      head.appendChild(document.createTextNode('epoch '+(i+1)));
      tip.appendChild(head);
      spec.series.forEach(function(s,k){
        var v=s.values[i];
        var row=document.createElement('div'); row.className='ttr';
        var key=document.createElement('i'); key.style.background=s.color;
        var val=document.createElement('b');
        /* series names come from directory names -- untrusted text */
        val.appendChild(document.createTextNode(
          v===null||v===undefined?'--':(+v).toFixed(4)));
        var nm=document.createElement('span');
        nm.appendChild(document.createTextNode(s.name));
        row.appendChild(key); row.appendChild(val); row.appendChild(nm);
        tip.appendChild(row);
        var d=dots[k];
        if(v===null||v===undefined){ d.style.display='none'; return; }
        d.setAttribute('cx',x);
        d.setAttribute('cy',yAt(s,i));
        d.style.display='block';
      });
      tip.hidden=false;
      var box=svg.getBoundingClientRect(), fb=fig.getBoundingClientRect();
      var px=box.left-fb.left+(x/(svg.viewBox.baseVal.width||780))*box.width;
      var tw=tip.offsetWidth;
      tip.style.left=Math.max(4,Math.min(fb.width-tw-4,px+12))+'px';
      tip.style.top='34px';
    }
    /* the y a value maps to, recomputed from the same range the server used */
    var all=[];
    spec.series.forEach(function(s){ s.values.forEach(function(v){
      if(v!==null&&v!==undefined) all.push(v); }); });
    var lo=Math.min.apply(null,all), hi=Math.max.apply(null,all);
    var lo0=lo;
    if(hi-lo<1e-9){ var p=Math.max(Math.abs(hi)*0.05,0.01); lo-=p; hi+=p; }
    else { var pad=(hi-lo)*0.08; lo-=pad; hi+=pad; }
    /* Python's _nice() clamps the low end to 0 for non-negative data. Without
       the same clamp here the axes differ and the hover dot sits off the line
       on every loss chart. */
    if(lo0>=0&&lo<0) lo=0;
    function yAt(s,i){
      var v=s.values[i];
      return spec.top+spec.h-spec.h*((v-lo)/((hi-lo)||1));
    }

    function hide(){
      cross.style.display='none'; tip.hidden=true;
      dots.forEach(function(d){ d.style.display='none'; });
    }
    hit.addEventListener('pointermove',function(e){ show(e.clientX); });
    hit.addEventListener('pointerleave',hide);
    /* keyboard gets the same readout as the pointer */
    fig.tabIndex=0;
    var kb=0;
    fig.addEventListener('keydown',function(e){
      if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight') return;
      kb=Math.max(0,Math.min(spec.n-1,kb+(e.key==='ArrowRight'?1:-1)));
      e.preventDefault();
      var box=svg.getBoundingClientRect();
      show(box.left+(xOf(kb)/(svg.viewBox.baseVal.width||780))*box.width);
    });
    fig.addEventListener('blur',hide);
  });

  /* filter and pagination are ONE pass: paging the raw row list would page
     rows the filter has already removed, and the reader would land on an
     empty page 3 of a two-row project. */
}

function initTracker(){
  bindCharts();
  var sel=document.getElementById('tproj'),
      rows=[].slice.call(document.querySelectorAll('.thist tbody tr')),
      pager=document.querySelector('.tpage'),
      pnum=pager&&pager.querySelector('.tpn'),
      PER=8, page=0;
  function draw(){
    var want=sel?sel.value:'';
    var keep=rows.filter(function(tr){
      return !want||tr.getAttribute('data-proj')===want;
    });
    var pages=Math.max(1,Math.ceil(keep.length/PER));
    if(page>=pages) page=pages-1;
    if(page<0) page=0;
    rows.forEach(function(tr){ tr.style.display='none'; });
    keep.slice(page*PER,(page+1)*PER).forEach(function(tr){
      tr.style.display='';
    });
    if(pager){
      pager.hidden=pages<2;
      if(pnum){
        while(pnum.firstChild) pnum.removeChild(pnum.firstChild);
        pnum.appendChild(document.createTextNode(
          (page*PER+1)+'\u2013'+Math.min(keep.length,(page+1)*PER)+
          ' of '+keep.length));
      }
      [].forEach.call(pager.querySelectorAll('.tpb'),function(b){
        var d=+b.getAttribute('data-d');
        b.disabled=(d<0&&page===0)||(d>0&&page>=pages-1);
      });
    }
  }
  if(pager){
    [].forEach.call(pager.querySelectorAll('.tpb'),function(b){
      b.addEventListener('click',function(){
        page+=+b.getAttribute('data-d'); draw();
      });
    });
  }
  if(sel) sel.addEventListener('change',function(){ page=0; draw(); });
  if(rows.length) draw();
  /* Clicking a run opens its metrics in the same region the live run uses --
     one detail view, not a second layout that could drift from it. The key is
     project/name and the server resolves it against the runs it already
     discovered; a path from the client is never accepted. */
  var det=document.getElementById('trkdet');
  function openRun(key,tr){
    if(!det) return;
    fetch('/api/training/run?key='+encodeURIComponent(key))
      .then(function(r){return r.json()}).then(function(j){
        if(!j||!j.html) return;
        det.innerHTML=j.html;
        window.__trkSel=key;
        [].forEach.call(document.querySelectorAll('.thist tbody tr'),
          function(x){ x.classList.toggle('sel',x===tr); });
        bindCharts();
        var back=document.getElementById('trkBack');
        if(back) back.addEventListener('click',function(){
          window.__trkSel=null; refreshTracker(true);
        });
        det.scrollIntoView({block:'nearest',
          behavior:(matchMedia('(prefers-reduced-motion:reduce)').matches
                    ?'auto':'smooth')});
      }).catch(function(){});
  }
  [].forEach.call(document.querySelectorAll('.thist tbody tr'),function(tr){
    var key=tr.getAttribute('data-key');
    if(!key) return;
    tr.addEventListener('click',function(){ openRun(key,tr); });
    tr.addEventListener('keydown',function(e){
      if(e.key==='Enter'||e.key===' '){ e.preventDefault(); openRun(key,tr); }
    });
  });
  if(window.__trkSel){
    var keep=document.querySelector(
      '.thist tbody tr[data-key="'+window.__trkSel.replace(/"/g,'\\"')+'"]');
    if(keep) openRun(window.__trkSel,keep);
  }
}
initTracker();
/* Re-render the section from the server on an interval. The markup comes from
   render_training() so the client never re-implements a chart; after the swap
   the hover layer is re-bound because the old nodes are gone. Paused while the
   tab is hidden -- a background tab polling a duckdb-backed endpoint every
   30s is pure waste. */
var refreshTracker;
(function(){
  var host=document.getElementById('trk');
  if(!host) return;
  var busy=false;
  function refresh(force){
    /* a forced refresh (back to the live run) runs even mid-poll */
    if((busy||document.hidden)&&!force) return;
    busy=true;
    fetch('/api/training').then(function(r){return r.json()}).then(function(j){
      /* keep the previous render on failure rather than blanking the panel */
      if(j&&j.html){
        var open=document.getElementById('tproj');
        var want=open?open.value:'';
        host.innerHTML=j.html;
        var sel=document.getElementById('tproj');
        if(sel&&want){ sel.value=want; }
        initTracker();
        if(sel&&want){ sel.dispatchEvent(new Event('change')); }
      }
    }).catch(function(){}).then(function(){ busy=false; });
  }
  refreshTracker=refresh;
  setInterval(refresh,30000);
  document.addEventListener('visibilitychange',function(){
    if(!document.hidden) refresh();
  });
})();

/* ── review queue depth in the header ── */
(function(){
  var btn=document.getElementById('revBtn'),
      num=document.getElementById('revN'),
      lab=document.getElementById('revL');
  if(!btn||!num) return;
  function paint(n){
    if(n===null||n===undefined){ num.textContent='\u2014'; return; }
    btn.classList.toggle('quiet',n===0);
    num.textContent=n===0?'Review':n.toLocaleString();
    lab.textContent=n===0?'nothing waiting':(n===1?'to review':'to review');
  }
  function poll(){
    if(document.hidden) return;
    fetch('/api/review/count').then(function(r){return r.json()})
      .then(function(j){ paint(j&&typeof j.left==='number'?j.left:null); })
      /* keep the last good number rather than blanking the button */
      .catch(function(){});
  }
  poll();
  setInterval(poll,30000);
  document.addEventListener('visibilitychange',function(){
    if(!document.hidden) poll();
  });
})();
function genCommands(){
  var region=(cmdRegion.value||'').trim();
  if(!region){cmdOut.innerHTML='';return;}
  fetch('/api/commands?region='+encodeURIComponent(region)).then(function(r){return r.json()}).then(function(j){
    if(j.error){cmdOut.innerHTML='<div style="color:#d8743a;padding:8px 2px">unknown region: '+esc(region)+'</div>';return;}
    var labels=['① Extract','② Coverage audit','③ Backfill metadata (no download)','④ Download images','⑤ Consolidate data → one drive (dry-run; add --execute)'];
    cmdOut.innerHTML=j.commands.map(function(c,i){
      return '<div class="cmdblock"><div class="cmdhead"><span>'+labels[i]+'</span><button type="button" class="cp" data-i="'+i+'" title="Copy command">'+COPY_SVG+'</button></div><pre>'+esc(c)+'</pre></div>';
    }).join('');
    cmdOut.querySelectorAll('.cp').forEach(function(b){b.addEventListener('click',function(e){e.preventDefault();copyText(j.commands[+b.dataset.i])})});
  }).catch(function(){cmdOut.innerHTML='<div style="color:#d8743a;padding:8px 2px">failed to generate</div>'});
}
if(cmdGen){cmdGen.addEventListener('click',genCommands);cmdRegion.addEventListener('keydown',function(e){if(e.key==='Enter')genCommands()});}
/* ── ground-animal density map (geo-anchored raster, zoom-adaptive) ── */
(function(){
  /* click-to-interact gate: the scrim eats wheel and drag, so the page keeps
     scrolling past the map until the user asks for the map instead. Esc, the
     lock button, or scrolling the map out of view re-arms it. */
  (function(){
    var gate=document.getElementById('mapGate'),lock=document.getElementById('mapLock'),
        wrap=gate&&gate.parentNode;
    if(!gate||!lock)return;
    function setLocked(on){
      gate.hidden=!on;lock.hidden=on;
    }
    gate.addEventListener('click',function(){setLocked(false)});
    lock.addEventListener('click',function(){setLocked(true)});
    document.addEventListener('keydown',function(e){
      if(e.key==='Escape'&&!lock.hidden)setLocked(true);
    });
    /* re-arm once the map leaves the viewport, so it is never left live
       under a scroll the user has moved on from */
    if(window.IntersectionObserver&&wrap){
      new IntersectionObserver(function(es){
        if(es[0]&&!es[0].isIntersecting&&!lock.hidden)setLocked(true);
      },{threshold:0}).observe(wrap);
    }
  })();
  var mapEl=document.getElementById('map');
  if(!mapEl||typeof echarts==='undefined')return;
  Promise.all([
    fetch('world.json').then(function(r){return r.json()}),
    fetch('map_points.json').then(function(r){return r.json()})
  ]).then(function(res){
    var world=res[0],md=res[1];
    echarts.registerMap('world',world);
    var levels=md.levels||{};
    if(!Object.keys(levels).length&&md.points)levels[String(md.res)]={res:md.res,max:md.max,points:md.points};
    var keys=Object.keys(levels).map(parseFloat).sort(function(a,b){return b-a}); // coarse→fine
    var cache={};
    function lvl(k){
      var s=String(k);
      if(!cache[s]){
        var L=levels[s];
        cache[s]={res:L.res,maxLog:Math.log10((L.max||1)+1),
          data:L.points.map(function(p){return {value:[p[0],p[1],Math.log10(p[2]+1)],cnt:p[2]}})};
      }
      return cache[s];
    }
    var cur=lvl(keys[0]);
    var ch=echarts.init(mapEl,null,{renderer:'canvas'});
    function cellPx(res){ /* pixel footprint of one res° cell at current zoom */
      try{
        var a=ch.convertToPixel({geoIndex:0},[0,0]),b=ch.convertToPixel({geoIndex:0},[res,res]);
        return [Math.max(Math.abs(b[0]-a[0]),1.1),Math.max(Math.abs(b[1]-a[1]),1.1)];
      }catch(e){return [3,3];}
    }
    ch.setOption({
      backgroundColor:'transparent',
      tooltip:{trigger:'item',backgroundColor:'#21262d',borderColor:'#2c333b',borderWidth:1,textStyle:{color:'#eef1f4'},formatter:function(p){return p.data?'<b>'+fmt(p.data.cnt)+'</b> ground animals<br><span style="color:#98a2ad">'+cur.res+'° cell</span>':''}},
      geo:{map:'world',roam:true,scaleLimit:{min:1,max:40},itemStyle:{areaColor:'#171c22',borderColor:'#2c333b',borderWidth:.5},emphasis:{disabled:true},select:{disabled:true}},
      visualMap:{type:'continuous',min:0,max:cur.maxLog,dimension:2,calculable:true,left:14,bottom:20,itemHeight:130,itemWidth:12,text:['dense','sparse'],textStyle:{color:'#98a2ad',fontSize:11},
        inRange:{color:['#160f3c','#451077','#7b2382','#b0357b','#e34e65','#fb8861','#fec287','#fcfdbf']}},
      series:[{name:'ground animals',type:'scatter',coordinateSystem:'geo',symbol:'rect',
        data:cur.data,symbolSize:3,itemStyle:{opacity:.92},progressive:6000,progressiveThreshold:10000}]
    });
    ch.setOption({series:[{symbolSize:cellPx(cur.res)}]});   // size once geo exists
    var t=null;
    ch.on('georoam',function(){
      if(t)clearTimeout(t);
      t=setTimeout(function(){
        var g=ch.getOption().geo[0],want=keys[0];
        for(var i=0;i<keys.length;i++)if(g.zoom>=(i===0?0:4.5*i))want=keys[i];
        var L=lvl(want),upd={series:[{symbolSize:cellPx(L.res)}]};
        if(L!==cur){cur=L;upd.visualMap={max:L.maxLog};upd.series[0].data=L.data;}
        ch.setOption(upd);
      },130);
    });
    window.addEventListener('resize',function(){ch.resize()});
  }).catch(function(){mapEl.innerHTML='<div style="color:#69727d;padding:40px;text-align:center">map data unavailable</div>'});
})();
var bEl=document.getElementById('bars');
if(typeof echarts==='undefined'){
  bEl.innerHTML='<div style="color:#69727d;padding:50px;text-align:center">Chart library not loaded.</div>';
}else{
var grid={lineStyle:{color:'rgba(130,140,150,.09)'}};
echarts.init(bEl,null,{renderer:'canvas'}).setOption({
  backgroundColor:'transparent',
  tooltip:{trigger:'axis',axisPointer:{type:'shadow'},backgroundColor:'#21262d',borderColor:'#2c333b',borderWidth:1,textStyle:{color:'#eef1f4'},formatter:function(p){return p[0].name+'<br/><b>'+fmt(p[0].value)+'</b> ground animals'}},
  grid:{left:6,right:56,top:6,bottom:6,containLabel:true},
  xAxis:{type:'value',axisLine:{show:false},axisTick:{show:false},axisLabel:{color:'#98a2ad',fontSize:11,formatter:fmt},splitLine:grid},
  yAxis:{type:'category',data:D.regions,axisLine:{lineStyle:{color:'#2c333b'}},axisTick:{show:false},axisLabel:{color:'#c4ccd4',interval:0,fontSize:11}},
  series:[{type:'bar',data:D.dogs,barWidth:'66%',itemStyle:{borderRadius:[0,4,4,0],color:new echarts.graphic.LinearGradient(0,0,1,0,[{offset:0,color:'#d8923a'},{offset:1,color:'#f0b85f'}])},label:{show:true,position:'right',color:'#828d98',fontSize:10,formatter:function(p){return fmt(p.value)}}}]});
}
window.addEventListener('resize',function(){var c=echarts.getInstanceByDom(bEl);if(c)c.resize()});
/* ── detection sweep panel (§7.4 / A.5) ── polls /api/detect every 5 s, but
   ONLY while its fold is open and the tab is visible — every fetch consumes a
   ThreadingHTTPServer thread. The img/s sparkline (last ~15 min at 5 s
   samples) draws as the subtle background of the "img/s (now)" KPI chip.
   Raw window keys (w60/w900) never reach the page — they render as
   "img/s (now)" / "img/s (sustained)". */
(function(){
  var fold=document.getElementById('f-detect');
  if(!fold)return;
  var off=document.getElementById('detOff'),on=document.getElementById('detOn'),
      dEl=document.getElementById('detDrives'),rEl=document.getElementById('detRegions'),
      rHead=document.getElementById('detRegHead'),hEl=document.getElementById('detHealth'),
      eEl=document.getElementById('detErrs'),mEl=document.getElementById('detMeta'),
      sEl=document.getElementById('detSpark'),
      hPct=document.getElementById('dhPct'),hEta=document.getElementById('dhEta'),
      hNow=document.getElementById('dhNow'),hSus=document.getElementById('dhSus'),
      hDone=document.getElementById('dhDone'),hRun=document.getElementById('dhRun'),
      hFill=document.getElementById('dhFill'),hCount=document.getElementById('dhCount'),
      hPos=document.getElementById('dhPos');
  var POLL=5000,SPARK_N=180,timer=null,inflight=false,spark=null,samples=[];
  var lastJ=null,errOpen=false,srz=null;
  window.addEventListener('resize',function(){
    if(srz)clearTimeout(srz);
    srz=setTimeout(function(){if(spark)spark.resize()},150);
  });
  /* The cards NEVER disappear: an idle sweep (or a field the writer has not
     published) renders this dash in the number's place, so the layout is
     identical before and after the sweep starts and nothing jumps. */
  var DASH='\\u2014',live=false;
  function n(v,f){return (!live||v==null||v!==v)?DASH:f(v)}
  /* An idle payload carries no drives/regions at all, so the row rosters would
     vanish with it. Remember the last one we saw (and across reloads) and draw
     it dashed instead. */
  var roster={drives:[],regions:[]};
  try{var _r=JSON.parse(localStorage.getItem('sdDetRoster'));
      if(_r&&_r.drives&&_r.regions)roster=_r;}catch(_){}
  function keep(k,names){
    if(!names.length||roster[k].join()===names.join())return;
    roster[k]=names;
    try{localStorage.setItem('sdDetRoster',JSON.stringify(roster))}catch(_){}
  }
  function hnum(v){v=+v||0;return v>=1e6?fmt(v):v.toLocaleString('en-US')}
  function etaTxt(s){if(s==null||!isFinite(s))return '\\u2014';
    if(s<=0)return 'done';
    if(s<60)return '<1 min left';
    if(s<5400)return Math.round(s/60)+' min left';
    if(s<129600)return (s/3600).toFixed(1)+' hours left';
    return (s/86400).toFixed(1)+' days left';}
  function agoTxt(s){s=Math.max(0,Math.round(s));
    if(s<90)return s+' s ago';
    if(s<5400)return Math.round(s/60)+' min ago';
    if(s<129600)return (s/3600).toFixed(1)+' h ago';
    return (s/86400).toFixed(1)+' days ago';}
  function bar(p,color){return '<div class="bar"><div class="fill" style="width:'+Math.max(0,Math.min(+p||0,100)).toFixed(1)+'%;background:'+(color||'var(--acc)')+'"></div></div>'}
  function chip(t,warn){return '<span class="chip"'+(warn?' style="border-color:rgba(216,116,58,.5);color:#d8743a"':'')+'>'+t+'</span>'}
  function errSum(j){var e=(j&&j.errors)||{},n=0,k;for(k in e)n+=e[k]||0;return n}
  function drawSpark(){
    if(typeof echarts==='undefined'||!sEl)return;
    /* `samples` is in-memory, so it restarts empty on every page load. Two
       points on a band-centred category axis paint a line from 25% to 75% of
       the card -- a hard-edged block that reads as a rendering fault. Show
       nothing until the series is long enough to be a trend. */
    if(samples.length<4){sEl.style.display='none';return}
    sEl.style.display='';
    if(!spark)spark=echarts.init(sEl,null,{renderer:'canvas'});
    /* The canvas is sized at init(). The KPI card is a grid track whose width
       is not final on the first paint, so without this the chart keeps the
       stale size and its fill stops mid-card. */
    spark.resize();
    spark.setOption({backgroundColor:'transparent',animation:false,
      grid:{left:0,right:0,top:2,bottom:0},
      /* boundaryGap:false -- a sparkline must run edge to edge, not sit
         inset by half a band at each end */
      xAxis:{type:'category',show:false,boundaryGap:false,
             data:samples.map(function(_,i){return i})},
      yAxis:{type:'value',show:false,min:0},
      tooltip:{show:false},
      series:[{type:'line',data:samples,symbol:'none',
        lineStyle:{width:1,color:'rgba(232,166,69,.55)'},
        areaStyle:{color:'rgba(232,166,69,.10)'}}]});
  }
  function render(j){
    lastJ=j;
    live=!!(j&&j.running);
    /* idle is a STATUS LINE now — the cards below stay put and go to dashes */
    off.style.display=live?'none':'';
    if(!live)off.textContent='sweep idle'+(j&&j.age_s!=null?' \\u2014 last run '+agoTxt(j.age_s):'')+
      (j&&j.state==='failed'?' (failed)':'');
    j=j||{};
    /* headline: % complete, human ETA, now/sustained throughput. imgs_done is
       GLOBAL (all-time, across restarts) — the %, the bar and the ETA are all
       against imgs_total, never against the per-process run_imgs_done. */
    var ips=j.img_per_sec||{},rNow=+ips.w60||0,rSus=+ips.w900||0;
    var tot=+j.imgs_total||0,pct=(live&&tot)?100*(+j.imgs_done||0)/tot:0;
    /* always 2 decimals: at ~50 img/s the third digit is the only one that
       visibly moves, and a field that switches precision at 10% jitters */
    hPct.textContent=(live&&tot)?(pct.toFixed(2)+'%'):DASH;
    hDone.textContent=n(j.imgs_done,hnum);
    hEta.textContent=live?etaTxt(j.eta_s):DASH;
    hNow.textContent=n(ips.w60,function(v){return (+v).toFixed(1)});
    hSus.textContent=n(ips.w900,function(v){return (+v).toFixed(1)});
    hFill.style.width=(live?Math.min(pct,100):0).toFixed(2)+'%';
    if(hPos)hPos.textContent=n(j.positives,hnum);
    /* every figure on this line is ALL-TIME, so say so -- the same line used
       to mix a global image count with a per-process positives count and
       reported a 0.1% hit rate against a true ~2.8% */
    hCount.textContent='all-time: '+n(j.imgs_done,hnum)+' of '+
      (tot?hnum(tot):DASH)+' images \\u00b7 '+n(j.positives,hnum)+' positives'+
      (j.positive_rate!=null?' ('+j.positive_rate+'%)':'');
    if(hRun)hRun.textContent='this run: '+n(j.run_imgs_done,hnum)+' images'+
      (j.run_positives!=null?' \\u00b7 '+n(j.run_positives,hnum)+' positives':'')+
      (live&&j.started_at?' \\u00b7 since '+j.started_at:'');
    if(live){samples.push(+rNow.toFixed(1));if(samples.length>SPARK_N)samples.shift();}
    drawSpark();
    /* per drive: name · bar · rate; queue only when nonzero, badge only when
       stalled. Idle keeps the rows (from the remembered roster) and dashes
       the numbers rather than dropping the block. */
    var dr=j.drives||{},dk=Object.keys(dr).sort();
    keep('drives',dk);
    dEl.innerHTML=(dk.length?dk:roster.drives).map(function(nm){
      var d=dr[nm]||{},known=live&&d.total!=null,p=known&&d.total?100*d.done/d.total:0;
      return '<div class="drow'+(known?'':' dmut')+'"><span class="dn">'+esc(nm)+'</span>'+
        bar(known?p:0,pctColor(known?p:0))+
        '<span class="dv">'+(known&&d.total?p.toFixed(0)+'% \\u00b7 ':'')+
        (live?n(d.rate,function(v){return (+v).toFixed(1)}):DASH)+' img/s'+
        (live&&d.queue_depth?' \\u00b7 queue '+fmt(d.queue_depth):'')+'</span>'+
        /* The 30 s threshold was tuned against a USB enclosure that parks its
           disk aggressively; the badge means "no bytes moved", which is a
           spin-down on that hardware but may be anything on yours. The tooltip
           says what was MEASURED, not what this one host's enclosure does. */
        (live&&d.stalled?'<span class="dbadge" title="no progress for 30 s">stalled</span>':'')+
        '</div>';
    }).join('')||'<div class="dnone">no drives reported</div>';
    /* regions: the FULL planned roster (scrolls past ~300px) so a partial
       sweep never reads as "only these regions exist".
       Order: IN PROGRESS first (that is the only group anything is happening
       to), nearest-to-done first within it; then finished; then untouched,
       muted. A plain % desc put every 100% region above the ones actually
       moving -- which is the comment this code carried while doing the
       opposite. */
    var rg=j.regions||{},rk=Object.keys(rg);
    keep('regions',rk);
    function rank(r){
      if(r.p!=null&&r.p>0&&r.p<100) return 0;   /* moving */
      if(r.p!=null&&r.p>=100) return 1;         /* done */
      return 2;                                  /* not started */
    }
    var all=(rk.length?rk:roster.regions).map(function(nm){
        return {n:nm,p:(live&&rg[nm]!=null)?+rg[nm]||0:null}})
      .sort(function(a,b){
        var ra=rank(a),rb=rank(b);
        if(ra!==rb) return ra-rb;
        if(ra===0) return (b.p||0)-(a.p||0)||a.n.localeCompare(b.n);
        return a.n.localeCompare(b.n);
      });
    var prog=all.filter(function(r){return r.p>0&&r.p<100});
    rHead.textContent=all.length?'Per region \\u2014 '+(live?prog.length:DASH)+
      ' of '+all.length+' in progress':'Per region';
    rEl.innerHTML=all.map(function(r){
      return '<div class="drow'+(r.p>0?'':' dmut')+'"><span class="dn">'+esc(r.n.replace(/_/g,' '))+'</span>'+
        bar(r.p||0,pctColor(r.p||0))+
        '<span class="dv">'+n(r.p,function(v){return v.toFixed(1)+'%'})+'</span></div>';
    }).join('')||'<div class="dnone">no per-region data</div>';
    /* classifier: gauge only once crops actually flow (A.5) */
    if(!live){
      hEl.innerHTML='<div class="drow"><span class="dn">not_a_dog</span>'+
        '<div class="dband"></div><span class="dv">'+DASH+'</span></div>';
    }else if((j.crops_classified||0)>0){
      var band=j.not_a_dog_band||{lo:7,hi:16},nd=j.not_a_dog_rate,SCALE=30;
      var bad=band.in_band===false;
      hEl.innerHTML='<div class="drow" title="share of detected crops that aren\\u2019t dogs \\u2014 expected '+band.lo+'\\u2013'+band.hi+'% from labelled data">'+
        '<span class="dn">not_a_dog</span>'+
        '<div class="dband"><div class="zone" style="left:'+(100*band.lo/SCALE)+'%;width:'+(100*(band.hi-band.lo)/SCALE)+'%"></div>'+
        (nd!=null?'<div class="cur'+(bad?' bad':'')+'" style="left:'+Math.min(100*nd/SCALE,99)+'%"></div>':'')+
        '</div><span class="dv">'+(nd!=null?nd+'%'+(bad?' \\u26a0 outside ':' \\u00b7 healthy ')+band.lo+'\\u2013'+band.hi+'%':'\\u2014')+'</span></div>'+
        '<div class="dnone">'+fmt(j.crops_classified)+' crops classified</div>';
    }else{
      hEl.innerHTML='<div class="dnone">classifier not wired in yet</div>';
    }
    /* errors: one muted line; details expand on click; green zero state */
    var errN=errSum(j),errs=j.errors||{};
    if(!live){
      eEl.innerHTML='<div class="derr">'+DASH+' errors</div>';
    }else if(!errN){
      eEl.innerHTML='<div class="derr ok">0 errors</div>';
    }else{
      eEl.innerHTML='<div class="derr"><span class="dt" data-t="err">'+fmt(errN)+
        ' error'+(errN===1?'':'s')+' \\u2014 '+(errOpen?'hide':'details')+'</span>'+
        '<div class="dchips"'+(errOpen?'':' style="display:none"')+'>'+
        Object.keys(errs).filter(function(k){return errs[k]>0}).map(function(k){
          return chip(esc(k)+' <b>'+fmt(errs[k])+'</b>',true)}).join('')+
        (j.last_error?chip('last: '+esc(j.last_error),true):'')+
        (j.publish_errors?chip('publish <b>'+fmt(j.publish_errors)+'</b>',true):'')+
        '</div></div>';
    }
    /* muted run meta */
    var g=j.gpu,meta=[];
    if(j.run_id)meta.push('run '+esc(j.run_id)+(j.gen!=null?' \\u00b7 gen '+j.gen:''));
    if(g)meta.push('GPU '+(g.util!=null?g.util+'%':'\\u2014')+(g.temp!=null?' \\u00b7 '+g.temp+'\\u00b0C':''));
    if(j.boxes_per_img!=null)meta.push(j.boxes_per_img+' boxes/img');
    if(j.started_at)meta.push('since '+esc(j.started_at));
    mEl.innerHTML=meta.join('&ensp;\\u00b7&ensp;')||DASH;
  }
  eEl.addEventListener('click',function(ev){var t=ev.target;
    if(t&&t.getAttribute&&t.getAttribute('data-t')==='err'){errOpen=!errOpen;if(lastJ)render(lastJ);}});
  function tick(){
    if(inflight)return;
    inflight=true;
    fetch('/api/detect').then(function(r){return r.json()}).then(render)
      .catch(function(){render(null)})
      .then(function(){inflight=false;schedule();});
  }
  function schedule(){
    if(timer)clearTimeout(timer);
    if(!fold.open||document.hidden)return;      /* no polling when unseen */
    timer=setTimeout(tick,POLL);
  }
  fold.addEventListener('toggle',function(){
    if(fold.open){tick();if(spark)spark.resize();}
    else if(timer)clearTimeout(timer);
  });
  document.addEventListener('visibilitychange',function(){
    if(!document.hidden&&fold.open)tick();
    else if(timer)clearTimeout(timer);
  });
  if(fold.open)tick();
})();
/* ── live detection crops (§7.4) ── a SEPARATE 60 s loop, deliberately NOT
   folded into the 5 s status poll: the writer emits ~4 crops/s and prunes at
   120 s, so re-fetching thumbnails every 5 s would be ~12x the bytes for the
   same handful of pictures. Same gating as the status poll (fold open + tab
   visible), and the grid sits inside #detOn so an idle sweep hides it with
   the rest of the panel. The sample is drawn server-side, so "↻ Shuffle" is
   just an immediate re-fetch.

   The API hands back up to CROP_CAP (24) but the grid shows only a WHOLE
   number of rows: 12 tiles in an 8-wide grid left a ragged 8+4 second row
   with dead space beside it. We measure the real column count after layout
   and slice to floor(available/cols)*cols, capped at ROWS. Re-fitted on a
   debounced resize and on every refresh/Shuffle.

   Clicking a tile whose full frame exists (has_full) opens the lightbox on
   /recent_crops/full/<name> — the SAME frame with the box already burned in by
   PreviewWriter. Nothing is drawn client-side; there is no canvas here. */
(function(){
  var fold=document.getElementById('f-detect'),
      grid=document.getElementById('dcropGrid'),
      sub=document.getElementById('dcropSub'),
      btn=document.getElementById('dcropShuffle'),
      fEl=document.getElementById('dcropFlagged');
  if(!fold||!grid)return;
  var POLL=60000,timer=null,inflight=false;
  var ROWS=2,MINW=110,GAP=8;   /* must match .dcrops minmax()/gap in the CSS */
  /* `shown` is what the grid currently holds; makeLightbox() takes its own
     SNAPSHOT of it when opened, so a 60 s refresh landing mid-view can neither
     close the overlay nor swap the picture out from under the arrow keys. */
  var shown=[];
  /* `pool` is everything the API returned; the grid paints a fitted slice */
  var pool=[],poolN=0,rz=null;
  /* false-positive flags. The server copies the pixels out of the rolling
     window at flag time, so this Set is only the view state; it is re-seeded
     from the payload on every refresh so a flag survives Shuffle and reload.
     `pending` holds names whose POST is still in flight, so a refresh landing
     mid-request cannot revert the tile the user just clicked. */
  var flagged=new Set(),pending=new Set(),flagTotal=0;
  var FLAG_ON='flagged as false positive \\u2014 click to undo',
      FLAG_OFF='flag as false positive (not a dog)';
  /* esc() leaves quotes alone; these values land in a title="" attribute */
  function att(s){return esc(s).replace(/"/g,'&quot;')}
  function cap(c){
    return 'image_id '+c.image_id+' \\u00b7 conf '+(+c.conf||0).toFixed(2)+
      ' \\u00b7 '+Math.max(0,Math.round(+c.age_s||0))+'s ago';
  }
  /* the tracks auto-fill actually resolved to; the arithmetic is only a
     fallback for a grid that has not been laid out yet (fold still closed) */
  function cols(){
    var c=0;
    try{
      var t=typeof getComputedStyle==='function'&&getComputedStyle(grid).gridTemplateColumns;
      if(t&&t!=='none')c=t.split(/\\s+/).filter(function(s){return s}).length;
    }catch(_){}
    if(!c)c=Math.floor(((grid.clientWidth||0)+GAP)/(MINW+GAP));
    return Math.max(1,c);
  }
  /* whole rows only: floor(have/cols)*cols, at most ROWS. If we cannot even
     fill one row we show what we have — that is a single short row, never a
     ragged row trailing a full one. */
  function fit(have){
    var c=cols(),full=Math.min(Math.floor(have/c),ROWS);
    return full>0?full*c:have;
  }
  function paint(){
    var cs=pool.slice(0,fit(pool.length));
    shown=cs;
    if(sub)sub.textContent='random sample from the last minute \\u00b7 '+poolN+' found';
    flagLine();
    if(!cs.length){
      grid.innerHTML='<div class="dnone" style="grid-column:1/-1">no detections in the last minute</div>';
      return;
    }
    grid.innerHTML=cs.map(function(c,i){
      var conf=(+c.conf||0).toFixed(2),age=Math.max(0,Math.round(+c.age_s||0)),
          hit=!!c.has_full,   /* no full frame on disk -> not clickable */
          fl=flagged.has(c.name);
      return '<div class="dcrop'+(hit?' cx':'')+(fl?' fl':'')+'"'+
        (hit?' data-i="'+i+'" role="button" tabindex="0"':'')+
        ' title="image_id '+att(c.image_id)+' \\u00b7 conf '+conf+' \\u00b7 '+age+
        's ago'+(fl?' \\u2014 flagged':hit?' \\u2014 click for the full frame':'')+'">'+
        '<button type="button" class="fx" data-fx="'+att(c.name)+'" title="'+
        (fl?FLAG_ON:FLAG_OFF)+'" aria-pressed="'+(fl?'true':'false')+
        '">\\u2691</button>'+
        '<img loading="lazy" alt="detection crop" src="/recent_crops/'+
        encodeURIComponent(''+c.name)+'"><span class="cf">'+conf+'</span></div>';
    }).join('');
  }
  function flagLine(){
    if(fEl)fEl.textContent=flagTotal>0?flagTotal+' flagged as false positive':'';
  }
  /* Optimistic: the tile flips at once and rolls back if the POST fails. The
     copy out of the rolling window happens server-side before it answers. */
  function toggleFlag(name){
    if(!name)return;
    var on=!flagged.has(name);
    function put(v){
      if(v)flagged.add(name);else flagged['delete'](name);
      flagTotal=Math.max(0,flagTotal+(v?1:-1));
      paint();syncLbFlag();
    }
    put(on);
    pending.add(name);
    function settle(){pending['delete'](name)}
    fetch('/api/detect/flag',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:name,label:'false_positive',undo:!on})})
      .then(function(r){return r.json()})
      .then(function(j){
        settle();
        if(!j||!j.ok){put(!on);return;}          /* server refused -> revert */
        if(j.flagged_total!=null){flagTotal=j.flagged_total;paint();syncLbFlag();}
      })
      .catch(function(){settle();put(!on)});
  }
  function renderCrops(j){
    pool=(j&&j.crops)||[];poolN=(j&&j.total_last_min)||0;
    /* the server is authoritative, except for flags still in flight */
    if(j&&j.flagged){
      var s=new Set(j.flagged);
      pending.forEach(function(nm){
        if(flagged.has(nm))s.add(nm);else s['delete'](nm);
      });
      flagged=s;
    }
    if(j&&j.flagged_total!=null)flagTotal=j.flagged_total;
    paint();syncLbFlag();
  }
  /* a resize changes the column count, so the whole-row slice changes with it */
  window.addEventListener('resize',function(){
    if(rz)clearTimeout(rz);
    rz=setTimeout(paint,150);
  });
  /* the shared component (see makeLightbox) — identical overlay on /review */
  var LB=makeLightbox({flagged:function(n){return flagged.has(n)},
                       toggle:function(n){toggleFlag(n)}});
  function openLb(i){if(shown.length)LB.open(shown,i)}
  function closeLb(){LB.close()}
  function step(d){LB.step(d)}
  function syncLbFlag(){LB.sync()}
  function hitOf(e){
    var t=e&&e.target;
    return t&&t.closest?t.closest('.dcrop.cx'):null;
  }
  function fxOf(e){
    var t=e&&e.target;
    return t&&t.closest?t.closest('.fx'):null;
  }
  /* The flag button sits INSIDE the clickable tile, so it must swallow the event or
     flagging would also open the lightbox. */
  function onGridClick(e){
    var fx=fxOf(e);
    if(fx){
      if(e.stopPropagation)e.stopPropagation();
      if(e.preventDefault)e.preventDefault();
      toggleFlag(fx.getAttribute('data-fx'));
      return;
    }
    var t=hitOf(e);
    if(t)openLb(+t.getAttribute('data-i')||0);
  }
  grid.addEventListener('click',onGridClick);
  grid.addEventListener('keydown',function(e){
    if(e.key!=='Enter'&&e.key!==' ')return;
    var fx=fxOf(e);
    if(fx){
      e.preventDefault();
      if(e.stopPropagation)e.stopPropagation();
      toggleFlag(fx.getAttribute('data-fx'));
      return;
    }
    var t=hitOf(e);
    if(t){e.preventDefault();openLb(+t.getAttribute('data-i')||0)}
  });
  /* NO lightbox listeners here: makeLightbox() binds the scrim, close, prev,
     next, flag and the Escape/arrow keys itself. Binding them a second time
     made every ‹/› click and every arrow press fire step() twice, so the
     overlay skipped a crop each time. */
  function load(){
    if(inflight)return;
    inflight=true;
    fetch('/api/detect/crops').then(function(r){return r.json()}).then(renderCrops)
      .catch(function(){renderCrops(null)})
      .then(function(){inflight=false;schedule();});
  }
  function schedule(){
    if(timer)clearTimeout(timer);
    if(!fold.open||document.hidden)return;     /* no polling when unseen */
    timer=setTimeout(load,POLL);
  }
  if(btn)btn.addEventListener('click',function(){load()});
  fold.addEventListener('toggle',function(){
    if(fold.open)load();else if(timer)clearTimeout(timer);
  });
  document.addEventListener('visibilitychange',function(){
    if(!document.hidden&&fold.open)load();
    else if(timer)clearTimeout(timer);
  });
  if(fold.open)load();
})();
/* ── sweep stop/resume ── confirmation required; stop is graceful
   (SIGTERM -> the sweep commits its contiguous prefix), resume replays only
   uncommitted ranges, so neither loses work. ── */
(function(){
  var b=document.getElementById('sweepBtn'),pill=document.getElementById('sweepState');
  if(!b)return;
  var running=null,busy=false;
  function paint(r){
    running=r;
    if(busy)return;                 /* don't stomp "Stopping…" mid-flight */
    b.disabled=false;
    b.className='rbtn sw '+(r?'stop':'go');
    b.textContent=r?'Stop sweep':'Resume sweep';
    b.title=r?'stop the detection sweep — asks for confirmation first'
             :'resume the detection sweep from where it stopped';
    if(pill){pill.className='swpill '+(r?'on':'off');
             pill.textContent=r?'running':'stopped'}
  }
  function poll(){
    fetch('/api/sweep').then(function(x){return x.json()})
      .then(function(j){paint(!!j.running)}).catch(function(){});
  }
  b.addEventListener('click',function(e){
    /* the button lives inside <summary>: a bare click would fold the panel */
    e.preventDefault();e.stopPropagation();
    if(running===null||busy)return;
    /* NOTE: TEMPLATE is a non-raw Python string -- a single \n here would be
       consumed by Python and emit a real newline INSIDE this JS string
       literal, which is a syntax error that kills the whole page script. */
    var msg=running
      ? 'Stop the detection sweep?\\n\\nIt finishes the images already in '
        + 'flight and commits them, so no work is lost. Resume picks up '
        + 'exactly where it left off.'
      : 'Resume the detection sweep?\\n\\nIt restarts from the last committed '
        + 'position — already-processed images are not redone.';
    if(!window.confirm(msg))return;
    var wasRunning=running;
    busy=true;b.disabled=true;
    b.textContent=wasRunning?'Stopping…':'Resuming…';
    if(pill){pill.className='swpill off';
             pill.textContent=wasRunning?'stopping':'starting'}
    fetch('/api/sweep',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:wasRunning?'stop':'resume'})})
     .then(function(x){return x.json()})
     .then(function(j){
        if(typeof toast==='function'&&j&&j.msg)toast(j.msg);
        /* stop is asynchronous (drain + commit): hold the busy label until
           /api/sweep actually reports the new state, or ~30 s have passed */
        var n=0,iv=setInterval(function(){
          fetch('/api/sweep').then(function(x){return x.json()})
            .then(function(s){
               if(!!s.running!==wasRunning||++n>20){
                 clearInterval(iv);busy=false;paint(!!s.running);
               }
            }).catch(function(){if(++n>20){clearInterval(iv);busy=false;poll()}});
        },1500);
     }).catch(function(){busy=false;poll()});
  });
  /* keyboard activation of the button must not fold the panel either */
  b.addEventListener('keydown',function(e){
    if(e.key===' '||e.key==='Enter')e.stopPropagation();
  });
  poll();setInterval(poll,10000);
})();
/* ── foldable sections ── state survives the hourly auto-refresh ── */
(function(){
  var KEY='sdFolds',saved={};
  try{saved=JSON.parse(localStorage.getItem(KEY))||{}}catch(_){}
  Array.prototype.forEach.call(document.querySelectorAll('details.fold'),function(d){
    if(saved[d.id]===false)d.open=false;
    d.addEventListener('toggle',function(){
      saved[d.id]=d.open;
      try{localStorage.setItem(KEY,JSON.stringify(saved))}catch(_){}
      /* a chart initialised while hidden has zero size — re-measure on open */
      if(d.open)Array.prototype.forEach.call(d.querySelectorAll('#map,#bars,#detSpark'),
        function(el){var c=echarts.getInstanceByDom(el);if(c)c.resize()});
    });
  });
})();
</script></body></html>"""

if __name__ == '__main__':
    main()
