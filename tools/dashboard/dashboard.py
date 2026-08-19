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
import atexit
import collections
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
import threading
import time
from datetime import datetime
from urllib.parse import parse_qs, quote, urlparse
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


def cfg_bool(key, default=False, env=None):
    """A yes/no config key, from a JSON boolean, a number or a spelling of one.

    Same reason cfg_int exists: cfg() returns strings only, so a key written
    the natural way -- true, not "true" -- warned and fell through to the
    default, which for a switch means the feature silently stayed as it was.
    JSON 1 and 0 count as spellings too: they are as natural a way to write a
    switch as "1"/"0", and refusing the unquoted form while accepting the
    quoted one is the exact silent fall-through this function was written to
    remove. Anything else that is present but unreadable WARNS before the
    default is used -- a written key must never die in silence.
    """
    envname = env or ('DASHBOARD_' + key.upper())
    raw = os.environ.get(envname)
    v = raw if raw else load_cfg().get(key)
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('1', 'true', 'yes', 'on'):
            return True
        if s in ('0', 'false', 'no', 'off', ''):
            return False
    if v is not None:
        src = envname if raw else CFG_PATH
        sys.stderr.write(
            f'warning: {src}: "{key}" is {v!r}, which does not read as a '
            f'boolean -- using the default ({default}). Write true/false, '
            f'1/0, or one of yes/no/on/off.\n')
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

# Regions the DASHBOARD does not show. Display only: the catalog, the
# harvest, the sweep and every command still know about them, and nothing
# here deletes a row. Both are empty in practice -- between them 130
# manifest rows, no downloaded images and no ground animals at all -- so
# they contributed two permanently-0% cards, two flat bars, two markers on
# the map and two entries in every region list, and no information.
HIDE_REGIONS = frozenset(cfg_list('hide_regions') or ('Arctic', 'Antarctica'))

# The LLM annotator is an experiment that is off by default, and off here means
# absent: no link in the header and no route claimed, so /llm is a 404 rather
# than a page nobody asked for. Nothing is deleted -- llm_page.py,
# llm_annotate.py, their guard and the store of answers already collected all
# stay, and turning this on brings the page back exactly as it was, which is
# the point of a switch rather than a revert.
#
#   tools/dashboard/dashboard.config.json:  "llm_page": true
#   then rebuild (the header is baked at build time) and restart the service.
LLM_PAGE = cfg_bool('llm_page', False)

# Kept whole rather than reassembled when the switch flips: the caption is
# doing a second job -- the two links beside it are a queue somebody judges and
# the datasets those judgements build, and an LLM's answers are neither. A
# reader who arrives at /llm expecting annotations has already formed the idea
# that whole store exists to prevent. The violet stays on that page; one
# experiment does not get a colour in the header of everything else.
#
# The HTML comment explaining the control travels inside the constant for the
# same reason: baked beside the substitution point it shipped on every page
# with the switch off, describing a button that was absent and a URL that
# 404s. Off means absent -- commentary included.
LLM_NAV = """    <!-- Quiet like the one above, and the caption is doing a second job. The
         two beside it are a queue somebody judges and the datasets those
         judgements build; an LLM's answers are neither, and saying so here,
         before the click, costs one line. A reader who arrives at /llm
         expecting annotations has already formed the idea that whole store
         exists to prevent. The violet is that page's and stays there -- one
         experiment does not get a colour in the header of everything else. -->
    <a class="revbtn quiet" href="/llm" title="What an LLM said about \
crops — experimental, kept in a store of its own, and never an annotation">
      <span class="rvf">&#9708;</span>
      <span class="rvn"><b>LLM annotator</b><em>experimental &mdash; not \
annotations</em>
      </span></a>"""

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
    } for r in rows if r[0] not in HIDE_REGIONS]
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


# The server-side twin of pctColor() in the page script: the same bar, drawn
# once at build time and once live, so the two must step the same ramp. They
# had already drifted -- same thresholds, but this one finished in #3fb27f
# and the other in #43b581, two greens nobody chose to be different.
# adv_dashboard_render.py checks them against each other.
PROGRESS_RAMP = ('#8a6529', '#a97c2e', '#c79536', '#e0ae45')


def bar_color(pct):
    """A completion bar's fill: one hue, dark to light, by how much is done."""
    if pct >= 100:
        return '#43b581'          # finished is a state, not a quantity
    return PROGRESS_RAMP[min(3, max(0, int(pct // 25)))]


# ── pipeline status board ───────────────────────────────────────────────────
STATUS_FILE = os.path.join(OUT, 'regions_status.json')
STATS_FILE = os.path.join(OUT, 'board_stats.json')
MAP_FILE = os.path.join(OUT, 'map_points.json')
# the 0.05° grid is ~4x the cells of the 0.15° one; it lives in its own file
# the page only fetches on deep zoom, so ordinary visits never pay for it
MAP_FINE_FILE = os.path.join(OUT, 'map_points_fine.json')
MAP_CONF_MIN = 0.5   # a sweep detection below this stays out of the dogs layer
# Outlier rule, measured against all 32.1M harvested points (2026-08-05):
#
#   OFF LAND     154,611 points (0.481%). Sequences with interpolated GPS
#                   string frames across open water.
#   GPS FLYER     53,537 points (0.167%), 25,951 of them ON LAND and so
#                   invisible to the land test. A Mapillary sequence is one
#                   capture session, yet 338 of them span more than a degree --
#                   every single one of those turned out to teleport between
#                   consecutive frames, the worst by 38,000 km. Not one wide
#                   sequence was a genuine long drive, so a span gate is safe:
#                   the widest continuous sequence in the whole harvest spans
#                   0.44 deg. Inside a wide sequence, frames further than
#                   MAP_SEQ_OFF from that sequence's median are the minority
#                   cluster -- the side that cannot be where it claims.
#
# Together 180,562 points, 0.562% of the harvest. Points are kept in the
# payload either way; the page filters, so "exclude" is a view and never a
# deletion.
MAP_SEQ_SPAN = 1.0   # deg: a capture session this wide is broken
MAP_SEQ_OFF = 0.5    # deg: how far off its own median a frame must sit
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


WORLD_JSON = os.path.join(OUT, 'world.json')


def country_totals(fine, res):
    """Per-country totals, keyed by the names the MAP uses.

    Joined against world.json rather than the Natural Earth shapefile next
    door, because the page identifies a country by whatever name echarts
    reports for the polygon that was clicked -- matching any other source
    means maintaining a synonym table for every "United States" vs "United
    States of America" the two disagree on.

    The join is per CELL at the finest grid, not per point: 270K centres
    against 217 polygons through a spatial index, instead of 32M. A cell is
    ~5.5 km, so a count is exact and only its attribution is approximate --
    a cell straddling a border lands wholly in whichever country holds its
    centre. Returns {} rather than raising if geopandas is absent; the popup
    is a convenience and the map has to draw without it.
    """
    if not os.path.exists(WORLD_JSON):
        return {}
    try:
        import geopandas as gpd
        import pandas as pd
    except Exception as e:
        print('map: no geopandas, skipping country totals:', e)
        return {}
    # Order IS the output layout -- slot n here is index n in every row, and
    # the page reads [frames, outlier frames, dog frames, outlier dog frames].
    keys = ('levels', 'out_levels', 'dog_levels', 'dog_out_levels')
    rk = str(res)
    seen = {}
    for slot, k in enumerate(keys):
        for x, y, n in ((fine.get(k) or {}).get(rk, {}).get('points') or []):
            row = seen.get((x, y))
            if row is None:
                row = seen[(x, y)] = [0, 0, 0, 0]
            row[slot] += n
    if not seen:
        return {}
    try:
        from shapely.geometry import shape
        # Built by hand, not gpd.read_file: echarts' map JSON omits the
        # per-feature "type": "Feature", and the GeoJSON reader answers a
        # frame of zero rows for it rather than failing.
        with open(WORLD_JSON) as fh:
            doc = json.load(fh)
        def rings_ok(polys):
            """Drop empty parts. world.json ships one (China) whose 15
            polygons include a 13th with no rings at all; shapely raises on
            it, and one raise used to cost every country."""
            out = []
            for poly in polys:
                keep = [r for r in poly if r and len(r) >= 4]
                if keep:
                    out.append(keep)
            return out

        names, geoms, skipped = [], [], []
        for f in doc.get('features') or []:
            nm = (f.get('properties') or {}).get('name')
            geo = f.get('geometry')
            if not nm or not geo:
                continue
            try:
                if geo.get('type') == 'MultiPolygon':
                    parts = rings_ok(geo.get('coordinates') or [])
                    if not parts:
                        continue
                    geo = {'type': 'MultiPolygon', 'coordinates': parts}
                elif geo.get('type') == 'Polygon':
                    parts = rings_ok([geo.get('coordinates') or []])
                    if not parts:
                        continue
                    geo = {'type': 'Polygon', 'coordinates': parts[0]}
                g = shape(geo)
                if not g.is_valid:
                    g = g.buffer(0)
                if g.is_empty:
                    continue
            except Exception:            # one bad outline is not 216 others
                skipped.append(nm)
                continue
            names.append(nm)
            geoms.append(g)
        if skipped:
            print(f'map: {len(skipped)} country outline(s) unusable: '
                  f'{", ".join(sorted(skipped)[:5])}')
        if not names:
            print('map: world.json has no named features, skipping countries')
            return {}
        world = gpd.GeoDataFrame({'name': names}, geometry=geoms,
                                 crs='EPSG:4326')
        cells = list(seen)
        pts = gpd.GeoDataFrame(
            {'i': range(len(cells))},
            geometry=gpd.points_from_xy([c[0] for c in cells],
                                        [c[1] for c in cells]),
            crs=world.crs)
        j = gpd.sjoin(pts, world, how='inner', predicate='within')
        j = j[~j.index.duplicated(keep='first')]   # a border point hits twice
    except Exception as e:
        print('map: country join failed:', e)
        return {}
    out = {}
    for i, nm in zip(j['i'], j['name']):
        if not isinstance(nm, str) or not nm:
            continue
        row = seen[cells[i]]
        acc = out.get(nm)
        if acc is None:
            acc = out[nm] = [0, 0, 0, 0, 0]
        for s in range(4):
            acc[s] += row[s]
        acc[4] += 1                                # cells with any frame
    # [frames, outlier frames, dog frames, outlier dog frames, cells]
    return {k: v for k, v in out.items() if v[0] or v[1]}


def build_map_points(res_list=(0.5, 0.15), fine_res=0.05):
    """Bin the harvest AND the sweep's dog calls into density grids.

    Schema 2. One pass reads ``computed_geometry`` (GeoJSON Point) from the
    ground-animal parquets into a temp table; three things aggregate from it:

      levels      harvested frames per cell, at each resolution
      dog_levels  frames with a >= MAP_CONF_MIN dog call, joined from the
                  sweep store by image_id -- where the detector actually
                  fired, and (client-side, dogs/harvest) the hit rate
      regions     one anchor point per catalog region: the MEDIAN of its
                  points, so one bad GPS sequence cannot drag a continent's
                  marker into the sea

    The browser renders cells as geo-anchored raster rects and swaps to the
    finer grid on zoom; ``fine_res`` goes to MAP_FINE_FILE, fetched only on
    deep zoom. Paths come from the lock-free catalog snapshot, so this never
    contends with the live catalog DB. READ-ONLY on the sweep store.
    """
    snap = os.path.join(REPO, 'data', 'catalog.parquet')
    if not os.path.exists(snap):
        return
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")  # stay polite to the running jobs
    con.execute("SET memory_limit='8GB'")
    con.execute("INSTALL json; LOAD json;")
    # Per-point land test: a few Mapillary sequences carry bad/interpolated
    # GPS that strings their images across open ocean (with spurious animal
    # detections), drawing fake lines on the map. Tested per point, so coastal
    # cities (whose 0.5° cell center may sit just offshore) are kept. This
    # used to be a WHERE clause; it is now one half of the outlier flag, so
    # the page can show what it hides.
    sea_test = "false"
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
        sea_test = "NOT is_land(lon, lat)"
    except Exception as e:
        print('map: land mask unavailable, no sea outliers:', e)
    rows = con.execute(
        "SELECT DISTINCT path, region FROM read_parquet(?) "
        "WHERE kind='ground_animals'", [snap]).fetchall()
    # hidden regions never enter the map at all -- not as cells, not as a
    # marker, and not in the country totals the card reads
    by_path = {p: r for p, r in rows
               if os.path.exists(p) and r not in HIDE_REGIONS}
    if not by_path:
        con.close()
        return
    con.execute("CREATE TEMP TABLE preg(path VARCHAR, region VARCHAR)")
    con.executemany("INSERT INTO preg VALUES (?, ?)", list(by_path.items()))
    # image_id rides along so the dogs layer can join the sweep store; region
    # rides along so each region gets a marker anchored where its data is;
    # sequence rides along so a frame can be judged against its own session
    con.execute(
        """
      CREATE TEMP TABLE raw AS
      SELECT lon, lat, iid, region, seq FROM (
        SELECT TRY_CAST(json_extract(computed_geometry,'$.coordinates[0]') AS DOUBLE) lon,
               TRY_CAST(json_extract(computed_geometry,'$.coordinates[1]') AS DOUBLE) lat,
               TRY_CAST(g.image_id AS UBIGINT) iid, p.region,
               CAST(g.sequence AS VARCHAR) seq
        FROM read_parquet(?, filename=true, union_by_name=true) g
        JOIN preg p ON g.filename = p.path
        WHERE computed_geometry IS NOT NULL)
      WHERE lon BETWEEN -180 AND 180 AND lat BETWEEN -90 AND 90""",
        [sorted(by_path)])
    # one row per capture session: where it sat, and how far it wandered
    con.execute("""
      CREATE TEMP TABLE sq AS
      SELECT seq, median(lon) mlon, median(lat) mlat,
             greatest(max(lat) - min(lat), max(lon) - min(lon)) span
      FROM raw WHERE seq IS NOT NULL GROUP BY seq""")
    # A sequence with no id cannot be judged against itself, so coalesce to
    # false: an unjudgeable frame is kept, never quietly called an outlier.
    con.execute(f"""
      CREATE TEMP TABLE pts AS
      SELECT r.lon, r.lat, r.iid, r.region,
             ({sea_test}) OR coalesce(
                s.span > {MAP_SEQ_SPAN}
                AND greatest(abs(r.lon - s.mlon),
                             abs(r.lat - s.mlat)) > {MAP_SEQ_OFF},
                false) AS bad
      FROM raw r LEFT JOIN sq s USING (seq)""")
    con.execute("DROP TABLE raw")
    con.execute("DROP TABLE sq")
    total, outlier_total = con.execute(
        "SELECT count(*) FILTER (WHERE NOT bad), count(*) FILTER (WHERE bad) "
        "FROM pts").fetchone()

    # frames where the sweep called a dog: distinct image_ids with a
    # confident detection, from the newest generation in the store
    dogs_total = dogs_outlier_total = 0
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
        import store as _store
        det = _store._sql_src(
            _store._store_globs(_store.get_detect_root(), 'det'))
        gen = con.execute(f"SELECT max(gen) FROM {det}").fetchone()[0]
        con.execute(
            f"""
          CREATE TEMP TABLE dpts AS
          SELECT p.lon, p.lat, p.bad FROM pts p
          JOIN (SELECT DISTINCT TRY_CAST(image_id AS UBIGINT) iid
                FROM {det} WHERE gen = ? AND conf >= ?) d USING (iid)""",
            [gen, MAP_CONF_MIN])
        dogs_total, dogs_outlier_total = con.execute(
            "SELECT count(*) FILTER (WHERE NOT bad), "
            "count(*) FILTER (WHERE bad) FROM dpts").fetchone()
    except Exception as e:
        print('map: sweep store unavailable, no dogs layer:', e)
        con.execute("CREATE TEMP TABLE IF NOT EXISTS dpts(lon DOUBLE, "
                    "lat DOUBLE, bad BOOLEAN)")

    def grid(table, res, bad):
        """Cells at one resolution, from the clean points or the outliers."""
        rows = con.execute(f"""
          SELECT round(floor(lon/{res})*{res}+{res / 2}, 4) x,
                 round(floor(lat/{res})*{res}+{res / 2}, 4) y, count(*) n
          FROM {table} WHERE bad = {'true' if bad else 'false'}
          GROUP BY 1, 2""").fetchall()
        pts = [[r[0], r[1], r[2]] for r in rows]
        return {'res': res, 'max': max((p[2] for p in pts), default=0),
                'points': pts}

    levels = {str(r): grid('pts', r, False) for r in res_list}
    dog_levels = {str(r): grid('dpts', r, False) for r in res_list}
    # The outlier grids ship alongside rather than instead: the page merges
    # them back in when the box is unticked, so nothing is hidden that cannot
    # be shown again without a rebuild. They are small -- a few thousand
    # cells against seventy thousand.
    out_levels = {str(r): grid('pts', r, True) for r in res_list}
    dog_out_levels = {str(r): grid('dpts', r, True) for r in res_list}
    # The ANCHOR is clean-only on purpose -- that median is what stops one
    # teleporting sequence dragging a continent's marker into the sea. The
    # COUNT ships both ways, because the marker's tooltip says "frames on the
    # map" and the map's own total changes with the outlier toggle.
    regions = [{'key': r[0], 'lon': round(r[1], 3), 'lat': round(r[2], 3),
                'n': r[3], 'n_bad': r[4]}
               for r in con.execute(
                   "SELECT region, median(lon) FILTER (WHERE NOT bad), "
                   "median(lat) FILTER (WHERE NOT bad), "
                   "count(*) FILTER (WHERE NOT bad), "
                   "count(*) FILTER (WHERE bad) "
                   "FROM pts GROUP BY region ORDER BY region").fetchall()
               if r[1] is not None]
    fine = {'levels': {str(fine_res): grid('pts', fine_res, False)},
            'dog_levels': {str(fine_res): grid('dpts', fine_res, False)},
            'out_levels': {str(fine_res): grid('pts', fine_res, True)},
            'dog_out_levels': {str(fine_res): grid('dpts', fine_res, True)}}
    countries = country_totals(fine, fine_res)
    con.close()

    out = {'schema': 3, 'total': total, 'dogs_total': dogs_total,
           'outlier_total': outlier_total,
           'dogs_outlier_total': dogs_outlier_total,
           'seq_span': MAP_SEQ_SPAN, 'seq_off': MAP_SEQ_OFF,
           'conf_min': MAP_CONF_MIN, 'fine_res': fine_res,
           'levels': levels, 'dog_levels': dog_levels,
           'out_levels': out_levels, 'dog_out_levels': dog_out_levels,
           'regions': regions, 'countries': countries,
           'country_res': fine_res,
           'built_at': time.strftime('%Y-%m-%d %H:%M')}
    os.makedirs(OUT, exist_ok=True)
    for path, payload in ((MAP_FILE, out), (MAP_FINE_FILE, fine)):
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp, path)


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
    } for r in sorted(set(data) | set(img)) if r not in HIDE_REGIONS]


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
    tabs collapse to <=0.5 file reads/s (§7.2). Absent / unparsable degrade
    to {'running': False, 'ever': False} — nothing is known, which is not the
    same answer as an idle sweep and must not render as one."""
    with _detect_lock:
        now = time.monotonic()
        if _detect_memo['body'] is not None and now - _detect_memo['t'] < 2.0:
            return _detect_memo['body']
        if _read_detect_status is None:
            body = {'running': False, 'ever': False}
        else:
            try:
                # Read the doc WHOLE, then decide what is still true. The
                # staleness rule used to throw the whole payload away, so a
                # finished sweep -- 32.5M images, 3.3M positives, a hundred
                # per cent complete -- rendered as six em-dashes. Those totals
                # are permanent facts: the run ended, it did not un-happen.
                # Only the instantaneous fields go, because those would be
                # lies about a process that is not running.
                body = dict(_read_detect_status(stale_after=1e12) or {})
                # A missing or unparsable status.json collapses to the same
                # {'running': False} a genuinely idle sweep gets, and the
                # panel drew that as "sweep idle -- 0 of 17 complete" over a
                # finished 32.5M-image harvest: a claim it had no document to
                # make. Only a real document carries an age, so that is what
                # says one was read.
                body['ever'] = 'age_s' in body
                fresh = float(body.get('age_s') or 0) <= 120.0
                live = fresh and str(body.get('state')) == 'running'
                body['running'] = live
                if not live:
                    done = float(body.get('imgs_done') or 0)
                    total = float(body.get('imgs_total') or 0)
                    body['finished'] = bool(total and done >= total)
                    # An ETA is a claim about a process, so it goes. So does
                    # the GPU sample: it is one instantaneous reading taken at
                    # the run's last publish, and left in it put "GPU 27%" on
                    # the panel two days after that run stopped, against the
                    # machine panel's live 86% three sections down the page.
                    body.pop('eta_s', None)
                    body.pop('gpu', None)
                    # Per drive, done/total is work that HAPPENED and stays --
                    # dropping the whole block took the bars with it. The rate,
                    # the queue depth and the stall flag all describe a reader
                    # that is not reading, so only those three go.
                    body['drives'] = {
                        d: {'done': v.get('done'), 'total': v.get('total')}
                        for d, v in (body.get('drives') or {}).items()
                        if isinstance(v, dict)
                    }
            except Exception:
                body = {'running': False, 'ever': False}
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
# The guesser needs torch and transformers; the dashboard does not, and on
# this box they are different environments. Resolved from config so no
# machine's layout ends up in a public repo.
TRIAGE_PYTHON = cfg('triage_python', sys.executable, env='TRIAGE_PYTHON')
# Confusion matrices live on Comet, so topping the cache up needs an
# interpreter with comet_ml and a file holding the key. Both are config, and
# BOTH must be set for the top-up to run at all -- a clone that has neither
# should never make a surprise network call, and the panel simply shows no
# matrix, which is what it did before any of this existed.
CONFUSION_PYTHON = cfg('confusion_python', '', env='CONFUSION_PYTHON')
# Scoring a run against its own val split needs ultralytics and torch, which
# is the sweep's environment rather than the dashboard's. Defaults to it for
# that reason; a checkout with neither simply never scores and the panel says
# so instead of pretending.
MISTAKES_PYTHON = cfg('mistakes_python', '', env='MISTAKES_PYTHON')
COMET_ENV_FILE = cfg('comet_env_file', '', env='COMET_ENV_FILE')
TRIAGE_WATCH = cfg_int('triage_watch', 300, env='TRIAGE_WATCH')
# Which model the Run button uses, and on what. Config rather than a constant:
# the better model needs a GPU to be practical, and the GPU on this kind of
# box is also the one training runs on. Empty means the tool's own defaults.
# The Run button is only offered where a run could actually work: an
# interpreter was named in config. A clone that never set one sees nothing,
# exactly as before this existed.
CONFIGURED_TRIAGE = bool(cfg('triage_python', '', env='TRIAGE_PYTHON'))
TRIAGE_MODEL = cfg('triage_model', '', env='TRIAGE_MODEL')
TRIAGE_DEVICE = cfg('triage_device', '', env='TRIAGE_DEVICE')
def leash_store():
    """The leash verdict store, or None if the tool is not present.

    Loaded lazily and never fatally: leash labelling is an extra axis on the
    review page, and a checkout without it should lose the two buttons, not
    the page.
    """
    if _LEASH.get('tried'):
        return _LEASH.get('mod')
    _LEASH['tried'] = True
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
        import leash_store
        _LEASH['mod'] = leash_store
    except Exception:
        _LEASH['mod'] = None
    return _LEASH['mod']


_LEASH = {}
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
            moved_crop = moved_full = False
            if name in _flag_names(other):
                _rewrite_labels(name, other)
                _flag_names(other).discard(name)
                ost = _store_for(other)
                # MOVE the copies rather than delete-and-recopy. The recopy
                # below reads the live pool, and a crop being re-judged from
                # the audit view left that pool minutes after it was first
                # flagged -- so deleting here and copying from the pool lost
                # the image entirely, leaving a ledger entry with no picture.
                for src, dst, flag_it in (
                        (os.path.join(ost['crops'], name),
                         os.path.join(st['crops'], name), 'crop'),
                        (os.path.join(ost['full'], name),
                         os.path.join(st['full'], name), 'full')):
                    try:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        os.replace(src, dst)
                        if flag_it == 'crop':
                            moved_crop = True
                        else:
                            moved_full = True
                    except OSError:
                        try:
                            os.remove(src)
                        except OSError:
                            pass
            # the two copies are independent: the full frame can survive the
            # prune a beat longer than the crop, or vice versa. And the source
            # is the crop's own directory -- reading recent_crops alone left
            # 368 harvested crops filed with copied:false and no picture.
            src = crop_dir(name)
            got_crop = moved_crop or _copy_out(
                os.path.join(src, name), os.path.join(st['crops'], name))
            got_full = moved_full or _copy_out(
                os.path.join(src, 'full', name),
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




COPY_JS = r"""
/* Copy that works off localhost. navigator.clipboard is undefined on a plain
   http origin, which is exactly how this dashboard is served, so the async API
   is tried first and a detached textarea carries the rest. Returns a promise
   resolving to whether it landed, so the button can say. */
function copyText(t){
  t=String(t==null?'':t);
  if(navigator.clipboard&&navigator.clipboard.writeText&&window.isSecureContext){
    return navigator.clipboard.writeText(t).then(function(){return true},
                                                 function(){return fallback(t)});
  }
  return Promise.resolve(fallback(t));
  function fallback(v){
    try{
      var ta=document.createElement('textarea');
      ta.value=v;
      /* off-screen but focusable: display:none or visibility:hidden makes the
         selection uncopyable */
      ta.setAttribute('readonly','');
      ta.style.position='fixed';ta.style.top='-1000px';ta.style.opacity='0';
      document.body.appendChild(ta);
      ta.select();ta.setSelectionRange(0,v.length);
      var ok=document.execCommand('copy');
      document.body.removeChild(ta);
      return !!ok;
    }catch(e){return false}
  }
}
/* Say what happened, on the button itself, and put its own label back. */
function copyOnto(btn,text,label){
  if(!btn)return;
  copyText(text).then(function(ok){
    btn.textContent=ok?'Copied':'Press \u2318C';
    btn.classList.toggle('done',ok);
    clearTimeout(btn.__t);
    btn.__t=setTimeout(function(){
      btn.textContent=label;btn.classList.remove('done');
    },1400);
  });
}
"""


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
border-bottom:1px solid var(--bd);transition:padding .16s ease}
/* ── the header at work ───────────────────────────────────────────────────
   Everything up here is setup, and setup is read once. Once the page has
   scrolled, the header keeps only what a reviewer ACTS on -- the count, the
   chips saying what is narrowing the queue, and the controls that change it
   -- and sheds the running tally, the two progress rows and the open panel.
   Those are ambient: worth a glance, not worth a third of the viewport on
   every screen of crops. */
/* 96px, not 1px. A one-pixel sentinel flipped the header the instant the
   page moved at all, so a small scroll near the top snapped ~60px of chrome
   in and out repeatedly. The threshold now sits below anything a reader
   nudges past by accident. */
.scrollcue{display:block;height:96px;margin-bottom:-96px;pointer-events:none}
body.compact header{padding-top:7px}
/* Collapsed, not deleted. display:none took the height in one frame and the
   grid jumped; a height transition makes the same change read as the header
   folding rather than the page lurching. */
.tally,.lines{overflow:hidden;
transition:max-height .18s ease,opacity .18s ease,padding .18s ease}
.tally{max-height:40px}
.lines{max-height:64px}
body.compact .tally,body.compact .lines{max-height:0;opacity:0;padding-top:0;
padding-bottom:0}
body.compact .npanel{display:none}
@media(prefers-reduced-motion:reduce){.tally,.lines{transition:none}}
body.compact h1{font-size:14px}
body.compact .score>b{font-size:19px;letter-spacing:-.3px}
body.compact .score>span{font-size:11px}
body.compact .cap{padding-top:7px}
body.compact .pagebar{padding:5px 0 8px}
@media(prefers-reduced-motion:reduce){header{transition:none}}
.hrow{display:flex;flex-wrap:wrap;align-items:center;gap:16px 20px}
h1{font-size:17px;font-weight:640;letter-spacing:-.2px;display:flex;align-items:center;gap:9px}
h1 .fl{color:var(--red)}
.back{color:var(--mut);text-decoration:none;font-size:12.5px;border:1px solid var(--bd);
border-radius:8px;padding:4px 10px;transition:color .12s,border-color .12s}
.back:hover{color:var(--acc);border-color:rgba(232,166,69,.35)}
/* The headline and the tally are different KINDS of number, so they are set
   as different kinds: one is what is left to do, the rest are what has been
   done. Ranking them by size in one row made six numbers of similar weight
   and left the reader to find the live one. */
.score{margin-left:auto;display:flex;flex-direction:column;align-items:flex-end;
gap:1px}
.score>b{font-size:26px;font-weight:660;letter-spacing:-.6px;
font-variant-numeric:tabular-nums;line-height:1.05}
.score>span{font-size:12px;color:var(--dim)}
.tally{display:flex;align-items:baseline;gap:5px;flex-wrap:wrap;
justify-content:flex-end}
.tally b{font-size:12px;font-weight:640;color:var(--mut);
font-variant-numeric:tabular-nums}
.tally b+span{margin-right:5px}
.tally span{font-size:11px;color:var(--dim)}
.tally b.pos{color:var(--green)}
.tally b.dup,.tally b.lea{color:var(--dim)}

/* ── toolbar ── */
/* ── the caption ──────────────────────────────────────────────────────────
   One line of running text, not a control strip. It states what the queue
   currently is; the count sits in it at the same size as the words, marked
   out by weight and colour rather than by being enormous. The GRID is this
   page's hero, and a display-sized number here competed with it. */
.cap{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 0 0}
.capline{margin:0;font-size:12.5px;line-height:1.5;color:var(--mut);
font-variant-numeric:tabular-nums}
.capline b{color:var(--tx);font-weight:640}
/* the narrowing readout: the one thing this block exists to do, and it
   appears only when something has actually been narrowed */
.capline i{font-style:normal;color:var(--dim)}
.capsp{flex:1}
.nbtn{display:inline-flex;align-items:center;gap:5px;background:transparent;
border:1px solid var(--bd);color:var(--mut);border-radius:8px;padding:5px 11px;
font-size:12px;font-family:inherit;cursor:pointer;transition:color .12s,border-color .12s}
.nbtn:hover{color:var(--tx);border-color:var(--dim)}
.nbtn:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.nbtn.on{color:var(--acc);border-color:rgba(232,166,69,.45)}
.ncar{display:inline-block;transition:transform .16s}
.nbtn.on .ncar{transform:rotate(90deg)}
/* the count of what is applied, so the button says something when the panel
   is shut */
.nbtn em{font-style:normal;font-weight:650;color:var(--acc)}

/* ── applied filters ──────────────────────────────────────────────────── */
.chips{display:flex;gap:6px;flex-wrap:wrap;padding:9px 0 0}
.chips[hidden]{display:none}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:11px;
line-height:1.7;padding:1px 4px 1px 9px;border-radius:999px;
background:rgba(232,166,69,.11);border:1px solid rgba(232,166,69,.32);
color:#e8b877;font-variant-numeric:tabular-nums}
.chipx{background:transparent;border:0;color:inherit;opacity:.6;cursor:pointer;
font-family:inherit;font-size:13px;line-height:1;padding:2px 5px;border-radius:999px}
.chipx:hover{opacity:1;background:rgba(232,166,69,.2)}
.chipx:focus-visible{outline:2px solid var(--acc);outline-offset:1px;opacity:1}

/* ── the panel ────────────────────────────────────────────────────────────
   Inline, not a popover. This page is driven by F/D/L/N with the hands on
   the keyboard, and a floating layer that traps focus would fight the work
   it is meant to serve. */
.npanel{display:grid;gap:13px;padding:14px 0 4px;margin-top:11px;
border-top:1px solid var(--bd)}
.npanel[hidden]{display:none}
.ngrp{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
/* trimGroups() hides a group whose every control is hidden, and it can only
   do that if the attribute wins: a bare display:flex outranks the UA's
   [hidden]{display:none}, so the heading over nothing stayed on screen */
.ngrp[hidden]{display:none}
.nlab{flex:none;width:118px;font-size:10.5px;letter-spacing:.055em;
text-transform:uppercase;color:var(--dim)}
.nrow{display:flex;gap:8px;align-items:center;flex-wrap:wrap;flex:1;min-width:220px}
@media(max-width:640px){.ngrp{display:block}.nlab{width:auto;display:block;
margin-bottom:6px}}
.rbtn{display:inline-flex;align-items:center;gap:6px;
background:rgba(232,166,69,.13);border:1px solid rgba(232,166,69,.34);
color:var(--acc);border-radius:8px;padding:5px 12px;font-size:12.5px;font-weight:600;
cursor:pointer;font-family:inherit;font-variant-numeric:tabular-nums;transition:background .12s}
/* Drawn, not typed. The arrows were text glyphs, and a font renders them at
   whatever weight and baseline it likes -- next to 600-weight 12.5px they
   came out as a thin squiggle riding high above the label. A stroked icon
   takes its weight from stroke-width and its position from the flex row. */
.bico{width:14px;height:14px;flex:none}
/* The model's guess. Dashed, lower-case, and in nobody's verdict colour --
   a reader must never mistake it for something that was decided. */
.sg{flex:none;max-width:46%;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap;font-size:10px;line-height:1.6;padding:0 5px;
border-radius:4px;border:1px dashed rgba(130,140,150,.42);
color:var(--mut);pointer-events:auto}
.sg-dog{border-color:rgba(232,166,69,.55);color:#e8b877}
.sg-animal{border-color:rgba(110,180,150,.5);color:#8fd0b4}
.sg-object{border-color:rgba(130,140,150,.45);color:var(--dim)}
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
.cnt{color:var(--mut);font-size:12.5px;font-variant-numeric:tabular-nums}
/* the folded legend */
#find{background:var(--panel);color:var(--tx);border:1px solid var(--bd);
border-radius:8px;padding:5px 10px;font-family:inherit;font-size:12px;
min-width:196px}
#find::placeholder{color:var(--dim)}
#find:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
#find.warn{border-color:rgba(232,166,69,.5)}
#findmsg{flex:1 1 100%;order:99;font-weight:500;font-size:11.5px;
  color:var(--acc);letter-spacing:.01em}
.keys{padding:7px 0 11px;border-top:1px solid var(--bd)}
.keys summary{display:flex;align-items:center;gap:12px;cursor:pointer;
list-style:none;color:var(--dim);font-size:11.5px}
.keys summary::-webkit-details-marker{display:none}
.keys summary:focus-visible{outline:2px solid var(--acc);outline-offset:3px;
border-radius:5px}
.klead{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.klead kbd{margin-left:7px}
.klead kbd:first-child{margin-left:0}
.kmore{margin-left:auto;color:var(--dim);opacity:.75}
.kmore::after{content:' ▾'}
.keys[open] .kmore::after{content:' ▴'}
.kbody{color:var(--dim);font-size:11.5px;padding-top:9px;display:flex;
flex-wrap:wrap;gap:6px 14px;align-items:center;max-width:1180px}
.hint{color:var(--dim);font-size:11.5px;padding-bottom:14px;display:flex;
flex-wrap:wrap;gap:6px 14px;align-items:center}
/* ── balance strip: one bar, two lines, no card chrome ── */
/* ── the two progress lines ───────────────────────────────────────────────
   Your progress, and the guesser's. They had a full-width strip each, with
   their own bar, dot, legend and 27px number -- four times the volume for two
   numbers, in a block that is supposed to recede behind the crops. Each is
   now one line of the same caption type, with a hairline track sharing the
   row rather than owning it. */
.lines{display:grid;gap:2px;padding:10px 0 0}
/* A FIXED height. The guesser's sub-line grows and shrinks every five
   seconds while a pass runs -- "Not running" one poll, "176 of 2,864 this
   pass · 39.2/s" the next -- and with a row free to wrap, the header changed
   height under the reader and shoved the grid down the page on a timer. */
.line{display:flex;align-items:center;gap:10px;margin:0;padding:5px 0;
min-height:22px;
font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}
.line[hidden]{display:none}
.line b{color:var(--tx);font-weight:640;flex:none}
/* shrinkable, and allowed to break: the longest sub this row produces is
   the "<other guesser> is guessing now · they share the card · N crops have
   no guess from this one" branch, and at flex:none its base size is its
   max-content width -- 492px, which pushed the sticky header wider than the
   viewport and gave the whole page a horizontal scrollbar below ~510px. */
.line .lsub{color:var(--dim);flex:0 1 auto;min-width:0;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.line .lend{flex:none;color:var(--mut)}
.line .track{flex:1;min-width:60px;height:3px;border-radius:2px;
background:rgba(130,140,150,.18);overflow:hidden;display:block}
.line .track i{display:block;height:100%;width:0;background:var(--red);
transition:width .45s ease}
#bal.ok b{color:var(--green)}
#bal.ok .track i{background:var(--green)}
#trg .track i{background:var(--acc)}
/* the guesser's own line: dimmer still, because it is a colleague's progress
   rather than yours */
.line.trg{color:var(--dim)}
.line.trg b{color:var(--mut);font-weight:600}
/* the track drops to its own row on a phone; the text still elides
   rather than wrapping, so the row keeps one height everywhere */
@media(max-width:700px){.line{flex-wrap:wrap}
.line .track{order:9;flex-basis:100%}}
.pagebar{display:flex;align-items:center;justify-content:flex-end;gap:10px;
padding:9px 0 12px}
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
.meta{display:flex;align-items:center;justify-content:space-between;gap:6px;
padding:5px 8px;font:400 10.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
color:var(--dim);font-variant-numeric:tabular-nums}
.meta .id{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.meta .cf{color:var(--mut);font-weight:600;flex:none}
/* two verdicts, side by side and equal weight -- neither is the default, and
   a hairline keeps them from reading as one wide button */
/* A CONTACT SHEET. At rest a tile is a photograph and one caption line --
   nothing else. The verdict and leash rows used to be permanent furniture: 94px
   of every 303px tile, identical on all fifty, which is two hundred buttons on
   screen saying the same four things while the pictures they are about get the
   remainder. They ride over the foot of the frame now, and only when the tile
   is under the cursor, selected, or carrying a mark. The keyboard never needed
   them; the mouse gets them exactly where the hand already is. */
.actwrap{position:absolute;left:0;right:0;bottom:0;padding:26px 5px 5px;
background:linear-gradient(to top,rgba(10,12,15,.94) 46%,rgba(10,12,15,.72) 74%,
rgba(10,12,15,0));opacity:0;transform:translateY(5px);pointer-events:none;
transition:opacity .13s ease,transform .13s ease}
.card:hover .actwrap,.card.sel .actwrap,.card:focus-within .actwrap,
.card.awaitleash .actwrap,.card.changed .actwrap,.card.unjudged .actwrap,
/* A tile that already carries a mark shows it. Hiding it until hover is right
   for a queue -- an unjudged crop has nothing to report -- and wrong the
   moment there IS something to report. */
.card:has(.fbtn.on) .actwrap,.card:has(.lbtn.on) .actwrap,
/* and audit mode is nothing BUT reading the marks: every tile there carries
   one, and hiding all fifty made the mode useless */
body.auditing .actwrap{
opacity:1;transform:none;pointer-events:auto}
@media(prefers-reduced-motion:reduce){.actwrap{transition:none}}
.acts{display:grid;grid-template-columns:1fr 1fr;gap:5px}
/* AUDIT MODE. Belongs in THIS stylesheet: /review is its own document with
   its own <style>, and the same rules in the dashboard's block styled nothing
   here -- the class was on the button and the button looked untouched. */
.fbtn.on{background:rgba(232,166,69,.22)!important;
border:1px solid rgba(232,166,69,.55)!important;color:var(--acc)!important;
font-weight:700}
.card.changed{box-shadow:inset 0 0 0 2px var(--acc)}
/* no verdict left on it: neither button is lit, and the tile says so rather
   than looking like a crop that was never reached */
/* judged a dog and still owing a leash call: dimming it would read as
   "done", so it keeps full weight and gets a rail instead */
.card.awaitleash{box-shadow:inset 0 0 0 2px rgba(67,181,129,.45)}
.card.awaitleash .acts.leash::before{content:'leash?';grid-column:1/-1;
font-size:10px;color:var(--green);letter-spacing:.04em;margin-bottom:-2px}
.card.unjudged{opacity:.62}
.card.unjudged .meta::after{content:'no verdict';margin-left:auto;
font-size:10px;color:var(--dim)}
/* The audit list is fetched with label= and leash= and nothing else, so
   a group that narrows only the queue would be a control that does
   nothing here. #find is one of them: loadAudit never sends the term, so
   the search box sat visible and accepted input that changed nothing --
   while quietly saving the term to prefs, to reorder the queue later as a
   surprise. Hidden the way the country and gate controls are. */
body.auditing #ngrpLooks,body.auditing #ngrpWhere,
body.auditing #country,body.auditing #unkeep,
body.auditing #find,body.auditing #findmsg{display:none}
#verdict{display:none}
body.auditing #verdict{display:inline-block}
/* The leash row. In THIS stylesheet for the reason the comment above gives:
   /review is its own document, and these rules in the dashboard's block
   styled nothing -- the buttons rendered as browser defaults. A second axis,
   so it is quieter than the verdict row it sits under. */
.acts.leash{border-top:0;padding:5px 0 0;gap:5px;display:grid;
grid-template-columns:1fr 1fr}
.lbtn{appearance:none;background:transparent;color:var(--dim);
border:1px dashed var(--bd);border-radius:7px;padding:5px 4px;font-size:11px;
font-family:inherit;cursor:pointer;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis;transition:color .12s,border-color .12s,background .12s}
.lbtn:hover{color:var(--tx);border-color:var(--dim)}
.lbtn:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.lbtn.le.on{border-style:solid;color:#5ec89a;border-color:rgba(67,181,129,.55);
background:rgba(67,181,129,.16);font-weight:600}
.lbtn.un.on{border-style:solid;color:var(--acc);border-color:rgba(232,166,69,.55);
background:rgba(232,166,69,.16);font-weight:600}
.sec.lea{color:var(--mut)}
.fbtn{border:1px solid rgba(130,140,150,.22);border-radius:7px;
background:rgba(20,24,30,.72);color:var(--mut);padding:6px 4px;
font-size:11px;cursor:pointer;font-family:inherit;font-weight:600;
transition:background .12s,color .12s,border-color .12s;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
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
/* one page, or none, hides the pager with the attribute -- and display:flex
   beat it, so a queue with nothing left in it still offered Next */
.foot[hidden]{display:none}

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
.lbcopy{font-variant-numeric:tabular-nums}
.lbcopy.done{color:var(--green);border-color:rgba(67,181,129,.5);
background:rgba(67,181,129,.14)}
.lbx{position:absolute;top:16px;right:18px}
.lbyes{background:rgba(67,181,129,.14);border-color:rgba(67,181,129,.4);color:var(--green)}
.lbyes:hover{background:rgba(67,181,129,.24)}
@media(max-width:560px){.score{margin-left:0;width:100%}.grid{grid-template-columns:repeat(auto-fill,minmax(140px,1fr))}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
/* crop-suggestion run progress: this filter lives on THIS page, so its
   progress does too. Was on the main dashboard, which is not where anyone
   sorting the queue is looking. */
.trgdot{width:8px;height:8px;border-radius:50%;background:var(--dim);flex:none}
.trg.on .trgdot{background:var(--green);animation:trgpulse 1.6s ease-in-out infinite}
.trg.warn .trgdot{background:var(--acc)}
@keyframes trgpulse{0%,100%{opacity:1}50%{opacity:.35}}
@media(prefers-reduced-motion:reduce){.trg.on .trgdot{animation:none}}
.trgbtn{flex:none;appearance:none;background:transparent;color:var(--mut);
border:1px solid var(--bd);border-radius:999px;padding:4px 13px;font-size:11.5px;
font-family:inherit;cursor:pointer;transition:color .12s,border-color .12s}
.trgbtn:hover:not(:disabled){color:var(--tx);border-color:var(--dim)}
.trgbtn:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.trgbtn:disabled{opacity:.5;cursor:default}
.trgerr{flex-basis:100%;font-size:11px;color:var(--red);margin-top:6px}
#trgModel{flex:none;appearance:none;background:transparent;color:var(--mut);
border:1px solid var(--bd);border-radius:999px;padding:4px 10px;
font-size:11.5px;font-family:inherit;cursor:pointer}
#trgModel:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
#trgModel[hidden]{display:none}
/* What the dropdown's percentage means, and what the chosen guesser is for.
   Under the strip, where it is read before the Run button rather than inside
   a title attribute nobody hovers. Folded: the summary is the answer, the
   body is the working. */
.trgnote{margin:-4px 0 12px;font-size:11px;line-height:1.55;color:var(--dim);
max-width:74ch}
.trgnote[hidden]{display:none}
.trgnote>summary{cursor:pointer;list-style:none;color:var(--mut);
padding:2px 0;transition:color .12s}
.trgnote>summary::-webkit-details-marker{display:none}
.trgnote>summary::before{content:"\203a";display:inline-block;width:12px;
color:var(--dim);transition:transform .15s}
.trgnote[open]>summary::before{transform:rotate(90deg)}
.trgnote>summary:hover{color:var(--tx)}
.trgnote>summary:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.trgnote p{margin:7px 0 0 12px}
.trgnote p[hidden]{display:none}
</style></head><body><div class="wrap">

<!-- Scrolled past this, you are working rather than setting up, and the
     header sheds everything you are not acting on. A sentinel, not a scroll
     handler: the browser reports the crossing once instead of the page
     measuring itself on every frame. -->
<i class="scrollcue" id="scrollcue" aria-hidden="true"></i>

<header>
  <div class="hrow">
    <h1><span class="fl">&#9873;</span> Not a dog <span class="cnt">&middot; detection review</span></h1>
    <a class="back" href="/">&larr; dashboard</a>
    <!-- Two counts, no ratio between them: "left" is the live retained pool
         (crops age out after 24 h / 3000), "flagged" is cumulative all-time.
         A bar dividing one by the other would be a made-up percentage. -->
    <!-- One of these six numbers is the job and five are trivia, and they
         were set in the same weight, so the one that decides whether to keep
         going had to be picked out of a row. The tally moves to a second,
         quieter line under it. -->
    <div class="score">
      <b id="left">&mdash;</b><span id="leftlab">left to review</span>
      <div class="tally">
        <b class="sec" id="done">&mdash;</b><span>flagged</span>
        <b class="sec pos" id="pos">&mdash;</b><span>marked dog</span>
        <b class="sec" id="seen">&mdash;</b><span>kept</span>
        <b class="sec dup" id="dups">&mdash;</b><span>repeats hidden</span>
        <b class="sec lea" id="leashN">&mdash;</b><span title="leashed / unleashed verdicts recorded — a separate axis from the dog verdicts, kept in its own database">leash calls</span>
      </div>
    </div>
  </div>
  <!-- ── what you are looking at ──────────────────────────────────────────
       Not a toolbar. Nine controls sat here in one row, styled identically,
       holding four different KINDS of thing: a view switch, four filters, two
       display options and two actions. Equal weight for unequal things is
       what made it unreadable, not the count.

       So: a caption states what the queue currently is, the filters you have
       actually applied appear as chips under it, and everything set once and
       forgotten moves behind one disclosure. Resting state is two quiet lines
       instead of a wall of pills. -->
  <!-- ── what you are looking at ──────────────────────────────────────────
       Not a toolbar. Nine controls sat in one row, styled identically,
       holding four different KINDS of thing: a view switch, four filters, two
       display options and two actions. Equal weight for unequal things is
       what made it unreadable — not the count.

       So the block became a caption. One line says what the queue currently
       is; the filters you have actually applied appear as chips beneath it;
       everything set once and forgotten sits behind one disclosure. Resting
       state is two quiet lines rather than a wall of pills. -->
  <div class="cap">
    <p class="capline" id="cap">&mdash;</p>
    <span class="capsp"></span>
    <!-- Free text over the queue. Not a filter: it ORDERS, so the near misses
         stay reachable -- they are the crops most worth a human's eye. It
         stays out here because typing is a different verb from picking, and
         it is reached constantly. -->
    <input id="find" type="search" list="findterms" hidden autocomplete="off"
           placeholder="find crops of&hellip;"
           title="type what you are looking for and the queue is reordered to bring it to the front — the same model that guesses the buckets, asked a different way">
    <datalist id="findterms"></datalist>
    <!-- A view switch, not a filter: it changes what the page IS. Set apart
         from the narrowing controls for that reason. -->
    <select id="mode" title="review new crops, or check the ones you already judged">
      <option value="queue">Unreviewed queue</option>
      <option value="audit">Check my annotations</option>
    </select>
    <!-- The audit view's own filter, and its ONLY one. It was folded into the
         panel under "On a leash", which is not a question it answers, so the
         one control that view has took a click to reach and sat under a
         heading about something else. CSS shows it only while auditing. -->
    <select id="verdict" title="which verdict to check">
      <option value="all">Both verdicts</option>
      <option value="false_positive">Only &ldquo;not a dog&rdquo;</option>
      <option value="true_positive">Only &ldquo;is a dog&rdquo;</option>
    </select>
    <button type="button" class="nbtn" id="narrow" aria-expanded="false"
            aria-controls="npanel"
            title="filters, sorting, and the guesser that fills them">Filter<span class="ncar" aria-hidden="true">&#8250;</span></button>
  </div>

  <!-- Says out loud when the search cannot work. A search that quietly
       ordered nothing was read as the model returning nonsense, which is the
       right conclusion from the evidence the page was giving. -->
  <b id="findmsg" hidden></b>

  <!-- Only what has actually been applied. No filters, no row: an empty
       filter bar spends a line telling you nothing is set. -->
  <div class="chips" id="chips" hidden></div>

  <!-- The controls, grouped by the QUESTION each answers rather than by what
       kind of widget it is. That grouping is the fix: "Any guess" and "50 per
       page" sat adjacent and identical, and one narrows the work while the
       other is a preference nobody sets twice. -->
  <div class="npanel" id="npanel" hidden>
    <div class="ngrp" id="ngrpLooks">
      <span class="nlab">What it looks like</span>
      <div class="nrow">
        <!-- A MODEL'S GUESS, and labelled as one. It sorts the queue so a
             reviewer can work through one kind of mistake at a time; it is
             never written to a ledger and never becomes a label. -->
        <select id="suggest" title="narrow by what the selected guesser thinks each crop is — a guess to sort by, never a label" hidden>
          <option value="">Any guess</option>
          <option value="dog">Looks like a dog</option>
          <option value="animal">Other animal</option>
          <option value="object">Not an animal</option>
          <option value="none">No guess yet</option>
        </select>
        <!-- The dog-bin gate on its own axis: it answers the question the
             REVIEWER is answering, where the guess filter answers what kind
             of thing it is. -->
        <select id="gatef" title="narrow by the trained dog/not-dog gate's verdict — its own axis, and always this model whatever the guesser below is set to" hidden></select>
      </div>
    </div>
    <div class="ngrp">
      <span class="nlab">On a leash</span>
      <div class="nrow">
        <select id="leashf" title="narrow by leash verdict — a separate axis from the dog verdict, kept in its own database" hidden>
          <option value="all">Any leash state</option>
          <option value="none">Needs a leash call</option>
          <option value="leashed">Leashed</option>
          <option value="unleashed">Unleashed</option>
        </select>
      </div>
    </div>
    <div class="ngrp" id="ngrpWhere">
      <span class="nlab">Where</span>
      <div class="nrow">
        <!-- Populated from /api/review, which lists only countries the sweep
             has actually produced crops for, with counts. -->
        <select id="country" title="only review crops from one country">
          <option value="">All countries</option>
        </select>
      </div>
    </div>
    <div class="ngrp">
      <span class="nlab">Show</span>
      <div class="nrow">
        <select id="sort" title="which crops to surface first">
          <option value="low" selected>Least confident first</option>
          <option value="conf">Most confident first</option>
          <option value="new">Newest first</option>
        </select>
        <select id="size"><option value="50">50 per page</option>
          <option value="100">100 per page</option></select>
        <button class="rbtn quiet" id="reload" title="pull in detections found since this page loaded"><svg class="bico" viewBox="-1 -1 26 26" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>Refresh pool</button>
        <!-- destructive, and deliberately NOT beside Prev/Next: a mis-click
             there would throw away every keep decision made so far -->
        <button class="rbtn danger" id="unkeep" title="put every crop you already judged a dog back into the queue"><svg class="bico" viewBox="-1 -1 26 26" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>Restore kept</button>
      </div>
    </div>
    <!-- data-own: the guesser strip decides whether this group is worth
         showing, from whether there is a guesser at all. trimGroups()
         must not also decide it from whether its controls are visible --
         the Run button never hides, so the two answers differ. -->
    <div class="ngrp" id="ngrpWho" data-own="1">
      <span class="nlab">Guesses by</span>
      <div class="nrow">
        <select id="trgModel" hidden title="which model's guesses the filter above shows, and which one Run starts"></select>
        <button type="button" class="trgbtn" id="trgRun">&mdash;</button>
      </div>
      <details class="trgnote" id="trgNote" hidden>
        <summary id="trgNoteSum"></summary>
        <p id="trgNoteBasis"></p>
        <p id="trgNoteCaveat"></p>
        <p id="trgNoteWhich"></p>
      </details>
    </div>
  </div>

  <!-- The page's own progress, and the guesser's, on one quiet line each.
       Two full-width strips with their own bars, dots and legends said the
       same two numbers at four times the volume. -->
  <div class="lines">
    <p class="line" id="bal" hidden>
      <b id="balNum">&mdash;</b><span id="balNumU">crops left to judge</span>
      <i class="track"><i id="balFill"></i></i>
      <span class="lend" id="balLeft"></span>
      <span class="lsub" id="balMain"></span>
    </p>
    <p class="line trg" id="trg" hidden>
      <span class="trgdot" id="trgDot"></span>
      <b id="trgState">&mdash;</b><span class="lsub" id="trgSub"></span>
      <i class="track"><i id="trgFill"></i></i>
      <span class="lend" id="trgPct"></span>
    </p>
  </div>

  <div class="pagebar">
    <span class="cnt" id="pg">&mdash;</span>
    <button class="rbtn quiet" id="next" title="bank this screen and bring up the next unjudged crops">Next &rsaquo;</button>
  </div>
</header>


<!-- The legend is a lesson, and a lesson stops being one after the first
     day. It kept two full lines of the viewport permanently to teach four
     keys. Folded away, remembered open or shut, and the four keys that carry
     the work stay on the summary line where the hand can find them. -->
<details class="keys" id="keys">
  <summary>
    <span class="klead"><kbd>F</kbd> not a dog<kbd>D</kbd> is a dog<kbd>L</kbd> leashed<kbd>N</kbd> no leash</span>
    <span class="kmore">more</span>
  </summary>
  <div class="kbody">
    <span><kbd>&larr;</kbd><kbd>&rarr;</kbd><kbd>&uarr;</kbd><kbd>&darr;</kbd> move</span>
    <span><kbd>&#9166;</kbd> full frame &amp; edit box</span>
    <span><kbd>&#8679;</kbd>+arrows nudge box &middot; saves itself</span>
    <span><kbd>U</kbd> undo</span>
    <span>Flag what is <b>not</b> a dog, and mark the low-confidence ones that <b>are</b>. Moving to another page passes on the rest, so nothing you have judged comes back.</span>
    <span>The bar under each crop is detector confidence.</span>
    <span>One crop per camera pass, and one per photo &mdash; repeat frames and
    duplicate shots of the same animal are hidden.</span>
  </div>
</details>

<div class="grid" id="grid"></div>
<div id="state"></div>
<div class="foot" id="foot" hidden>
  <span class="cnt" id="pg2"></span>
  <button class="rbtn quiet" id="next2" title="bank this screen and bring up the next unjudged crops">Next &rsaquo;</button>
</div>
</div>
<script>
__COPY_JS__

/* sel = -1 means NOTHING is selected. The page opens that way on purpose:
   a pre-selected first tile looks like a choice the user did not make. The
   first arrow press picks tile 0 and keyboard flow takes over from there. */
var page=0,size=50,sort='low',country='',countryName='',items=[],reserve=[],pages=1,sel=-1,
    smallN=0,minPx=0,harvestN=0,mode='queue',verdict='all',suggest='',leashf='all',find='',gatef='all',loading=false,
    /* Which guesser the guess filter is reading. Page scope, not inside the
       progress strip's closure: the queue request needs it too, and the two
       must never disagree about whose opinions are on screen. */
    BACKEND='siglip',
    todoN=0,flaggedN=0,posN=0,seenN=0,dupN=0,session=0,lastUndo=null,toastT=null,lb=null,busy={};
/* leash verdicts for what is on screen, and whether the store exists at all.
   LEASH_ON stays false on a checkout without the tool, and the two buttons
   simply never render. */
var LEASH={},LEASH_ON=false,leashN={leashed:0,unleashed:0};
function leash(name,label){
  if(!LEASH_ON)return;
  /* clicking the label a crop already has takes it back -- the same gesture
     the verdict buttons use, and the reason this is a database rather than an
     append-only log */
  var had=LEASH[name]===label;
  var body=had?{name:name,remove:true}:{name:name,label:label};
  if(had)delete LEASH[name]; else LEASH[name]=label;
  paintLeash(name);
  fetch('/api/review/leash',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)})
    .then(function(r){return r.json()})
    .catch(function(){return null})
    .then(function(j){
      if(!j||!j.ok){
        /* put the button back where it was: an optimistic paint that the
           server refused is a lie about what was recorded */
        if(had)LEASH[name]=label; else delete LEASH[name];
        paintLeash(name);
        leashNote((j&&j.error)?('Leash verdict not saved: '+j.error)
                              :'Leash verdict not saved.');
        return;
      }
      leashN.leashed=j.leashed;leashN.unleashed=j.unleashed;
      paintLeashCount();
      var held=document.querySelector('.card.awaitleash[data-name="'+
        name.replace(/"/g,'\\"')+'"]');
      if(!held)return;
      held.classList.remove('awaitleash');
      /* Both axes answered, so the tile has nothing left to ask and leaves
         like any other judged crop. It was only kept back to make the leash
         askable; taking the leash back (had) leaves it in place, still owing
         one. */
      if(had){held.classList.add('awaitleash');return;}
      releaseHeld(name);
    });
}
function releaseHeld(name){
  /* the same surgical removal flag() does: splice the item, drop the node,
     pull one from reserve so the grid stays full, and leave the rest of the
     DOM untouched so nothing reflows under the cursor */
  var i=idx(name);
  if(i<0)return;
  var card=cardAt(i);
  items.splice(i,1);
  if(card&&card.parentNode)card.parentNode.removeChild(card);
  var nx=reserve.shift();
  if(nx){items.push(nx);$('grid').appendChild(tile(nx))}
  /* the undo toast still on screen belongs to the DOG verdict that put this
     tile on hold; if undoing it now has to hand a reserve crop back, its
     bookkeeping has to know that happened here */
  if(lastUndo&&lastUndo.crop&&lastUndo.crop.name===name&&nx)lastUndo.pulled=true;
  if(sel>=items.length)sel=items.length-1;
  if(!items.length)render();else mark();
}
function leashNote(msg){
  /* the page has no general notice -- showUndo is about a verdict and offers
     to undo it, which is the wrong thing to say when nothing was recorded */
  var t=$('tbox');
  if(!t){t=document.createElement('div');t.className='toast';t.id='tbox';
    document.body.appendChild(t)}
  t.innerHTML='';
  var sp=document.createElement('span');sp.className='tt';sp.textContent=msg;
  t.appendChild(sp);
  clearTimeout(toastT);
  toastT=setTimeout(function(){lastUndo=null;hideToast()},4000);
}
function paintLeash(name){
  var card=document.querySelector('.card[data-name="'+name.replace(/"/g,'\\"')+'"]');
  if(!card)return;
  var le=card.querySelector('.lbtn.le'),un=card.querySelector('.lbtn.un');
  if(le)le.classList.toggle('on',LEASH[name]==='leashed');
  if(un)un.classList.toggle('on',LEASH[name]==='unleashed');
}
function paintFind(j){
  var el=$('find'), dl=$('findterms'), msg=$('findmsg');
  if(!el)return;
  /* the control only exists once there is something to search */
  var ready=(j.find_terms||[]).length>0;
  el.hidden=!ready&&!j.find;
  if(dl&&ready)dl.innerHTML=(j.find_terms||[]).map(function(t){
    return '<option value="'+esc(t)+'">'}).join('');
  var st=j.find_state, cov=j.find_cover||[0,0];
  /* Every state that is not 'ordered the queue' gets a sentence, because the
     one thing they have in common on screen is that the queue does not move.
     Without this they are all the same event to a reader. */
  /* textContent, so the raw term goes in -- esc() here would show a search
     for "cats & dogs" as "cats &amp; dogs" */
  var say=st==='learning'
    ? 'working out what “'+j.find+'” looks like — try again in a moment'
    : st==='unknown'
    ? 'nothing has encoded “'+j.find+'” yet'
    : st==='failed'
    ? 'could not encode “'+j.find+'” — see data/crop_search.log'
    : st==='cold'
    /* Both numbers. 'cold' means nothing in THIS view is searchable, which is
       not the same as nothing being embedded at all -- a country or leash
       filter can select a slice the guesser has not reached. Naming only the
       pool size told a reviewer to start a guesser that was already running
       and had covered most of the pool. */
    ? (cov[0] ? n(cov[0])+' of '+n(cov[1])+' crops are embedded, but none in '+
                'this view — clear a filter, or let the guesser catch up'
              : 'none of the '+n(cov[1])+' crops in the queue have been '+
                'embedded yet — start the guesser above and it embeds as it '+
                'works')
    : st==='novectors'
    ? 'no crops have been embedded yet — start the guesser above'
    : st==='mismatch'
    ? 'the crops were embedded with a different model — re-encoding the search '+
      'words to match, try again in a moment'
    : '';
  el.classList.toggle('warn',!!say);
  if(msg){msg.hidden=!say;msg.textContent=say;}
  el.title=st==='on'
    ? n(j.find_hits)+' of '+n(cov[1])+' crops ranked by how much they look '+
      'like '+j.find
    : say||('type what you are looking for and the queue is reordered to '+
            'bring it to the front');
}
/* ── the caption, and the filters you have applied ────────────────────────
   FILTERS is the single description of what can narrow the queue: the control
   that sets it, how to read its current value, and what resets it. Every part
   of the block below is derived from this one list, so a filter cannot appear
   in the chips and be missing from the panel, or be removable and not
   resettable. Adding one means adding one entry. */
function optText(el,strip){
  var o=el.options&&el.options[el.selectedIndex];
  if(!o)return '';
  return strip?o.text.replace(/\s*\(.*\)$/,''):o.text;
}
/* `where` is which VIEW each filter actually narrows, and it is not
   decoration: the two views send different requests. The audit list is
   fetched with label= and leash= and nothing else, so a guess or a country
   left set from the queue narrows nothing there -- and the chip row was
   advertising both while hiding the verdict filter, the one that does apply.
   A chip that names a filter the request never carried is worse than no chip:
   it explains an empty list with a cause that is not the cause. */
var FILTERS=[
  {id:'suggest', off:'',    where:'queue', strip:1},
  {id:'gatef',   off:'all', where:'queue', strip:1},
  {id:'country', off:'',    where:'queue', strip:1},
  {id:'leashf',  off:'all', where:'both', strip:1},
  /* shown and hidden by CSS on body.auditing rather than by the hidden
     attribute, so its availability is the view, not el.hidden */
  {id:'verdict', off:'all', where:'audit', css:1}
];
function activeFilters(){
  var out=[];
  for(var i=0;i<FILTERS.length;i++){
    var f=FILTERS[i],el=$(f.id);
    if(!el)continue;
    if(f.where!=='both'&&f.where!==mode)continue;
    if(!f.css&&el.hidden)continue;
    if(el.value===f.off)continue;
    var t=optText(el,f.strip);
    if(t)out.push({id:f.id,off:f.off,text:t});
  }
  return out;
}
/* The sentence. It says what the queue IS, and -- only when something has
   actually been narrowed -- what it was narrowed from. That readout is the
   one thing this block exists to do, so it is the only thing in it set in the
   page's own voice rather than as a control. */
function paintCap(j){
  var el=$('cap');if(!el)return;
  /* the two payloads name their count differently; the caption is one
     sentence and must not read an undefined field in either mode */
  var act=activeFilters();
  var showing=(mode==='audit')?(j.total||0):(j.total_unflagged||0);
  var word=mode==='audit'?'annotation':'crop';
  var head=mode==='audit'
    ? 'Checking <b>'+n(showing)+'</b> '+word+(showing===1?'':'s')
    : 'Reviewing <b>'+n(showing)+'</b> unjudged '+word+(showing===1?'':'s');
  var from=(act.length&&j.pool_unfiltered&&j.pool_unfiltered>showing)
    ? ' <i>&middot; narrowed from '+n(j.pool_unfiltered)+'</i>' : '';
  el.innerHTML=head+from;
  var btn=$('narrow');
  if(btn)btn.innerHTML='Filter'+(act.length?' <em>'+act.length+'</em>':'')+
    '<span class="ncar" aria-hidden="true">›</span>';
}
/* Only what is applied, each one removable where it is read. A filter you
   cannot see is a filter you will not think to clear, which is how an empty
   queue becomes a bug report. */
function paintChips(){
  var box=$('chips');if(!box)return;
  var act=activeFilters();
  box.hidden=!act.length;
  if(!act.length){box.innerHTML='';return;}
  box.innerHTML=act.map(function(a){
    return '<span class="chip">'+esc(a.text)+
      '<button type="button" class="chipx" data-f="'+att(a.id)+
      '" title="'+att('clear '+a.text)+'" aria-label="'+
      att('clear '+a.text)+'">×</button></span>'}).join('');
}
function paintGate(j){
  var el=$('gatef');if(!el)return;
  /* only once the gate has verdicts over this queue: a dropdown that filters
     nothing is worse than no dropdown */
  if(!j.gate_ready){el.hidden=true;return;}
  el.hidden=false;
  var c=j.gate_counts||{},who=j.gate_label||'Gate';
  var L=[['all',who+': any',null],
         ['dog','Gate says dog','dog'],
         ['not_dog','Gate says not a dog','not_dog'],
         ['none','No gate verdict','none']];
  var html='';
  for(var i=0;i<L.length;i++){
    var k=L[i][2],n2=(k===null)?null:(c[k]||0);
    html+='<option value="'+att(L[i][0])+'">'+esc(L[i][1])+
      (n2===null?'':' ('+n(n2)+')')+'</option>';
  }
  el.innerHTML=html;
  el.value=j.gate||'all';
}
function paintLeashOptions(counts){
  var sel=$('leashf');
  if(!sel)return;
  sel.hidden=false;
  /* each option says how many it would show: "Needs a leash call" is the
     number this axis is worked from, and it should be readable without
     selecting it first */
  var lab={all:'Any leash state',none:'Needs a leash call',
           leashed:'Leashed',unleashed:'Unleashed'};
  [].forEach.call(sel.options,function(o){
    var c=counts[o.value];
    o.textContent=lab[o.value]+(c==null?'':'  ('+n(c)+')');
    o.disabled=(o.value!=='all'&&c===0&&leashf!==o.value);
  });
  sel.value=leashf;
}
function paintLeashCount(){
  var e=$('leashN');
  if(e)e.textContent=n(leashN.leashed)+' / '+n(leashN.unleashed);
}
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
  /* The grid is honest while loading -- it shows skeletons -- but the count
     line went on describing the PREVIOUS mode, so switching to Check my
     annotations and back read "1,642 annotated" over a queue that was still
     arriving. And `items` still held the old list, so a keyboard F or D
     landed on a crop from the view being left. Both are neutralised until the
     response the page is actually waiting for lands. */
  loading=true;
  items=[];reserve=[];
  $('pg').textContent=$('pg2').textContent='loading\u2026';
  $('next').disabled=$('next2').disabled=true;
}
function toTop(){
  try{
    window.scrollTo({top:0,left:0,behavior:SOFT?'smooth':'auto'});
  }catch(_){
    window.scrollTo(0,0);          /* older engines: no options object */
  }
}
/* AUDIT MODE. A misannotation does not wait in a queue -- it goes into a
   dataset as ground truth and teaches the wrong thing, and until now nothing
   could look at one again: flagging removed a crop from the queue for good.
   This reads the ledgers instead of the pool, shows each crop's current
   verdict, and lets it be changed in place. */
/* EVERY loader stamps its request and drops a response the page has moved
   past. Both write the same globals -- items, pages, the count line, the
   pager -- so a slower fetch landing after a faster one repainted the grid
   with the other mode's data: switching to Check my annotations and back
   left the queue showing fifty lit verdict buttons and an "annotated" count,
   with mode already back to queue. Same race on sort, size, verdict and
   country, which fire the same way. */
var reqSeq=0;
function loadAudit(){
  var my=++reqSeq;
  skeleton();
  return fetch('/api/review/annotated?page='+page+'&size='+size+
               '&sort='+auditSort()+'&label='+encodeURIComponent(verdict)+
               '&leash='+encodeURIComponent(leashf))
  .then(function(r){if(!r.ok)throw 0;return r.json()})
  .then(function(j){
    if(my!==reqSeq)return;          /* superseded before it landed */
    if(j.error)throw 0;
    loading=false;
    items=j.items||[];reserve=[];page=j.page||0;pages=j.pages||1;
    if(j.leash)LEASH=j.leash;
    if(j.leash_totals){LEASH_ON=true;leashN=j.leash_totals;paintLeashCount();}
    if(j.leash_counts)paintLeashOptions(j.leash_counts);
    var only=verdict==='false_positive'?' marked not a dog':
             verdict==='true_positive'?' marked a dog':' annotated';
    var lab=items.length?
      (n(items.length)+' shown \u00b7 '+n(j.total)+only+
       (verdict==='all'?' \u00b7 '+n(j.n_false_positive)+' not a dog, '+
        n(j.n_true_positive)+' a dog':'')):
      (verdict==='all'?'nothing annotated yet':'none with that verdict');
    $('pg').textContent=lab;$('pg2').textContent=lab;
    $('next').disabled=$('next2').disabled=page>=pages-1;
    $('foot').hidden=pages<=1;
    paintChips();paintCap(j);
    if(sel>=items.length)sel=items.length-1;
    render();toTop();
  }).catch(function(){
    if(my!==reqSeq)return;          /* an old failure must not paint over a
                                       render that has already succeeded */
    $('state').innerHTML='<div class="state"><b>Could not load annotations</b>'+
      'The ledgers under data/hard_negatives and data/hard_positives could '+
      'not be read.</div>';
    $('grid').innerHTML='';
  });
}
/* the queue's sort names are about confidence; only two of them mean
   anything for a list that is already judged */
function auditSort(){return (sort==='conf'||sort==='low')?sort:'recent'}
function load(){
  if(mode==='audit')return loadAudit();
  var my=++reqSeq;
  skeleton();
  /* returns the promise: callers (and the test harness) can await a settled
     grid instead of guessing at microtask depth */
  return fetch('/api/review?page='+page+'&size='+size+'&sort='+sort+
    '&suggest='+encodeURIComponent(suggest)+
    '&leash='+encodeURIComponent(leashf)+
    '&gate='+encodeURIComponent(gatef)+
    '&find='+encodeURIComponent(find)+
    '&backend='+encodeURIComponent(BACKEND)+
               '&country='+encodeURIComponent(country))
  .then(function(r){if(!r.ok)throw 0;return r.json()})
  .then(function(j){
    if(my!==reqSeq)return;          /* superseded before it landed */
    if(j.error)throw 0;
    loading=false;
    items=j.items||[];reserve=j.reserve||[];page=j.page||0;pages=j.pages||1;
    todoN=j.total_unflagged||0;flaggedN=j.flagged_total||0;
    smallN=j.too_small||0;minPx=j.min_px||0;
    harvestN=j.harvested_available||0;
    if(j.seen_total!=null)seenN=j.seen_total;
    if(j.positive_total!=null)posN=j.positive_total;
    if(j.collapsed!=null)dupN=j.collapsed;
    if(j.leash)LEASH=j.leash;
    if(j.leash_totals){LEASH_ON=true;leashN=j.leash_totals;paintLeashCount();}
    if(j.leash_counts)paintLeashOptions(j.leash_counts);
    paintCountries(j.countries,j.country,j.country_coverage);
    paintSuggest(j);
    paintGate(j);
    paintChips();
    paintCap(j);
    trimGroups();
    paintFind(j);
    score();
    /* "Page 3 of 47" described an offset that no longer moves. What the
       reader actually needs is how much is left after this screen. */
    var more=Math.max(0,todoN-items.length);
    var lab=items.length?(n(items.length)+' shown \u00b7 '+n(more)+' left'):
      'nothing left to review';
    /* Held-back crops are stated, not silently dropped -- and the threshold
       is named so the number can be argued with. */
    if(smallN)lab+=' \u00b7 '+n(smallN)+' too small to judge (under '+minPx+'px)';
    /* the queue is two sources now, and which one a crop came from changes
       what it is: the pool is whatever the sweep passed in the last while,
       the harvested set was chosen from all 1.2M positives on purpose */
    if(harvestN)lab+=' \u00b7 '+n(harvestN)+' harvested from the full sweep';
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
    if(my!==reqSeq)return;          /* an old failure must not paint over a
                                       render that has already succeeded */
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
    dz.parentNode.title=n(dupN)+' crops hidden because the same picture is '
      +'already accounted for: another frame from the same Mapillary sequence, '
      +'or a pixel-identical copy of a crop already in the queue or already '
      +'judged. The same photo reaches this queue through more than one '
      +'sequence, which is how near-identical crops used to be judged twice.';}
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
    el.className='line';
    $('balFill').style.width='0%';
    $('balNum').textContent='—';$('balNumU').textContent='dataset not found';
    $('balLeft').textContent='';
    $('balMain').textContent=b.error||('missing '+(b.dataset||'dataset'));
    el.title='';
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
  /* One fill, one meaning: not-dog crops standing against the dog target,
     counting both those already built into the dataset and those your flags
     have earned since. Splitting the two put the difference between
     bookkeeping states in front of the number that matters. */
  var got=Math.min(want,have+pend);
  var pct=want?Math.min(100,100*got/want):0;
  $('balFill').style.width=pct.toFixed(1)+'%';
  var short=Math.max(0,want-have-pend);
  /* JUDGEMENTS, not flags. Dividing the shortfall by the yield answers a
     different question -- how many more NOT-DOGS are needed -- and that only
     equals the reviewer's workload if every future verdict is "not a dog".
     About a fifth are not, and each of those joins the very class being
     chased, so a judgement closes the gap by yield x (negatives - positives).
     Mirrored from the server so banking a verdict updates it without a round
     trip. */
  var nf=b.new_flags||0,np=b.new_positive_flags||0,jd=nf+np;
  var share=(jd>=(b.mix_min_sample||50))?np/jd:0;
  var net=y*(1-2*share);
  var need=short?(net>0?Math.ceil(short/net):null):0;
  el.className='line'+(short?'':' ok');
  var ds=b.dataset||'the dataset';
  if(!short){
    $('balNum').textContent='0';
    $('balNumU').textContent='crops left to judge';
    $('balLeft').textContent='100%';
    $('balMain').textContent='balanced';
  }else if(need===null){
    /* every judgement adds to both sides at the same rate: reviewing alone
       cannot close this, and a number here would be a lie */
    $('balNum').textContent='—';
    $('balNumU').textContent='not closing';
    $('balLeft').textContent=Math.round(pct)+'%';
    $('balMain').textContent='not closing at this rate';
  }else{
    $('balNum').textContent=n(need);
    $('balNumU').textContent='crops left to judge';
    /* One number at the end of the track, in the track's own unit. The line
       used to carry three readings of one thing -- the crops to judge, the
       crops the bar is short, and the percentage with its fraction spelled
       out -- in three different units, which is what made a one-line summary
       longer than the thing it summarised. */
    $('balLeft').textContent=Math.round(pct)+'%';
    $('balLeft').title=n(short)+' more not-dog crops fill this bar. That is '+
      'not the same as the '+n(need)+' on the left: those are crops to JUDGE, '+
      'and only some of what you judge ends up a usable not-dog crop.';
    $('balMain').textContent='';
  }
  /* The breakdown that used to occupy four legend swatches. It explains the
     number rather than competing with it, so it lives on hover. */
  el.title=n(have)+' not-dog crops in '+ds+', plus '+n(pend)+' earned from '+
    n(nf)+' flag'+(nf===1?'':'s')+' since that build, against '+n(want)+
    ' dog crops'+(pendPos?' (+'+n(pendPos)+' from crops you marked as dogs)':'')+
    '.\n\nAbout '+Math.round(y*100)+'% of what you flag survives into the '+
    'dataset — the rest is held back for acceptance, near-duplicate, under '+
    'the size floor, or ambiguous'+
    (share?'; and '+Math.round(share*100)+'% of what you judge comes back "a '+
      'dog", which raises the target':'')+
    (b.reserved_ids?'.\n\n'+n(b.reserved_ids)+' crops are reserved to test '+
      'the gate and never trained on.':'.');
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
  /* PREVIEW first, HQ when it lands. The HQ cut (from the ORIGINAL -- the
     preview thumbnails are cut from the 1280 letterbox and capped at 160px,
     which throws away 3.6-5.3x the pixels actually available) is cut on
     demand, and a cold cut opens a 12MP frame plus two duckdb scans: measured
     on the live audit view, 8 of 50 tiles sat as BLANK cards for 25-30s with
     nothing to say a load was even happening. So the tile paints the cheap
     preview immediately and swaps to the HQ crop the moment the server has
     it; if the preview itself is missing it falls the other way, to /hq.
     Every crop is clickable: has_full only says a burned-in preview frame was
     saved, and the editor reads the original either way. */
  var hqURL='/hq?name='+encodeURIComponent(c.name);
  d.innerHTML='<img class="thumb zoom" loading="lazy" alt="detection crop" '+
      'src="'+
      (c.label?('/flagged?label='+encodeURIComponent(c.label)+'&name='):
       c.harvested?'/review_set/':'/recent_crops/')+
      encodeURIComponent(c.name)+'" '+
      "onerror=\"this.onerror=null;this.src='"+hqURL+"'\">"+
    '<div class="rail"><i style="width:'+pc+'%"></i></div>'+
    /* One caption line, contact-sheet style: frame slug, the model's guess,
       exposure. The guess used to be a chip pinned over the top-left of the
       photograph -- covering the only thing on the card worth looking at, and
       on a queue where most guesses read "nothing" it covered it for nothing.
       Dotted and muted here so it still never reads as a verdict. */
    '<div class="meta"><span class="id" title="'+att(c.image_id)+'">'+esc(c.image_id)+
      '</span>'+
      (c.sg?'<span class="sg sg-'+c.sg+'" title="'+att('a general-purpose image '+
        'model guessed '+(c.sgl||c.sg)+((c.sgp!=null)?' ('+Math.round(c.sgp*100)+
        '% of its confidence on '+SG_WORD[c.sg]+')':'')+
        '. A suggestion for sorting the queue — not a label, and not recorded.')+
      '">'+esc(c.sgl||SG_WORD[c.sg])+'</span>':'')+
      '<span class="cf">'+(+c.conf||0).toFixed(2)+'</span></div>'+
    '<div class="actwrap">'+
    '<div class="acts">'+
      '<button class="fbtn no'+(c.label==='false_positive'?' on':'')+
        '" type="button" title="'+(c.label==='false_positive'?
          'click again to remove this annotation':'false positive (F)')+'">'+
        '&#9873; Not a dog</button>'+
      '<button class="fbtn yes'+(c.label==='true_positive'?' on':'')+
        '" type="button" title="'+(c.label==='true_positive'?
          'click again to remove this annotation':
          'a real dog the detector was unsure about (D)')+
        '">&#10003; Is a dog</button>'+
    '</div>'+
    /* A SECOND axis, kept visually apart from the verdict row above it. A
       leash label says "this is a dog, and here is whether it is on a leash" --
       it is stored on its own and never touches the dog/not-dog ledgers. */
    (LEASH_ON?('<div class="acts leash">'+
      '<button class="lbtn le'+(LEASH[c.name]==='leashed'?' on':'')+
        '" type="button" title="'+(LEASH[c.name]==='leashed'?
          'click again to remove this leash verdict':'on a leash (L)')+
        '">Leashed</button>'+
      '<button class="lbtn un'+(LEASH[c.name]==='unleashed'?' on':'')+
        '" type="button" title="'+(LEASH[c.name]==='unleashed'?
          'click again to remove this leash verdict':'no leash (N)')+
        '">Unleashed</button>'+
    '</div>'):'')+
    '</div>';
  var im=d.querySelector('.thumb');
  /* The upgrade starts when the PREVIEW has painted, so the lazy loader still
     decides which tiles cost anything at all, and the HQ requests trickle in
     at the pace the grid scrolls rather than 50 at once. {once:true}: the
     swap itself fires 'load' again. Guarded like prefetchNext_ -- the test
     harness has no Image, and an optimisation must not take down the tile. */
  if(typeof Image!=='undefined')
    im.addEventListener('load',function(){
      var up=new Image();
      up.onload=function(){if(im.src!==up.src)im.src=up.src};
      up.src=hqURL;
    },{once:true});
  im.onclick=function(){openLb(idx(c.name))};
  d.querySelector('.fbtn.no').onclick=function(e){
    e.stopPropagation();flag(idx(c.name),false,'false_positive')};
  d.querySelector('.fbtn.yes').onclick=function(e){
    e.stopPropagation();flag(idx(c.name),false,'true_positive')};
  if(LEASH_ON){
    d.querySelector('.lbtn.le').onclick=function(e){
      e.stopPropagation();leash(c.name,'leashed')};
    d.querySelector('.lbtn.un').onclick=function(e){
      e.stopPropagation();leash(c.name,'unleashed')};
  }
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
    /* Two different facts share this empty grid. With a filter set, the POOL
       is not judged -- only the slice is empty -- and "Every detection has
       been judged" under a header reading "narrowed from 2,716" was the page
       disagreeing with itself. Say which one is true. */
    if(activeFilters().length){
      $('state').innerHTML='<div class="state"><b>Nothing matches these filters</b>'+
        (mode==='audit'
          ?'None of your annotations fit this slice. '
          :'The queue still holds unjudged crops; this slice of it is empty. ')+
        'Clear a chip above to widen the view.'+
        '<div><button class="rbtn" id="rl2">'+ICO_REFRESH+'Check again</button></div></div>';
      $('rl2').onclick=load;return;
    }
    $('state').innerHTML='<div class="state"><b>Queue is clear</b>'+
      'Every detection in the pool has been judged. New crops appear here as the '+
      'sweep finds them.<div><button class="rbtn" id="rl2">'+ICO_REFRESH+'Check for more</button></div></div>';
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
  if(loading)return;               /* the list under this index is on its way out */
  var c=items[i];if(!c||busy[c.name])return;
  label=label||'false_positive';
  /* Auditing is not consuming. The crop keeps its place in the grid and its
     buttons restate the verdict, so a screenful can be checked without the
     list resequencing under the reader after every click. Re-deciding is
     already handled server-side: flag_crop rewrites the other label's ledger
     rather than filing one image under both. */
  if(mode==='audit'){
    if(busy[c.name])return;
    busy[c.name]=1;
    var was=c.label;
    /* Clicking the verdict a crop ALREADY has takes it back: the annotation
       is removed and the crop returns to the unreviewed queue. Auditing is
       for the ones you got wrong, and "wrong" includes having judged
       something that should never have been judged at all -- a crop too
       blurred to call, or one flagged by a mis-click. Without this the only
       way out of a bad annotation was to assert the opposite one, which is
       just a second guess wearing a verdict. */
    var undo=(was===label);
    return fetch('/api/detect/flag',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:c.name,label:label,undo:undo})})
     .then(function(r){return r.json()}).then(function(j){
        delete busy[c.name];
        if(!j||j.ok===false)return;
        c.label=undo?null:label;
        var el=cardAt(i);
        if(el){
          var no=el.querySelector('.fbtn.no'),ys=el.querySelector('.fbtn.yes');
          if(no)no.classList.toggle('on',c.label==='false_positive');
          if(ys)ys.classList.toggle('on',c.label==='true_positive');
          /* the border marks "this differs from what the ledger held when the
             page loaded", which a removal does too */
          el.classList.toggle('changed',was!==c.label);
          el.classList.toggle('unjudged',!c.label);
        }
        if(j.flagged_total!=null)flaggedN=j.flagged_total;
        if(j.positive_total!=null)posN=j.positive_total;
        score();
        /* leashNote, not toast: this page has no toast(). The audit branch is
           the only caller that reached for one, so every verdict changed here
           threw a ReferenceError into an empty catch and confirmed nothing. */
        leashNote(undo?'annotation removed \u2014 back in the queue':
                  'changed to '+(label==='true_positive'?'a dog':'not a dog'));
     }).catch(function(){delete busy[c.name]});
  }
  /* "Is a dog" is the point at which a leash call becomes askable, so the
     crop stays where it is instead of being consumed -- you can look again,
     open it, fix the box and answer the leash, all on the tile you just judged.
     Only for that verdict, only while the leash store is on, and only until it
     has a leash call: everything else still leaves the queue on click, which
     is what makes the queue drain. */
  var hold=(LEASH_ON&&label==='true_positive'&&!LEASH[c.name]);
  busy[c.name]=1;
  var card=cardAt(i);
  if(card&&!hold)card.classList.add('go');
  return fetch('/api/detect/flag',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:c.name,label:label})})
   .then(function(r){return r.json()}).then(function(j){
      delete busy[c.name];
      if(!j||j.ok===false){if(card)card.classList.remove('go');return}
      session++;
      todoN=Math.max(0,todoN-1);
      if(label==='true_positive')posN++;else flaggedN++;
      score();bumpBal(label==='true_positive'?0:1,label==='true_positive'?1:0);
      if(hold){
        /* judged, kept: the verdict shows on the tile and the leash row is
           now the only thing left to answer on it */
        c.label=label;
        if(card){
          var ys=card.querySelector('.fbtn.yes');
          if(ys)ys.classList.add('on');
          card.classList.add('awaitleash');
        }
        if(!viaKey)sel=-1; else mark();
        showUndo(c,i,false,label,true);
        return;
      }
      /* Surgical removal + backfill: the rest of the grid does not re-render,
         so nothing reflows under the cursor and no image reloads. */
      items.splice(i,1);
      if(card&&card.parentNode)card.parentNode.removeChild(card);
      var nx=reserve.shift();
      if(nx){items.push(nx);$('grid').appendChild(tile(nx))}
      if(!viaKey)sel=-1;                         /* mouse: no auto-advance */
      if(sel>=items.length)sel=items.length-1;   /* stays -1 if unset */
      if(!items.length)render();else mark();
      showUndo(c,i,!!nx,label);
   }).catch(function(){delete busy[c.name];if(card)card.classList.remove('go')});
}

/* ── undo ───────────────────────────────────────────────────────────────── */
function showUndo(c,at,pulled,label,held){
  /* `pulled` records whether the flag consumed a crop from `reserve` to keep
     the grid full; undo has to hand that one back or the page grows on every
     flag/undo cycle.

     `held` records the other shape: the leash hold deliberately leaves the
     crop on the grid, so there is nothing to give back and nothing to
     re-insert. Undo assumed every verdict had removed its tile, so undoing a
     held one put a second copy in `items` and a second tile on the grid, both
     lit for a record the server had just deleted. */
  lastUndo={crop:c,at:at,pulled:pulled,label:label||'false_positive',
            held:!!held};
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
      var at;
      if(u.held){
        /* the tile never left: take the verdict back off the one already
           there rather than adding another */
        u.crop.label=null;
        at=idx(u.crop.name);
        var hel=cardAt(at);
        if(hel){
          hel.classList.remove('awaitleash');
          var hy=hel.querySelector('.fbtn.yes');
          if(hy)hy.classList.remove('on');
        }
      }else{
        /* back where it was, not at the front: the eye is still on that spot */
        at=Math.min(u.at,items.length);
        if(!items.length)$('state').innerHTML='';
        items.splice(at,0,u.crop);
        g.insertBefore(tile(u.crop),g.children[at]||null);
      }
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
        /* Only here, and only here on purpose: the id is a thing you want
           while looking hard at one crop, and a copy button on every tile in
           a 50-crop grid would be 50 controls nobody asked for. */
        '<button class="rbtn quiet lbcopy" id="lbcopy" '+
          'title="copy this image\u2019s Mapillary id">Copy ID</button>'+
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
    $('lbcopy').onclick=function(e){
      /* the footer sits over the image; a click here must not also count as
         a click on the backdrop or the frame behind it */
      e.stopPropagation();
      copyOnto(this,this.dataset.id||'','Copy ID');
    };
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
  /* the button carries the id it will copy, so stepping to the next crop
     cannot leave it pointing at the one before */
  var cb=$('lbcopy');
  if(cb){cb.dataset.id=c.image_id||'';cb.textContent='Copy ID';
    cb.classList.remove('done');}
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
    /* Leash calls work in here too, and this is the view where they should be
       made: a leash is a few pixels wide, and at thumbnail size it is simply
       not visible -- the same reason the dashboard-thumbnail table reads 54.9%
       where the full-resolution crops read 81.3% for the same detections.
       Unlike F and D these do NOT close the lightbox: deciding the leash does
       not decide the dog, and you usually want to answer both while looking at
       the same frame. */
    else if((e.key==='l'||e.key==='L')&&sel>=0&&LEASH_ON){
      leash(items[sel].name,'leashed');e.preventDefault()}
    else if((e.key==='n'||e.key==='N')&&sel>=0&&LEASH_ON){
      leash(items[sel].name,'unleashed');e.preventDefault()}
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
  if((e.key==='l'||e.key==='L')&&sel>=0&&LEASH_ON){
    e.preventDefault();leash(items[sel].name,'leashed');return;
  }
  if((e.key==='n'||e.key==='N')&&sel>=0&&LEASH_ON){
    e.preventDefault();leash(items[sel].name,'unleashed');return;
  }
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
  /* auditing pages through a fixed list -- there is nothing to bank, the same
     reason go(d) has. And banking it is not merely useless: a crop whose
     annotation was just removed is back in the queue by the promise the
     button makes, and this would retire it unjudged and unreachable. */
  if(mode==='audit')return Promise.resolve();
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
  if(mode==='audit'||!items.length||!navigator.sendBeacon)return;
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
function go(d){
  /* auditing pages through a fixed list -- there is nothing to bank, and the
     list does not shrink as you look at it */
  if(mode==='audit')return function(){page=Math.max(0,page+d);sel=-1;load()};
  return nav(function(){page=0;sel=-1;load()});
}
$('next').onclick=$('next2').onclick=function(){return go(1)()};
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
/* The sort choice survives the visit. Validated against the <select>'s own
   options on the way back in, so a stale value left by an older build -- or
   anything else that ends up in storage -- cannot put the page into a sort
   the server does not know. */
var SG_WORD={dog:'a dog',animal:'an animal',object:'not an animal'};
var ICO_REFRESH='<svg class="bico" viewBox="-1 -1 26 26" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>';
var SORT_KEY='sdReviewSort';          /* the older single-key store */
var PREFS_KEY='sdReview';
function loadPrefs(){
  var o=null;
  try{o=JSON.parse(localStorage.getItem(PREFS_KEY))}catch(_){}
  if(o&&typeof o==='object')return o;
  /* carry a sort chosen under the old key forward, so the one preference
     that already survived a visit does not get dropped by this change */
  var s=null;try{s=localStorage.getItem(SORT_KEY)}catch(_){}
  return s?{sort:s}:{};
}
function savePref(k,v){
  var o=loadPrefs();o[k]=v;
  try{localStorage.setItem(PREFS_KEY,JSON.stringify(o))}catch(_){}
}
/* Applies a stored value only if the control actually offers it. A value left
   by an older build -- or anything else that ends up in storage -- must not
   put the page into a state the server does not know. */
function restoreSel(id,val){
  var el=$(id);
  if(!el||val==null)return null;
  for(var i=0;i<el.options.length;i++){
    if(el.options[i].value===String(val)){el.value=String(val);return el.value;}
  }
  return null;
}
function restorePrefs(){
  var o=loadPrefs(),v;
  if((v=restoreSel('sort',o.sort))!==null)sort=v;
  if((v=restoreSel('size',o.size))!==null)size=parseInt(v,10)||size;
  if((v=restoreSel('verdict',o.verdict))!==null)verdict=v;
  /* the options are rebuilt from the response, like country's, so the value
     rides along in the first request instead of being matched now */
  if(o.suggest)suggest=o.suggest;
  if((v=restoreSel('leashf',o.leashf))!==null)leashf=v;
  if(typeof o.find==='string'){find=o.find;if($('find'))$('find').value=find;}
  if(typeof o.backend==='string'&&o.backend)BACKEND=o.backend;
  if(typeof o.gatef==='string'&&o.gatef)gatef=o.gatef;
  if(o.npanel&&$('npanel')){$('npanel').hidden=false;
    if($('narrow')){$('narrow').classList.add('on');
      $('narrow').setAttribute('aria-expanded','true');}}
  if((v=restoreSel('mode',o.mode))!==null){
    mode=v;
    document.body.classList.toggle('auditing',mode==='audit');
  }
  /* The country <select> is built from the response, so its options do not
     exist yet. Carry the value into the first request instead and let
     paintCountries either select it or, if the pool no longer holds that
     country, clear it -- see the self-heal there. */
  if(o.country)country=o.country;
}
/* CHANGING THE VIEW IS NOT REVIEWING.
   markSeen() banks every crop on screen -- it is how "I am done with these"
   is recorded, and Next earns it. These three do not: showing 100 instead of
   50, reordering the same queue, or narrowing to one country all mean "show
   me this differently", and banking first threw the current screenful out of
   the queue unjudged. Switching 50 -> 100 quietly consumed 50 crops. */
$('sort').onchange=function(){var v=this.value;
  savePref('sort',v);sort=v;page=0;sel=-1;load()};
$('size').onchange=function(){var v=parseInt(this.value,10)||50;
  savePref('size',v);size=v;page=0;sel=-1;load()};
/* Rebuilt from every response so the hourly refresh reaches an open tab, but
   only when the option set actually CHANGED -- rewriting the <select> on each
   page turn would drop the open dropdown and reset the caret mid-click. */
var countrySig='';
function paintCountries(list,cur,cov){
  if(!list)return;
  /* What share of the queue the index can place, ON the control it limits.
     Picking a country drops every crop with no country -- there is no
     unknown-country escape -- so a coverage of 0.74 means a quarter of the
     queue is unreachable with any country selected, and the option counts
     cannot say so: they are tallied over crops that HAVE a country, so each
     one promises exactly what it delivers. The server has computed this
     number all along and nothing displayed it, which is how an entire crop
     directory sat unindexed without a symptom. */
  var csel=$('country');
  if(csel)csel.title='only review crops from one country'+
    (cov==null?'':' — the index can place '+Math.round(cov*100)+
     '% of this queue; the rest are only reachable with the filter off');
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
  /* A remembered country the pool no longer holds would filter every request
     down to nothing, and the <select> cannot even show it -- the option does
     not exist -- so the page would look unfiltered while returning an empty
     queue forever. Drop it and reload, once. */
  if(cur&&!countryName){
    country='';savePref('country','');
    $('country').value='';
    if(!countryHealed){countryHealed=true;load();}
    return;
  }
  if(cur!=null)$('country').value=cur;
}
var countryHealed=false;
/* Only offered once the triage tool has produced predictions: an empty
   dropdown that silently filters nothing is worse than no dropdown. Counts
   come from the server, tallied on the post-collapse queue, so each option
   promises exactly what it delivers. */
/* A group whose every control is hidden is a heading over nothing: an
   uppercase label and 29px of panel naming a category that offers no choice.
   The flat row it replaced left no trace when its selects were hidden. */
function trimGroups(){
  var gs=document.querySelectorAll('.ngrp');
  for(var i=0;i<gs.length;i++){
    var g=gs[i],row=g.querySelector('.nrow'),live=0;
    if(!row)continue;
    for(var k=0;k<row.children.length;k++)if(!row.children[k].hidden)live++;
    /* a group with an owner decides for itself; see data-own in the markup */
    if(!g.dataset.own)g.hidden=!live;
  }
}
function paintSuggest(j){
  var el=$('suggest');if(!el)return;
  /* The server is the authority on what it applied. It drops a guess filter
     whose bucket the chosen guesser cannot produce, and the page has to adopt
     that or it re-sends the dropped value forever -- which is how a hidden
     control went on emptying the queue across reloads, the preference
     outliving every surface that could show or clear it. */
  if(typeof j.suggest==='string'&&j.suggest!==suggest){
    suggest=j.suggest;savePref('suggest',suggest);
  }
  /* The gate has its own control, so it must not also appear here: two
     dropdowns filtering by one model's verdict is a choice the reader has to
     work out is not a choice. Selecting it in the toggle still runs it and
     still shows its coverage — that is what the toggle is for. */
  if(j.backend==='dogbin'&&j.gate_ready){el.hidden=true;return;}
  if(!j.suggest_ready){el.hidden=true;return;}
  el.hidden=false;
  /* The options are the chosen guesser's own vocabulary, sent with the
     payload. Hard-coding three buckets here offered the dog-bin gate an
     "Other animal" filter it can never fill, and hid the "Not a dog" one it
     answers with. */
  var c=j.suggest_counts||{},L=[['','Any guess',null]];
  var bk=j.buckets||[{key:'dog',label:'Looks like a dog'},
                     {key:'animal',label:'Other animal'},
                     {key:'object',label:'Not an animal'}];
  for(var b=0;b<bk.length;b++)L.push([bk[b].key,bk[b].label,bk[b].key]);
  L.push(['none','No guess yet','none']);
  var html='';
  for(var i=0;i<L.length;i++){
    var k=L[i][2],n2=(k===null)?null:(c[k]||0);
    html+='<option value="'+att(L[i][0])+'">'+esc(L[i][1])+
      (n2===null?'':' ('+n(n2)+')')+'</option>';
  }
  el.innerHTML=html;
  el.value=j.suggest||'';
}
$('suggest').onchange=function(){var v=this.value;
  savePref('suggest',v);suggest=v;page=0;sel=-1;load()};
$('country').onchange=function(){var v=this.value;
  savePref('country',v);country=v;page=0;sel=-1;load()};
/* Switching mode does NOT bank the screen, for the same reason the other view
   controls do not: nothing on it has been judged by looking at it. Audit mode
   also hides the controls that mean nothing there -- a country filter over
   crops chosen by verdict, and a Next that would bank annotations as if they
   were fresh work. */
/* guarded: the control is absent on a checkout with no leash store, and an
   unguarded assignment there throws and takes the rest of the script with it */
if($('find')){
  var ft=null;
  $('find').addEventListener('input',function(){
    /* debounced: every keystroke would re-rank the whole pool server-side */
    clearTimeout(ft);
    var v=this.value;
    ft=setTimeout(function(){
      if(v===find)return;
      find=v;savePref('find',find);page=0;sel=-1;load();
    },420);
  });
  $('find').addEventListener('search',function(){
    clearTimeout(ft);find=this.value;savePref('find',find);page=0;sel=-1;load();
  });
}
/* Compact once the top of the page has scrolled away. Guarded because the
   test harness has no IntersectionObserver, and a page that threw here would
   lose every handler bound after it. */
(function(){
  var cue=$('scrollcue');
  if(!cue||typeof IntersectionObserver!=='function')return;
  var at=null,hold=0;
  new IntersectionObserver(function(es){
    var want=!es[0].isIntersecting;
    if(want===at)return;
    /* Hysteresis. Compacting removes height from a sticky header, which moves
       everything below it — and that settling can cross the sentinel again
       and ask for the opposite. Refusing a reversal for a moment turns a
       flutter into one change. */
    /* the observer's own timestamp, not the wall clock: it is what the
       browser measured the crossing at, and it can be driven in a test */
    var now=(typeof es[0].time==='number')?es[0].time:Date.now();
    if(at!==null&&now-hold<260)return;
    at=want;hold=now;
    document.body.classList.toggle('compact',want);
  },{threshold:0}).observe(cue);
})();
if($('narrow'))$('narrow').addEventListener('click',function(){
  /* Inline, not a popover: this page is driven by F/D/L/N with the hands on
     the keyboard, and a floating layer that took focus would fight the work
     it exists to serve. */
  var pan=$('npanel'),open=pan.hidden;
  pan.hidden=!open;
  this.classList.toggle('on',open);
  this.setAttribute('aria-expanded',open?'true':'false');
  savePref('npanel',open?'1':'');
});
/* Delegated, because the chips are rebuilt on every load and listeners bound
   to the old nodes would die with them. */
if($('chips'))$('chips').addEventListener('click',function(e){
  var b=e.target&&e.target.closest&&e.target.closest('.chipx');
  if(!b)return;
  var id=b.getAttribute('data-f'),el=$(id);
  if(!el)return;
  for(var i=0;i<FILTERS.length;i++)if(FILTERS[i].id===id)el.value=FILTERS[i].off;
  /* go through the control's own handler so one filter has one code path */
  if(el.onchange)el.onchange.call(el);
  else{page=0;sel=-1;load();}
});
if($('gatef'))$('gatef').onchange=function(){
  gatef=this.value;savePref('gatef',gatef);page=0;sel=-1;load();
};
if($('leashf'))$('leashf').onchange=function(){
  leashf=this.value;savePref('leashf',leashf);page=0;sel=-1;load();
};
$('mode').onchange=function(){
  mode=this.value;savePref('mode',mode);page=0;sel=-1;
  document.body.classList.toggle('auditing',mode==='audit');
  load();
};
/* A flip made while this filter is on leaves the tile where it is rather than
   yanking it out from under the pointer -- it carries the changed border, and
   the next load drops it. */
$('verdict').onchange=function(){verdict=this.value;
  savePref('verdict',verdict);page=0;sel=-1;load()};
restorePrefs();
load();loadBal();
/* ── crop-suggestion run progress ── polls /api/triage while the tab is
   visible. No fold to gate on here (unlike the dashboard's sweep panel), so
   it just runs whenever the page is shown. Hidden until a run exists. */
(function(){
  var el=document.getElementById('trg');
  if(!el)return;
  function setPara(id,text){
    var el=$(id); if(!el)return;
    el.textContent=text||''; el.hidden=!text;
  }
  function paintBackends(j){
    var sel=$('trgModel'), note=$('trgNote'), list=(j&&j.backends)||[];
    if(!sel)return;
    /* one guesser is not a choice, so do not draw a control that offers one */
    sel.hidden=list.length<2;
    if(sel.hidden){if(note)note.hidden=true;return;}
    var want=list.map(function(b){
      return b.key+'|'+b.label+'|'+b.recall}).join(',');
    /* rebuilt only when the offer itself changes: this repaints every 5s and
       an unconditional innerHTML would close the dropdown under the pointer */
    if(sel.dataset.sig!==want){
      sel.dataset.sig=want;
      sel.innerHTML=list.map(function(b){
        /* BOTH numbers. One alone is not a claim: a guesser that called
           everything a dog would read 100% here. The set they came from is
           the fold below — an <option> cannot hold a sentence. */
        var pct=b.recall==null?'':' · finds '+Math.round(b.recall*100)+
                '%, clears '+Math.round((b.clears||0)*100)+'%';
        return '<option value="'+esc(b.key)+'">'+esc(b.label)+pct+
               '</option>'}).join('');
    }
    if(sel.value!==BACKEND)sel.value=BACKEND;
    var cur=list.filter(function(b){return b.key===BACKEND})[0];
    if(note){
      note.hidden=!cur;
      if(cur){
        /* The summary answers the question the dropdown raises — 75% of
           WHICH dogs — in one line. Everything else waits behind the fold. */
        var pct=cur.recall==null?'':Math.round(cur.recall*100)+'%';
        $('trgNoteSum').textContent=pct
          ? cur.label+' finds '+pct+' of the dogs in a fixed test set, and '+
            'clears '+Math.round((cur.clears||0)*100)+'% of the not-dogs — '+
            'what that means'
          : 'about '+cur.label;
        /* Set through a helper that tolerates an absent node. paint() runs
           inside a promise chain, so a throw here is swallowed and the strip
           simply stops repainting -- the whole of it, not just the legend.
           One missing paragraph is not worth the progress bar. */
        setPara('trgNoteBasis',j.recall_basis);
        setPara('trgNoteCaveat',j.recall_caveat);
        setPara('trgNoteWhich',cur.note);
      }
    }
  }
  function paint(j){
    /* shown once a run has happened OR once one could be started -- the
       button lives in here, so hiding it on an empty file left no way back */
    /* The controls moved out of this row into the panel, and the only thing
       that ever hid them was being inside it. On a checkout with no guesser
       at all the row correctly disappears and this returns -- leaving a Run
       button on screen showing the markup's raw placeholder, enabled, and
       clickable: it reads its own label to decide direction, so the dash was
       treated as "start". The group goes with the row. */
    var who=$('ngrpWho');
    if(!j||(!j.ever&&!j.can_run)){
      el.hidden=true;if(who)who.hidden=true;return;
    }
    el.hidden=false;if(who)who.hidden=false;
    paintBackends(j);
    var running=!!j.running, cov=Math.round((j.coverage||0)*100),
        gap=Math.max(0,(j.pool||0)-(j.guessed||0)), state, sub='';
    /* 'line trg' is the base: the strip became one of the two caption lines,
       and rewriting className wholesale used to drop the class that gives it
       its type and layout */
    el.className='line trg'+(running?' on':(j.stalled?' warn':''));
    if(j.starting){
      /* spawned, but the model is still loading and it has not written a
         count yet; "0 of 0" would read as nothing to do */
      state='Starting the guesser';
      sub='loading the model';
    }else if(running&&j.total&&(j.done||0)>=j.total){
      /* the pass finished and it is sleeping until the next look -- saying
         "Guessing crops" through that is the same lie as saying "stopped" */
      state='Watching for new crops';
      sub='last pass did '+(j.total||0).toLocaleString()+
          (j.watch?' \u00b7 looks again every '+j.watch+'s':'');
    }else if(running&&j.total){
      state='Guessing crops';
      sub=(j.done||0).toLocaleString()+' of '+(j.total||0).toLocaleString()+
          ' this pass'+(j.rate?' \u00b7 '+j.rate+'/s':'');
    }else if(running){
      state='Watching for new crops';
      sub='nothing waiting'+(j.watch?' \u00b7 rechecks every '+j.watch+'s':'');
    }else if(j.stalled){
      state='Run stopped';
      sub='no progress for '+Math.round((j.age_s||0)/60)+' min';
    }else if(gap>0&&j.why){
      /* it stopped for a reason it left behind -- name it, because "not
         running" alone sends the reader to look at the dashboard */
      state='Guessing stopped';
      sub=j.why+' \u00b7 '+gap.toLocaleString()+' crop'+(gap===1?'':'s')+
          ' still have no guess';
      el.className='line trg warn';
    }else if(j.busy_with){
      /* the OTHER guesser has the card. Saying "Not running" here is true and
         useless: it invites a Run press that can only be refused. */
      state='Waiting for '+j.busy_with;
      sub=j.busy_with+' is guessing now · they share the card'+
          (gap>0?' · '+gap.toLocaleString()+' crop'+(gap===1?'':'s')+
                 ' have no guess from this one':'');
    }else if(gap>0){
      /* no run active says nothing about coverage: do not claim "up to date"
         while the queue is only partly guessed */
      state='Not running';
      sub=gap.toLocaleString()+' crop'+(gap===1?'':'s')+' have no guess yet';
      el.className='line trg warn';
    }else if(!j.ever){
      state='No guesses yet';
      sub=(j.pool||0).toLocaleString()+' crops in the queue';
    }else{
      state='Guesses up to date';
      sub=j.model?j.model.split('/').pop():'';
    }
    $('trgState').textContent=state;
    $('trgSub').textContent=sub;
    /* the percentage alone. "3,385 crops have no guess yet" and "1,633 of
       5,018 crops guessed" are the same fact twice, and they sat either side
       of the bar that is a third telling of it. The counts move to the hover,
       where a breakdown belongs. */
    $('trgPct').textContent=cov+'%';
    el.title=(j.guessed||0).toLocaleString()+' of '+
      (j.pool||0).toLocaleString()+' crops in the queue have a guess from '+
      (j.model?String(j.model).split('/').pop():'this model')+'.';
    $('trgFill').style.width=cov+'%';
    /* the button reflects what the run IS doing, so the label is the action
       it would take -- not the state it is in */
    var btn=$('trgRun');
    /* `dataset.busy`, not `disabled`: the guard means "a click of mine is in
       flight, do not overwrite its label", and now that paint() also disables
       the button for the other guesser, reading disabled would latch it off
       forever. */
    if(btn&&btn.dataset.busy!=='1'){
      btn.textContent=running?'Pause':'Run guesses';
      btn.title=running
        ?'Stop guessing. Everything already guessed is kept.'
        :'Guess the crops that have none yet, then watch for new ones.';
      /* Pressing it while the other guesser holds the card can only be
         refused, so do not offer the press. Not hidden -- a control that
         vanishes reads as a broken page -- just plainly unavailable. */
      var blocked=!running&&!!j.busy_with;
      btn.disabled=blocked;
      if(blocked)btn.title='Pause '+j.busy_with+' first — the two guessers '+
                           'share the card.';
    }
  }
  /* Bumped whenever the run is started or stopped. A poll issued BEFORE an
     action can land after it and repaint the state the action just changed,
     which showed "Not running" for a few seconds under a button already
     reading Pause. Responses from an older generation are dropped. */
  var gen=0;
  function poll(){
    if(document.hidden)return;
    var mine=gen;
    /* catch is for a dropped request only, never a bug in paint() */
    fetch('/api/triage?backend='+encodeURIComponent(BACKEND))
      .then(function(r){return r.json()})
      .catch(function(){return null})
      .then(function(j){if(j&&mine===gen)paint(j)});
  }
  poll(); setInterval(poll,5000);
  document.addEventListener('visibilitychange',function(){if(!document.hidden)poll()});
  /* Reachable from outside the closure. The strip is otherwise repainted only
     by a 5s timer, which means nothing can ask it to catch up after an event
     that changed what it should say -- and a harness with a stubbed
     setInterval could never see it repaint at all. */
  window.__trgPoll=poll;

  var msel=$('trgModel');
  if(msel) msel.addEventListener('change',function(){
    BACKEND=this.value; savePref('backend',BACKEND);
    /* Both halves move together. The strip's coverage and the queue's guess
       filter are two views of the same guesser, and repainting one without
       the other is how a page ends up reporting SigLIP's coverage over
       RF-DETR's guesses. */
    gen++; page=0; sel=-1;
    poll(); load();
  });

  var btn=$('trgRun');
  if(btn) btn.addEventListener('click',function(){
    /* what the button says is what it does: read the label, not a cached
       flag that a poll may have moved underneath it */
    var stopping=btn.textContent.indexOf('Pause')===0;
    var old=document.querySelector('.trgerr');
    if(old) old.remove();
    gen++;
    btn.dataset.busy='1';
    btn.disabled=true; btn.textContent=stopping?'Pausing\u2026':'Starting\u2026';
    fetch('/api/triage',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:stopping?'stop':'start',backend:BACKEND})})
      .then(function(r){return r.json()})
      .catch(function(){return {ok:false,msg:'the dashboard did not answer'}})
      .then(function(j){
        btn.dataset.busy='';
        btn.disabled=false;
        if(j&&j.ok){
          /* the server already said what happened; showing it now beats
             waiting up to 5s for a poll to say the same thing */
          btn.textContent=j.running?'Pause':'Run guesses';
        }
        if(j&&!j.ok&&j.msg){
          /* a start that fails does so for a reason the reader can act on --
             usually an interpreter without torch -- so say it rather than
             silently going back to "Not running" */
          /* beside the button that failed, not in the progress row it used
             to live in -- that row is hidden in exactly the cases a start is
             refused */
          var e=document.createElement('div');
          e.className='trgerr'; e.textContent=j.msg;
          ($('ngrpWho')||el).appendChild(e);
        }
        /* the status file is written by the run itself, so it lags the
           spawn; poll now for the button, and again once it has caught up */
        poll(); setTimeout(poll,1500);
      });
  });
})();
</script></body></html>"""
# Substituted at import time, so the server and the tests both read a
# finished document -- a placeholder left in REVIEW_HTML would parse as a
# bare identifier and only fail when a button was pressed.
REVIEW_HTML = REVIEW_HTML.replace('__COPY_JS__', COPY_JS)

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
# Bounded: the sweep writes ~4 crops/s for days. Sized for who hashes now --
# the whole review pool (retention 3000) plus every crop both ledgers have
# judged, which is ~1800 and grows with each flag. It used to be a handful of
# sequence-less crops, so 8000 was generous; at ~4800 per request it was close
# enough to the ceiling that ordinary sweep churn would trip the wipe every
# few minutes and re-read every one of them from disk.
DHASH_MAX = 24000


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


# Visual-duplicate collapse: the Hamming distance between two 64-bit dHashes
# at which they are the same picture. 0 is exact-match only; -1 turns it off.
#
# 0 because it was measured, twice. Every duplicate pair that survived the
# sequence collapse sat at distance 0 -- the same photo reached through two
# different Mapillary sequences -- and nothing sat between 1 and 5. The two
# pairs that turned up at distance 6 were BOTH wrong on sight: a blue-toned
# dog matched to a person sitting on the ground, and a small pale crop matched
# to a dark silhouette. These crops are small, blurry and low-contrast, which
# is where a 64-bit perceptual hash is weakest, so the tolerance that looks
# generous is the one that hides a crop nobody has judged.
DUP_BITS = cfg_int('review_dup_bits', 0, env='REVIEW_DUP_BITS')
def _dup_bands(h, nb):
    """`nb` slices of a 64-bit hash, tagged by position, as evenly as they cut.

    Two hashes differing in at most nb-1 bits cannot differ in ALL nb bands, so
    sharing no band proves they are further apart than the tolerance. That turns
    the comparison from every-crop-against-every-crop into nb dict lookups --
    the difference between ~9 million popcounts per request and a few hundred.

    The band count is derived from the tolerance rather than fixed, because a
    fixed one is a trap: with 8 bands the guarantee holds only to 7 bits, and a
    tolerance of 8 or 10 set in config would have quietly started missing real
    pairs -- measured at 3 and 12 misses per 400 -- with nothing to show for it.
    """
    out, lo = [], 0
    for i in range(nb):
        hi = 64 * (i + 1) // nb
        out.append((i, (h >> lo) & ((1 << (hi - lo)) - 1)))
        lo = hi
    return out


class DupIndex:
    """Hashes seen so far, searchable by near-equality."""

    def __init__(self, bits=None):
        self.bits = DUP_BITS if bits is None else bits
        # one more band than the tolerance: see _dup_bands
        self.nb = max(1, min(64, self.bits + 1))
        self.bands = {}
        self.exact = set()

    def hit(self, h):
        """Is a hash within `bits` of this one already in the index?"""
        if h is None or self.bits < 0:
            return False
        if self.bits == 0:
            return h in self.exact       # the default: no popcount needed
        seen = set()
        for key in _dup_bands(h, self.nb):
            for other in self.bands.get(key, ()):
                if other in seen:
                    continue
                seen.add(other)
                if bin(h ^ other).count('1') <= self.bits:
                    return True
        return False

    def add(self, h):
        if h is None or self.bits < 0:
            return
        if self.bits == 0:
            self.exact.add(h)
            return
        for key in _dup_bands(h, self.nb):
            self.bands.setdefault(key, []).append(h)


_JUDGED_DUPS = {'at': None, 'index': None}


def judged_dup_index():
    """A DupIndex over the crop copies both flag ledgers keep.

    The sequence collapse already stops a judged crop's own siblings coming
    back; this is the same guarantee for the case it cannot see -- the same
    photo reached through a different sequence. That case is the one that put
    near-identical crops on both sides of a verdict.

    Rebuilt when either ledger's crop directory changes, which is once per
    flag, and the hashes themselves are cached by name.
    """
    dirs = [os.path.join(HN_DIR, 'crops'), os.path.join(HP_DIR, 'crops')]
    try:
        stamp = tuple(os.stat(d).st_mtime_ns if os.path.isdir(d) else 0
                      for d in dirs)
    except OSError:
        stamp = None
    if _JUDGED_DUPS['at'] == stamp and _JUDGED_DUPS['index'] is not None:
        return _JUDGED_DUPS['index']
    idx = DupIndex()
    if DUP_BITS >= 0:
        for d in dirs:
            try:
                names = os.listdir(d)
            except OSError:
                continue
            for nm in names:
                if nm.lower().endswith('.jpg'):
                    idx.add(_dhash(os.path.join(d, nm), 'judged/' + nm))
    _JUDGED_DUPS.update(at=stamp, index=idx)
    return idx


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
        # Drop the oldest half rather than everything. A full wipe made the
        # next request re-read every crop it had just hashed; keeping the
        # newer half means the working set survives its own overflow.
        # pop, not del: two review requests are two threads, both can take
        # the same snapshot, and the second del would raise KeyError on a key
        # the first already removed. dict.clear() was idempotent and this
        # replaced it, so the property has to be put back deliberately.
        for k in list(_dhash_cache)[:len(_dhash_cache) // 2]:
            _dhash_cache.pop(k, None)
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
# A run that finished one epoch is a smoke test, not a result: it "ran to last
# epoch" only because its budget WAS one epoch. Two of them sat in the history
# beside 300-epoch runs with a best epoch of 1 and a metric nobody should
# compare against anything.
TRK_MIN_EPOCHS_DEFAULT = 2


def min_epochs():
    return max(0, cfg_int('training_min_epochs', TRK_MIN_EPOCHS_DEFAULT,
                          env='TRAINING_MIN_EPOCHS'))


def is_real_run(r, floor):
    """Whether a run belongs in the history.

    A LIVE run always does, whatever its epoch count -- it is on epoch 1 for a
    while, and hiding the thing the section exists to watch would be the worst
    possible reading of "too short to matter".
    """
    if r['status'] == 'running':
        return True
    return r['status'] in REAL and (r['epochs_done'] or 0) >= floor


def best_per_project(runs):
    """{project: run name} -- the highest deciding metric in each project.

    Only runs of the SAME task are compared, because mAP50-95 and top-1
    accuracy are not the same axis. Ties keep the earlier row, matching how
    ultralytics breaks a fitness tie.

    This is a measurement over what is on disk, and it can disagree with the
    registry's `best` -- which is worth seeing rather than hiding. The registry
    records what was PROMOTED, on reserved data and after a human looked; this
    column records what merely scored highest on each run's own validation
    split. When they differ, the promoted one is usually still right.
    """
    top = {}
    for r in runs:
        v = r.get('best_headline')
        if v is None or not r.get('task'):
            continue
        key = (r['project'], r['task'])
        if key not in top or v > top[key][0]:
            top[key] = (v, r['name'])
    # one winner per project: if a project somehow holds two tasks, the
    # higher number would be meaningless across them, so keep them separate
    out = {}
    for (proj, _task), (v, name) in top.items():
        if proj not in out or v > out[proj][0]:
            out[proj] = (v, name)
    return {p: n for p, (_v, n) in out.items()}


def _history(runs):
    """Finished runs as a table -- also the non-hover route to every value."""
    top = best_per_project(runs)
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
        star = ''
        if top.get(r['project']) == r['name']:
            star = (f'<span class="bstar"'
                    + _t(f'Highest {r.get("headline_label") or "metric"} among '
                         f'this project\'s runs, on each run\'s own '
                         f'validation split. Not the same claim as the '
                         f'registry column, which records what was promoted, '
                         f'on reserved data.')
                    + '>&#9733; best</span>')
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
            f'<td class="tbest">{star}</td>'
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
        f'<th{_t("The highest deciding metric among this projects runs, "
                  "measured from results.csv. Compared only within a task, and "
                  "on each run own validation split -- two runs trained on "
                  "different datasets are not strictly comparable.")}>'
        'best in project</th>'
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


def find_run(key, runs=None):
    """The run a key names, preferring the one that actually ran.

    run_key is project/name and is NOT unique. ultralytics honours its own
    runs_dir, so project=dogdetection can land at <root>/runs/detect/... while
    a stale directory of the same name sits at <root>/... -- discover()'s own
    docstring describes this exact case, and it is live on this box.

    Two resolvers disagreeing about which one they mean is worse than the
    ambiguity itself: render_run_detail scanned in order and got the
    405-epoch run, render_run_diff built a dict and got the 0-epoch stub, so
    the comparison reported "epochs run 0" for a run whose own detail page,
    one click above it, showed 405. Everything resolves through here now, and
    the one that recorded epochs wins -- a directory with no results.csv is
    the leftover in every case this arises from.
    """
    runs = training_runs() if runs is None else runs
    hits = [r for r in runs if run_key(r) == key]
    if not hits:
        return None
    # max() keeps the first on a tie, so discovery order still decides
    return max(hits, key=lambda r: (r.get('epochs_done') or 0))


def _compare_picker(key, runs):
    """The control that turns one run's detail into a comparison.

    Sorted with the same project first: the comparison anyone actually wants is
    against the run before this one, not against a detector from another
    project scored on a different metric.
    """
    here = key.split('/', 1)[0]
    others = [run_key(r) for r in runs if run_key(r) != key]
    others.sort(key=lambda k: (k.split('/', 1)[0] != here, k))
    if not others:
        return ''
    opts = ''.join(f'<option value="{esc_html(k)}">{esc_html(k)}</option>'
                   for k in others)
    return (f'<div class="dpick"><label for="trkCmp">Compare with</label>'
            f'<select id="trkCmp" data-a="{esc_html(key)}">'
            f'<option value="">&mdash; pick a run &mdash;</option>{opts}'
            f'</select></div>')


def render_run_detail(key):
    """One run's detail region, resolved by key against what was discovered."""
    runs = training_runs()
    r = find_run(key, runs)
    if r is not None:
        return (_past_head(r) + _compare_picker(key, runs)
                + _charts(r) + render_confusion(r) + render_mistakes(r))
    return '<div class="mnone">That run is no longer on disk.</div>'


# ── comparing two runs ──────────────────────────────────────────────────────
# Bookkeeping, not settings: these differ between any two runs and say nothing
# about why one won. Leaving them in buries the three lines that matter under
# twenty that do not.
DIFF_SKIP = frozenset((
    'name', 'project', 'save_dir', 'exist_ok', 'resume', 'save_period',
    'device', 'workers', 'verbose', 'plots', 'save', 'save_json', 'save_txt',
    'source', 'show', 'val', 'split', 'time',
))


def _diff_args(a, b):
    """[(key, a value, b value)] for settings the two runs did not share."""
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
        import training_tracker
    except Exception:
        return []
    out = []
    ra = training_tracker.read_args(os.path.join(a['dir'], 'args.yaml'))
    rb = training_tracker.read_args(os.path.join(b['dir'], 'args.yaml'))
    for k in sorted(set(ra) | set(rb)):
        if k in DIFF_SKIP:
            continue
        va, vb = ra.get(k), rb.get(k)
        if str(va) != str(vb):
            out.append((k, va, vb))
    return out


def _delta(av, bv, digits=4, higher_better=True, fmt=None):
    """A relative to B, as a signed cell that says which way is good.

    A is the run being examined -- the first column, the one named first in
    the heading. It used to subtract the other way, so a run that scored 0.37
    against a promoted 0.50 reported "+0.1270" in the colour of an
    improvement. Every regression read as a win.

    Formatted with the row's own formatter when it has one: a wall clock shown
    as "1h 55m" whose change reads "-559" makes the reader do arithmetic in a
    unit the row never used.
    """
    if av is None or bv is None:
        return '<td class="dnum dmid">&mdash;</td>'
    d = av - bv
    if abs(d) < 10 ** -(digits + 1):
        return '<td class="dnum dmid">no change</td>'
    good = (d > 0) if higher_better else (d < 0)
    mag = fmt(abs(d)) if fmt else f'{abs(d):.{digits}f}'
    # A change the row's own formatter cannot represent is not a change worth
    # printing: 0.2s between two epochs both shown as "35s" came out as a
    # portentous "-0s". The test is whether the row renders the two values
    # identically -- comparing against fmt(0) does not work, because _hms(0)
    # is the "--" placeholder rather than a zero.
    if fmt and fmt(av) == fmt(bv):
        return '<td class="dnum dmid">no change</td>'
    return (f'<td class="dnum {"dup" if good else "ddn"}">'
            f'{"+" if d > 0 else "&minus;"}{mag}</td>')


def render_run_diff(a_key, b_key):
    """Two runs side by side: what differed, and what it bought.

    Resolved by exact match against discovered runs, like every other run
    endpoint -- a directory arriving from the client is a traversal waiting to
    happen.
    """
    runs = training_runs()
    # the same resolver the detail view uses -- see find_run
    a, b = find_run(a_key, runs), find_run(b_key, runs)
    if not a or not b:
        return '<div class="mnone">One of those runs is no longer on disk.</div>'
    if a_key == b_key:
        return '<div class="mnone">Pick two different runs to compare.</div>'

    rows = []
    # A run that is still training has not won on cost. Its epoch count, the
    # position of its peak and its wall clock are partial totals, and against a
    # finished run every one of them comes out smaller and therefore green --
    # three of ten rows in the improvement colour on the screen a promotion is
    # decided from. The values are true and stay; the comparison does not.
    live = [r for r in (a, b) if r.get('status') == 'running']

    def line(label, av, bv, fmt=None, digits=4, higher_better=True,
             hint='', partial=False, notes=(None, None)):
        def cell(v, note=None):
            if v is None:
                return '<td class="dnum dmid">&mdash;</td>'
            # a peak belongs to an epoch, and two runs peak at different ones;
            # without the epoch beside it the row reads as one model's score
            tag = f'<em>{esc_html(str(note))}</em>' if note else ''
            return f'<td class="dnum">{fmt(v) if fmt else v}{tag}</td>'
        chg = ('<td class="dnum dmid">&mdash;</td>' if partial and live
               else _delta(av, bv, digits, higher_better, fmt))
        rows.append(f'<tr{_t(hint)}><th>{label}</th>'
                    f'{cell(av, notes[0])}{cell(bv, notes[1])}{chg}</tr>')

    am = {m['key']: m for m in (a.get('latest') or [])}
    bm = {m['key']: m for m in (b.get('latest') or [])}
    same_metric = a.get('headline_key') == b.get('headline_key')
    mlabel = a.get('headline_label') or 'headline metric'

    # On the csv column, which is the metric's one identity. This used to look
    # up by headline_label against a table keyed by the SHORT name, and the
    # two vocabularies only agree for detect: HEADLINE['classify'] labels it
    # 'top-1 accuracy' and LATEST['classify'] calls it 'accuracy_top1', so
    # every classifier comparison -- dogbin_008 vs dogbin_007, leash_v2_003 vs
    # leash_v2_002 -- missed, and both epoch notes came out None. The row then
    # read exactly the way the comment in line() says it must not: two peaks
    # from two different epochs of two different-length runs, printed side by
    # side with nothing saying so. The label fallback stays for a run payload
    # cached before `col` existed.
    def headline_metric(r, table):
        hk = r.get('headline_key')
        for m in (r.get('latest') or []):
            if hk and m.get('col') == hk:
                return m
        return table.get(r.get('headline_label')) or {}

    ah = headline_metric(a, am)
    bh = headline_metric(b, bm)
    line(f'best {esc_html(mlabel)}' if same_metric else 'best (differing metric)',
         a.get('best_headline'), b.get('best_headline'),
         fmt=lambda v: f'{v:.4f}', digits=4,
         notes=(f'epoch {ah["peak_epoch"]}' if ah.get('peak_epoch') else None,
                f'epoch {bh["peak_epoch"]}' if bh.get('peak_epoch') else None),
         hint='The peak the run reached, not where it finished.')
    # Every metric the run recorded, not just the one that picks the
    # checkpoint. A detector is promoted on mAP but SHIPPED on recall -- a
    # missed dog is unrecoverable and a false positive is one click -- and a
    # comparison that shows only the headline cannot answer the question the
    # retrain was for. Reported at each run's BEST-FITNESS epoch, because
    # that is the checkpoint that would be promoted; the peak of a metric at
    # some other epoch belongs to a model nobody would ship.
    for key in [k for k in ('recall', 'precision', 'mAP50')
                if k in am or k in bm]:
        hint = {'recall': 'Of the dogs really there, the share it found. The '
                          'error this project cannot recover from.',
                'precision': 'Of what it called a dog, the share that was '
                             'one. Costs a click, not a dog.',
                'mAP50': 'The same question as mAP50-95 but forgiving about '
                         'how tightly the box fits.'}[key]
        line(f'{key} at best epoch',
             (am.get(key) or {}).get('at_best'),
             (bm.get(key) or {}).get('at_best'),
             fmt=lambda v: f'{v:.4f}', digits=4, hint=hint)
    # and the ceiling each run reached. Only recall carried one of these, which
    # read as a run scoring worse than it ever had: a run whose precision
    # touched 0.8393 at epoch 64 was reported at the 0.7855 its promoted
    # checkpoint happened to hold, and nothing on the screen said otherwise.
    # Every metric gets its ceiling, and every ceiling names its epoch --
    # because these four peaks land on four different epochs and no single
    # checkpoint holds them all. The at-best-epoch rows above are the model
    # that would actually ship; these are the ceilings it was drawn from.
    for key in [k for k in ('recall', 'precision', 'mAP50')
                if k in am or k in bm]:
        ap, bp = am.get(key) or {}, bm.get(key) or {}
        if ap.get('peak') is None and bp.get('peak') is None:
            continue
        line(f'best {key} at any epoch',
             ap.get('peak'), bp.get('peak'),
             fmt=lambda v: f'{v:.4f}', digits=4,
             notes=(f'epoch {ap["peak_epoch"]}' if ap.get('peak_epoch') else None,
                    f'epoch {bp["peak_epoch"]}' if bp.get('peak_epoch') else None),
             hint='The highest this metric ever reached, and when. A different '
                  'epoch from the promoted one means no shipped checkpoint '
                  'ever held this number.')
    line('epochs run', a.get('epochs_done'), b.get('epochs_done'),
         digits=0, higher_better=False, partial=True,
         hint='More epochs is not better; it is what the run cost.')
    line('best epoch', a.get('best_epoch'), b.get('best_epoch'), digits=0,
         higher_better=False, partial=True,
         hint='How early the peak arrived. Earlier is cheaper.')
    line('seconds per epoch', a.get('secs_per_epoch'), b.get('secs_per_epoch'),
         fmt=lambda v: _hms(v), digits=1, higher_better=False)
    line('wall clock', a.get('wall_clock_s'), b.get('wall_clock_s'),
         fmt=lambda v: _hms(v), digits=0, higher_better=False, partial=True)
    line('final validation loss', a.get('latest_val_loss'),
         b.get('latest_val_loss'), fmt=lambda v: f'{v:.4f}', digits=4,
         higher_better=False)

    live_note = ''
    if live:
        live_note = ('<div class="dwarn">'
                     + esc_html(' and '.join(r['name'] for r in live))
                     + (' is' if len(live) == 1 else ' are')
                     + ' still training. Every figure below is true as of '
                       'now, but the cost rows are a snapshot rather than a '
                       'total, so their change is left blank rather than '
                       'reported as a saving.</div>')

    metric_note = ''
    if not same_metric:
        metric_note = ('<div class="dwarn">These runs are scored on different '
                       'metrics &mdash; ' + esc_html(str(a.get("headline_key")))
                       + ' against ' + esc_html(str(b.get("headline_key")))
                       + '. The rows below still describe each run correctly, '
                         'but the difference between them is not a '
                         'comparison.</div>')

    chart = ''
    if a.get('curve') and b.get('curve') and same_metric:
        # Padded to a common length on purpose. _pts spreads a series across
        # the full width using ITS OWN length, which is right when every series
        # in a chart is the same run's, and wrong here: a 180-epoch run and a
        # 197-epoch one would both stretch edge to edge under an axis labelled
        # 1..197, drawing the shorter run's peak at an epoch it never reached.
        # Padding with None keeps one x scale and leaves the shorter run's
        # tail blank, which is what actually happened.
        span = max(len(a['curve']), len(b['curve']))

        def pad(v):
            return list(v) + [None] * (span - len(v))

        chart = _chart('trk-diff', f'{a_key} vs {b_key}', mlabel,
                       [{'name': a['name'], 'values': pad(a['curve']),
                         'color': TRK_A},
                        {'name': b['name'], 'values': pad(b['curve']),
                         'color': TRK_B}], fmt='.3f')

    diffs = _diff_args(a, b)
    if diffs:
        drows = ''.join(
            f'<tr><th>{esc_html(k)}</th>'
            f'<td class="dnum">{esc_html(str(va)) if va is not None else "&mdash;"}</td>'
            f'<td class="dnum">{esc_html(str(vb)) if vb is not None else "&mdash;"}</td>'
            f'<td></td></tr>' for k, va, vb in diffs)
        settings = (f'<div class="dsub">settings that differ '
                    f'({len(diffs)})</div>'
                    f'<table class="dtab"><tbody>{drows}</tbody></table>')
    else:
        settings = ('<div class="dsub">settings that differ</div>'
                    '<div class="mnone">Identical settings &mdash; whatever '
                    'separates these two runs, it is not in args.yaml.</div>')

    return (f'<div class="tlive past dcmp">'
            f'<div class="tlhead"><span class="tst idle">vs</span>'
            f'<b>{esc_html(a_key)} &nbsp;vs&nbsp; {esc_html(b_key)}</b>'
            f'<button type="button" class="rbtn quiet tback" id="trkBack">'
            f'&larr; back to the live run</button></div>'
            f'{live_note}{metric_note}'
            f'<table class="dtab"><thead><tr><th></th>'
            f'<th class="dnum">{esc_html(a["name"])}</th>'
            f'<th class="dnum">{esc_html(b["name"])}</th>'
            f'<th class="dnum">change</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
            f'{chart}{settings}</div>')


# ── what a run got wrong ────────────────────────────────────────────────────
# Written by tools/detect/run_mistakes.py, which scores a run against its own
# val split. Read here, never computed here: a panel must not wait on
# inference, and the answer does not change once a run has finished.
MISTAKE_DIR = os.path.join(REPO, 'data', 'mistakes')
_MISS = {}
_FLAGS = {}


def flag_store():
    """The wrong-label store, or None if the tool is not in this checkout."""
    if _FLAGS.get('tried'):
        return _FLAGS.get('mod')
    _FLAGS['tried'] = True
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
        import label_flags
        _FLAGS['mod'] = label_flags
    except Exception:
        _FLAGS['mod'] = None
    return _FLAGS['mod']


def flagged_now():
    mod = flag_store()
    if not mod:
        return None
    try:
        return mod.flagged_files()
    except Exception:
        return None


# One scoring attempt per run per this many seconds. Scoring is a minute or
# two of CPU, and a run that fails to score should not be retried on every
# render of its own panel.
MISS_RETRY_S = 900
# key -> (when it was last tried, the process, or None if none ever started).
# The process is kept because "in flight" and "fell over" are different things
# to say, and holding only the timestamp made them the same one.
_MISS_TRIED = {}


def mistakes_python():
    return MISTAKES_PYTHON or SWEEP_PYTHON


_DET_VAL = {}
DET_VAL_TTL = 300


def _detect_val_ready(data, run_dir):
    """Does this detector's dataset yaml point at a val split with images?

    The same question run_mistakes.discover() asks, asked the same way, so the
    panel and the scorer cannot disagree about which runs are scorable.

    Cached: answering means parsing a yaml and listing a split of thousands of
    files, and the panel that asks re-renders on a timer.
    """
    ck = (data, run_dir)
    hit = _DET_VAL.get(ck)
    if hit and time.time() - hit[0] < DET_VAL_TTL:
        return hit[1]
    ok = False
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
        import run_mistakes
        ds = run_mistakes.resolve_data(data, run_dir, training_root())
        ok = bool(ds) and run_mistakes._val_images(ds) > 0
    except Exception:
        ok = False
    _DET_VAL[ck] = (time.time(), ok)
    return ok


def scorable(r):
    """Can this run be scored: finished, weights and val split still on disk.

    Detectors count. run_mistakes.py grew score_detect() when the detector
    panels turned up empty, but this gate still said classify-only, so no
    detector was ever scored on its own: the one that has a grid was scored by
    hand from a shell, and every other one showed 'only classification runs
    can be scored this way' -- a sentence about a limit that no longer exists.
    """
    if r.get('task') not in ('classify', 'detect'):
        return False
    if r.get('status') in ('running', 'never_started'):
        return False
    ds = r.get('data') or ''
    if not os.path.exists(os.path.join(r['dir'], 'weights', 'best.pt')):
        return False
    if r.get('task') == 'classify':
        return os.path.isdir(os.path.join(ds, 'val'))
    # A detector's `data` is a yaml, and a bare `dataset.yaml` resolves against
    # several bases -- one of which can turn up an unrelated file of that name.
    # run_mistakes.py checks the split has images before it will score, and
    # this has to agree with it or the panel promises a grid that never comes.
    return _detect_val_ready(ds, r['dir'])


def mistakes_topup(r):
    """Score this run in the background if it has never been scored.

    A finished run with weights and its split still on disk is a run whose
    mistakes are knowable, and the panel used to show nothing at all until
    somebody remembered the tool -- which is exactly what happened to the
    leash runs. Triggered by opening the run, so nothing is scored that nobody
    looked at, and detached, so the panel renders now and the grid appears on
    the next look.
    """
    key = run_key(r)
    if not scorable(r) or mistakes_for(key):
        return False
    py = mistakes_python()
    script = os.path.join(REPO, 'tools', 'detect', 'run_mistakes.py')
    if not py or not os.path.exists(script):
        return False
    now = time.time()
    if now - _MISS_TRIED.get(key, (0.0, None))[0] < MISS_RETRY_S:
        return False
    # stamped BEFORE the spawn, so a Popen that raises is still a try and the
    # panel does not fork one per render
    _MISS_TRIED[key] = (now, None)
    try:
        log = open(os.path.join(REPO, 'data', 'mistakes_run.log'), 'a')
        proc = subprocess.Popen(
            [py, script, '--run', key, '--root', training_root()],
            cwd=REPO, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            start_new_session=True)
        _SPAWNED.append(proc)
        _MISS_TRIED[key] = (now, proc)
        return True
    except Exception:
        return False


def mistakes_pending(key):
    """Is a scoring run for this key still in flight?

    A scorer that exited non-zero -- an interpreter without ultralytics dies on
    the import in under a second -- is not in flight, it is finished and it
    failed. Answering yes for the whole retry window made those two states the
    same sentence, so the panel said 'scoring it now, reload in a minute'
    forever and the line naming the command to run by hand was unreachable.
    """
    at, proc = _MISS_TRIED.get(key, (0.0, None))
    if not at or proc is None:
        return False
    return proc.poll() in (None, 0) and (time.time() - at) < MISS_RETRY_S


def _mistake_path(key):
    """The cache file for a run key, tolerating how the project is spelled.

    The same project reaches this file two ways -- args.yaml says
    `dogdetection` where the directory is `DogDetection` -- and an exact match
    silently found nothing for the one detector on this box that had been
    scored.
    """
    want = str(key).replace('/', '__') + '.json'
    exact = os.path.join(MISTAKE_DIR, want)
    if os.path.exists(exact):
        return exact
    try:
        for f in os.listdir(MISTAKE_DIR):
            if f.lower() == want.lower():
                return os.path.join(MISTAKE_DIR, f)
    except OSError:
        pass
    return exact


def mistakes_for(key):
    """The cached mistakes for a run key, or None if it has not been scored."""
    path = _mistake_path(key)
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        _MISS.pop(key, None)
        return None
    hit = _MISS.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get('items'), list):
        return None
    _MISS[key] = (stamp, doc)
    return doc


def mistake_item(key, i):
    """(absolute path, item) for one mistake, or (None, None).

    Resolved from the run's OWN cache by index, so nothing the client sends is
    ever joined onto a path. The realpath check is belt and braces for a
    dataset that contains a symlink pointing out of itself.
    """
    doc = mistakes_for(key)
    if not doc:
        return None, None
    try:
        i = int(i)
        if i < 0:
            # Python would wrap -1 onto the LAST item -- the one index shape
            # the page never asks for served a crop instead of a 404
            return None, None
        item = doc['items'][i]
    except (ValueError, TypeError, IndexError, KeyError):
        return None, None
    root = os.path.realpath(doc.get('dataset') or '')
    full = os.path.realpath(os.path.join(root, item.get('file') or ''))
    if not root or not full.startswith(root + os.sep):
        return None, None
    return (full, item) if os.path.isfile(full) else (None, None)


def mistake_bytes(key, i):
    """JPEG bytes for one mistake: the crop itself, or the box within a frame.

    A classifier is wrong about a whole picture and the file IS the evidence.
    A detector is wrong about a REGION of a much larger street photo, so
    serving that photo would show a street and leave the reader hunting for
    what the argument is about. The box is cut out, with a margin, because a
    box with no surroundings cannot be judged either -- whether a thing is a
    dog is partly a question about what is next to it.
    """
    path, item = mistake_item(key, i)
    if not path:
        return None
    box = item.get('box')
    if not box or len(box) != 4:
        try:
            with open(path, 'rb') as fh:
                return fh.read()
        except OSError:
            return None
    try:
        from PIL import Image
        import io
        with Image.open(path) as im:
            W, H = im.size
            x1, y1, x2, y2 = (float(v) for v in box)
            pad = 0.35 * max(x2 - x1, y2 - y1) + 8
            crop = im.convert('RGB').crop((
                max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                min(W, int(x2 + pad)), min(H, int(y2 + pad))))
            crop.thumbnail((512, 512))
            buf = io.BytesIO()
            crop.save(buf, 'JPEG', quality=86)
            return buf.getvalue()
    except Exception:
        return None


def render_mistakes(r):
    """The crops a run got wrong, grouped by which way it went wrong."""
    key = run_key(r)
    doc = mistakes_for(key)
    if not doc:
        # The section is part of what a finished run looks like, so it appears
        # and says where it is up to. Returning nothing made a run that had
        # simply never been scored indistinguishable from one that had nothing
        # to show.
        if not scorable(r):
            why = ('still training &mdash; a run is scored once it stops'
                   if r.get('status') == 'running' else
                   'only classify and detect runs are scored this way'
                   if r.get('task') not in ('classify', 'detect') else
                   'its weights or its validation split are no longer on disk')
            return (f'<div class="wrwrap"><div class="wrhead">'
                    f'<b>What it got wrong</b>'
                    f'<span class="wrsub">{why}</span></div></div>')
        started = mistakes_topup(r)
        note = ('scoring it against its validation split now &mdash; reload in '
                'a minute' if started or mistakes_pending(key) else
                'not scored yet. Run tools/detect/run_mistakes.py --run '
                + esc_html(key))
        return (f'<div class="wrwrap"><div class="wrhead">'
                f'<b>What it got wrong</b>'
                f'<span class="wrsub">{note}</span></div></div>')
    items = doc.get('items') or []
    if not items:
        return (f'<div class="wrwrap"><div class="wrhead"><b>Nothing wrong</b>'
                f'<span class="wrsub">all {doc.get("n", 0):,} validation crops '
                f'classified correctly</span></div></div>')
    detect = doc.get('task') == 'detect'
    flags = flagged_now()
    # A classifier's mistake is a direction between two classes, and there is
    # one group per off-diagonal cell of the confusion matrix so the two
    # readings line up. A detector's is a kind: a box it invented, or one it
    # never found. Different question, different grouping.
    groups = {}
    for i, it in enumerate(items):
        k = ((it.get('kind'), it.get('cls')) if detect
             else (it.get('true'), it.get('pred')))
        groups.setdefault(k, []).append((i, it))
    order = sorted(groups, key=lambda k: -len(groups[k]))

    def gname(k):
        if not detect:
            return f'{esc_html(k[0])} &rarr; {esc_html(k[1])}'
        return ('invented a ' if k[0] == 'invented' else 'missed a ') + \
            esc_html(str(k[1]))

    def ghint(k):
        if not detect:
            return f'crops that really were {k[0]} and the model called {k[1]}'
        return ('boxes it drew where the labels say there was nothing'
                if k[0] == 'invented' else
                'labelled boxes it never found')

    chips = ['<button type="button" class="wrchip on" data-g="">'
             f'all {len(items):,}</button>']
    for k in order:
        chips.append(
            f'<button type="button" class="wrchip" '
            f'data-g="{esc_html(str(k[0]))}|{esc_html(str(k[1]))}"'
            + _t(ghint(k))
            + f'>{gname(k)} <em>{len(groups[k]):,}</em></button>')

    tiles = []
    for k in order:
        for i, it in groups[k]:
            p = it.get('p')
            if detect:
                # invented is the model's doing, missed is the model's
                # omission -- the colour marks which is the model's claim
                lead = ('<span class="wrsaid">invented</span>'
                        if k[0] == 'invented' else
                        '<span class="wrwas">missed</span>')
                dir_html = (f'<span class="wrdir">{lead}'
                            f'<span class="wrarr">&middot;</span>'
                            f'<span class="wrwas">{esc_html(str(k[1]))}</span>'
                            f'</span>')
                sure = (f'{p * 100:.0f}% sure' if p is not None
                        else 'never fired')
            else:
                dir_html = (f'<span class="wrdir">'
                            f'<span class="wrwas">{esc_html(str(k[0]))}</span>'
                            f'<span class="wrarr">&rarr;</span>'
                            f'<span class="wrsaid">{esc_html(str(k[1]))}</span>'
                            f'</span>')
                sure = f'{(p if p is not None else 0) * 100:.0f}% sure'
            # A crop the reviewer can say is not the model's fault. Only
            # where the label is a thing that can BE wrong: a detector's miss
            # is about a box, not a class, and there is nothing to relabel.
            fl = ''
            if flags is not None and not detect:
                f = it.get('file') or ''
                on = ' on' if f in flags else ''
                fl = (f'<button type="button" class="wrflag{on}" '
                      f'data-f="{esc_html(f)}" data-was="{esc_html(str(k[0]))}"'
                      f' data-should="{esc_html(str(k[1]))}"'
                      + _t('the DATASET is wrong here, not the model \u2014 '
                           'flag it and the next build leaves it out')
                      + '>label is wrong</button>')
            tiles.append(
                f'<figure class="wrtile{" flagged" if fl and " on" in fl else ""}" '
                f'data-g="{esc_html(str(k[0]))}|{esc_html(str(k[1]))}">'
                f'<img loading="lazy" alt="what the model got wrong" '
                f'src="/api/training/wrong?key={quote(key)}&amp;i={i}">'
                f'<figcaption>{dir_html}'
                f'<span class="wrp">{sure}</span>{fl}'
                f'</figcaption></figure>')

    n, wrong = doc.get('n', 0), doc.get('wrong', len(items))
    more = ('' if not doc.get('truncated') else
            f' &middot; the {len(items)} it was surest about')
    if detect:
        hit = doc.get('hit', 0)
        sub = (f'{hit:,} of {n:,} labelled boxes found &middot; '
               f'{wrong:,} wrong{more}')
        keyline = (f'<span class="wrkeyi"><span class="wrsaid">invented</span>'
                   f'a box where the labels say there was nothing</span>'
                   f'<span class="wrkeyi"><span class="wrwas">missed</span>'
                   f'a labelled box it never found</span>')
    else:
        sub = (f'{wrong:,} of {n:,} validation crops &middot; surest '
               f'first{more}')
        keyline = (f'<span class="wrkeyi"><span class="wrwas">grey</span>'
                   f'what the crop really was</span>'
                   f'<span class="wrkeyi"><span class="wrarr">&rarr;</span>'
                   f'<span class="wrsaid">orange</span>what the model called '
                   f'it</span>')
    # A bounded panel with a pager. Every miss on one page ran the tiles down
    # the section with nothing holding them, and at the 240 this keeps that is
    # a page you scroll past rather than read.
    return (f'<div class="wrwrap" id="wrong" data-run="{esc_html(key)}" '
            f'data-dataset="{esc_html(str(doc.get("dataset") or ""))}">'
            f'<div class="wrhead"><b>What it got wrong</b>'
            f'<span class="wrsub">{sub}</span></div>'
            f'<div class="wrbox">'
            f'<div class="wrbar"><div class="wrchips">{"".join(chips)}</div>'
            f'<div class="wrpage"><button type="button" class="wrnav" '
            f'data-d="-1" aria-label="previous page">&lsaquo;</button>'
            f'<span class="wrat">&mdash;</span>'
            f'<button type="button" class="wrnav" data-d="1" '
            f'aria-label="next page">&rsaquo;</button></div></div>'
            f'<div class="wrgrid">{"".join(tiles)}</div>'
            # The key sits inside the panel, under the grid it explains, and
            # is drawn with the caption's own classes -- so it is a sample of
            # the thing rather than a description of it, and it cannot drift
            # out of step with what the tiles actually look like.
            f'<div class="wrkey">{keyline}'
            f'<span class="wrkeyi wrflagn" id="wrflagn" hidden></span></div>'
            f'</div>'
            f'<details class="wrfoot"><summary>why this order</summary>'
            f'<p>Sorted by how sure it was. A confident mistake is worth more '
            f'than a hesitant one &mdash; the model is not undecided about '
            f'these, and whatever it has learnt to see there it has learnt '
            f'firmly.</p></details></div>')


# ── confusion matrices ──────────────────────────────────────────────────────
# Cached from Comet by tools/detect/fetch_confusion.py. Read here rather than
# fetched: the dashboard must render without a Comet key and without a network
# round trip in the request path.
CONFUSION_FILE = os.path.join(REPO, 'data', 'confusion.json')
_CONF = {'at': None, 'runs': {}, 'fold': {}}


def confusion_index():
    """{run key: {labels, matrix, orientation, ...}} from the cache file."""
    try:
        st = os.stat(CONFUSION_FILE)
        stamp = (st.st_mtime, st.st_size)
    except OSError:
        # fold too: clearing only `runs` left confusion_for() answering from a
        # cache built out of a file that is no longer there
        _CONF.update(at=None, runs={}, fold={})
        return _CONF['runs']
    if _CONF['at'] == stamp:
        return _CONF['runs']
    runs = {}
    try:
        with open(CONFUSION_FILE) as fh:
            got = json.load(fh)
        if isinstance(got, dict) and isinstance(got.get('runs'), dict):
            runs = got['runs']
    except (OSError, ValueError):
        runs = {}
    # Folded once here, not per lookup. The same project reaches this file
    # spelled two ways -- args.yaml says `dogdetection` where Comet logged
    # `DogDetection` -- and an exact match silently found nothing for four of
    # the five detector runs that had a matrix waiting.
    _CONF.update(at=stamp, runs=runs,
                 fold={k.lower(): v for k, v in runs.items()})
    return runs


def confusion_for(key):
    """The cached matrix for a run key, tolerating how the project is spelled."""
    runs = confusion_index()
    got = runs.get(key)
    if got is None:
        got = (_CONF.get('fold') or {}).get(str(key).lower())
    return got


def _is_bg(label, diag):
    """Is this class ultralytics' structural "nothing was here" pseudo-class?

    The NAME alone is not enough. In a detection matrix `background` is a
    bookkeeping row whose diagonal is zero by construction -- there is no
    correctly detecting nothing -- and its rates are undefined. But a
    classifier can be trained with a real class called background, and that
    one has a diagonal like any other: judging by name alone reported 55
    correct predictions and their recall as em-dashes. The structural zero is
    the actual invariant, so test for it.
    """
    return str(label).strip().lower() == 'background' and not diag


def _conf_stats(labels, matrix):
    """Per-class precision and recall, given rows=predicted, cols=true.

    Which way round the matrix sits is the whole ballgame here: with rows as
    the prediction, a ROW sums to everything the model called that class (so
    precision) and a COLUMN sums to everything that truly was it (so recall).
    Transposing them silently swaps the two, and a model that misses half the
    dogs would read as one that over-calls them.
    """
    n = len(labels)
    rows = [sum(matrix[i]) for i in range(n)]
    cols = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    # 'background' is not a class the model can be right about. In a detection
    # matrix its row is the misses and its column the false alarms, and its
    # diagonal is zero BY CONSTRUCTION -- there is no such thing as correctly
    # detecting nothing. Computing rates off that zero printed "0.0% precision,
    # 0.0% recall" next to background, which reads as a catastrophic failure
    # rather than a cell that has no meaning.
    prec, rec = [], []
    for i in range(n):
        undef = _is_bg(labels[i], matrix[i][i])
        prec.append(None if undef or not rows[i] else matrix[i][i] / rows[i])
        rec.append(None if undef or not cols[i] else matrix[i][i] / cols[i])
    return rows, cols, prec, rec


def _pct(x):
    return '&mdash;' if x is None else f'{x * 100:.1f}%'


def render_confusion(r):
    """One run's confusion matrix, drawn in the dashboard's own palette.

    Ultralytics also writes a confusion_matrix.png, and it is deliberately not
    what is shown: it is a light-background raster of a fixed size, so it
    cannot be themed, hovered, or read for the derived rates that are the
    reason to look at a confusion matrix at all.
    """
    got = confusion_for(run_key(r))
    if not got:
        return ''
    labels = got.get('labels') or []
    matrix = got.get('matrix') or []
    n = len(labels)
    if not n or len(matrix) != n or any(len(row) != n for row in matrix):
        return ''
    rows, cols, prec, rec = _conf_stats(labels, matrix)
    # Counted down the columns of the REAL classes. Summing every cell counts
    # the background column too, and in a detection matrix that column is the
    # false positives the model invented -- not crops the split contains. It
    # made one split report three different sizes: birds-yolov9e16, -yolov8n
    # and -yolov8n2 all score against 1,121 instances and the header read
    # 1,235, 1,251 and 1,271, moving only with each model's false-alarm count.
    # A classify matrix has no background column, so nothing changes there.
    real = [j for j in range(n) if not _is_bg(labels[j], matrix[j][j])]
    total = sum(cols[j] for j in real)
    if not total:
        return ''
    correct = sum(matrix[j][j] for j in real)
    invented = sum(rows) - total

    head = [f'<th class="cx"></th><th class="cx cxax" colspan="{n}">'
            f'true class &rarr;</th><th class="cx"></th>']
    sub = ['<th class="cx cxax cxrow">predicted &darr;</th>']
    for j in range(n):
        sub.append(f'<th class="cxt" title="{esc_html(labels[j])}: '
                   f'{cols[j]} in the validation set">'
                   f'{esc_html(labels[j])}</th>')
    sub.append(
        '<th class="cxt cxr hcue"'
        + _t('Of everything the model CALLED this class, the share that really '
             'was it. Read along a row. The diagonal cell above is the other '
             'half of the story -- recall, the share of everything that really '
             'was this class that the model found. A model can reach high '
             'precision by only ever calling the easy cases.')
        + '>precision</th>')

    body = []
    for i in range(n):
        cells = [f'<th class="cxl" title="the model predicted '
                 f'{esc_html(labels[i])} {rows[i]} times">'
                 f'{esc_html(labels[i])}</th>']
        for j in range(n):
            v = matrix[i][j]
            # Normalised down each column, which is the standard form and the
            # one ultralytics' own confusion_matrix_normalized.png uses: a
            # column is one true class, so the cell reads "this share of the
            # real dogs was called X". Raw counts cannot be compared between
            # columns when the classes are different sizes, and ours are.
            share = (v / cols[j]) if cols[j] else None
            agree = (i == j) and not _is_bg(labels[i], matrix[i][i])
            hue = 'var(--green)' if agree else 'var(--red)'
            # tint follows the share, floored so a small but real mistake stays
            # visible rather than fading into the panel
            a = 0.0 if not v or share is None else max(0.13, share ** 0.6)
            what = ('correctly called' if i == j else 'wrongly called')
            # the background diagonal is the same undefined cell the rates
            # skip -- printing "0.0%" there says the model failed at something
            # it was never asked to do
            body_txt = ('&mdash;' if share is None or (i == j and not agree)
                        else f'{share * 100:.1f}%')
            cells.append(
                f'<td class="cxc{" dg" if agree else ""}{" z" if not v else ""}"'
                f' style="--w:{a:.3f};--h:{hue}"'
                f' title="{v:,} of the {cols[j]:,} true {esc_html(labels[j])} '
                f'were {what} {esc_html(labels[i])}">'
                f'{body_txt}</td>')
        if prec[i] is None:
            why = ('Not defined for background: its diagonal is zero by '
                   'construction, so there is nothing to take a share of.'
                   if _is_bg(labels[i], matrix[i][i]) else
                   f'The model never predicted {labels[i]}.')
            cells.append(f'<td class="cxn hcue"{_t(why)}>{_pct(prec[i])}</td>')
        else:
            cells.append(
                f'<td class="cxn hcue"'
                + _t(f'The model called {rows[i]:,} crops '
                     f'{labels[i]}; {matrix[i][i]:,} of them really were '
                     f'({prec[i] * 100:.1f}%). The other '
                     f'{rows[i] - matrix[i][i]:,} are the red cells across '
                     f'this row.')
                + f'>{_pct(prec[i])}</td>')
        body.append(f'<tr>{"".join(cells)}</tr>')
    # NOT recall: normalising down the column makes the diagonal cell recall
    # already, so a recall row printed the same two numbers a few pixels below
    # themselves. What normalising actually hides is how many crops each column
    # stands for -- 81.7% of a class means something different at 169 than at
    # 12 -- so the bottom row carries the support instead.
    foot = ['<th class="cxl cxr hcue"'
            + _t('How many crops in the validation set really were each class, '
                 'and for a background column how many false alarms the model '
                 'raised. The percentages above are shares of these, so a '
                 'column with few crops moves in big jumps.')
            + '>crops</th>']
    for j in range(n):
        bgcol = _is_bg(labels[j], matrix[j][j])
        foot.append(f'<td class="cxn hcue"'
                    + _t(f'{cols[j]:,} detections the model made where there '
                         f'was nothing to find -- false alarms, not crops the '
                         f'split contains.' if bgcol else
                         f'{cols[j]:,} crops in the validation set really were '
                         f'{labels[j]}.')
                    + f'>{cols[j]:,}</td>')
    foot.append('<td class="cxn"></td>')

    src = got.get('experiment')
    note = (f' &middot; from Comet run {esc_html(src)}' if src else '')
    bg_note = ('' if not any(_is_bg(labels[i], matrix[i][i])
                            for i in range(n)) else
               ' The background row counts what the detector missed and the '
               'background column what it invented; there is no correctly '
               'detecting nothing, so that class has no precision or recall.')
    return (f'<div class="cxwrap"><div class="cxhead">'
            f'<b>Confusion matrix</b>'
            f'<span class="cxsub">normalised by true class &middot; '
            f'{total:,} labelled instances &middot; '
            f'{correct / total * 100:.1f}% found'
            f'{f" &middot; {invented:,} false alarms" if invented else ""}'
            f'{note}</span></div>'
            f'<div class="cxscroll"><table class="cx">'
            f'<thead><tr>{"".join(head)}</tr><tr>{"".join(sub)}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody>'
            f'<tfoot><tr>{"".join(foot)}</tr></tfoot></table></div>'
            # Folded. It is a note you read once to trust the orientation,
            # not a paragraph the matrix has to be read past every time.
            f'<details class="cxfoot"><summary>how to read this</summary>'
            f'<p>Each column is one true class and sums to 100%, so a cell '
            f'reads &ldquo;this share of the real X was called Y&rdquo; '
            f'&mdash; which makes the diagonal each class&rsquo;s recall. '
            f'Hover a cell for the raw count.</p>'
            f'<p>Rows are what the model predicted, columns what the crop '
            f'actually was. Comet labels its rows &ldquo;Actual Category&rdquo;; '
            f'for an ultralytics matrix that is wrong, and believing it would '
            f'swap every miss with a false alarm.{bg_note}</p></details>'
            f'</div>')


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


# ── the dog-bin gate over the whole store ───────────────────────────────────
# tools/detect/gate_store.py judges every detection the sweep committed. It
# publishes no status of its own, and does not need to: it writes one parquet
# shard per 20,000 images, so what has actually been done is on disk. Reading
# that is ground truth, where a self-reported counter is a claim -- and it
# needs no change to a job already twelve hours into a run.
GATE_DIR = os.path.join(REPO, 'data', 'gate')
# Mirrors STAGES in tools/detect/gate_store.py, which is the authority. Only
# what the PANEL needs is here: where the shards are, what the positive class
# is called, and which stage has to finish before this one can start. The two
# lists are checked against each other by a guard rather than kept in step by
# hand -- they are read by different interpreters and cannot import each other.
# `card` and `hint` were here too, and nothing read them: the page is static
# HTML, so its labels live in the markup, and a second copy on the server was
# a source of truth that no code consulted and no guard could keep honest.
GATE_STAGES = {
    'gate': {'dir': 'gate', 'title': 'dog-bin gate', 'positive': 'dog',
             'project': 'dog-bin', 'feeds_on': None},
    'leash': {'dir': 'leash', 'title': 'leash model', 'positive': 'leashed',
              'project': 'leash-models', 'feeds_on': 'gate'},
}
_GATE = {}                     # per stage: {'at': .., 'doc': ..}
# A shard is immutable once written -- the runner builds it under a dot-name
# and os.replace()s it into place, and a resumed run skips the indices it
# already has. So a file only ever needs reading ONCE, and the scan costs one
# glob plus a footer per file that is new since the last call. That is what
# makes a 2-second window affordable: with a 20-second cache a shard landing
# could sit invisible for twenty seconds after the fact, which on a panel that
# only moves when a shard lands is most of the time between moves.
_GATE_FILES = {}
GATE_TTL_S = 2


def gate_dir(stage='gate'):
    return os.path.join(REPO, 'data', GATE_STAGES[stage]['dir'])


def _gate_shards(stage='gate'):
    """The written record, read incrementally."""
    now = time.time()
    memo = _GATE.setdefault(stage, {'at': 0.0, 'doc': None})
    files = _GATE_FILES.setdefault(stage, {})
    if memo['doc'] is not None and now - memo['at'] < GATE_TTL_S:
        return memo['doc']
    d, pre = gate_dir(stage), GATE_STAGES[stage]['dir']
    doc = None
    try:
        with open(os.path.join(d, 'plan.json')) as fh:
            plan = json.load(fh)
    except (OSError, ValueError):
        memo.update(at=now, doc=doc)
        return doc
    try:
        import glob
        import pyarrow.parquet as pq
    except Exception:
        memo.update(at=now, doc=doc)
        return doc
    fs = sorted(glob.glob(os.path.join(d, f'{pre}-*.parquet')))
    for f in fs:
        try:
            mt = os.path.getmtime(f)
        except OSError:
            continue
        if files.get(f, (None,))[0] == mt:
            continue
        try:
            # Both numbers in one pass, once per shard for the life of the
            # process. Counting dogs used to mean re-reading a label column
            # every 20 seconds, which is why it was limited to the newest
            # three shards and why the share was a sample rather than the
            # whole record -- read once, it can be all of it.
            t = pq.read_table(f, columns=['label'])
            col = t.column('label').to_pylist()
            pos = GATE_STAGES[stage]['positive']
            files[f] = (mt, len(col), sum(1 for v in col if v == pos))
        except Exception:
            continue
    for f in list(files):                # a shard that went away is not a fact
        if f not in fs:
            files.pop(f, None)
    rows = sum(v[1] for v in files.values())
    dogs = sum(v[2] for v in files.values())
    seen = rows
    times = sorted(v[0] for v in files.values())
    total = int(plan.get('rows') or 0)
    images = int(plan.get('images') or 0)
    # The runner shards by IMAGE and skips any shard index whose file is
    # already there, so "every planned shard exists" is the same test it uses
    # to decide there is nothing left to do. Rows are not that test: a frame it
    # could not decode, or a drive that dropped mid-run, leaves a shard a few
    # rows short that no rerun can ever top up -- and two missing rows out of
    # 4.7M then blocked the leash stage forever while the panel read 100.0%.
    # The size is whatever the job pinned on its first run; the plan's value is
    # the fallback for a job written before that pin existed.
    try:
        with open(os.path.join(d, 'shard_size')) as fh:
            per_shard = int(fh.read().strip())
    except (OSError, ValueError):
        per_shard = int(plan.get('shard_rows') or 0)
    shards_total = -(-images // per_shard) if images and per_shard else 0
    # rate from the shard timestamps: the run writes one every few minutes,
    # so the last handful is a real recent throughput and not a lifetime mean
    rate = sus = 0.0
    per = (rows / len(files)) if files else 0
    if len(times) >= 2:
        # Measured over the gaps BETWEEN shards, not first-to-last mtime.
        # Stop/Resume is an ordinary workflow here -- the gate shares the GPU
        # with training, so an overnight stop is the usual reason to press it
        # -- and spanning the whole run counted those hours as working time:
        # after a 12 h pause the ETA card read 22 h for work with 10 h left.
        # A gap far longer than the usual one is a pause, so it and the shard
        # that closed it are dropped rather than averaged in.
        gaps = sorted(b - a for a, b in zip(times, times[1:]))
        typical = gaps[len(gaps) // 2]
        kept = [g for g in gaps if 0 < g <= 5 * typical]
        if kept:
            sus = per * len(kept) / sum(kept)
        k = min(len(times) - 1, 5)
        span = times[-1] - times[-1 - k]
        if span > 0:
            rate = per * k / span
    doc = {'rows': rows, 'total': total, 'shards': len(files),
           'shards_total': shards_total,
           # a plan too old to say how many shards it wants falls back to the
           # row test, which is what this was before
           'done': (len(files) >= shards_total if shards_total
                    else bool(total) and rows >= total),
           'dogs': dogs, 'seen': seen, 'last': times[-1] if times else 0.0,
           'rate': rate, 'sustained': sus,
           'model': plan.get('model') or 'dog-bin gate',
           'images': images,
           'created': plan.get('created') or ''}
    memo.update(at=now, doc=doc)
    return doc


def _gate_beat(stage='gate'):
    """What the run is doing BETWEEN shards.

    A shard is 20,000 frames, so on a cold start the footers say nothing for
    several minutes: the panel read 0%, 0 judged, no rate and no share while
    sixteen decoders were flat out, and pressing Run looked like it had done
    nothing. The shards stay the record; this fills the gap between them, and
    only while it is fresh enough to be about a process that still exists (a
    killed run cannot clean up after itself). Read every call, never cached --
    it is the one part of this panel that changes by the second.
    """
    try:
        with open(os.path.join(gate_dir(stage), 'progress.json')) as fh:
            b = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(b, dict):
        return None
    try:
        fresh = time.time() - float(b.get('updated') or 0) <= 30.0
    except (TypeError, ValueError):
        return None
    return b if fresh else None


def gate_upstream(stage):
    """How far the stage this one FEEDS ON has got, or None.

    The leash model judges the gate's dogs, so it cannot be planned until the
    gate is done: shard numbering is fixed at plan time, and a plan built
    early would permanently omit every dog the gate had not reached. The panel
    has to be able to say that rather than offer a button that fails.
    """
    up = GATE_STAGES[stage]['feeds_on']
    if not up:
        return None
    s = _gate_shards(up)
    if s is None:
        return {'stage': up, 'title': GATE_STAGES[up]['title'],
                'rows': 0, 'total': 0, 'shards': 0, 'shards_total': 0,
                'ready': False}
    return {'stage': up, 'title': GATE_STAGES[up]['title'],
            'rows': s['rows'], 'total': s['total'],
            'shards': s['shards'], 'shards_total': s['shards_total'],
            'ready': s['done']}


def gate_progress(stage='gate'):
    """How far a stage has got: the shards it has written, plus the shard it
    is in the middle of."""
    if stage not in GATE_STAGES:
        return {'ever': False, 'error': f'unknown stage {stage}'}
    shards = _gate_shards(stage)
    up = gate_upstream(stage)
    if shards is None:
        return {'ever': False, 'stage': stage, 'upstream': up,
                'planned': False}
    beat = _gate_beat(stage) or {}
    rows = shards['rows'] + int(_num_or(beat.get('rows_flight'), 0))
    total, dogs, seen = shards['total'], shards['dogs'], shards['seen']
    rate, sus = shards['rate'], shards['sustained']
    if beat:
        # measured by the run itself, not inferred from file timestamps
        rate = _num_or(beat.get('box_s'), 0.0)
        if not sus:
            sus = rate
        # Real counts, not a share and a 1.0 -- the panel shows both the
        # percentage and the two numbers under it, and a synthesised
        # denominator would put "0.213 of 1 boxes" in the tooltip.
        boxes = _num_or(beat.get('boxes'), 0)
        if seen <= 0 and boxes > 0:
            # `dogs` is newer than `dog_share`. A run started before it
            # existed keeps publishing the share, and reading the absent
            # count as zero would report a gate that calls nothing a dog --
            # so the count is derived from the share when it has to be.
            n = beat.get('dogs')
            dogs = (_num_or(n, 0) if n is not None
                    else round(_num_or(beat.get('dog_share'), 0) * boxes))
            seen = boxes
    left = max(0, total - rows)
    warm = bool(shards['last']) and (time.time() - shards['last']) < 900
    return {'ever': bool(shards['shards']) or total > 0,
            'stage': stage, 'planned': True, 'upstream': up,
            'running': bool(beat) or warm,
            # every planned shard is on disk: the stage is finished, which is
            # not the same as one stopped at shard 82 even though both have
            # no process behind them
            'done': shards['done'],
            'rows': rows, 'total': total, 'shards': shards['shards'],
            'shards_total': shards['shards_total'],
            'pct': (rows / total) if total else 0,
            'dog_share': (dogs / seen) if seen else None,
            # the two numbers the share is made of: a percentage alone cannot
            # say whether it is drawn from a thousand boxes or four million
            'dogs': int(dogs) if seen else None,
            'dogs_of': int(seen) if seen else None,
            'rate': round(rate, 1), 'sustained': round(sus, 1),
            'eta_s': (left / sus) if sus > 0 else None,
            'model': shards['model'], 'images': shards['images'],
            # frames opened, which moves every second -- the judged count only
            # moves when a shard lands, and a run whose first shard is still
            # minutes away needs something true to show
            'images_done': int(_num_or(beat.get('images'), 0)) if beat else None,
            'img_s': _num_or(beat.get('img_s'), 0.0) if beat else None,
            'created': shards['created']}


# ── drive health ────────────────────────────────────────────────────────────
# The harvest is spread across six roots on separate disks, and the two ways
# that goes wrong are silent. A drive that is not mounted does not error: the
# catalog simply finds no cells under it and every region it held reads as
# zero. A drive that is full does not error either until a write fails, by
# which point a sweep has been dropping frames for hours.
#
# Both are visible in one number each -- is it there, and how much room is
# left -- so this section is one row per drive and nothing else. Where each
# region's data lives is a different question, and the section above answers
# it.
DRIVE_TIGHT_PCT = 0.90        # used share at which a disk needs attention
DRIVE_TIGHT_GB = 50.0         # ...or an absolute floor, whichever bites first
# SMART is read, never started. `smartctl -H` asks the drive for the verdict it
# already holds; it does not unmount, does not interrupt I/O and does not touch
# the filesystem. `smartctl -t`, which would START a self-test, is deliberately
# not used anywhere here.
SMART_CACHE = os.path.join(REPO, 'data', 'dashboard', 'drive_smart.json')
SMART_TTL_S = 900
SMART_CACHE_V = 2       # bumped when the cached shape changes
SMART_TIMEOUT_S = 12


def _mount_devices():
    """{mountpoint: (device, bus)} from lsblk, or {} if it is not there."""
    out = {}
    try:
        got = subprocess.run(['lsblk', '-J', '-o', 'NAME,PATH,TRAN,MOUNTPOINT'],
                             capture_output=True, text=True, timeout=15)
        tree = json.loads(got.stdout or '{}')
    except (OSError, ValueError, subprocess.SubprocessError):
        return out

    def walk(nodes, bus=None):
        for n in nodes:
            b = n.get('tran') or bus
            if n.get('mountpoint'):
                out[n['mountpoint']] = (n.get('path'), b)
            walk(n.get('children') or [], b)
    walk(tree.get('blockdevices') or [])
    return out


def _device_for(path, mounts):
    """(device, bus) holding `path`, by longest mountpoint match.

    realpath FIRST. One of these roots is a symlink onto another disk, and
    matching the literal path walked up to '/' instead -- reporting the root
    filesystem's device, and its SMART verdict, for an 18 TB drive.
    """
    real = os.path.realpath(path)
    best = ''
    for mp in mounts:
        if (real == mp or real.startswith(mp.rstrip(os.sep) + os.sep)) \
                and len(mp) > len(best):
            best = mp
    return mounts.get(best, (None, None))


ATA_WANT = {5: ('reallocated', 'sectors the drive had to move'),
            197: ('pending', 'sectors it cannot read and has not moved yet'),
            198: ('uncorrectable', 'sectors it gave up on'),
            199: ('CRC errors', 'garbled transfers — usually the cable or the '
                                'USB bridge, not the disk')}


def _num(x):
    try:
        return int(str(x).split()[0].split('h')[0])
    except (TypeError, ValueError, IndexError):
        return None


def _smart_facts(doc):
    """[(label, value, level)] — what the drive says about its own wear.

    The same four questions whichever kind of disk answered: how hot, how long
    it has been running, how worn, and how much damage it has already found.
    NVMe and ATA report those under completely different names, so they are
    normalised here rather than in the markup.
    """
    out = []
    t = ((doc.get('temperature') or {}).get('current'))
    # 0 is what a bridge returns when it did not really answer, not a drive
    # sitting at freezing; showing it as a reading would be inventing one
    if t:
        out.append((f'{t} \u00b0C', 'temperature',
                    'bad' if t >= 65 else 'warn' if t >= 55 else 'ok'))
    hrs = ((doc.get('power_on_time') or {}).get('hours'))
    if hrs:
        yrs = hrs / 8760.0
        out.append((f'{yrs:.1f} yr powered' if yrs >= 1
                    else f'{hrs:,} h powered', 'how long it has been running',
                    'ok'))
    nv = doc.get('nvme_smart_health_information_log') or {}
    if nv:
        w = nv.get('percentage_used')
        if w is not None:
            out.append((f'{w}% worn', 'share of its rated write life used',
                        'bad' if w >= 90 else 'warn' if w >= 80 else 'ok'))
        sp = nv.get('available_spare')
        if sp is not None and sp < 100:
            out.append((f'{sp}% spare', 'spare blocks left',
                        'bad' if sp < 20 else 'warn'))
        me = nv.get('media_errors')
        if me is not None:
            out.append((f'{me:,} media error' + ('' if me == 1 else 's'),
                        'unrecoverable data errors',
                        'bad' if me else 'ok'))
        us = nv.get('unsafe_shutdowns')
        if us:
            out.append((f'{us:,} unsafe shutdowns',
                        'lost power with writes in flight', 'ok'))
        return out
    for a in ((doc.get('ata_smart_attributes') or {}).get('table') or []):
        want = ATA_WANT.get(a.get('id'))
        if not want:
            continue
        v = _num((a.get('raw') or {}).get('string'))
        if v is None:
            continue
        label, why = want
        # a zero here is the good news and worth stating; a non-zero one is
        # the whole reason to read SMART at all
        lvl = 'ok' if v == 0 else ('warn' if a['id'] == 199 else 'bad')
        out.append((f'{v:,} {label}', why, lvl))
    return out


def _smart_read(dev, bus):
    """('passed'|'failing'|'unreadable', detail) for one device.

    Read-only: -H returns the drive's own stored self-assessment. USB bridges
    often need to be told how to talk to the disk behind them, so one retry
    with -d sat before giving up -- that is still a read.
    """
    sm = shutil.which('smartctl')
    if not dev or not sm:
        return 'unreadable', 'smartctl is not installed', []
    # Unprivileged first, and only then through sudo -n -- which fails
    # immediately when no rule grants it rather than waiting on a password
    # prompt no page build could answer. Without this second form the sudoers
    # rule the section asks for would change nothing: the rule permits
    # `sudo smartctl`, and nothing here was running sudo.
    base = ['-H', '-j', '-A']
    # `-d sat` for every device, not just the ones lsblk calls USB. Auto
    # detection answers for three of the six roots with a verdict and an empty
    # attribute table -- including one on a plain SATA port -- and an empty
    # table reads as a healthy drive with nothing to report, which is the one
    # thing SMART must never be mistaken for.
    tries = [[sm] + base + [dev], [sm] + base + ['-d', 'sat', dev]]
    tries += [['sudo', '-n'] + t for t in list(tries)]
    last, best = '', None
    for argv in tries:
        try:
            got = subprocess.run(argv, capture_output=True, text=True,
                                 timeout=SMART_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return 'unreadable', 'smartctl could not be run', []
        try:
            doc = json.loads(got.stdout or '{}')
        except ValueError:
            doc = {}
        st = (doc.get('smart_status') or {})
        if 'passed' in st:
            got = ('passed' if st['passed'] else 'failing',
                   'the drive reports itself FAILING — replace it'
                   if not st['passed'] else '', _smart_facts(doc))
            # a verdict with no attributes behind it is worth keeping only
            # until something fuller answers
            if got[2]:
                return got
            best = best or got
            continue
        msgs = ' '.join(m.get('string', '')
                        for m in (doc.get('smartctl') or {}).get('messages')
                        or [])
        last = msgs or (got.stderr or '').strip() or 'no verdict returned'
        if 'permission' in last.lower() and argv[0] == 'sudo':
            # sudo refused too: the rule is absent, so stop asking
            return best or ('unreadable', 'needs root', [])
        if 'sudo:' in last.lower() or 'a password is required' in last.lower():
            return best or ('unreadable', 'needs root', [])
    return best or ('unreadable', last[:120], [])


def drive_smart(force=False):
    """{device: {state, detail}}, cached — one probe per drive per 15 minutes.

    Cached because the page rebuilds far more often than a disk's verdict
    changes, and six subprocess round trips inside a page build is a cost paid
    for nothing.
    """
    now = time.time()
    if not force:
        try:
            with open(SMART_CACHE) as fh:
                doc = json.load(fh)
            # versioned: a cache written before the attributes were collected
            # would keep every card factless until it aged out on its own
            if (doc.get('v') == SMART_CACHE_V
                    and now - float(doc.get('at') or 0) < SMART_TTL_S):
                return doc.get('by_device') or {}
        except (OSError, ValueError, TypeError):
            pass
    mounts = _mount_devices()
    by_dev = {}
    for _label, root in sorted(_grid_roots().items()):
        dev, bus = _device_for(root, mounts)
        if not dev or dev in by_dev:
            continue
        state, detail, facts = _smart_read(dev, bus)
        by_dev[dev] = {'state': state, 'detail': detail, 'bus': bus,
                       'facts': facts}
    try:
        os.makedirs(os.path.dirname(SMART_CACHE), exist_ok=True)
        tmp = SMART_CACHE + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump({'v': SMART_CACHE_V, 'at': now,
                       'by_device': by_dev}, fh)
        os.replace(tmp, SMART_CACHE)
    except OSError:
        pass
    return by_dev


def _drive_free(path):
    """(total, used, free) bytes for the filesystem holding `path`, or None."""
    try:
        u = shutil.disk_usage(path)
        return u.total, u.used, u.free
    except OSError:
        return None


# ── the machine itself ──────────────────────────────────────────────────────
# Every long job on this box is one of three things: waiting on a disk,
# waiting on the CPU, or waiting on the GPU. Which one decides whether a knob
# is worth turning, and the answer is not guessable -- the gate runs at 75
# boxes/s with the GPU at 0%, because decoding an 8000x4000 panorama is 98% of
# the work and the card is idle waiting for pixels. So the four figures here
# are chosen to answer that question rather than to fill a row of gauges.
_CPU = {'t': 0.0, 'idle': 0.0, 'total': 0.0, 'pct': None}
# The card is sampled CONTINUOUSLY, not once per request, and this is the
# whole reason the panel is worth having. Measured on this box while the gate
# ran: 44 readings over 22 s came back 0, 0, 0, ... 63, 0, 0, 29 -- forty
# zeroes and four bursts. The work is real (mean 4.2%) but it arrives in
# fractions of a second between long decode stalls, so a reading taken at the
# moment a browser happens to ask lands on a zero nine times in ten. The card
# read "0%" forever and looked broken. A window, not a glance.
_GPU = {'proc': None, 'samples': None, 'meta': {}, 'retry_at': 0.0,
        'lock': threading.Lock()}
GPU_WINDOW = 30          # readings kept; nvidia-smi -l 1 gives one a second
GPU_RETRY_S = 60.0       # a box with no card must not respawn nvidia-smi hourly


def _gpu_reader(proc):
    """Drain one nvidia-smi -l 1 into the ring. Runs until the pipe closes."""
    try:
        for line in proc.stdout:
            f = [x.strip() for x in line.split(',')]
            if len(f) < 7:
                continue
            _GPU['meta'] = {'name': f[0],
                            'mem_used': _num_or(f[2], None),
                            'mem_total': _num_or(f[3], None),
                            'temp': _num_or(f[4], None),
                            'power': _num_or(f[5], None),
                            'power_max': _num_or(f[6], None)}
            u = _num_or(f[1], None)
            if u is not None and _GPU['samples'] is not None:
                # stamped, because "the last thirty readings" is only the last
                # half minute while they keep arriving -- an nvidia-smi that
                # wedges with its pipe open leaves `proc` set, so nothing here
                # respawns and nothing else could age the window out
                _GPU['samples'].append((time.time(), u))
    except Exception:
        pass
    finally:
        with _GPU['lock']:
            if _GPU['proc'] is proc:
                _GPU['proc'] = None
                # the window goes with the process it was read from. Left
                # behind, a dead reader's last frame was served as the live
                # card for the whole back-off -- 90% util and "gpu bound" in
                # the header from an nvidia-smi that had already exited
                _GPU['samples'] = None
                _GPU['meta'] = {}
                _GPU['retry_at'] = time.time() + GPU_RETRY_S


def _gpu_start():
    """One long-lived nvidia-smi for the life of the server. Forking one per
    request cost 30 ms of a 2 s poll and still only bought a single glance."""
    with _GPU['lock']:
        if _GPU['proc'] is not None or time.time() < _GPU['retry_at']:
            return
        try:
            proc = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,'
                 'memory.total,temperature.gpu,power.draw,power.limit',
                 '--format=csv,noheader,nounits', '-l', '1'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1)
        except (OSError, ValueError):
            _GPU['retry_at'] = time.time() + GPU_RETRY_S
            return
        _GPU['proc'] = proc
        _GPU['samples'] = collections.deque(maxlen=GPU_WINDOW)
        threading.Thread(target=_gpu_reader, args=(proc,), daemon=True).start()


def _cpu_pct():
    """Busy share since the last SAMPLE. /proc/stat is a set of counters since
    boot, so a single read says what the machine has averaged over days --
    only the delta between two reads is about now.

    The window is held at a second minimum whatever the request rate. The
    delta is consumed by whoever reads it, so two open tabs polling every two
    seconds would have measured alternating one-second windows, and ten of
    them windows short enough to be mostly scheduling noise. The sampling
    cadence has to be the server's, not the audience's.
    """
    # On time, not on "have we got a number yet": gating on the latter let the
    # call right after the baseline through, and it measured a window of
    # microseconds -- one scheduler tick either way reads as 0% or 100%.
    if _CPU['t'] and time.time() - _CPU['t'] < 1.0:
        return _CPU['pct']
    try:
        with open('/proc/stat') as fh:
            parts = fh.readline().split()
    except OSError:
        return None
    if len(parts) < 5 or parts[0] != 'cpu':
        return None
    try:
        v = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    total = sum(v)
    idle = v[3] + (v[4] if len(v) > 4 else 0.0)     # idle + iowait
    first = _CPU['total'] <= 0
    dt, di = total - _CPU['total'], idle - _CPU['idle']
    _CPU.update(t=time.time(), idle=idle, total=total)
    # The first call has nothing to subtract from, and the delta against zero
    # is the counter itself -- the machine's average since BOOT, which on a
    # box that has been up for days is a small number that looks like an
    # answer. One tick of "unknown" beats a plausible wrong number.
    if first or dt <= 0:
        return None if first else _CPU['pct']
    _CPU['pct'] = max(0.0, min(100.0, 100.0 * (1.0 - di / dt)))
    return _CPU['pct']


def _meminfo():
    """Bytes: (total, available, swap_total, swap_free). MemAvailable, not
    MemFree -- the page cache is free memory that happens to be useful, and
    counting it as used reports 60 GB in use on an idle box."""
    want = {'MemTotal': 0, 'MemAvailable': 0, 'SwapTotal': 0, 'SwapFree': 0}
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                k, _, rest = line.partition(':')
                if k in want:
                    want[k] = float(rest.split()[0]) * 1024.0
    except (OSError, IndexError, ValueError):
        return None
    return want


def _pressure(kind):
    """PSI: the share of the last 10 s in which EVERY task was stalled on this
    resource. `full` is the honest one -- `some` counts a single blocked
    thread on a box with fifteen others working."""
    try:
        with open(f'/proc/pressure/{kind}') as fh:
            for line in fh:
                if line.startswith('full'):
                    for tok in line.split():
                        if tok.startswith('avg10='):
                            return float(tok[6:])
    except (OSError, ValueError):
        pass
    return None


def _gpu():
    """The card over the last half minute. None when there is no card or no
    driver -- a machine without one is not a broken dashboard."""
    _gpu_start()
    now = time.time()
    # readings older than the window are not "the last half minute", they are
    # the last half minute the reader managed before it stopped answering
    s = [u for t, u in list(_GPU['samples'] or ()) if now - t <= GPU_WINDOW]
    if not _GPU['meta'] or not s:
        return None
    doc = dict(_GPU['meta'])
    # The MEAN is the headline: it is what share of the window the card had
    # work, which is the question. The peak goes beside it, because a mean of
    # 4% built from bursts of 63% is a different machine from one sitting at a
    # flat 4%, and only the two together say which this is.
    doc['util'] = sum(s) / len(s)
    doc['util_peak'] = max(s)
    doc['window'] = len(s)
    return doc


def sys_stats():
    """CPU, memory, GPU and what the machine is stalled on. Never raises: a
    missing card or an older kernel means one figure is unknown, not that the
    panel is."""
    m = _meminfo() or {}
    tot, avail = m.get('MemTotal') or 0, m.get('MemAvailable') or 0
    sw_t, sw_f = m.get('SwapTotal') or 0, m.get('SwapFree') or 0
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = None
    return {'cpu': _cpu_pct(), 'cores': os.cpu_count() or 0, 'load': load1,
            'mem_used': tot - avail, 'mem_total': tot,
            'swap_used': sw_t - sw_f, 'swap_total': sw_t,
            'io_stall': _pressure('io'), 'cpu_stall': _pressure('cpu'),
            'gpu': _gpu(), 'ts': time.time()}


def drive_health():
    """One record per configured root: is it there, is it sound, is there room.

    Never raises. This runs inside the page build, and a drive that has gone
    away is the case it exists to report -- failing the whole dashboard
    because one root is unplugged would hide the answer behind the symptom.
    """
    mounts = _mount_devices()
    smart = drive_smart()
    out = []
    for label, root in sorted(_grid_roots().items()):
        dev, bus = _device_for(root, mounts)
        sm = smart.get(dev) or {}
        rec = {'label': label, 'root': root, 'mounted': os.path.isdir(root),
               'device': dev, 'bus': bus,
               'smart': sm.get('state') or 'unreadable',
               'smart_detail': sm.get('detail') or '',
               'smart_facts': sm.get('facts') or [],
               'total': None, 'used': None, 'free': None}
        if rec['mounted']:
            got = _drive_free(root)
            if got:
                rec['total'], rec['used'], rec['free'] = got
        out.append(rec)
    return out


def _gb(n):
    return n / (1024.0 ** 3)


def _drive_tight(r):
    if not r.get('total'):
        return False
    return (r['used'] / r['total'] >= DRIVE_TIGHT_PCT
            or _gb(r['free']) < DRIVE_TIGHT_GB)


def _drive_verdict(r):
    """(rank, word, class) -- the one thing to know about this drive."""
    if not r['mounted']:
        return 0, 'not mounted', 'bad'
    if r['smart'] == 'failing':
        return 0, 'SMART failing', 'bad'
    if _drive_tight(r):
        return 1, 'nearly full', 'warn'
    return 2, 'healthy', 'ok'


def render_drives():
    """One card per root, worst first. A verdict, then the numbers behind it."""
    rows = drive_health()
    if not rows:
        return ('<div class="mnone">no data/catalog_dirs.txt &mdash; nothing '
                'to check</div>')
    rows.sort(key=lambda r: (_drive_verdict(r)[0], r['label']))

    cards = []
    for r in rows:
        _, word, kind = _drive_verdict(r)
        pct = (r['used'] / r['total']) if r.get('total') else None
        meter = ''
        if pct is not None:
            meter = ('<i class="dhmeter"><i style="width:'
                     + f'{min(100.0, pct * 100):.1f}' + '%"></i></i>')
        # the meter already carries the proportion; a percentage beside it is
        # the third telling of one number
        room = ((f'<b>{_gb(r["free"]):,.0f} GB</b><span> free of '
                 f'{_gb(r["total"]):,.0f} GB</span>') if pct is not None
                else ('<span>capacity unreadable while unmounted</span>'
                      if not r['mounted']
                      else '<span>capacity unreadable</span>'))
        # Only when SMART has something to say. Six cards each reading "SMART
        # passed" is six repetitions of the normal case; that it was read at
        # all is one fact about the whole section, and it is stated once.
        smrow = ''
        if r['smart'] != 'passed':
            sm = ('SMART FAILING' if r['smart'] == 'failing'
                  else 'SMART unavailable')
            smcls = 'bad' if r['smart'] == 'failing' else 'dim'
            why = (f' &middot; {esc_html(r["smart_detail"])}'
                   if r['smart_detail'] else '')
            smrow = f'<div class="dhsm {smcls}">{sm}{why}</div>'
        # What the drive says about its own wear. A zero here is the good
        # news and worth stating -- "0 reallocated" is the reason to keep
        # using the disk -- so the values are shown whatever they are, and
        # only the ones that are not zero take a colour.
        facts = ''.join(
            f'<span class="dhf {lvl}" title="{esc_html(why)}">'
            f'{esc_html(val)}</span>'
            for val, why, lvl in (r.get('smart_facts') or []))
        if facts:
            facts = f'<div class="dhfacts">{facts}</div>'
        cards.append(
            f'<div class="dh {kind}">'
            f'<div class="dhtop"><b class="dhname">{esc_html(r["label"])}</b>'
            f'<span class="dhverdict">{word}</span></div>'
            f'{meter}'
            f'<div class="dhroom">{room}</div>'
            f'{facts}'
            f'{smrow}'
            f'</div>')

    # That SMART was consulted at all is worth one line, so a section with
    # nothing to report cannot be mistaken for one that never checked.
    npass = sum(1 for r in rows if r['smart'] == 'passed')
    nfail = sum(1 for r in rows if r['smart'] == 'failing')
    if nfail:
        lead = (f'<p class="dhlead">SMART reports {nfail} of {len(rows)} '
                f'drive{"" if nfail == 1 else "s"} FAILING</p>')
    elif npass == len(rows):
        lead = f'<p class="dhsum">SMART passed on all {len(rows)} drives</p>'
    elif npass:
        lead = (f'<p class="dhsum">SMART passed on {npass} of {len(rows)} '
                f'&middot; the rest could not be read</p>')
    else:
        lead = '<p class="dhsum">SMART could not be read on any drive</p>'

    note = ''
    if any(r['smart'] == 'unreadable' and r['smart_detail'] == 'needs root'
           for r in rows):
        # The rule is narrow on purpose: one read-only subcommand, no wildcard
        # over smartctl's whole surface, and nothing that could start a test.
        who = esc_html(os.environ.get('USER') or 'you')
        smpath = esc_html(shutil.which('smartctl') or '/usr/sbin/smartctl')
        note = (
            '<p class="dhnote">SMART needs root to read. It is a read-only '
            'query &mdash; it does not unmount anything, interrupt I/O or '
            'touch the filesystem &mdash; but the kernel still gates it. '
            'Grant just that one command in its own drop-in file, which '
            '<code>visudo</code> syntax-checks before saving:<br>'
            '<code>sudo visudo -f /etc/sudoers.d/smartctl-dashboard</code>'
            '<br>and put one line in it:<br>'
            '<code>' + who + ' ALL=(root) NOPASSWD: ' + smpath
            + ' -H -j *</code><br>'
            'Until then the capacity and mount checks above still hold.</p>')
    return lead + '<div class="dhs">' + ''.join(cards) + '</div>' + note


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


# At most one top-up attempt per this many seconds, however many runs finish.
CONF_TOPUP_S = 600
_CONF_TOPUP = {'at': 0.0}


def confusion_topup():
    """Fetch matrices for finished runs that have none cached, in the background.

    The cache was a manual snapshot, so every run that finished after the last
    fetch showed no confusion matrix until somebody remembered to re-run the
    tool -- which is exactly what happened to dogbin_008. A finished run whose
    matrix is missing is a fact the dashboard can see for itself, so it acts on
    it.

    Detached and debounced, and it never blocks the render: the worst case is
    that the matrix appears on the next page load instead of this one.
    """
    if not CONFUSION_PYTHON or not COMET_ENV_FILE:
        return False
    now = time.time()
    if now - _CONF_TOPUP['at'] < CONF_TOPUP_S:
        return False
    try:
        have = confusion_index()
        want = set()
        for r in training_runs():
            # only a run that has FINISHED has a matrix to fetch; a live one
            # has not run its final validation yet
            if r.get('status') in ('running', 'never_started'):
                continue
            if confusion_for(run_key(r)):
                continue
            # ultralytics writes confusion_matrix.png beside the run at the
            # same moment it logs the numbers to Comet, so the PNG is a free
            # local answer to "did this run ever produce one". Without this
            # test every interrupted run, and every run never logged to Comet,
            # counts as missing forever -- and the top-up would go back to
            # Comet every ten minutes for a matrix that does not exist.
            if not os.path.exists(os.path.join(r['dir'],
                                               'confusion_matrix.png')):
                continue
            want.add(r['project'])
        if not want:
            return False
        _CONF_TOPUP['at'] = now
        script = os.path.join(REPO, 'tools', 'detect', 'fetch_confusion.py')
        if not os.path.exists(script):
            return False
        env = dict(os.environ, COMET_ENV_FILE=COMET_ENV_FILE)
        log = open(os.path.join(REPO, 'data', 'confusion_fetch.log'), 'a')
        _SPAWNED.append(subprocess.Popen(
            [CONFUSION_PYTHON, script, '--update',
             '--projects', ','.join(sorted(want))],
            cwd=REPO, env=env, stdout=log, stderr=log,
            stdin=subprocess.DEVNULL, start_new_session=True))
        return True
    except Exception:
        # a cache top-up is never worth failing a page render over
        return False


def render_training():
    """Runs on disk: what is training now, how it compares, what came before.

    The live run leads because it is the only part that changes while the page
    is open, and the only question with a deadline attached.
    """
    # A finished run with no cached matrix is a fact this function can see;
    # acting on it here means the panel repairs itself instead of waiting for
    # someone to remember the tool. Detached, debounced, never blocking.
    confusion_topup()
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

    # Decided before anything reads it: the charts, the project dropdown and
    # the table all have to agree on which runs count.
    floor = min_epochs()
    shown = [r for r in runs if is_real_run(r, floor)]
    hidden = sum(1 for r in runs if r['status'] not in REAL)
    too_short = len(runs) - len(shown) - hidden

    focus = next((r for r in runs if r['live'] and r['epochs_done']), None) \
        or next((r for r in shown if r['epochs_done']), None) \
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
                         sorted({r['project'] for r in shown}))
               + '</select>'
                 # The table below says how each run scored; the one thing it
                 # cannot say is what the run was scored ON. That is a
                 # directory, and this is the way to it.
                 '<a href="/datasets" class="rbtn nav" title="open the '
                 'datasets these runs trained on and look inside">'
                 '&#9638; Datasets</a>'
               + '<span class="tnote">Numbers here are each run\'s '
                 'own validation split &mdash; the split that drove its early '
                 'stopping. What a model is <em>accepted</em> on is the '
                 'reserved set, in Best models above.</span></div>')
    out.append(_history(shown))
    # Never silently, and never with one reason standing in for two: a run
    # left out for being unfinished and a project retired in the config are
    # different facts, and one sentence covering both said something untrue
    # about each.
    why = []
    if hidden:
        why.append(f'{hidden} interrupted before patience or the epoch '
                   f'budget, or never finished an epoch')
    if too_short:
        why.append(f'{too_short} shorter than {floor} epoch'
                   f'{"s" if floor != 1 else ""} &mdash; a smoke test, not a '
                   f'result')
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
            # Two panels on this page put a "best" number on the same run: this
            # card carries Comet's metrics as logged at promotion, the run
            # registry below reads the per-epoch results.csv, and the two
            # bookkeepings can differ in the third decimal (train-30: 0.4973 vs
            # 0.4968). Saying which ledger each number is from beats a page
            # that silently disagrees with itself by 0.0005.
            f' &middot; <span title="The registry table computes its best '
            f'column from each run\'s per-epoch results.csv; Comet logs its '
            f'own final validation, and the two can differ in the third '
            f'decimal.">metrics as logged by Comet &mdash; the run registry '
            f'reads results.csv, so the same run can read a shade different '
            f'there</span>'
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
    # one temp name per writer: the browser prefetches four at a time over the
    # same names warm_hq is cutting, and on a shared '<name>.part' the second
    # os.replace hit a file the first had already moved -- hq_crop returned
    # None, the handler 404'd, and the tile was judged at the 160px preview for
    # the rest of the page's life
    tmp = '%s.%d.%d.part' % (out, os.getpid(), threading.get_ident())
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
        r.save(tmp, 'JPEG', quality=92, optimize=True)
        os.replace(tmp, out)
        return out
    except Exception as e:
        # the prune below leaves .part names alone now, so a half-written temp
        # would sit in the cache dir forever if nobody dropped it here
        try:
            os.remove(tmp)
        except OSError:
            pass
        sys.stderr.write('hq_crop(%s): %s\n' % (name, e))
        return None


_hq_lock = threading.Lock()
_hq_busy = False
# The warmer outlives the request that started it, which is the point -- but it
# must not outlive the interpreter. A daemon thread holding a half-read parquet
# when the process exits takes duckdb's static teardown down with it: every
# check passes, the summary prints, and then the run dies of SIGABRT on
# `TProtocolException: Invalid data`, so the exit code says failure about work
# that succeeded. Rare on a server that never exits; near-certain in a script
# that imports this module, builds one review page and returns.
_hq_stop = threading.Event()
_hq_thread = None


def _hq_shutdown():
    """Ask the warmer to stop, and give it a moment to notice."""
    _hq_stop.set()
    t = _hq_thread
    if t is not None and t.is_alive():
        t.join(timeout=2.0)


atexit.register(_hq_shutdown)


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
                if _hq_stop.is_set():
                    return
                try:
                    hq_crop(n)
                except Exception:
                    pass
            if _hq_stop.is_set():
                return
            # drop cached cuts whose crop has aged out of the rolling pool --
            # the preview writer prunes recent_crops, nothing pruned this.
            # every directory the queue serves counts as "still live": pruning
            # against recent_crops alone deleted each harvested crop's cut in
            # the same pass that made it, so 37% of the queue was permanently
            # cold and re-decoded a 12 MP original on every page turn
            pool = set()
            for d in (CROPS, review_extra_dir()):
                try:
                    pool |= set(os.listdir(d))
                except OSError:
                    pass
            try:
                cached = os.listdir(HQ_DIR) if pool else []
            except OSError:
                cached = []
            for f in cached:
                # an in-flight '<name>.jpg.<pid>.<tid>.part' is a writer's
                # temp, never a pool name -- removing it 404'd the tile
                if f not in pool and not f.endswith('.part'):
                    try:
                        os.remove(os.path.join(HQ_DIR, f))
                    except OSError:
                        pass
        finally:
            _hq_busy = False

    global _hq_thread
    _hq_thread = threading.Thread(target=work, daemon=True)
    _hq_thread.start()


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
#
# MEASURED ON THE PRE-9f642b92 HARVEST. That commit recovered flags the match
# stage was mis-filing as ambiguous: re-run read-only over today's ledgers,
# match-stage survival moves from 2,640/2,778 (95.0%) to 2,757/2,778 (99.2%),
# so this value now runs ~4-5% conservative -- the panel over-asks slightly,
# which wastes minutes and never data. Re-measure at the next dogbin build,
# the first made end-to-end with the fixed pipeline.
FLAG_YIELD = 0.460
# Verdicts needed before the observed dog/not-dog mix is used to project how
# much reviewing is left. Below this the share is noise, and a couple of
# early "a dog" calls would swing the estimate by thousands.
MIX_MIN_SAMPLE = 50


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

    # How many more crops must be JUDGED, which is the thing the reviewer
    # actually does. Dividing the shortfall by FLAG_YIELD answers a
    # different question -- how many more NEGATIVES are needed -- and only
    # matches reality if every future verdict is "not a dog". It is not:
    # a fifth of them come back "a dog", each one adding to the class being
    # chased. So a judgement closes the gap by yield x (negatives - positives)
    # rather than by yield, and the honest figure is meaningfully larger.
    judged = fresh + fresh_pos
    # Under a small sample the observed mix is noise; fall back to the
    # negatives-only figure rather than extrapolating from a handful.
    share = (fresh_pos / judged) if judged >= MIX_MIN_SAMPLE else 0.0
    net = FLAG_YIELD * (1 - 2 * share)
    if not still:
        judgements = 0
    elif net > 0:
        judgements = int(-(-still // net))
    else:
        judgements = None       # marking dogs at least as fast: never closes
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
        'judgements_needed': judgements,
        'positive_share': round(share, 4),
        'mix_min_sample': MIX_MIN_SAMPLE,
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
        # the directories THIS server serves crops from, not the index's own
        # guess at them: whatever review_pool_names() walks is what the
        # country filter has to be able to place.
        ci.build(REPO, COUNTRY_INDEX,
                 extra_dirs=[d for d in (review_extra_dir(),) if d])
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


# ── the harvested review set ────────────────────────────────────────────────
# recent_crops/ is a rolling window: 2 crops/s written, newest 3,000 kept, and
# 0.24% of the sweep's positives held at any moment. Everything else is pruned
# unreviewed. build_review_set.py picks crops from the WHOLE store on purpose
# -- confidence band, size floor, spread over cells -- and writes them here,
# where nothing prunes them.
#
# Same filename convention, so every existing mechanism works untouched: the
# flag ledgers, /hq, the box editor, the country filter, the size floor.
def review_extra_dir():
    d = cfg('review_extra_dir', '', env='REVIEW_EXTRA_DIR')
    # resolved against the REPO, never the cwd: the server is started from
    # wherever, and a relative path that depends on that is a path that works
    # until someone starts it from somewhere else
    return os.path.join(REPO, d) if d and not os.path.isabs(d) else d


def review_pool_names():
    """[(name, directory)] across the live pool and the harvested set.

    The live pool comes first so a fresh detection still surfaces promptly;
    within a directory the sort decides the order anyway.
    """
    out = []
    for d in (CROPS, review_extra_dir()):
        if not d:
            continue
        try:
            for n in os.listdir(d):
                out.append((n, d))
        except OSError:
            continue          # absent or unreadable is not an error here
    return out


def crop_dir(name):
    """Which of the queue's directories a crop file is actually in.

    Everything that copies a crop out at verdict time has to ask. Naming
    recent_crops alone was right for the rolling pool and wrong for every
    harvested crop, which is not a race but a certainty: the copy simply never
    happened, and the verdict was recorded anyway with no image beside it.

    Falls back to the pool, which is where a crop that has just aged out was.
    """
    for d in (CROPS, review_extra_dir()):
        if d and os.path.exists(os.path.join(d, name)):
            return d
    return CROPS


# ── crops too small for anyone to judge ─────────────────────────────────────
# /hq cuts each tile from the ORIGINAL at native box resolution, so what the
# reviewer sees is the box's true pixel size. A 17x15 box in a 1920x1080 frame
# is ~255 pixels of animal blown up to 640 -- not a rendering problem, and not
# a judgement any person or model can make. Asking for one is worse than
# skipping it: a coin-flip "not a dog" can turn the whole frame into a
# detector NEGATIVE, teaching it to miss a dog it did find.
#
# Off by default (0). A fresh clone behaves exactly as before.
_SZ = {'by_key': {}, 'lock': threading.Lock()}
SZ_MAX = 50000        # ten times the live queue; a refill needs only the pool


def min_review_px():
    return max(0, cfg_int('review_min_px', 0, env='REVIEW_MIN_PX'))


def box_short_sides(keys):
    """{(image_id, conf2): shorter side in ORIGINAL pixels}.

    One query for the whole page, not two per crop like box_for. Keyed on
    (image_id, conf x100) because that is how a crop filename names its own
    detection everywhere else in this file -- an image with a big box and a
    tiny one must not have the tiny crop judged by the big box's size.
    """
    # A box's short side is a fact about a frame that was decoded once: it
    # cannot go stale, so the cache is bounded by size rather than by age. The
    # wall-clock wipe was pure loss -- `at` was refreshed only when it fired,
    # so a header polling the queue depth every 30 s paid a full scan of the
    # detection store every fourth minute for the life of the server.
    #
    # One scan at a time, too: this is seconds of multi-core duckdb against the
    # drives the sweep is already reading, and concurrent callers were each
    # paying it in full for the same answer. The second one waits and then
    # finds the cache filled.
    with _SZ['lock']:
        if len(_SZ['by_key']) > SZ_MAX:
            _SZ['by_key'].clear()
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
                        f"FROM {det} WHERE CAST(image_id AS VARCHAR) "
                        f"IN ({lst}) GROUP BY 1, 2").fetchall()
                    con.close()
                    for iid, c2, side in rows:
                        _SZ['by_key'][(iid, int(c2))] = float(side)
                except Exception:
                    # the store is the source; if it cannot be read the floor
                    # simply does not apply. Never drop a crop on a failed
                    # lookup.
                    pass
    return _SZ['by_key']


ANNOT_SORTS = {
    'recent': lambda r: -r['flagged_at'],
    'oldest': lambda r: r['flagged_at'],
    'conf': lambda r: (-r['conf'], -r['ts']),
    'low': lambda r: (r['conf'], -r['ts']),
}


def annotated_payload(page=0, size=REVIEW_PAGE, label='all', sort='recent',
                      leash=''):
    """Crops that already carry a verdict, for auditing the annotations.

    A misannotation is worse than an unjudged crop: it does not sit in a queue
    waiting, it goes into a dataset as ground truth and teaches the wrong
    thing. Nothing in this project could look at one again -- once flagged, a
    crop left the queue for good.

    Reads the ledgers rather than the pool. The pool rotates every few
    minutes; these crops were copied out of it when they were flagged, and
    their originals are still in the store, so an audit works on crops the
    live queue has long forgotten.
    """
    size = 100 if int(size) >= 100 else REVIEW_PAGE
    page = max(0, int(page))
    sort = sort if sort in ANNOT_SORTS else 'recent'
    want = [label] if label in FLAG_LABELS else list(FLAG_LABELS)

    items, seen_names = [], set()
    for lb in want:
        st = _store_for(lb)
        try:
            fh = open(st['labels'])
        except OSError:
            continue
        with fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if not isinstance(r, dict):
                    continue
                nm = r.get('crop') or ''
                m = _CROP_RE.match(nm)
                if not m:
                    continue
                # LAST LINE WINS. The ledger is append-only and a re-decision
                # rewrites the other label's file, but a name can still appear
                # twice within one file; the newest line is the verdict.
                items.append({
                    'name': nm, 'image_id': m.group(2),
                    'ts': int(m.group(1)),
                    'conf': round(int(m.group(3)) / 100.0, 2),
                    'label': lb,
                    'flagged_at': int(r.get('flagged_at') or 0),
                    'has_crop': True})
    # collapse to one entry per crop, keeping the newest verdict
    by_name = {}
    for it in items:
        prev = by_name.get(it['name'])
        if prev is None or it['flagged_at'] >= prev['flagged_at']:
            by_name[it['name']] = it
    # and drop any whose verdict the in-memory set no longer agrees with --
    # an undo rewrites the file, but a stale line could survive a crash
    live = {lb: _flag_names(lb) for lb in FLAG_LABELS}
    items = [it for it in by_name.values() if it['name'] in live[it['label']]]

    items.sort(key=ANNOT_SORTS[sort])
    # Counted before the filter narrows them, so the option can say how many
    # it would show -- and counted over the SAME list the filter is applied to.
    # Taken over the dogs alone, "needs a leash call" advertised 112 and handed
    # back 2,765, because the filter also ran over every crop judged not a dog
    # and all of them satisfy "no leash verdict".
    want_leash = leash if leash in LEASH_FILTERS else 'all'
    leash_offer = {k: len(_leash_keep(items, k)) for k in LEASH_FILTERS}
    if want_leash != 'all':
        items = _leash_keep(items, want_leash)
    total = len(items)
    pages = max(1, (total + size - 1) // size)
    page = min(page, pages - 1)
    lo = page * size
    shown = items[lo:lo + size]
    return {'items': shown, 'page': page, 'size': size,
            'leash': _leash_for([c['name'] for c in shown]),
            'leash_totals': _leash_counts(),
            'leash_filter': want_leash, 'leash_counts': leash_offer,
            'pages': pages, 'total': total, 'sort': sort, 'label': label,
            # What the list holds before EITHER filter. len(by_name) was
            # wrong: `want` decides which ledger files are read at all, so the
            # verdict filter narrows upstream of the baseline and the baseline
            # collapsed onto the total exactly when the one filter this view
            # has was on. Counted from the live sets instead, which do not
            # move with the request.
            'pool_unfiltered': sum(len(live[lb]) for lb in FLAG_LABELS),
            'n_false_positive': sum(1 for i in by_name.values()
                                    if i['label'] == FLAG_LABEL
                                    and i['name'] in live[FLAG_LABEL]),
            'n_true_positive': sum(1 for i in by_name.values()
                                   if i['label'] == POS_LABEL
                                   and i['name'] in live[POS_LABEL])}


# ── model suggestions: a way to sort the queue, never a label ───────────────
# Written by tools/detect/triage_crops.py, read here and nowhere else. The
# file is a sibling of the ledgers, not part of them: nothing that builds a
# training set opens it, and tools/detect/tests/adv_triage_isolation.py
# asserts that against the source. Every record carries unverified=True.
TRIAGE_FILE = os.path.join(OUT, 'triage.jsonl')
TRIAGE_STATUS = os.path.join(OUT, 'triage_status.json')
# A run that dies leaves its last position behind and no way to know it died,
# so a status older than this is reported as stopped rather than running. The
# writer touches the file every batch, which is seconds apart even on CPU.
TRIAGE_STALE_S = 90
TRIAGE_BUCKETS = ('dog', 'animal', 'object', 'not_dog')
# What each guesser can actually SAY, in the order the filter offers it. Per
# backend because they do not answer the same question: the dog-bin gate is
# binary and has no opinion about what a not-dog is, and folding its 'not_dog'
# into 'object' would file every cow under "not an animal".
BUCKET_LABELS = {'dog': 'Looks like a dog', 'animal': 'Other animal',
                 'object': 'Not an animal', 'not_dog': 'Not a dog'}
OPEN_BUCKETS = ('dog', 'animal', 'object')
# The guessers, and what it takes to run each. A backend is only offered when
# an interpreter for it is named in config -- they are different environments
# on purpose: RF-DETR wants transformers>=5 and the SigLIP backend does not
# run on that, so installing them together would break the one that works.
TRIAGE_BACKENDS = ('siglip', 'dogbin', 'rfdetr')
RFDETR_PYTHON = cfg('rfdetr_python', '', env='RFDETR_PYTHON')
RFDETR_MODEL = cfg('rfdetr_model', 'rfdetr', env='RFDETR_MODEL')
# The dog-bin gate runs under ultralytics, a third environment again. Its
# weights are NOT config: the promoted checkpoint is already recorded in
# data/best_models.json, and a second place to write it down is a second place
# for it to go stale.
DOGBIN_PYTHON = cfg('dogbin_python', '', env='DOGBIN_PYTHON')
# Share of the 120 crops in data/hard_positives -- ones the DETECTOR was
# unsure about and a human then confirmed are dogs -- that each guesser files
# under 'dog'. Both measured the same way on the same set, which is the only
# thing that makes them comparable: the first number here was 0.98, carried
# over from a different measurement (how many real dogs land in 'other
# animals', 2.0%), and putting it beside RF-DETR's overstated the gap by
# twenty points on the one screen meant to prevent exactly that.
#
# These are the HARD positives by construction -- detector confidence 0.05 to
# 0.10 -- so neither number is this pipeline's accuracy on an average crop.
# They are a like-for-like comparison between the two guessers, which is what
# the dropdown is for.
BACKEND_INFO = {
    'siglip': {'label': 'SigLIP 2', 'recall': 0.977, 'clears': 0.943,
               'buckets': OPEN_BUCKETS,
               'note': 'a general-purpose model asked our question in our own '
                       'words. Reads the whole crop, always has an opinion, '
                       'and is the only one that leaves behind the vectors '
                       'the search box needs.'},
    'dogbin': {'label': 'Dog-bin gate', 'recall': 0.936, 'clears': 0.943,
               'buckets': ('dog', 'not_dog'),
               'note': 'the classifier this project trained on its own '
                       'reviewers\' verdicts — the only one of the three that '
                       'has seen nothing but this data. It answers a narrower '
                       'question, dog or not, with no opinion on what a '
                       'not-dog is. It does not win: SigLIP finds more of the '
                       'dogs and clears the same share of not-dogs. Writes no '
                       'search vectors.'},
    'rfdetr': {'label': 'RF-DETR', 'recall': 0.678, 'clears': 0.957,
               'buckets': OPEN_BUCKETS,
               'note': 'a COCO detector: names a concrete class or says '
                       'nothing. Weakest at finding dogs — most crops are '
                       'under 64px and a detector needs pixels on target — '
                       'but it is the least likely to call a non-dog a dog, '
                       'so it is the one for finding cows, horses and people. '
                       'Writes no search vectors.'},
}
# What the percentage beside each guesser IS. Sent to the page rather than
# written into it, so the sentence and the numbers it explains come from one
# place and cannot drift apart. '% of known dogs' on its own does not say
# which dogs, or what the guesser had to do to count -- and a number nobody
# can interpret is not the safeguard it was put there to be.
RECALL_BASIS = (
    'One fixed test, the same crops for all three: the 342 dogs and 300 '
    'not-dogs of the dog-bin validation split, every one of them labelled by '
    'a reviewer. "Finds" is the share of the dogs a guesser files under '
    '"dog"; "clears" is the share of the not-dogs it does not. A guesser that '
    'called everything a dog would score 100% and 0%, so the pair is the '
    'claim, never one number alone.')
# The part that is easy to leave out and expensive to leave out. Two of the
# three had a hand in choosing themselves against this split, which flatters
# exactly the two that lead.
RECALL_CAVEAT = (
    'None of the three trained on these crops, but two of them were TUNED '
    'against them: the dog-bin gate picked its best epoch here, and SigLIP\'s '
    'bucket rule was chosen here too. RF-DETR is the only one that has never '
    'seen this split in any form. Read the gap between the leaders as the '
    'optimistic end of its range.')
_triage_cache = {'mtime': None, 'by': {}}


def backend_of(model_id):
    """Which guesser wrote a record, from the model it names.

    52,000 records predate the backend field and carry only `model`, so the
    answer has to be derivable from that or they all vanish from a filter.
    Kept in step with triage_crops.backend_of by adv_cross_module_signatures.
    """
    m = str(model_id or '')
    if m.startswith('rfdetr'):
        return 'rfdetr'
    if m.startswith('dogbin'):
        return 'dogbin'
    if m == 'imagenet' or m.startswith('efficientnet'):
        return 'imagenet'
    return 'siglip'


def dogbin_weights():
    """The promoted dog-bin checkpoint, or '' if there is none on disk.

    Read from data/best_models.json every time rather than cached: promoting a
    new gate rewrites that file, and a guesser still running the old one
    because the dashboard read the path at import is the kind of staleness
    nobody notices.
    """
    try:
        with open(BEST_MODELS) as fh:
            best = (json.load(fh).get('projects') or {}) \
                .get('dog-bin', {}).get('best') or {}
    except (OSError, ValueError, AttributeError):
        return ''
    rel = str(best.get('weights') or '')
    if not rel:
        return ''
    p = rel if os.path.isabs(rel) else os.path.join(training_root(), rel)
    return p if os.path.exists(p) else ''


def backends_available():
    """Backends this checkout can actually RUN, in offer order."""
    out = []
    if CONFIGURED_TRIAGE:
        out.append('siglip')
    # weights as well as an interpreter: the gate is a checkpoint on this
    # machine, and offering to run one that is not there wastes a click
    if DOGBIN_PYTHON and dogbin_weights():
        out.append('dogbin')
    if RFDETR_PYTHON:
        out.append('rfdetr')
    return out


def backends_offered():
    """Backends the page may select: runnable, plus any with guesses on file.

    A backend whose interpreter is gone still has opinions worth reading, and
    the legacy `--model imagenet` one has no interpreter of its own at all --
    filtering the dropdown by runnability alone made its guesses invisible to
    every index while the tool still documented how to write them.
    """
    out = list(backends_available())
    for b in TRIAGE_BACKENDS:
        if b not in out and triage_seen(b):
            out.append(b)
    return out


def pick_backend(want):
    """The backend a request actually gets. ONE rule, used by every caller.

    The strip and the queue validated this differently -- one against the list
    of names that exist, the other against the list that can run -- so a saved
    preference for a backend this checkout cannot run had the strip describing
    one guesser while the queue served another's opinions, with nothing on
    screen admitting it.
    """
    offered = backends_offered()
    if want in offered:
        return want
    return offered[0] if offered else 'siglip'


def triage_index(backend='siglip'):
    """{crop name: {bucket, p, top}} for ONE guesser, reloaded on change.

    Per backend, because the two disagree on purpose and a filter that mixed
    them would answer with whichever ran last. Same mtime discipline as
    country_index(): the hourly rebuild has to reach a server that has been up
    for days, and a triage run appends to this file while that server is
    running.
    """
    return _triage_load(backend)[0]


def triage_seen(backend='siglip'):
    """Crops this guesser has LOOKED at, guess or no guess.

    Not the same set as the index. RF-DETR is allowed to find nothing and
    writes that down; 922 of its records on this box say exactly that. Those
    crops are finished work -- it will not re-run them -- but they carry no
    bucket, so counting coverage off the index alone left the strip reporting
    a permanent shortfall and sitting in the warn state no amount of guessing
    could clear.
    """
    return _triage_load(backend)[1]


def _triage_load(backend):
    """(guesses, looked-at) for one backend, reloaded when the file changes."""
    backend = backend if backend in TRIAGE_BACKENDS else 'siglip'
    try:
        mtime = os.path.getmtime(TRIAGE_FILE)
    except OSError:
        return _triage_cache['by'].get(backend, ({}, set()))
    # Kept per backend rather than one slot: two tabs on different guessers
    # would otherwise evict each other on every poll and re-read 52,000 lines
    # each time.
    if _triage_cache['mtime'] != mtime:
        _triage_cache.update(mtime=mtime, by={})
    if backend not in _triage_cache['by']:
        doc, seen = {}, set()
        try:
            with open(TRIAGE_FILE) as fh:
                for ln in fh:
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    nm = isinstance(r, dict) and r.get('name')
                    if not nm:
                        continue
                    if backend_of(r.get('backend') or r.get('model')) \
                            != backend:
                        continue
                    seen.add(nm)
                    if r.get('bucket') not in TRIAGE_BUCKETS:
                        # A detector is allowed to find nothing, and it writes
                        # that down. The record proves the crop was looked at,
                        # which is why it is not simply absent -- but it is
                        # not a guess, so it must not become one here. Dropping
                        # any earlier guess for the crop is deliberate: last
                        # line wins, and 'this backend saw nothing' is the
                        # later word.
                        doc.pop(nm, None)
                        continue
                    # last line wins, so a --refresh re-run supersedes
                    # prefer the in-bucket name; fall back to the overall
                    # top-1 for records written before that field existed
                    doc[nm] = {'b': r['bucket'], 'p': r.get('p'),
                               # 'guess' is the current key; 'label' was used
                               # briefly and is read only so records written
                               # then still render
                               'top': r.get('guess') or r.get('label')
                               or (r.get('top') or [[None]])[0][0]}
        except OSError:
            doc, seen = {}, set()
        _triage_cache['by'][backend] = (doc, seen)
    return _triage_cache['by'][backend]


# Failures worth repeating back to the reader, most specific first.
_TRIAGE_REASONS = (
    ('out of memory', 'the GPU was full \u2014 something else on this box has it'),
    ('No module named', 'that interpreter is missing a package it needs'),
    ('CUDA driver', 'the CUDA driver would not initialise'),
    ('No such file or directory', 'a file it needed was not there'),
)


def _triage_last_error():
    """A short reason from the tail of the run log, or None.

    The tail only, and only if it is recent: the log is appended to across
    runs, and a reason lifted from one that ended yesterday is worse than
    saying nothing.
    """
    path = os.path.join(REPO, 'data', 'triage_run.log')
    try:
        if time.time() - os.path.getmtime(path) > 3600:
            return None
        with open(path, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4096))
            tail = fh.read().decode('utf-8', 'replace')
    except OSError:
        return None
    for needle, said in _TRIAGE_REASONS:
        if needle in tail:
            return said
    return None


def _num_or(v, default):
    """A number out of a status file, or the default.

    Every caller reads a file some other process is writing, so a truncated
    write, a changed schema or a JSON NaN has to degrade to "unknown" rather
    than reach arithmetic. NaN is the one that hides: it is a float, it
    passes this cast, and it poisons everything downstream silently.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def triage_status(backend='siglip'):
    """Progress of the suggestion run, plus how much of the queue it covers.

    Coverage is the useful half: a finished run still leaves crops unguessed
    because the live pool keeps growing underneath it, and that is the number
    that decides whether the filter is worth trusting right now. Coverage is
    per backend for the same reason the index is: RF-DETR having guessed the
    pool says nothing about whether SigLIP has.
    """
    backend = pick_backend(backend)
    doc = {}
    try:
        with open(TRIAGE_STATUS) as fh:
            doc = json.load(fh) or {}
    except (OSError, ValueError):
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    age = time.time() - float(doc.get('updated') or 0)
    # Liveness comes from the process table, not from the pid in the file.
    # os.kill(pid, 0) was the obvious check and it is wrong twice over: it
    # succeeds for a ZOMBIE, which is a dead process the parent has not waited
    # on, and it succeeds for whatever unrelated process later inherits a
    # recycled pid. Scanning for a python actually running triage_crops.py
    # answers the question that was meant -- a zombie has no command line, so
    # it cannot match.
    _reap()
    alive = bool(triage_pids())
    # A --watch run is SUPPOSED to go quiet: it finishes a pass and sleeps for
    # its whole interval before looking again. With a 90s threshold against a
    # 300s interval it read as stopped for 210 of every 300 seconds while
    # perfectly healthy, so the silence a run has announced it will keep is
    # added to the grace rather than counted against it.
    quiet_ok = TRIAGE_STALE_S + max(0, _num_or(doc.get('watch'), 0))
    # WHOSE run this is. One status file serves both guessers, so a live run
    # answers for the backend that started it and for no other. Without this
    # the strip reported an RF-DETR run as SigLIP's the moment the dropdown
    # moved -- same progress bar, same Pause button, and pressing it would
    # have stopped a run the reader was not looking at.
    run_backend = (doc.get('backend') or backend_of(doc.get('model'))) \
        if doc else None
    live = bool(doc.get('running')) and age < quiet_ok and alive
    running = live and run_backend == backend
    # STALLED means genuinely hung: the process is still alive but has stopped
    # writing. A DEAD pid is not stalled -- the run simply ended (a kill -9
    # leaves running=True with no chance to write finished=True), so it falls
    # through to the plain "not running" state instead of alarming with
    # "Run stopped, no progress for 114 min" when nothing is wrong.
    # ...and a stall belongs to the run that stalled, for the same reason
    stalled = (bool(doc.get('running')) and alive and age >= quiet_ok
               and run_backend == backend)
    tri = triage_index(backend)
    # Coverage is "has this guesser dealt with the crop", which includes the
    # ones it looked at and had nothing to say about. Counting only the crops
    # with a bucket made RF-DETR's coverage permanently short by the 922 it
    # had honestly declined, and the strip could never leave its warning.
    done = triage_seen(backend)
    pool = review_pool_names()
    have = sum(1 for n, _ in pool if n in done)
    return {'ever': bool(doc) or bool(tri),
            # Which guesser this coverage is about, what else could be asked,
            # and how each did against the crops a human has already ruled on.
            # The recall belongs on the page: switching to the weaker guesser
            # without being told it is the weaker one is a trap.
            # The OTHER guesser, when it is the one running. Only one runs at a
            # time -- they share a GPU and a status file -- so a Run press
            # while the other works can only be refused, and a button that
            # refuses without saying why is worse than one that is not there.
            'busy_with': (BACKEND_INFO.get(run_backend, {}).get('label')
                          or run_backend) if (live and run_backend != backend)
                         else None,
            'backend': backend, 'recall_basis': RECALL_BASIS,
            'recall_caveat': RECALL_CAVEAT, 'backends': [
                dict(BACKEND_INFO.get(b, {}), key=b,
                     running=(b == run_backend and live))
                for b in backends_offered()],
            # Whether a run could be STARTED, which is not the same fact as
            # whether one ever has. The strip hides itself until something has
            # run, and it carries the Run button -- so clearing the guesses hid
            # the only control that could put them back.
            'can_run': backend in backends_available(),
            # Why it is NOT running, when the run left a reason behind. A run
            # can die long after the start call returned -- a GPU filling up
            # takes as long as a model takes to load -- and the strip just went
            # quiet, so a full graphics card looked like a broken dashboard.
            'why': None if running else _triage_last_error(),
            'running': running,
            'starting': bool(doc.get('starting')) and running,
            'stalled': stalled,
            'idle': bool(doc.get('idle')),
            'watch': doc.get('watch') or 0,
            'model': doc.get('model') or '',
            'done': doc.get('done') or 0,
            'total': doc.get('total') or 0,
            'rate': doc.get('rate') or 0,
            'passes': doc.get('passes') or 0,
            'age_s': int(age) if doc else None,
            'pool': len(pool), 'guessed': have,
            'coverage': round(have / len(pool), 3) if pool else 0}


LEASH_FILTERS = ('all', 'none', 'leashed', 'unleashed')

# ── free-text search over the queue ─────────────────────────────────────────
# The crops carry a SigLIP vector each (triage_crops.py keeps the one it
# already computes) and terms carry one too (crop_search.py encodes them). A
# search is therefore a dot product here -- no model in this process, and the
# dashboard's environment has no torch to run one with.
VEC_FILE = os.path.join(OUT, 'triage_vecs.npz')
TERM_FILE = os.path.join(OUT, 'search_terms.npz')
_VEC = {'at': None, 'names': None, 'vecs': None, 'model': ''}
_TERM = {'at': None, 'terms': {}, 'model': ''}
SEARCH_RETRY_S = 120
SEARCH_TERM_MAX = 48
_TERM_TRIED = {}
# The single in-flight encoder: 'sent' is the batch it carries, 'want' the
# words queued behind it, 'bad' the ones whose last attempt exited non-zero.
_TERM_JOB = {'proc': None, 'want': set(), 'sent': set(), 'bad': set()}


def search_term_ok(term):
    """Whether a word is one this dashboard will hand to the encoder.

    Anything accepted is permanent: crop_search.py has --add and no --remove,
    and the whole store comes back as the review page's suggestion list. A
    plain GET reaches this -- a typo'd URL, a bookmark, an <img src> on any
    page the operator visits -- so an unvalidated term meant a stranger could
    put a sentence in front of the next reviewer. '../../etc/passwd' is in the
    live store because a route probe asked for it once.

    A search phrase, then: words, spaces, and the punctuation that occurs
    inside words. Nothing shaped like a path, a flag or a shell.
    """
    return (2 <= len(term) <= SEARCH_TERM_MAX
            and all(c.isalnum() or c in " '-" for c in term))


def _npz(path, state, load):
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        state.update(at=None)
        return False
    if state['at'] == stamp:
        return True
    try:
        import numpy as np
        d = np.load(path, allow_pickle=False)
    except Exception:
        state.update(at=None)
        return False
    load(d)
    state['at'] = stamp
    return True


def crop_vectors():
    """(names, matrix, model) for the pool, or (None, None, '')."""
    def load(d):
        _VEC.update(names=[str(x) for x in d['names']], vecs=d['vecs'],
                    model=str(d['model']))
    if not _npz(VEC_FILE, _VEC, load):
        return None, None, ''
    return _VEC['names'], _VEC['vecs'], _VEC['model']


def search_terms():
    """({term: vector}, model) that have been encoded."""
    def load(d):
        _TERM.update(terms={str(t): d['vecs'][i]
                            for i, t in enumerate(d['terms'])},
                     model=str(d['model']))
    if not _npz(TERM_FILE, _TERM, load):
        return {}, ''
    return _TERM['terms'], _TERM['model']


def search_ready():
    """Terms that can be searched right now, for the page's datalist."""
    terms, tmodel = search_terms()
    _, _, cmodel = crop_vectors()
    # a term encoded by one model cannot be compared with crops embedded by
    # another; the numbers would still come out and mean nothing
    return sorted(terms) if (terms and tmodel == cmodel) else []


def search_scores(term):
    """({crop name: similarity}, why) -- `why` is None when scores are usable.

    The two failure modes are worth telling apart, because one is a wait and
    the other is a job to do. 'unknown' means nobody has encoded this word
    yet, which fixes itself. 'mismatch' means the crops and the words come
    from different models, which nothing on this page can fix.
    """
    term = (term or '').strip()
    if not term:
        return None, 'unknown'
    terms, tmodel = search_terms()
    names, vecs, cmodel = crop_vectors()
    if names is None:
        return None, 'novectors'
    if tmodel and cmodel and tmodel != cmodel:
        return None, 'mismatch'
    if term not in terms:
        return None, 'unknown'
    import numpy as np
    sims = vecs.astype('float32') @ terms[term].astype('float32')
    return {names[i]: float(sims[i]) for i in range(len(names))}, None


def search_coverage(pooled=None):
    """(crops with a vector, crops in the pool) -- how much of it is searchable.

    A vector belongs to a crop FILE, and the review pool rotates: 3,000 crops,
    turned over in under an hour while the harvest runs. Vectors are written
    by the guesser as it works, so leaving it stopped lets coverage fall to
    nothing while every part of the search still looks healthy. It did: 4,513
    vectors, 3,010 crops in the pool, zero crops in both, and a search that
    silently reordered nothing. This number is what makes that visible.
    """
    names, _, _ = crop_vectors()
    have = set(names or ())
    # the caller has usually just listed the pool; listing it again is five
    # thousand stat entries per page turn for an answer already in hand
    pool = {n for n, _ in (review_pool_names() if pooled is None else pooled)}
    return len(pool & have), len(pool)


def search_learn(term):
    """Ask for a term to be encoded in the background. Returns what to say.

    'learning' one is running, or has just been started
    'failed'   the last attempt at this word exited badly
    'unknown'  there is no interpreter here that could encode it

    ONE encoder at a time, with the rest queued. Every unknown word spawned
    its own process, each of which loads SigLIP; typing three words in a row
    put three copies of it in memory and -- before crop_search.py took a lock
    -- let them overwrite each other's work.

    A failed attempt has to be sayable. The encoder needs an environment with
    transformers in it, and when it does not have one it dies on the import in
    under a second -- while the page went on promising an answer 'in a moment'
    for as long as anyone cared to wait.
    """
    term = (term or '').strip()
    # TRIAGE_PYTHON, not the scorer's: encoding a word needs transformers,
    # and the environment that made the crop vectors is the one env guaranteed
    # to have it. MISTAKES_PYTHON is an ultralytics env and on this box has no
    # transformers at all, so preferring it meant every newly typed word died
    # on the import in under a second while the page said 'learning' forever.
    py = TRIAGE_PYTHON if CONFIGURED_TRIAGE else (mistakes_python() or
                                                  TRIAGE_PYTHON)
    script = os.path.join(REPO, 'tools', 'detect', 'crop_search.py')
    # A term that will not be stored is answered exactly like one nobody has
    # encoded, which is what it is and always will be -- there is no path from
    # here that ends with it in the store.
    if not term or not search_term_ok(term):
        return 'unknown'
    if not py or not os.path.exists(script):
        return 'unknown'
    now = time.time()
    with _SPAWN_LOCK:
        job = _TERM_JOB
        if job['proc'] is not None and job['proc'].poll() is None:
            # one is already running: ride along with it rather than start a
            # second copy of the model
            job['want'].add(term)
            return 'learning'
        if job['proc'] is not None:
            # It has exited: judge the batch it was carrying before starting
            # another, or a broken environment retries in silence forever.
            # Judged PER TERM, by whether the word actually landed in the
            # cache -- not by the exit code over the whole batch. One word
            # argparse would not take (anything starting with '-') sinks the
            # whole --add, and blaming the exit code told a reviewer that the
            # perfectly good word they had typed 'could not be encoded'.
            done = set(search_terms()[0])
            for t in job['sent']:
                if t in done:
                    job['bad'].discard(t)
                else:
                    job['bad'].add(t)
            job.update(proc=None, sent=set())
        # keyed by whatever anyone typed, so it needs a ceiling for the same
        # reason _TERM_TRIED does
        if len(job['bad']) > 512:
            job['bad'] = set(sorted(job['bad'])[-256:])
        if now - _TERM_TRIED.get(term, 0) < SEARCH_RETRY_S:
            return 'failed' if term in job['bad'] else 'learning'
        job['want'].add(term)
        batch = sorted(job['want'])
        # `pop`, not `del`: two threads reaching this under different terms
        # both prune, and the second one used to raise KeyError
        if len(_TERM_TRIED) > 512:
            for k, _ in sorted(_TERM_TRIED.items(),
                               key=lambda kv: kv[1])[:256]:
                _TERM_TRIED.pop(k, None)
        for t in batch:
            _TERM_TRIED[t] = now
        try:
            log = open(os.path.join(REPO, 'data', 'crop_search.log'), 'a')
            # --add-json, not --add: a word beginning with '-' is a word
            # somebody can type, and as an --add value argparse reads it as a
            # flag and rejects the batch it was in
            proc = subprocess.Popen(
                [py, script, '--add-json', json.dumps(batch)], cwd=REPO,
                stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                start_new_session=True)
            _SPAWNED.append(proc)
            job.update(proc=proc, want=set(), sent=set(batch))
            return 'learning'
        except Exception:
            job['want'].discard(term)
            job['bad'].add(term)
            return 'failed'


def _leash_keep(items, want, key='name'):
    """Narrow a list of crops by leash state. 'none' = no verdict recorded.

    Kept separate from the dog verdict on purpose: "a dog I have not decided
    the leash for" is the question this whole axis exists to answer, and it is
    not expressible as a value of the dog verdict.
    """
    if want not in LEASH_FILTERS or want == 'all':
        return items
    got = _leash_for([c[key] for c in items])
    if want == 'none':
        return [c for c in items if c[key] not in got]
    return [c for c in items if got.get(c[key]) == want]


def _leash_for(names):
    """{crop: 'leashed'|'unleashed'} for the crops about to be shown."""
    mod = leash_store()
    if not mod:
        return {}
    try:
        return mod.labels_for(names)
    except Exception:
        return {}


def _leash_counts():
    mod = leash_store()
    if not mod:
        return None            # None means "no store", not "none recorded"
    try:
        return mod.counts()
    except Exception:
        return None


GATE_FILTERS = ('all', 'dog', 'not_dog', 'none')


def _gate_covers(backend):
    """Does the gate's own filter already answer this backend's question?

    The dog-bin gate has a dedicated control, so the page does not also offer
    it as a guess filter -- two dropdowns over one model's verdict is a choice
    the reader has to work out is not a choice. Written once, here, because
    the page's decision to hide a control and the server's decision to ignore
    its value have to be the same decision.
    """
    return backend == 'dogbin' and bool(triage_index('dogbin'))


def review_payload(page=0, size=REVIEW_PAGE, sort=None, country='',
                   suggest='', leash='', find='', backend='siglip', gate=''):
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

    ``size=0`` is the header's queue-depth poll: the total, and no crops. It
    used to land on REVIEW_PAGE one line down, so a poll every 30 s built the
    full page AND its reserve and warmed a hundred HQ cuts for a page nobody
    was looking at.
    """
    count_only = int(size) == 0
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
    pooled = review_pool_names()
    if not pooled:
        return empty
    names = [n for n, _ in pooled]
    # which directory each crop came from, so the client can be served the
    # right bytes without guessing
    where = {n: d for n, d in pooled}
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
    want_backend = pick_backend(backend)
    _tri = triage_index(want_backend)
    want = (country or '').upper()
    want_sg = suggest if suggest in TRIAGE_BUCKETS or suggest == 'none' else ''
    # A filter the chosen guesser cannot honour is dropped, not applied. The
    # control hides itself when a backend has no guesses, so switching to a
    # fresh one while filtered to 'Looks like a dog' emptied the queue and
    # took away the only thing that could put it back.
    if not _tri and want_sg:
        want_sg = ''
    # ...and neither is a bucket this guesser cannot produce. The page hides
    # the guess filter whenever the dog-bin gate's own axis covers it, but
    # hiding a control does not unset it: the value stayed in the request and
    # the server kept honouring it, so choosing the gate could empty the queue
    # with no chip, no cross to clear it, no control on screen and no
    # "narrowed from" -- every surface the redesign added to make an empty
    # queue explainable, silent at once. The server decides, and echoes what
    # it decided so the page can follow.
    _can_say = set(BACKEND_INFO.get(want_backend, {}).get('buckets')
                   or OPEN_BUCKETS) | {'none'}
    if want_sg and (want_sg not in _can_say or _gate_covers(want_backend)):
        want_sg = ''
    want_leash = leash if leash in LEASH_FILTERS else 'all'
    # Always the dog-bin gate, whatever the toggle above is set to: this axis
    # is ONE model's verdict on the reviewer's own question, not "the current
    # guesser's opinion", so it must not move when the toggle does.
    _gate = triage_index('dogbin')
    want_gate = gate if gate in GATE_FILTERS else 'all'
    if not _gate:
        want_gate = 'all'
    want_find = (find or '').strip()
    cands = []
    for name in names:
        m = _CROP_RE.match(name)
        if not m or name in flagged or name in positives:
            continue
        iid = m.group(2)
        if iid in judged:      # flagged, or already looked at and kept
            continue
        iso = by_country.get(iid, '')
        sg = _tri.get(name) or {}
        cands.append({'name': name, 'image_id': iid,
                      'ts': int(m.group(1)),
                      'conf': round(int(m.group(3)) / 100.0, 2),
                      'country': iso,
                      # a GUESS, carried under its own keys so no reader can
                      # confuse it with the verdict fields
                      'sg': sg.get('b', ''), 'sgp': sg.get('p'),
                      'sgl': sg.get('top'),
                      # harvested crops have no burned-in preview frame, so
                      # the lightbox opens on the ORIGINAL through /hq
                      'harvested': where.get(name) != CROPS,
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
    kept, seen_seq, collapsed = [], {}, 0
    # Pool-wide, not a fallback. The perceptual hash used to run only for a
    # crop whose sequence was unknown, so two crops from DIFFERENT sequences
    # showing the same animal in the same frame both survived -- which is how
    # near-identical crops ended up judged twice. Measured on the live pool:
    # 127 of 1,742 hashable survivors had a twin, every one of them from
    # another sequence.
    dup_seen = DupIndex()
    dup_judged = judged_dup_index()

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
        if sq and near(seen_seq, sq, ts):
            collapsed += 1
            continue
        # Hashed from the directory the crop is actually in. Joining CROPS
        # unconditionally silently returned None for every harvested crop --
        # 1,023 of 2,757, 37% of the pool -- so the hash check had no effect
        # on more than a third of the queue.
        h = _dhash(os.path.join(where.get(c['name'], CROPS), c['name']),
                   c['name'])
        if h is not None:
            if dup_judged.hit(h) or dup_seen.hit(h):
                collapsed += 1
                continue
            dup_seen.add(h)
        if sq:
            seen_seq.setdefault(sq, []).append(ts if ts is not None else 0.0)
            c['seq'] = sq
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
    # Counted on the same population the filter will select from, and after
    # the collapse, so the number on the option is the number it delivers.
    sg_offer = {'none': 0}
    for c in kept:
        k = c.get('sg') or 'none'
        sg_offer[k] = sg_offer.get(k, 0) + 1
    # What the queue would hold with nothing narrowed, for the caption's
    # "narrowed from" readout. Taken here, after the dedup and the sequence
    # collapse, because those are not filters -- they are what the queue IS.
    # Comparing against the raw pool would advertise a reduction the reviewer
    # never chose and could not undo.
    pool_unfiltered = len(kept)
    if want:
        kept = [c for c in kept if c['country'] == want]
    if want_sg:
        kept = [c for c in kept
                if (c.get('sg') or 'none') == want_sg]
    # The gate is its own axis, not a rival taxonomy. It answers the question
    # the REVIEWER is answering -- is this a dog -- where the guess filter
    # answers what kind of thing it is, so narrowing by one while reading the
    # other is the useful move and swapping between them is not. Same shape as
    # the leash axis, and for the same reason.
    gate_offer = {'all': len(kept), 'none': 0, 'dog': 0, 'not_dog': 0}
    for c in kept:
        k = (_gate.get(c['name']) or {}).get('b') or 'none'
        if k in gate_offer:
            gate_offer[k] += 1
    if want_gate and want_gate != 'all':
        kept = [c for c in kept
                if ((_gate.get(c['name']) or {}).get('b') or 'none')
                == want_gate]
    # after the collapse and the country/guess/gate filters, for the same
    # reason those are: the number an option advertises has to be the number it
    # hands back, and every filter above this one has already removed crops
    leash_offer = {k: len(_leash_keep(kept, k))
                   for k in ('all', 'none', 'leashed', 'unleashed')}
    if want_leash and want_leash != 'all':
        kept = _leash_keep(kept, want_leash)
    # Search orders the queue, it does not cut it. "Find me cats" means "put
    # the cat-looking ones first so I can work through them", and a hard cut
    # would also throw away the near-misses, which are exactly the crops worth
    # a human's eye.
    find_state, find_hits = 'off', 0
    if want_find:
        scores, why = search_scores(want_find)
        if scores is None:
            # A mismatch has to ask for an encode too, or it is terminal.
            # crop_search.add() re-encodes the whole vocabulary under whichever
            # model the CROP vectors carry, so this is exactly what clears it
            # -- and search_learn() is the only thing in the dashboard that
            # runs it. Gating the call on 'unknown' meant the one state that
            # cannot fix itself was the one state that never asked for help,
            # and free-text search stayed dead until somebody ran the tool by
            # hand. Which is the state this box was left in earlier today.
            if why in ('unknown', 'mismatch'):
                asked = search_learn(want_find)
                find_state = why if (why == 'mismatch' and asked == 'learning')\
                    else asked
            else:
                find_state = why
        else:
            scored = [(scores.get(c['name']), c) for c in kept]
            have = [(v, c) for v, c in scored if v is not None]
            miss = [c for v, c in scored if v is None]
            have.sort(key=lambda t: -t[0])
            for v, c in have:
                c['find'] = round(v, 4)
            kept = [c for _, c in have] + miss
            # 'on' only if it actually ordered something. A word encoded
            # against a vector store that covers none of this queue used to
            # report success and change nothing, which is indistinguishable
            # from a model returning nonsense -- and is what it looked like.
            find_state, find_hits = ('on' if have else 'cold'), len(have)
    items = kept
    total = len(items)
    pages = max(1, -(-total // size))
    page = min(page, pages - 1)
    lo = page * size
    shown = [] if count_only else items[lo:lo + size]
    ahead = [] if count_only else items[lo + size:lo + 2 * size]
    if not count_only:
        warm_hq([c['name'] for c in items[lo:lo + 2 * size]])
    return {'items': shown, 'reserve': ahead,
            'page': page, 'size': size, 'sort': sort, 'total_unflagged': total,
            # images judged, not crop FILES judged -- so it is the same unit as
            # total_unflagged and the two sum to a meaningful denominator
            'pages': pages, 'flagged_total': len(flagged_ids),
            # never a silent cap: the page says how many it is holding back
            'too_small': too_small, 'min_px': floor,
            # what the model guessed, and how many of each are in the queue.
            # 'suggest_ready' is how the page knows whether to offer the
            # control at all -- an empty file means nobody has run the tool.
            'suggest': want_sg, 'suggest_counts': sg_offer,
            'suggest_ready': bool(_tri),
            # what the queue holds before any filter, so the caption can
            # say what it was narrowed from
            'pool_unfiltered': pool_unfiltered,
            # which guesser's opinions the filter above is showing, and what
            # that guesser is able to say. The dog-bin gate answers dog or
            # not-dog and nothing else, so offering it 'Other animal' would be
            # offering an option that can only ever return nothing.
            'backend': want_backend,
            'buckets': [{'key': b, 'label': BUCKET_LABELS.get(b, b)}
                        for b in (BACKEND_INFO.get(want_backend, {})
                                  .get('buckets') or OPEN_BUCKETS)],
            # The gate's own axis. Offered only once it has verdicts covering
            # this queue -- an empty dropdown that filters nothing is worse
            # than no dropdown, the same rule the guess filter follows.
            'gate': want_gate, 'gate_counts': gate_offer,
            'gate_ready': bool(_gate),
            'gate_label': BACKEND_INFO.get('dogbin', {}).get('label')
                          or 'Dog-bin gate',
            # crops only: the set's manifest sits in the same directory
            'harvested_available': sum(1 for n, d in pooled
                                       if d != CROPS and _CROP_RE.match(n)),
            'positive_total': n_pos, 'seen_total': len(seen_ids),
            'collapsed': collapsed,
            # leash verdicts for the crops on THIS page, so a button can show
            # what was already decided. Its own axis and its own store: a crop
            # can be a dog and unjudged for leash, or the reverse.
            'leash': _leash_for([c['name'] for c in shown + ahead]),
            'leash_totals': _leash_counts(),
            'leash_filter': want_leash, 'leash_counts': leash_offer,
            'find': want_find, 'find_state': find_state,
            'find_hits': find_hits, 'find_terms': search_ready(),
            # searchable / in the pool, so a search that can't work says why
            'find_cover': search_coverage(pooled),
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
            # that says whether the filter is usable at all -- shown on the
            # country <select>'s own tooltip, because for a long time it was
            # computed here and rendered nowhere, and a crop directory the
            # index never walked cost the filter 1,593 crops in silence.
            'country_coverage': coverage}


# Children this server started, kept only so they can be reaped. A Popen whose
# object is dropped is never waited on, so when the child exits it stays in the
# table as a zombie -- and a zombie still has a /proc entry, so os.kill(pid, 0)
# reports it alive. That is how a finished guesser went on claiming to be
# "Guessing crops" with a Pause button underneath it.
_SPAWNED = []
# Guards the check-then-spawn in triage_control, and _SPAWNED with it. This is
# a ThreadingHTTPServer: two tabs pressing Run land in two threads, both see no
# process running, and both start one. Measured -- two simultaneous POSTs both
# answered "guessing started", and two guessers then worked the same queue and
# appended to the same file.
_SPAWN_LOCK = threading.Lock()


def _reap():
    """Collect any spawned child that has exited. Cheap and idempotent."""
    for proc in list(_SPAWNED):
        try:
            if proc.poll() is not None:
                _SPAWNED.remove(proc)
        except Exception:
            _SPAWNED.remove(proc)


def _script_pids(script, *need_args):
    """PIDs of processes that ARE `python .../<script>`, with those arguments.

    Matched on argv structure, not on the command line as one string. The
    substring version matched any process whose command line merely CONTAINED
    the script's name and the word python -- which is true of every shell
    running a command that mentions it, including `pgrep -f triage_crops.py`
    typed with a full interpreter path. Harmless while only reading, and not
    harmless at all one function later, where the answer is handed to SIGTERM.

    argv[0] must be a python, and some later argument must BE the script --
    a `python -c` whose code quotes the filename carries it inside one large
    argument and no longer matches.
    """
    out = []
    try:
        listing = os.listdir('/proc')
    except OSError:
        return out
    for d in listing:
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/cmdline', 'rb') as f:
                argv = [a for a in f.read().decode('utf-8', 'replace').split('\0')
                        if a]
        except OSError:
            continue
        if len(argv) < 2 or 'python' not in os.path.basename(argv[0]):
            continue
        if not any(a == script or a.endswith('/' + script) for a in argv[1:]):
            continue
        if any(w not in argv[1:] for w in need_args):
            continue
        out.append(int(d))
    return out


def sweep_pids():
    """PIDs of running sweep processes (never matches this server)."""
    return _script_pids('sweep.py', 'run')


def triage_pids():
    """PIDs running the crop guesser.

    Found by scanning /proc rather than read from the status file: a killed run
    never gets to clear its own `running: true`, so that file's pid outlives it
    and can be recycled onto something else entirely. Signalling a pid because
    a stale JSON named it is how you kill an unrelated process.
    """
    return _script_pids('triage_crops.py')


def triage_control(action, backend='siglip'):
    """stop = SIGTERM; start = relaunch detached in --watch mode.

    Stopping loses nothing: the guesser appends each batch to triage.jsonl as
    it goes, and a fresh run skips crops already in there.

    Serialised: deciding whether one is running and starting one has to be a
    single step, or two clicks race into two guessers.
    """
    with _SPAWN_LOCK:
        return _triage_control(action, backend)


def _running_backend():
    """Which guesser the live status file belongs to, or None if none is."""
    try:
        with open(TRIAGE_STATUS) as fh:
            doc = json.load(fh) or {}
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    # the run says so itself when it can; backend_of(model) is the fallback
    # for a status file written before the field existed
    return doc.get('backend') or backend_of(doc.get('model'))


def _triage_control(action, backend='siglip'):
    _reap()
    backend = pick_backend(backend)
    pids = triage_pids()
    if action == 'stop':
        if not pids:
            return {'ok': True, 'running': False, 'msg': 'already stopped'}
        # Stop MY guesser, never the other one. SIGTERM went to every
        # triage_crops.py alive whatever backend it was running, and the
        # button reads Pause for a moment after the dropdown moves, before
        # the next poll corrects it -- so a fast click could end the other
        # guesser's run with nothing on screen saying it had.
        other = _running_backend()
        if other and other != backend:
            return {'ok': False, 'running': False,
                    'msg': f'that is the '
                           f'{BACKEND_INFO.get(other, {}).get("label") or other}'
                           f' run — switch to it to stop it'}
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        return {'ok': True, 'running': False,
                'msg': 'stopping — guesses already written are kept'}
    if action == 'start':
        if pids:
            # Which guesser is already on the card. 'already running' under a
            # dropdown set to the other one reads as a bug in the button.
            other = _running_backend()
            if other and other != backend:
                return {'ok': False, 'running': False,
                        'msg': f'the {BACKEND_INFO.get(other, {}).get("label") or other} '
                               f'guesser is running — they share the card, so '
                               f'pause it first'}
            return {'ok': True, 'running': True, 'msg': 'already running'}
        # Each backend has its own interpreter, because they cannot share one:
        # RF-DETR needs transformers>=5 and SigLIP 2 is loaded by the 4.x the
        # rest of this pipeline runs on. Refusing here beats launching a
        # process that dies on an import.
        if backend not in backends_available():
            return {'ok': False, 'running': False,
                    'msg': f'no interpreter configured for the {backend} '
                           f'guesser (set {backend}_python in the dashboard '
                           f'config)' if backend != 'siglip' else
                           'no interpreter configured for the guesser'}
        py = {'rfdetr': RFDETR_PYTHON,
              'dogbin': DOGBIN_PYTHON}.get(backend) or TRIAGE_PYTHON
        script = os.path.join(REPO, 'tools', 'detect', 'triage_crops.py')
        if not os.path.exists(script):
            return {'ok': False, 'running': False, 'msg': 'triage_crops.py is missing'}
        logp = os.path.join(REPO, 'data', 'triage_run.log')
        try:
            log = open(logp, 'a')
            argv = [py, script, '--watch', str(TRIAGE_WATCH)]
            model = {'rfdetr': RFDETR_MODEL,
                     'dogbin': 'dogbin'}.get(backend) or TRIAGE_MODEL
            if model:
                argv += ['--model', model]
            if backend == 'dogbin':
                argv += ['--weights', dogbin_weights()]
            if TRIAGE_DEVICE:
                argv += ['--device', TRIAGE_DEVICE]
            proc = subprocess.Popen(
                argv,
                cwd=REPO, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                start_new_session=True)
            _SPAWNED.append(proc)
        except Exception as e:
            return {'ok': False, 'running': False, 'msg': str(e)}
        # The usual failure is a python without torch or transformers, and it
        # dies on the import -- instantly, and after this call has already
        # returned "started". Wait long enough to catch that and hand back the
        # reason instead of a strip that says "not running" for no visible
        # cause.
        try:
            code = proc.wait(timeout=2.5)
            if proc in _SPAWNED:
                _SPAWNED.remove(proc)      # wait() already reaped it
        except subprocess.TimeoutExpired:
            # Stamp the status file as ours before returning. The guesser
            # rewrites it, but only once the model is loaded, which is tens of
            # seconds -- and until then the file still describes the PREVIOUS
            # run. A stale `updated` next to a process that is now alive is
            # exactly the shape of "stalled", so the strip announced "Run
            # stopped" the moment you pressed Run.
            try:
                tmp = TRIAGE_STATUS + '.tmp'
                now = time.time()
                with open(tmp, 'w') as fh:
                    # Whose run this is, said outright. Everything that asks
                    # reads it, and a stamp that omitted it fell back to
                    # backend_of(model) over an absent model -- which answers
                    # 'siglip', so every freshly started RF-DETR run was
                    # attributed to the other guesser for the tens of seconds
                    # its model takes to load.
                    json.dump({'running': True, 'starting': True,
                               'backend': backend, 'model': model,
                               'pid': proc.pid, 'started': now,
                               'updated': now, 'done': 0, 'total': 0,
                               'watch': TRIAGE_WATCH, 'schema': 1}, fh)
                os.replace(tmp, TRIAGE_STATUS)
            except OSError:
                pass
            return {'ok': True, 'running': True, 'msg': 'guessing started'}
        tail = ''
        try:
            with open(logp) as fh:
                lines = [x.strip() for x in fh.readlines() if x.strip()]
            tail = lines[-1] if lines else ''
        except OSError:
            pass
        hint = ('' if 'triage_python' in load_cfg() or
                os.environ.get('TRIAGE_PYTHON') else
                ' Set "triage_python" in dashboard.config.json to an '
                'interpreter that has torch and transformers.')
        return {'ok': False, 'running': False,
                'msg': f'exited immediately (code {code}). {tail}{hint}'.strip()}
    return {'ok': False, 'msg': 'unknown action'}


def gate_pids(stage='gate'):
    """PIDs running gate_store.py for THIS stage.

    Both stages are the same script with the same subcommand, so matching on
    those alone would have shown the gate's twelve-hour run as the leash
    model's -- and offered a Stop button that killed it. The stage is read off
    the argv the process was actually started with; an absent --stage is the
    gate, which is the default the flag carries.
    """
    out = []
    for pid in _script_pids('gate_store.py', 'run'):
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                argv = [a for a in
                        f.read().decode('utf-8', 'replace').split('\0') if a]
        except OSError:
            continue
        got = 'gate'
        for i, a in enumerate(argv):
            if a == '--stage' and i + 1 < len(argv):
                got = argv[i + 1]
            elif a.startswith('--stage='):
                got = a.split('=', 1)[1]
        if got == stage:
            out.append(pid)
    return out


def _gate_plan(stage, sp):
    """Spawn the planner for a stage that has none, and say so."""
    script = os.path.join(REPO, 'tools', 'detect', 'gate_store.py')
    if _script_pids('gate_store.py', 'plan'):
        return {'ok': True, 'running': False,
                'msg': f'planning the {sp["title"]} — a few minutes over the '
                       f'whole store; the Run button appears when it lands'}
    try:
        import duckdb                                       # noqa: F401
    except ImportError:
        return {'ok': False, 'running': False,
                'msg': f'no plan yet, and this interpreter has no duckdb to '
                       f'build one — run `gate_store.py plan --stage {stage}`'}
    try:
        log = open(os.path.join(REPO, 'data', f'{sp["dir"]}_run.log'), 'a')
        env = dict(os.environ)
        env['TRAINING_ROOT'] = training_root()
        proc = subprocess.Popen(
            [sys.executable, script, 'plan', '--stage', stage],
            cwd=REPO, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            start_new_session=True, env=env)
        _SPAWNED.append(proc)
    except Exception as e:
        return {'ok': False, 'running': False, 'msg': str(e)}
    try:
        # the planner refuses an unfinished upstream, and it does so instantly
        code = proc.wait(timeout=2.5)
        if proc in _SPAWNED:
            _SPAWNED.remove(proc)
        tail = ''
        try:
            with open(os.path.join(REPO, 'data', f'{sp["dir"]}_run.log')) as fh:
                lines = [x.strip() for x in fh if x.strip()]
            tail = lines[-1] if lines else ''
        except OSError:
            pass
        # An exit inside the window is not a failure -- it is a planner that
        # finished. The leash plan is a 1-second query over the gate's own
        # shards, so it is USUALLY done before this wait is, and reporting a
        # clean exit as "planning failed (code 0)" told the reader the
        # opposite of what had just happened.
        if code == 0:
            return {'ok': True, 'running': False,
                    'msg': f'{sp["title"]} planned. ' + tail[:200]}
        return {'ok': False, 'running': False,
                'msg': f'planning failed (code {code}). ' + tail[:200]}
    except subprocess.TimeoutExpired:
        pass
    return {'ok': True, 'running': False,
            'msg': f'planning the {sp["title"]} — a few minutes over the '
                   f'whole store; the Run button appears when it lands'}


_AUDIT = {'mod': None, 'tried': False}


def _audit():
    """The audit module, imported once and only when asked for.

    It pulls in duckdb and the pool, neither of which a dashboard that nobody
    has opened /audit on should pay for -- and a missing pool is a state the
    page reports rather than an import error at boot.
    """
    if not _AUDIT['tried']:
        _AUDIT['tried'] = True
        try:
            sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
            import audit as _a
            _AUDIT['mod'] = _a
        except Exception:
            _AUDIT['mod'] = None
    # Readiness is per stage and belongs to the caller. Answering None unless
    # the GATE pool existed made data/fn_audit/pool.parquet a hard dependency
    # of the leash routes too: with only the leash pool built, /audit/leash
    # fell through to the static handler as a 404 and its endpoints reported
    # judged:0 rather than naming the pool that was actually missing.
    return _AUDIT['mod']


_DATASETS = {'mod': None, 'tried': False}


def _datasets():
    """The datasets module, imported once and only when asked for.

    It reaches back into this one -- what turns a directory into a dataset is
    the run list, and that is training_runs() -- so an import at module scope
    here is a cycle that breaks the dashboard at boot. On the first request
    this module is already built and the cycle closes harmlessly. Missing, it
    is a page naming the two files rather than a 500.
    """
    if not _DATASETS['tried']:
        _DATASETS['tried'] = True
        try:
            sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
            import datasets as _d
            _DATASETS['mod'] = _d
        except Exception:
            _DATASETS['mod'] = None
    return _DATASETS['mod']


_LLM = {'mod': None, 'tried': False}


def _llm():
    """The experimental LLM annotator page, imported once and only when asked.

    It pulls in tools/detect/llm_annotate.py at its own module scope, and that
    module owns a store nothing else in this dashboard reads or should read.
    Lazy for the same reason the two above are: a checkout missing either file
    is a page naming both rather than a dashboard that will not boot, and
    nobody who never opens /llm pays for the import.
    """
    if not _LLM['tried']:
        _LLM['tried'] = True
        try:
            sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
            import llm_page as _l
            _LLM['mod'] = _l
        except Exception:
            _LLM['mod'] = None
    return _LLM['mod']


def gate_planning():
    """True while a planner is building a work list."""
    return bool(_script_pids('gate_store.py', 'plan'))


def gate_control(action, stage='gate'):
    """stop = SIGTERM (the shard in flight is lost, the written ones are not);
    start = relaunch detached, which resumes at the first unwritten shard."""
    if stage not in GATE_STAGES:
        return {'ok': False, 'msg': f'unknown stage {stage}'}
    sp = GATE_STAGES[stage]
    with _SPAWN_LOCK:
        _reap()
        pids = gate_pids(stage)
        if action == 'stop':
            if not pids:
                return {'ok': True, 'running': False, 'msg': 'already stopped'}
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
            return {'ok': True, 'running': False,
                    'msg': 'stopping — finished shards are kept'}
        if action != 'start':
            return {'ok': False, 'msg': 'unknown action'}
        if pids:
            return {'ok': True, 'running': True, 'msg': 'already running'}
        py = DOGBIN_PYTHON
        script = os.path.join(REPO, 'tools', 'detect', 'gate_store.py')
        if not py:
            return {'ok': False, 'running': False,
                    'msg': f'no interpreter for the {sp["title"]} (set '
                           f'dogbin_python in the dashboard config)'}
        if not os.path.exists(script):
            return {'ok': False, 'running': False,
                    'msg': 'tools/detect/gate_store.py is missing'}
        # A stage that reads another's output cannot start before that one is
        # done, and saying so beats a button that fails.
        up = gate_upstream(stage)
        if up and not up['ready']:
            return {'ok': False, 'running': False,
                    'msg': f'the {up["title"]} has judged '
                           f'{up["rows"]:,} of {up["total"]:,} boxes — the '
                           f'{sp["title"]} reads its verdicts and cannot be '
                           f'planned until it finishes'}
        if not os.path.exists(os.path.join(gate_dir(stage), 'work.parquet')):
            # Plan it here rather than sending someone to a terminal. The
            # planner needs duckdb, and THIS interpreter has it -- it is the
            # only one that can both parse this file and open the catalog, so
            # the "run it where duckdb lives" instruction was pointing at the
            # process reading it. Detached, because a 4.8M-row join is minutes
            # and a request handler is not the place to spend them.
            return _gate_plan(stage, sp)
        try:
            log = open(os.path.join(REPO, 'data',
                                    f'{sp["dir"]}_run.log'), 'a')
            env = dict(os.environ)
            # the runner resolves the promoted weights against this
            env['TRAINING_ROOT'] = training_root()
            proc = subprocess.Popen(
                [py, script, 'run', '--stage', stage],
                cwd=REPO, stdout=log, stderr=log,
                stdin=subprocess.DEVNULL, start_new_session=True, env=env)
            _SPAWNED.append(proc)
        except Exception as e:
            return {'ok': False, 'running': False, 'msg': str(e)}
        # the usual failure is an interpreter without ultralytics, which dies
        # on the import in under a second and after this call has returned
        try:
            code = proc.wait(timeout=2.5)
            if proc in _SPAWNED:
                _SPAWNED.remove(proc)
            tail = ''
            try:
                with open(os.path.join(REPO, 'data',
                                       f'{sp["dir"]}_run.log')) as fh:
                    lines = [x.strip() for x in fh if x.strip()]
                tail = lines[-1] if lines else ''
            except OSError:
                pass
            # A run that ends this fast has done nothing -- but say WHICH
            # kind of nothing: a clean exit means there was no work left,
            # not that it fell over.
            if code == 0:
                return {'ok': True, 'running': False,
                        'msg': f'the {sp["title"]} finished immediately — '
                               f'nothing left to judge. ' + tail[:160]}
            return {'ok': False, 'running': False,
                    'msg': f'the {sp["title"]} exited immediately '
                           f'(code {code}). ' + tail[:160]}
        except subprocess.TimeoutExpired:
            pass
        return {'ok': True, 'running': True, 'msg': f'{sp["title"]} started'}


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
            _SPAWNED.append(subprocess.Popen(
                [py, os.path.join(REPO, 'tools', 'detect', 'sweep.py'),
                 'run', '--gen', '1'],
                cwd=REPO, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                start_new_session=True))
        except Exception as e:
            return {'ok': False, 'running': False, 'msg': str(e)}
        return {'ok': True, 'running': True, 'msg': 'resuming'}
    return {'ok': False, 'msg': 'unknown action'}


# The tab icon, served from memory at /favicon.ico. One dog, drawn by the
# system's emoji font.
FAVICON_SVG = (b"<svg xmlns='http://www.w3.org/2000/svg' "
               b"viewBox='0 0 100 100'>"
               b"<text y='0.9em' font-size='90'>\xf0\x9f\x90\x95</text></svg>")

# What the static fallback may serve out of OUT, which doubles as the server's
# working directory. Everything the five pages actually fetch by a bare path
# is named here; serve.log, history.duckdb, triage.jsonl and the rest of the
# working files are the server's own and stay unreachable.
STATIC_FILES = frozenset({'/', '/index.html', '/echarts.min.js',
                          '/world.json', '/map_points.json',
                          '/map_points_fine.json'})
STATIC_DIRS = ('/recent_crops/', '/review_set/')


def _static_allowed(path):
    return path in STATIC_FILES or any(
        path.startswith(d) for d in STATIC_DIRS)


class BoardHandler(SimpleHTTPRequestHandler):
    """Serve the static dashboard plus a tiny JSON board API."""

    db = 'data/catalog.duckdb'
    # Documents whose markup changes whenever this file does. Served with
    # only a Last-Modified, a browser is free to reuse them without asking
    # -- HTTP lets it guess a lifetime from the age of the file -- so a
    # rebuilt page kept showing the previous build until a hard reload.
    # Every visible change here landed on disk and stayed invisible.
    # no-cache is not no-store: the copy is kept, it just has to be
    # revalidated, so an unchanged page still answers 304.
    _NO_CACHE_PATHS = ('/', '/index.html', '/review')
    # The built page carries the training section between these, so the server
    # can swap in a fresh render without parsing HTML -- the section nests
    # divs, so "up to the next </div>" would cut it in the wrong place.
    _TRK_OPEN = b'<!--TRK-->'
    _TRK_CLOSE = b'<!--/TRK-->'

    def end_headers(self):
        try:
            if self.path.split('?', 1)[0] in self._NO_CACHE_PATHS:
                self.send_header('Cache-Control', 'no-cache, must-revalidate')
        except Exception:
            pass          # a header is never worth failing a response over
        SimpleHTTPRequestHandler.end_headers(self)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')  # live data (§7.2)
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache, must-revalidate')
        self.end_headers()
        self.wfile.write(body)

    def _audit_stage(self, q=None):
        """Which audit a request is for. Unknown names are the default, never
        a directory: the stage reaches a path."""
        a = _audit()
        if a is None:
            return None
        p = self.path.split('?', 1)[0]
        # /audit/<stage>, and /audit/<anything>/<stage>/... -- named rather
        # than listed, because the list was crop/ alone and the day a frame/
        # route appeared it silently answered for the gate on every stage.
        parts = [x for x in p.split('/') if x]
        for name in a.STAGES:
            if p == f'/audit/{name}':
                return name
            if len(parts) >= 3 and parts[0] == 'audit' and parts[2] == name:
                return name
        v = (q or {}).get('stage', [None])[0]
        return v if v in a.STAGES else a.DEFAULT_STAGE

    def _audit_get(self):
        """The audit pages, their crops and their two read endpoints."""
        path = self.path.split('?', 1)[0]
        a = _audit()
        pages = ['/audit'] + ([f'/audit/{k}' for k in a.STAGES] if a else [])
        if path in pages:
            if a is None:
                self._html(b'<!doctype html><meta charset=utf-8>'
                           b'<body style="background:#13151a;color:#98a2ad;'
                           b'font:14px system-ui;padding:40px">'
                           b'The audit module would not load. Check '
                           b'<code>tools/dashboard/audit.py</code> and '
                           b'<code>tools/detect/fn_audit.py</code>.</body>')
                return True
            stage = self._audit_stage()
            if not a.pool_ready(stage):
                self._html(
                    ('<!doctype html><meta charset=utf-8>'
                     '<body style="background:#13151a;color:#98a2ad;'
                     'font:14px system-ui;padding:40px;line-height:1.6">'
                     f'No audit pool for the {a.STAGES[stage]["title"]} yet.'
                     '<br>Build one from whatever it has judged so far:<br><br>'
                     f'<code>python tools/detect/fn_audit.py build '
                     f'--stage {stage}</code></body>').encode('utf-8'))
                return True
            self._html(a.page_html(stage).encode('utf-8'))
            return True
        if path.startswith('/audit/crop/') and path.endswith('.jpg'):
            stage = self._audit_stage()
            rest = path[len('/audit/crop/'):-4]
            if a and rest.startswith(stage + '/'):
                rest = rest[len(stage) + 1:]
            # the key is matched against the shape it is minted in; nothing a
            # client sends is ever joined onto a directory
            p = a.crop_path(rest, stage) if a else None
            if not p:
                self.send_error(404)
                return True
            try:
                with open(p, 'rb') as fh:
                    body = fh.read()
            except OSError:
                self.send_error(404)
                return True
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'public, max-age=86400')
            self.end_headers()
            self.wfile.write(body)
            return True
        if path == '/api/audit/page':
            q = parse_qs(urlparse(self.path).query)
            stage = self._audit_stage(q)
            if a is None or not a.pool_ready(stage):
                self._json({'error': 'pool not built'})
                return True
            try:
                i = int((q.get('i', ['-1'])[0]))
            except ValueError:
                i = -1
            self._json(a.api_page(i, n=a.page_size(q.get('n', [None])[0]),
                                  band=a.band_arg(q.get('band', [None])[0]),
                                  stage=stage))
            return True
        if path.startswith('/audit/frame/') and path.endswith('.jpg'):
            stage = self._audit_stage()
            rest = path[len('/audit/frame/'):-4]
            if a and rest.startswith(stage + '/'):
                rest = rest[len(stage) + 1:]
            body, meta = (a.frame_view(rest, stage) if a else (None, None))
            if not body:
                self.send_error(404)
                return True
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            # The geometry rides along with the picture. Asking for it
            # separately meant opening the same 8000x4000 frame twice for one
            # click -- a quarter of a second of decoding to learn four numbers
            # that the decode had already produced.
            self.send_header('X-Audit-Meta', json.dumps(meta))
            self.send_header('Access-Control-Expose-Headers', 'X-Audit-Meta')
            self.end_headers()
            self.wfile.write(body)
            return True
        if path == '/api/audit/judged':
            q = parse_qs(urlparse(self.path).query)
            stage = self._audit_stage(q)
            if a is None or not a.pool_ready(stage):
                self._json({'items': [], 'total': 0, 'pages': 1, 'page': 0})
                return True
            which = q.get('which', ['all'])[0]
            if which not in a.judged_views(stage):
                which = 'all'
            try:
                pg = int(q.get('page', ['0'])[0])
            except ValueError:
                pg = 0
            self._json(a.judged(stage, which, pg,
                                a.page_size(q.get('n', [None])[0])))
            return True
        if path == '/api/audit/stats':
            q = parse_qs(urlparse(self.path).query)
            stage = self._audit_stage(q)
            self._json(a.stats(stage) if a and a.pool_ready(stage)
                       else {'judged': 0, 'bands': []})
            return True
        return False

    def _datasets_get(self):
        """The datasets page, its three listings and its two pictures.

        `key` and `rel` go through verbatim. They are resolved in there,
        against a realpathed root that refuses anything landing outside it,
        and a second opinion formed here would only be somewhere for the two
        to disagree -- which is how a traversal gets through.
        """
        path = self.path.split('?', 1)[0]
        d = _datasets()
        if path == '/datasets':
            if d is None:
                self._html(b'<!doctype html><meta charset=utf-8>'
                           b'<body style="background:#13151a;color:#98a2ad;'
                           b'font:14px system-ui;padding:40px">'
                           b'The datasets module would not load. Check '
                           b'<code>tools/dashboard/datasets.py</code> and '
                           b'<code>tools/detect/dataset_index.py</code>.'
                           b'</body>')
                return True
            self._html(d.page_html().encode('utf-8'))
            return True
        if d is None:
            return False       # the page above already said which file is out
        q = parse_qs(urlparse(self.path).query)
        if path == '/api/datasets':
            # refresh is the page's rescan button, and only that: it forces a
            # live re-walk of every root, which is seconds on a cold cache.
            self._json(d.api_list(q.get('refresh', ['0'])[0]
                                  in ('1', 'true')))
            return True
        if path == '/api/datasets/tree':
            self._json(d.api_tree(q.get('key', [''])[0]))
            return True
        if path == '/api/datasets/files':
            try:
                pg = int(q.get('page', ['0'])[0])
            except ValueError:
                pg = 0
            self._json(d.api_files(q.get('key', [''])[0],
                                   q.get('rel', [''])[0], pg,
                                   d.page_size(q.get('n', [None])[0])))
            return True
        if path == '/datasets/thumb':
            body, ctype = d.thumb(q.get('key', [''])[0],
                                  q.get('rel', [''])[0])
            if not body:
                # a refused path and a file deleted mid-walk answer the same
                # thing, and the client draws the same broken tile for both
                self.send_error(404)
                return True
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            # Same day the review page's crops get. `v` on the URL is the
            # source file's mtime and it is aimed at THIS cache: a picture
            # replaced at the same path is a new URL, so nothing here has to
            # read the parameter -- or trust it.
            self.send_header('Cache-Control', 'private, max-age=86400')
            self.end_headers()
            self.wfile.write(body)
            return True
        if path == '/datasets/image':
            body, ctype = d.full(q.get('key', [''])[0], q.get('rel', [''])[0])
            if not body:
                self.send_error(404)
                return True
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'private, max-age=300')
            self.end_headers()
            self.wfile.write(body)
            return True
        return False

    def _llm_get(self):
        """The experimental LLM annotator page, its four reads and its crops.

        Everything is answered by llm_page, including the words: what the model
        said is spelled with the store's own tokens and this handler never
        translates one into 'dog' or into a verdict. It is a third tier below a
        person's answer and below a model's score, and a route here that
        rephrased it would be the first place the three got mixed up.

        `source` and `key` go through verbatim. The source is checked against
        llm_annotate.SOURCES over there and the key is LOOKED UP in that pool's
        dictionary, so nothing a client sends is joined onto a directory --
        and a second opinion formed here would only be somewhere for the two to
        disagree, which is how a traversal gets through.
        """
        path = self.path.split('?', 1)[0]
        lp = _llm()
        if path == '/llm':
            if lp is None:
                self._html(b'<!doctype html><meta charset=utf-8>'
                           b'<body style="background:#13151a;color:#98a2ad;'
                           b'font:14px system-ui;padding:40px">'
                           b'The LLM annotator module would not load. Check '
                           b'<code>tools/dashboard/llm_page.py</code> and '
                           b'<code>tools/detect/llm_annotate.py</code>.'
                           b'</body>')
                return True
            self._html(lp.page_html().encode('utf-8'))
            return True
        if lp is None:
            return False       # the page above already said which file is out
        q = parse_qs(urlparse(self.path).query)
        if path == '/api/llm':
            self._json(lp.api_overview())
            return True
        if path == '/api/llm/status':
            self._json(lp.api_status())
            return True
        if path == '/api/llm/disagreements':
            # An absent limit is passed on as it arrives: the module answers
            # its own ceiling to anything that is not a number, and a default
            # spelled again here is the copy that drifts from GRID_MAX.
            self._json(lp.api_disagreements(
                q.get('source', [''])[0] or None,
                q.get('direction', [''])[0] or None,
                q.get('limit', [None])[0]))
            return True
        if path == '/api/llm/unparsed':
            self._json(lp.api_unparsed(q.get('source', [''])[0] or None,
                                       q.get('limit', [None])[0]))
            return True
        if path == '/llm/crop':
            body, ctype = lp.crop(q.get('source', [''])[0],
                                  q.get('key', [''])[0])
            if not body:
                # a refused pool, a key that has left it and a crop
                # deduplicated off the disk are one answer here, and the tile
                # draws the same missing picture for all three
                self.send_error(404)
                return True
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            # Private, like the review and datasets pictures: these are crops
            # off someone's harvest, not something a proxy should hold.
            self.send_header('Cache-Control', 'private, max-age=86400')
            self.end_headers()
            self.wfile.write(body)
            return True
        return False

    def _llm_post(self):
        """Start and stop a batch, and nothing else.

        These are the only two routes on this page that are not a read, and
        neither writes an annotation: they signal llm_annotate, which appends
        to its own ledger and refuses any path outside its own store. There is
        no route here that promotes a verdict into a dataset, not even a
        refused one -- that decision is a person's, by hand, later.
        """
        path = self.path.split('?', 1)[0]
        if path not in ('/api/llm/run', '/api/llm/stop'):
            return False
        lp = _llm()
        if lp is None:
            self._json({'ok': False,
                        'msg': 'the LLM annotator module would not load — '
                               'check tools/dashboard/llm_page.py and '
                               'tools/detect/llm_annotate.py'})
            return True
        if path == '/api/llm/stop':
            self._json(lp.api_stop())
            return True
        try:
            n = int(self.headers.get('Content-Length', 0) or 0)
            data = json.loads(self.rfile.read(n) or b'{}')
            if not isinstance(data, dict):
                raise ValueError('body is not an object')
            # Nothing in the body reaches a path, a prompt or a command line.
            # The pool is one of four names, matched over there; the size is
            # clamped to one the interface offers, here, the way the audit and
            # datasets pages clamp a page size at this boundary.
            args = {'n': lp.run_size(data.get('n'))}
            # A body that never named a pool gets the module's default rather
            # than the empty string, which api_run answers as "unknown pool
            # ''" -- a message about a bug in the page, for a request that
            # simply left the choice open.
            if data.get('source'):
                args['source'] = str(data['source'])
            self._json(lp.api_run(**args))
        except Exception as e:
            self._json({'ok': False, 'msg': str(e)})
        return True

    def do_GET(self):
        # split('?') so cache-busting query strings still match (§7.2 —
        # /api/board's == match 404s on ?t=1 and that bit us before). The two
        # routes the comment is ABOUT were the two it was never applied to.
        if self.path.split('?', 1)[0] == '/api/board':
            try:
                self._json(board_payload())
            except Exception as e:
                self._json({'error': str(e)}, 500)
            return
        if self.path.split('?', 1)[0] == '/api/refresh':
            self._json(_refresh)
            return
        if self.path.split('?', 1)[0] == '/api/detect':
            try:
                self._json(detect_payload())
            except Exception:
                # 404-safe by construction, and honest about it: this says
                # nothing is known, not that the sweep is idle
                self._json({'running': False, 'ever': False})
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
                                          str(q.get('country', '')),
                                          str(q.get('suggest', '')),
                                          str(q.get('leash', '')),
                                          str(q.get('find', '')),
                                          str(q.get('backend', 'siglip')),
                                          str(q.get('gate', ''))))
            except Exception as e:
                self._json({'items': [], 'error': str(e)})
            return
        _p = self.path.split('?', 1)[0]
        if _p.startswith('/audit') or _p.startswith('/api/audit'):
            if self._audit_get():
                return
        # The /datasets prefix carries its two picture routes as well, and all
        # of it has to be claimed before the static handler below, which is
        # what turned /audit/leash into a 404 the day it was added.
        if _p.startswith('/datasets') or _p.startswith('/api/datasets'):
            if self._datasets_get():
                return
        # /llm carries its crop route under the same prefix, so both are
        # claimed here rather than beside the other JSON endpoints below --
        # anything left to the static handler is a 404 that looks like a
        # missing page instead of an unclaimed route.
        if LLM_PAGE and (_p.startswith('/llm')
                         or _p.startswith('/api/llm')):
            if self._llm_get():
                return
        if self.path.split('?', 1)[0] == '/review':
            body = REVIEW_HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split('?', 1)[0] == '/api/review/annotated':
            try:
                q = parse_qs(urlparse(self.path).query)
                self._json(annotated_payload(
                    int(q.get('page', [0])[0]),
                    int(q.get('size', [REVIEW_PAGE])[0]),
                    str(q.get('label', ['all'])[0]),
                    str(q.get('sort', ['recent'])[0]),
                    str(q.get('leash', [''])[0])))
            except Exception as e:
                self._json({'items': [], 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/flagged':
            # The copy taken when the crop was flagged. The client tries /hq
            # first (cut from the ORIGINAL), and falls back here for an image
            # the store can no longer resolve -- which is the whole reason the
            # copy is made at flag time.
            q = parse_qs(urlparse(self.path).query)
            nm = q.get('name', [''])[0]
            lb = q.get('label', [''])[0]
            if not _CROP_RE.match(nm or '') or lb not in FLAG_LABELS:
                self.send_error(404)
                return
            try:
                with open(os.path.join(_store_for(lb)['crops'], nm), 'rb') as f:
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
        if self.path.split('?', 1)[0] == '/api/review/count':
            # Just the queue depth. review_payload does one listdir over a
            # pool capped at 3000 plus the ledgers, so asking for size=0 is
            # the whole computation without serialising any crop.
            try:
                self._json({'left': review_payload(0, 0)['total_unflagged']})
            except Exception as e:
                self._json({'left': None, 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/training/wrong':
            q = parse_qs(urlparse(self.path).query)
            body = mistake_bytes(q.get('key', [''])[0], q.get('i', [''])[0])
            if not body:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(body)))
            # immutable: a val crop does not change under a finished run
            self.send_header('Cache-Control', 'public, max-age=86400')
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split('?', 1)[0] == '/api/training/diff':
            try:
                q = parse_qs(urlparse(self.path).query)
                self._json({'html': render_run_diff(q.get('a', [''])[0],
                                                    q.get('b', [''])[0])})
            except Exception as e:
                self._json({'html': '', 'error': str(e)})
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
        if self.path.split('?', 1)[0] == '/api/sys':
            try:
                self._json(sys_stats())
            except Exception as e:
                self._json({'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/gate':
            try:
                q = parse_qs(urlparse(self.path).query)
                # the name is checked against the table before it reaches a
                # path; an unknown one is an error, never a directory
                stage = (q.get('stage', ['gate'])[0]
                         if q.get('stage', ['gate'])[0] in GATE_STAGES
                         else 'gate')
                doc = gate_progress(stage)
                # liveness from the process table, not from the file dates:
                # a run killed mid-shard leaves recent mtimes behind
                doc['running'] = bool(gate_pids(stage))
                doc['planning'] = gate_planning()
                doc['can_run'] = bool(DOGBIN_PYTHON) and (
                    not doc.get('upstream')
                    or doc['upstream']['ready'])
                self._json(doc)
            except Exception as e:
                self._json({'ever': False, 'error': str(e)})
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
        if self.path.split('?', 1)[0] == '/api/triage':
            try:
                bq = {}
                if '?' in self.path:
                    bq = {k: v[0] for k, v in
                          parse_qs(self.path.split('?', 1)[1]).items()}
                self._json(triage_status(str(bq.get('backend', 'siglip'))))
            except Exception as e:
                self._json({'ever': False, 'error': str(e)})
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
        if self.path.split('?', 1)[0] in ('/', '/index.html'):
            if self._serve_index_fresh():
                return
        if self.path.split('?', 1)[0] == '/favicon.ico':
            # No page carried a <link rel="icon">, so every tab load in a real
            # browser fired a request that 404'd into the log. Served from
            # memory; browsers take an SVG here.
            self.send_response(200)
            self.send_header('Content-Type', 'image/svg+xml')
            self.send_header('Content-Length', str(len(FAVICON_SVG)))
            self.send_header('Cache-Control', 'public, max-age=604800')
            self.end_headers()
            self.wfile.write(FAVICON_SVG)
            return
        if not _static_allowed(self.path.split('?', 1)[0]):
            # The static fallback's directory is OUT, and OUT doubles as the
            # server's working directory: serve.log (client IPs), a 2MB
            # history.duckdb, a 38MB triage.jsonl and the search-term vectors
            # all sit beside the page, and SimpleHTTPRequestHandler hands out
            # anything in its directory by name. The pages fetch exactly the
            # names in the allow-list; everything else here is the server's.
            self.send_error(404)
            return
        super().do_GET()

    def _serve_index_fresh(self):
        """index.html with the training section re-rendered for THIS request.

        The page is a build artefact written on an interval, so the one section
        whose entire purpose is to be current was baked stale into it. The
        client polls /api/training, but on a 30s timer and not at all on load,
        so opening the dashboard showed an hour-old "running" row -- naming a
        run that had already been cancelled -- for the first half minute.
        Splicing here makes the very first paint correct, and correct without
        JavaScript.

        Returns False to fall through to the plain file whenever anything is
        off: a page built before the sentinels existed, or a render that
        raises. Serving the page as built is a worse page, not a broken one.
        """
        try:
            with open(os.path.join(OUT, 'index.html'), 'rb') as fh:
                page = fh.read()
        except OSError:
            return False
        i = page.find(self._TRK_OPEN)
        j = page.find(self._TRK_CLOSE, i + 1) if i >= 0 else -1
        if i < 0 or j < 0:
            return False
        try:
            fresh = render_training().encode('utf-8')
        except Exception:
            return False
        body = page[:i + len(self._TRK_OPEN)] + fresh + page[j:]
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True

    def do_POST(self):
        # The same prefix guard do_GET uses, so both verbs claim /llm the same
        # way and neither can end up owning half of it.
        _p = self.path.split('?', 1)[0]
        if LLM_PAGE and (_p.startswith('/llm')
                         or _p.startswith('/api/llm')):
            if self._llm_post():
                return
        if self.path.split('?', 1)[0] == '/api/audit/box':
            try:
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                a = _audit()
                stage = self._audit_stage(parse_qs(urlparse(self.path).query))
                self._json(a.save_correction(
                    data.get('key'), data.get('box') or [], stage)
                    if a and a.pool_ready(stage)
                    else {'ok': False, 'msg': 'pool not built'})
            except Exception as e:
                self._json({'ok': False, 'msg': str(e)})
            return
        if self.path.split('?', 1)[0] in ('/api/audit/draw',
                                          '/api/audit/verdict'):
            try:
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                a = _audit()
                stage = (self._audit_stage(parse_qs(urlparse(self.path).query))
                         if a else None)
                if a is None:
                    self._json({'error': 'the audit module would not load'})
                elif not a.pool_ready(stage):
                    # which pool, so a leash annotator is not sent to build
                    # the gate's
                    self._json({'error': f'the {stage} audit pool is not '
                                         f'built — run tools/detect/'
                                         f'fn_audit.py build --stage {stage}'})
                elif self.path.startswith('/api/audit/draw'):
                    self._json(a.api_draw(
                        n=a.page_size(data.get('n')),
                        band=a.band_arg(data.get('band')), stage=stage))
                else:
                    v = data.get('verdict')
                    self._json(a.record(
                        data.get('key'),
                        None if v is None else str(v),
                        {'band': data.get('band'), 'p_dog': data.get('p_dog'),
                         'seq': data.get('seq')}, stage=stage))
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/board':
            try:
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                ok = set_stage(data.get('region', ''), data.get('stage', ''))
                self._json({'ok': ok}, 200 if ok else 400)
            except Exception as e:
                self._json({'error': str(e)}, 500)
            return
        if self.path.split('?', 1)[0] == '/api/training/diff':
            try:
                q = parse_qs(urlparse(self.path).query)
                self._json({'html': render_run_diff(q.get('a', [''])[0],
                                                    q.get('b', [''])[0])})
            except Exception as e:
                self._json({'html': '', 'error': str(e)})
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
        if self.path.split('?', 1)[0] == '/api/gate':
            # The argv is built here, from config. The client chooses one of
            # two words and nothing else reaches a command line.
            try:
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                stage = str(data.get('stage') or 'gate')
                self._json(gate_control(str(data.get('action') or ''), stage))
            except Exception as e:
                self._json({'ok': False, 'msg': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/triage':
            # The argv is built here, from config -- nothing the client sends
            # reaches it. The only thing it chooses is which of two words.
            try:
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                self._json(triage_control(
                    str(data.get('action') or ''),
                    str(data.get('backend') or 'siglip')))
            except Exception as e:
                self._json({'ok': False, 'msg': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/training/relabel':
            # Says the DATASET is wrong here, not the model. Its own store,
            # removable, and it never edits a dataset in place -- the next
            # build reads the export and leaves those ids out.
            try:
                mod = flag_store()
                if not mod:
                    self._json({'ok': False, 'error': 'flag store missing'})
                    return
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                f = str(data.get('file') or '')
                if data.get('remove'):
                    body, code = mod.remove(f)
                else:
                    body, code = mod.add(
                        f, dataset=str(data.get('dataset') or ''),
                        class_was=str(data.get('was') or ''),
                        should_be=str(data.get('should') or ''),
                        run=str(data.get('run') or ''))
                self._json(body, code)
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
            return
        if self.path.split('?', 1)[0] == '/api/review/leash':
            # Its own store and its own axis. Nothing here touches the
            # dog/not-dog ledgers: those answer a different question, feed a
            # different model, and hard_positives in particular means "a dog
            # the detector was unsure about", not "a dog".
            try:
                mod = leash_store()
                if not mod:
                    self._json({'ok': False, 'error': 'leash store missing'})
                    return
                n = int(self.headers.get('Content-Length', 0) or 0)
                data = json.loads(self.rfile.read(n) or b'{}')
                name = str(data.get('name') or '')
                if data.get('remove'):
                    body, code = mod.remove(name)
                else:
                    # the crop's OWN directory: the queue serves the harvested
                    # set too, and those never pass through recent_crops, so
                    # the store's "a label whose image has aged out is not
                    # trainable" copy found nothing every single time
                    src = crop_dir(name)
                    body, code = mod.record(
                        name, str(data.get('label') or ''),
                        copy_from={'crop': src,
                                   'full': os.path.join(src, 'full')})
                self._json(body, code)
            except Exception as e:
                self._json({'ok': False, 'error': str(e)})
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
        if self.path.split('?', 1)[0] == '/api/refresh':
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
    <span class="rpct">{p['pct']:.0f}%</span></div>
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
            .replace('__LLMNAV__', LLM_NAV if LLM_PAGE else '')
            .replace('__NOW__', now.strftime('%Y-%m-%d %H:%M'))
            .replace('__PROG__', f"{ov['pct']:.1f}")
            .replace('__LB_CSS__', LB_CSS)
            .replace('__LB_HTML__', LB_HTML)
            .replace('__LB_JS__', LB_JS)
            .replace('__COPY_JS__', COPY_JS)
            .replace('__MODELS__', render_models())
            .replace('__TRAINING__', render_training())
            .replace('__DRIVES__', render_drives())
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
    # The atlas refreshes on its own clock. Its full-parquet scan is too
    # heavy for every hourly build (that is what tied it to --images-every),
    # but its dogs layer is fed by the LIVE sweep -- left to --images-every
    # alone it would fossilize while the sweep adds positives. Cap staleness.
    map_stale = (not os.path.exists(MAP_FILE)
                 or not os.path.exists(MAP_FINE_FILE)
                 or time.time() - os.path.getmtime(MAP_FILE) > 6 * 3600)
    if map_stale or (getattr(args, 'images', False) and not getattr(
            args, 'no_refresh', False)):
        try:
            build_map_points()
        except Exception as e:
            print('map build error:', e)
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, 'echarts.min.js')
    if os.path.exists(ECHARTS_SRC) and not os.path.exists(dst):
        shutil.copy(ECHARTS_SRC, dst)
    # Atomically, because the server keeps serving through a rebuild: a page
    # fetched mid-write got a truncated script, and a browser that had just
    # loaded the previous script against the new map_points.json threw on
    # every frame. The data files already replace this way; the page did not.
    page = os.path.join(OUT, 'index.html')
    with open(page + '.tmp', 'w') as f:
        f.write(render(ov, per, trend(), now, region_locations(args.db)))
    os.replace(page + '.tmp', page)
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

    # A serving process holds the module it imported at startup, and its
    # interval rebuild REGENERATES index.html from that copy. So editing this
    # file and rebuilding by hand looked fine, and then an hour later the page
    # silently reverted to the old one -- the map came back upside down with
    # controls that had been added since simply missing. Whoever edits the
    # source is not necessarily watching the server, so the server checks.
    # Not only this file. The pages this process serves also come out of
    # audit.py, datasets.py and llm_page.py, imported once and held -- so an
    # edit to a sibling sat invisible behind a healthy-looking server until
    # somebody restarted it by hand, which is the same failure this watcher
    # exists to close.
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    _watched = {os.path.abspath(__file__)}
    for _m in ('audit.py', 'datasets.py', 'llm_page.py'):
        _q = os.path.join(_src_dir, _m)
        if os.path.exists(_q):
            _watched.add(_q)
    _src_mtimes = {q: os.path.getmtime(q) for q in _watched}

    def _reexec_if_stale():
        """Restart in place when any served source file changes underneath us."""
        for q, was in _src_mtimes.items():
            try:
                if os.path.getmtime(q) == was:
                    continue
                with open(q) as fh:
                    body = fh.read()
                compile(body, q, 'exec')  # never exec into a half-written file
            except (OSError, SyntaxError) as e:
                print(f'{os.path.basename(q)} changed but will not load ({e}); '
                      f'staying on the running copy', file=sys.stderr)
                return
            print(f'{os.path.basename(q)} changed on disk -- restarting to '
                  f'serve it', flush=True)
            os.execv(sys.executable, [sys.executable] + sys.argv)

    def loop():
        cyc = 1
        while True:
            time.sleep(args.interval)
            _reexec_if_stale()
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
.lbcopy{font-variant-numeric:tabular-nums}
.lbcopy.done{color:var(--green);border-color:rgba(67,181,129,.5);
background:rgba(67,181,129,.14)}
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
    <button class="rbtn quiet lbcopy" id="cropLbCopy" title="copy this image’s Mapillary id">Copy ID</button>
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
      bf=document.getElementById('cropLbFlag'),
      bc=document.getElementById('cropLbCopy');
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
    /* re-armed on every step, so the button can never copy the id of the
       crop you were looking at a moment ago */
    if(bc){bc.dataset.id=c.image_id||'';bc.textContent='Copy ID';
      bc.classList.remove('done');}
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
  if(bc)bc.addEventListener('click',function(e){
    e.stopPropagation();
    copyOnto(this,this.dataset.id||'','Copy ID');
  });
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
<title>Street Dogs — harvest to model</title>
<script src="echarts.min.js"></script>
<style>
:root{--bg:#13151a;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;
/* used by .tst.halt, .tag.warn and the diff's down-arrow long before it was
   declared here -- as a plain colour that silently inherited, so an
   interrupted run never actually rendered rust. It only became visible when
   color-mix() got hold of it: an undefined custom property makes the whole
   declaration invalid, so every error cell in the confusion matrix came out
   fully transparent.
   It was rust, #d8743a, which is 12.5 OKLab units from the amber accent --
   under the 15 an average reader needs to tell two colours apart, so a
   failure was rendered in very nearly the page's ambient colour and did not
   read as one. The review page had already hit this and kept its own red for
   the "no" button; this is the same fix, made once. */
--red:#ef5350;
/* The progress ramp is NOT declared here. Every bar that uses it is filled
   from script, inline, so a custom property would be five names no rule ever
   reads -- and the page already carried two of those. It lives in
   PROGRESS_RAMP and pctColor(), which a guard holds to each other. */
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
  border-bottom:1px solid var(--bd);transition:padding .18s ease}
h1{font-size:clamp(21px,3vw,29px);font-weight:650;letter-spacing:-.4px;
transition:font-size .18s ease,letter-spacing .18s ease}
/* ── the header at work ───────────────────────────────────────────────────
   The top of this page is identity, and identity is read once. Once the page
   has scrolled the header keeps only what someone ACTS on -- the queue count,
   the two links, the refresh button -- and sheds the tagline, the "updated
   ... auto-refreshes hourly" line, the sub-labels under the two button
   numbers, and most of a 29px title. Those are ambient: worth a glance, not
   worth 100px of every screen on a page seven sections tall. */
/* 96px, not 1px. A one-pixel sentinel flips the header the instant the page
   moves at all, so a small scroll near the top snapped ~60px of chrome in and
   out repeatedly on /review. 96px of content past the header is more than
   anyone nudges by accident. The negative margin cancels the height, so the
   sentinel costs no layout. */
.scrollcue{display:block;height:96px;margin-bottom:-96px;pointer-events:none}
/* 24px shorter once folded, so the point the header unfolds at sits a little
   below the point it folded at. Without that they are the SAME point to
   within a pixel -- the sentinel rides under the header, so folding moves it
   and the browser's scroll anchoring moves the viewport by the same amount,
   and the crossing lands back exactly on itself -- and an exact boundary
   ping-pongs on a rounding error. */
body.compact .scrollcue{height:72px;margin-bottom:-72px}
body.compact header{padding-top:8px;padding-bottom:8px}
/* Shrunk, not hidden. display:none takes the height in one frame and every
   section below lurches, and this header is sticky so that is the whole page
   moving under the cursor. Type rather than the max-height /review folds its
   rows with, because all three of these are one run of text: max-height needs
   a cap to transition FROM, and any cap is a clip waiting for the width where
   the tagline wraps to a third line or "what runs trained on" to a second.
   Shrinking the type collapses both axes and needs no such number -- and the
   "updated ..." line shares a row with the button it belongs to, so what it
   costs is width, which a folded max-height would have left as a hole. The
   words stay in the accessibility tree either way, which is the point: they
   are ambient on screen, not gone from the page. */
.sub,.updt,.revbtn em{transition:font-size .18s ease,opacity .18s ease,
margin .18s ease}
body.compact .sub,body.compact .updt,body.compact .revbtn em{font-size:0;
opacity:0;margin-top:0}
/* A flex item of zero width still takes the gap on BOTH sides, so shedding
   the sentence left the live dot floating 14px off the Refresh button it
   annotates, reading as a stray light rather than a status one. */
body.compact .updt{margin-right:-7px}
/* the count is what the button is for and stays full size; the padding
   around it is what gives way */
body.compact .revbtn{padding:4px 10px 4px 11px}
/* /review drops its h1 by 3px and can snap it. 29px to 15px is a fifth of
   the header's height arriving in one frame while the padding around it
   eases, which read as a jolt, so the title is eased too (above). */
body.compact h1{font-size:15px;letter-spacing:-.2px}
/* Last, not first: a media query adds no specificity, so this block sitting
   above h1's own rule left the biggest text on the page as the ONE thing
   still easing for the reader who asked for no motion. */
@media(prefers-reduced-motion:reduce){
  header,h1,.sub,.updt,.revbtn em{transition:none}
}
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
transition:transform .12s ease,box-shadow .12s ease,background .12s ease,
padding .18s ease}
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
/* A bare display:flex outranks the UA's [hidden]{display:none}, so the stage
   switch hid neither control and the header carried four buttons: Resume
   sweep and Run gate side by side, only one of which the visible stage was
   about. The section shows ONE stage, so it offers ONE run to start. */
.swctl[hidden]{display:none}
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
/* ── atlas chrome: layer chips, fly-to, surveyor HUD, ramp legend ── */
.mapbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 12px}
.mchip{appearance:none;background:transparent;border:1px solid var(--bd);color:var(--mut);
border-radius:999px;padding:4px 13px;font-size:11.5px;font-family:inherit;cursor:pointer;
transition:color .12s,border-color .12s,background .12s}
.mchip:hover{color:var(--tx)}
.mchip:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
/* the active chip wears its layer's ink: amber = harvest, green = the
   detector, blue = the rate — the same coding the ramps use */
.mchip.on{color:#f0b85f;border-color:rgba(240,184,95,.55);background:rgba(232,166,69,.08)}
.mchip.on[data-l=dogs]{color:#43b581;border-color:rgba(67,181,129,.55);background:rgba(67,181,129,.08)}
.mchip.on[data-l=rate]{color:#7fb2d8;border-color:rgba(127,178,216,.55);background:rgba(127,178,216,.08)}
.mtog{display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--mut);cursor:pointer;margin-left:4px}
.mtog input{accent-color:var(--acc)}
#mapFind{margin-left:auto;background:var(--panel2);border:1px solid var(--bd);border-radius:8px;
color:var(--tx);font-family:inherit;font-size:12px;padding:5px 10px;width:180px}
#mapFind:focus{outline:none;border-color:rgba(232,166,69,.45)}
#mapFind::placeholder{color:var(--dim)}
.mreset{appearance:none;background:transparent;border:1px solid var(--bd);color:var(--mut);
border-radius:8px;padding:5px 11px;font-size:11.5px;font-family:inherit;cursor:pointer;
transition:color .12s,border-color .12s}
.mreset:hover{color:var(--tx);border-color:rgba(232,166,69,.4)}
.mreset:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
/* the lede says what the visible layer MEANS, for the reader who never
   hovers a chip; it swaps with the layer */
.maplede{margin:0 0 12px;font-size:11.5px;line-height:1.5;color:var(--dim);max-width:76ch}
.maphud{position:absolute;left:12px;bottom:10px;z-index:1;pointer-events:none;
font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;
color:var(--mut);background:rgba(13,17,23,.62);border:1px solid var(--bd);
border-radius:7px;padding:3px 9px;letter-spacing:.02em}
.maphud:empty{display:none}
/* country card: sits inside the map, opposite the lock button so the two
   never collide, and never taller than the panel it floats in */
.cpop{position:absolute;left:12px;top:12px;z-index:4;width:250px;
max-height:calc(100% - 24px);overflow:auto;background:#161b22;
border:1px solid var(--bd);border-radius:12px;padding:12px 13px 11px;
box-shadow:0 10px 28px rgba(0,0,0,.45)}
.cpop[hidden]{display:none}
.cpx{position:absolute;right:6px;top:5px;background:none;border:0;color:var(--dim);
font-size:17px;line-height:1;cursor:pointer;padding:2px 6px;border-radius:6px}
.cpx:hover{color:var(--tx)}
.cpx:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.cpname{font-size:13.5px;font-weight:620;color:var(--tx);padding-right:18px;
margin-bottom:1px}
.cprank{font-size:10.5px;color:var(--dim);margin-bottom:9px}
.cpbig{font-size:21px;font-weight:640;font-variant-numeric:tabular-nums;
line-height:1.15}
.cpunit{font-size:10.5px;color:var(--mut);margin-bottom:9px}
.cprow{display:flex;justify-content:space-between;gap:10px;font-size:11px;
padding:3px 0;border-top:1px solid rgba(130,140,150,.10)}
.cprow span:first-child{color:var(--dim)}
.cprow span:last-child{color:var(--mut);font-variant-numeric:tabular-nums}
.cpbar{height:4px;border-radius:3px;background:rgba(130,140,150,.14);
margin:2px 0 9px;overflow:hidden}
.cpbar i{display:block;height:100%;border-radius:3px}
.cpnote{font-size:10.5px;color:var(--dim);margin-top:8px;line-height:1.45}
.maplegend{display:flex;align-items:center;gap:9px;margin-top:10px;font-size:11px;
color:var(--dim);flex-wrap:wrap}
.mramp{width:170px;height:7px;border-radius:4px;border:1px solid rgba(130,140,150,.18)}
.mlmax{font-variant-numeric:tabular-nums;color:var(--mut)}
.mstats{margin-left:auto;font-variant-numeric:tabular-nums;text-align:right}
/* wrap, because the sweeps header carries a stage switch and a run control:
   at phone widths their fixed minimums pushed the page to 535px of sideways
   scroll while the heading collapsed to one word per line. With room, no
   wrap ever happens. */
.sect{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;font-size:15px;font-weight:620;margin:8px 0 14px}
.sect span{font-size:12.5px;font-weight:400;color:var(--dim)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(292px,1fr));gap:13px}
.rcard{background:linear-gradient(180deg,#1c2128,#181c22);border:1px solid var(--bd);border-radius:14px;padding:15px 17px;transition:border-color .15s,transform .15s}
.rcard:hover{border-color:rgba(232,166,69,.4);transform:translateY(-2px)}
.rtop{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;gap:8px}
.rname{font-size:14.5px;font-weight:620}
.rpct{font-size:14px;font-weight:680;font-variant-numeric:tabular-nums;
  color:var(--tx)}
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
/* SCOPED to the pipeline rail. Unscoped, this rule also captured the header's
   status dot -- .upd is position:static, so left:-27px resolved against the
   page and parked a pulsing green ring off the left edge of the window, the
   only part of it still visible. Same collision as .spk vs .kpi.ok.spk. */
.stg .dot{position:absolute;left:-27px;top:5px;width:12px;height:12px;
border-radius:50%;background:var(--bg);border:2px solid var(--green)}
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
/* ── drive health ─────────────────────────────────────────────────────────
   A verdict, then the numbers behind it. The first version was a flat list of
   identical grey rows: six readouts of equal weight, when five of the six
   facts on a healthy disk are things you never act on. Each drive is a card
   now, led by the one word that decides whether to do anything, and the cards
   sort worst first so the top of the section is the story.

   The capacity meter is the only ink that carries state, and it carries it in
   the drive's own colour -- so a glance across the grid is the whole report.
   The cell count went: it answered a question nobody was asking. */
.dhs{display:grid;gap:10px;
grid-template-columns:repeat(auto-fill,minmax(232px,1fr))}
.dh{border:1px solid var(--bd);border-radius:11px;padding:12px 13px 11px;
background:var(--panel);display:grid;gap:8px;align-content:start}
.dhtop{display:flex;align-items:baseline;gap:9px}
.dhname{font-size:13.5px;font-weight:660;color:var(--tx);letter-spacing:-.1px}
.dhverdict{margin-left:auto;font-size:10.5px;letter-spacing:.05em;
text-transform:uppercase;color:var(--dim)}
.dhmeter{display:block;height:5px;border-radius:3px;
background:rgba(130,140,150,.18);overflow:hidden}
.dhmeter i{display:block;height:100%;border-radius:3px;background:var(--green);
transition:width .4s ease}
.dhroom{font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums}
.dhroom b{color:var(--tx);font-weight:650;font-size:13px}
/* The drive's own numbers, as chips: they are read by scanning for the one
   that is not grey, which a sentence of them would not allow. */
.dhfacts{display:flex;flex-wrap:wrap;gap:4px}
.dhf{font-size:10.5px;line-height:1.7;padding:0 6px;border-radius:4px;
background:rgba(130,140,150,.10);border:1px solid var(--bd);color:var(--dim);
font-variant-numeric:tabular-nums;cursor:help}
.dhf.warn{color:var(--acc);border-color:rgba(232,166,69,.42);
background:rgba(232,166,69,.10)}
.dhf.bad{color:var(--red);border-color:rgba(216,116,58,.5);
background:rgba(216,116,58,.12);font-weight:650}
.dhsm{font-size:11px;color:var(--dim)}
.dhsm.ok{color:var(--green)}
.dhsm.bad{color:var(--red);font-weight:650}
/* the two states worth interrupting a glance for */
.dh.warn{border-color:rgba(232,166,69,.42)}
.dh.warn .dhmeter i{background:var(--acc)}
.dh.warn .dhverdict{color:var(--acc)}
.dh.bad{border-color:rgba(216,116,58,.55);background:rgba(216,116,58,.07)}
.dh.bad .dhmeter i{background:var(--red)}
.dh.bad .dhverdict{color:var(--red);font-weight:650}
.dhsum{margin:0 0 11px;font-size:11.5px;color:var(--dim)}
.dhlead{margin:0 0 11px;font-size:12.5px;color:var(--red);font-weight:650}
.dhnote{margin:12px 0 0;font-size:11.5px;line-height:1.6;color:var(--dim);
max-width:78ch}
.dhnote code{font-size:11px;background:var(--panel2);border:1px solid var(--bd);
border-radius:4px;padding:1px 5px;color:var(--mut)}
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
/* The percentage wore the bar's colour, which put 11px text at 3.4:1 on the
   dim end of the ramp -- unreadable exactly where a region is furthest
   behind and you most want to read it. The bar beside it already carries the
   magnitude; the number only has to be legible. */
.rc .rpc{color:var(--tx);font-weight:600}
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
/* ── the stage switch ─────────────────────────────────────────────────────
   Two passes over one store, one section. The switch is a pair of tabs and
   not a dropdown: there are two, they are peers, and which one is showing
   should be readable without opening anything. */
.stagesw{display:inline-flex;gap:2px;margin-left:14px;padding:2px;
border:1px solid var(--bd);border-radius:999px;vertical-align:middle}
.stagebtn{appearance:none;background:transparent;border:0;color:var(--dim);
font-family:inherit;font-size:11.5px;padding:3px 11px;border-radius:999px;
cursor:pointer;transition:color .12s,background .12s}
.stagebtn:hover{color:var(--tx)}
.stagebtn.on{background:rgba(232,166,69,.15);color:var(--acc);font-weight:640}
.stagebtn:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
/* ── detection sweep panel (§7.4) ── */
.dnone{color:var(--dim);font-size:12.5px;padding:2px}
.dsub{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);margin:16px 0 8px}
/* headline: compact KPI chips (the page's existing .kpi look); the img/s
   sparkline draws as the subtle BACKGROUND of the img/s (now) chip */
/* One number leads. Six cards at one weight is six headlines, so the panel
   had no answer to the question it exists to answer -- how far along is it --
   and "boxes/s (sustained)" shouted as loudly as "complete". Size carries the
   rank, not colour: the accent is already the progress bar directly below,
   and a hero in the same amber would just be the loudest thing twice. */
.kpi.lead{grid-column:span 2}
.kpi.lead .kpi-val{font-size:34px;line-height:1.12;letter-spacing:-1.2px}
@media(max-width:620px){.kpi.lead{grid-column:span 1}
  .kpi.lead .kpi-val{font-size:26px}}
.kpi.spk{position:relative;overflow:hidden}
.kpi.spk .kpi-label,.kpi.spk .kpi-val{position:relative;z-index:1}
/* ── the machine ──
   Four cards, one row, same shape as every other KPI on the page: this is a
   readout, not a feature, and a section that invented its own look would
   claim more attention than it earns. What it adds is the second line -- the
   figure under each headline that says what the headline is against (16
   cores, 63 GB, a 16 GB card), because a bare "48%" is not actionable. */
.sykpis .kpi{padding:12px 15px}
.sykpis .kpi-val{font-size:20px;margin-top:3px;position:relative;z-index:1}
.sysub{font-size:10.5px;color:var(--dim);margin-top:3px;position:relative;
  z-index:1;font-variant-numeric:tabular-nums;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.sysub.warn{color:var(--acc)}
.sysub.bad{color:var(--red)}
.symeta{font-size:11px;color:var(--dim);margin-top:11px}
.symeta:empty{display:none}
/* one word in the header: what the box is waiting on right now */
.syverdict{margin-left:auto;align-self:center;font-size:11px;font-weight:600;
  letter-spacing:.04em;text-transform:uppercase;color:var(--dim);
  border:1px solid var(--bd);border-radius:999px;padding:3px 10px}
.syverdict:empty{display:none}
.syverdict.io{color:var(--acc);border-color:rgba(232,166,69,.35)}
.syverdict.gpu{color:var(--green);border-color:rgba(67,181,129,.3)}
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
.dbadge{background:rgba(239,83,80,.13);border:1px solid rgba(239,83,80,.45);color:var(--red);border-radius:7px;padding:0 7px;font-size:10.5px;font-weight:620;flex:none}
/* not_a_dog gauge: the 7–16% labelled prior renders as a shaded healthy zone
   (Addendum A.5) so in/out of band is legible without reading numbers */
.dband{position:relative;height:10px;border-radius:6px;background:rgba(130,140,150,.16);flex:1}
.dband .zone{position:absolute;top:0;bottom:0;background:rgba(67,181,129,.3);border-radius:2px}
.dband .cur{position:absolute;top:-3px;bottom:-3px;width:3px;border-radius:2px;background:var(--acc)}
.dband .cur.bad{background:var(--red)}
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
/* ── comparing two runs ── */
.dpick{display:flex;align-items:center;gap:9px;margin:12px 0 4px;font-size:12px}
.dpick label{color:var(--mut)}
.dpick select{background:var(--panel);color:var(--tx);border:1px solid var(--bd);
border-radius:8px;padding:5px 9px;font-family:inherit;font-size:12px;max-width:340px}
.dpick select:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.dtab{width:100%;border-collapse:collapse;font-size:12.5px;margin:10px 0 4px}
.dtab th,.dtab td{padding:6px 10px;text-align:left;
border-bottom:1px solid var(--bd)}
.dtab thead th{color:var(--dim);font-weight:600;font-size:11px;
text-transform:lowercase;letter-spacing:.04em}
.dtab tbody th{color:var(--mut);font-weight:500}
.dtab .dnum{text-align:right;font-variant-numeric:tabular-nums;color:var(--tx)}
.dtab .dnum em{font-style:normal;font-size:10px;color:var(--dim);margin-left:6px}
.dtab thead .dnum{color:var(--dim)}
.dtab .dmid{color:var(--dim)}
/* direction, not sentiment: green means the second run moved the number the
   way you want for THAT row, which for wall clock is downwards */
.dtab .dup{color:var(--green)}
.dtab .ddn{color:var(--red)}
.dsub{margin:16px 0 0;font-size:11px;color:var(--dim);
text-transform:lowercase;letter-spacing:.04em}
.dwarn{margin:10px 0 0;padding:8px 11px;border-radius:9px;font-size:12px;
color:var(--tx);background:rgba(216,116,58,.10);
border:1px solid rgba(216,116,58,.30)}
/* ── what a run got wrong ──
   The confusion matrix counts them; this is the same fact in the only form
   you can act on. Tiles are small and dense because the value is in seeing a
   KIND emerge across twenty of them, not in studying one. */
.wrwrap{margin:22px 0 2px;padding-top:18px;border-top:1px solid var(--bd)}
.wrhead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:11px}
.wrhead b{font-size:13px;color:var(--tx)}
.wrsub{font-size:11.5px;color:var(--dim)}
/* the panel that holds them: a bounded object in the section, not a run of
   tiles down the page */
.wrbox{background:var(--panel2);border:1px solid var(--bd);border-radius:11px;
padding:11px 12px 12px}
.wrbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;
margin-bottom:11px}
.wrpage{margin-left:auto;display:flex;align-items:center;gap:4px;flex:none}
.wrat{font-size:11px;color:var(--dim);font-variant-numeric:tabular-nums;
min-width:92px;text-align:center}
.wrnav{appearance:none;background:transparent;border:1px solid var(--bd);
color:var(--mut);border-radius:7px;width:24px;height:24px;line-height:1;
font-size:14px;font-family:inherit;cursor:pointer;
transition:color .12s,border-color .12s}
.wrnav:hover:not(:disabled){color:var(--tx);border-color:var(--dim)}
.wrnav:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.wrnav:disabled{opacity:.35;cursor:default}
.wrchips{display:flex;gap:7px;flex-wrap:wrap}
.wrchip{appearance:none;background:transparent;border:1px solid var(--bd);
color:var(--mut);border-radius:999px;padding:4px 12px;font-size:11.5px;
font-family:inherit;cursor:pointer;transition:color .12s,border-color .12s,
background .12s}
.wrchip em{font-style:normal;color:var(--dim);margin-left:5px;
font-variant-numeric:tabular-nums}
.wrchip:hover{color:var(--tx);border-color:var(--dim)}
.wrchip:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.wrchip.on{color:var(--tx);border-color:rgba(216,116,58,.5);
background:rgba(216,116,58,.13)}
.wrchip.on em{color:var(--red)}
.wrgrid{display:grid;grid-template-columns:repeat(8,1fr);gap:9px;
grid-auto-rows:1fr}
@media(max-width:1500px){.wrgrid{grid-template-columns:repeat(6,1fr)}}
@media(max-width:1100px){.wrgrid{grid-template-columns:repeat(4,1fr)}}
@media(max-width:720px){.wrgrid{grid-template-columns:repeat(3,1fr)}}
.wrtile{margin:0;border:1px solid var(--bd);border-radius:9px;overflow:hidden;
background:var(--panel2);transition:border-color .12s}
.wrtile:hover{border-color:rgba(216,116,58,.45)}
.wrtile[hidden]{display:none}
.wrtile img{width:100%;aspect-ratio:1;object-fit:contain;display:block;
background:#0c0e11}
.wrtile figcaption{padding:5px 7px 6px;
font:400 10px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
font-variant-numeric:tabular-nums}
.wrdir{display:flex;align-items:baseline;gap:4px;min-width:0}
/* what it SAID is the mistake, so that is the word that carries the colour */
.wrsaid{color:var(--red);font-weight:600;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
.wrwas{color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.wrarr{color:var(--dim);flex:none;opacity:.7}
.wrp{display:block;color:var(--mut);margin-top:1px}
/* the reviewer's own verdict on the label, so it is the one thing on the tile
   that is a control rather than a report */
.wrflag{display:block;width:100%;margin-top:5px;appearance:none;
background:transparent;border:1px dashed var(--bd);border-radius:6px;
color:var(--dim);padding:3px 4px;font-family:inherit;font-size:9.5px;
cursor:pointer;transition:color .12s,border-color .12s,background .12s}
.wrflag:hover{color:var(--tx);border-color:var(--dim)}
.wrflag:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.wrflag.on{border-style:solid;color:var(--acc);
border-color:rgba(232,166,69,.55);background:rgba(232,166,69,.15);
font-weight:600}
.wrtile.flagged{border-color:rgba(232,166,69,.5)}
.wrtile.flagged img{opacity:.55}
.wrflagn{color:var(--acc)}
/* the key: the caption's own type and colours, so it is a sample of the
   thing rather than a description that can drift out of step with it */
.wrkey{display:flex;gap:18px;flex-wrap:wrap;align-items:baseline;
margin-top:11px;padding-top:10px;border-top:1px solid var(--bd);
font:400 10px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.wrkeyi{display:flex;align-items:baseline;gap:5px;color:var(--dim)}
/* saytally() hides the flag count with the attribute, and a bare display:flex
   outranks the UA's [hidden]{display:none} -- the empty span kept its 18px of
   the key row */
.wrkeyi[hidden]{display:none}
.wrfoot{margin-top:11px;font-size:11px;color:var(--dim);max-width:640px;
line-height:1.5}
/* the matrix's own off-diagonal cells jump here */
.cxc.err{cursor:pointer}
.cxc.err:hover{outline:1px solid rgba(216,116,58,.55);outline-offset:-1px}
/* ── confusion matrix ── */
.cxwrap{margin:18px 0 2px}
.cxhead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;
margin-bottom:9px}
.cxhead b{font-size:13px;color:var(--tx)}
.cxsub{font-size:11.5px;color:var(--dim)}
.cxscroll{overflow-x:auto}
table.cx{border-collapse:separate;border-spacing:4px;font-size:13px}
table.cx th{font-weight:500;color:var(--mut);padding:4px 9px;white-space:nowrap}
.cxax{color:var(--dim);font-size:11.5px;text-align:center;letter-spacing:.04em}
.cxrow{text-align:right;white-space:nowrap}
.cxt{font-size:12.5px;text-align:center}
.cxl{text-align:right;font-size:12.5px}
.cxr{color:var(--dim);font-size:11.5px}
/* the cell tint is the count's share of the biggest cell; the hue says
   whether the cell is agreement or a mistake, so errors read warm at a
   glance without having to compare numbers */
.cxc{text-align:center;font-variant-numeric:tabular-nums;color:var(--tx);
min-width:104px;padding:20px 16px;border-radius:10px;font-size:16px;
background:color-mix(in srgb, var(--h) calc(var(--w) * 100%), transparent);
border:1px solid transparent}
.cxc.dg{border-color:rgba(67,181,129,.30)}
.cxc.z{color:var(--dim)}
.cxn{text-align:center;font-variant-numeric:tabular-nums;color:var(--mut);
font-size:12.5px;padding:8px 10px}
.cxfoot{margin-top:10px;font-size:11px;color:var(--dim);max-width:640px;
line-height:1.5}
.cxfoot summary,.wrfoot summary{cursor:pointer;list-style:none;
display:inline-flex;align-items:center;gap:4px;opacity:.85}
.cxfoot summary::-webkit-details-marker,
.wrfoot summary::-webkit-details-marker{display:none}
.cxfoot summary::after,.wrfoot summary::after{content:'▾'}
.cxfoot[open] summary::after,.wrfoot[open] summary::after{content:'▴'}
.cxfoot summary:hover,.wrfoot summary:hover{color:var(--mut)}
.cxfoot summary:focus-visible,.wrfoot summary:focus-visible{
outline:2px solid var(--acc);outline-offset:3px;border-radius:4px}
.cxfoot p,.wrfoot p{margin:8px 0 0}
/* there is no way to tell a label that explains itself from one that does
   not, except the cursor */
.cx .hcue{cursor:help}
/* --bd is rgba(...,.13) and vanished against the panel: a cue nobody can
   see is not a cue */
.cx th.hcue{text-decoration:underline dotted var(--dim);
text-underline-offset:3px}
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
/* Tracks share the row so four cards fit on one line; the CARD is what gets
   capped, not the track. Capping the track at 340px meant four could not fit
   at panel width and the fourth orphaned onto a row of its own -- but a 1fr
   track with no cap on the card was the original bug, where a lone card
   spanned the full width and smeared its sparkline flat. */
.mcards{display:grid;gap:12px;justify-content:start;
grid-template-columns:repeat(auto-fit,minmax(238px,1fr))}
.mcard{max-width:360px}
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
/* the measured winner, said quietly: the registry column beside it carries
   the stronger claim and should keep the louder styling */
.bstar{display:inline-flex;align-items:center;gap:4px;font-size:11px;
color:var(--acc);white-space:nowrap}
.tbest{white-space:nowrap}
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
  /* wall clock is column 7 now that "best in project" sits at 6 -- an
     nth-child that does not move with the table hides the wrong column */
  .thist th:nth-child(7),.thist td:nth-child(7){display:none}
}
@media (prefers-reduced-motion:reduce){
  .tmtrack i{transition:none}
}
</style></head><body><div class="wrap">

<header>
  <div><h1>Street Dogs &middot; <span class="o">harvest to model</span></h1>
    <div class="sub">worldwide Mapillary survey &mdash; collecting, detecting,
    judging, training</div></div>
  <div class="hact">
    <!-- The count is the point: a queue depth is what makes someone open the
         page, and an empty queue should say so quietly rather than shout a
         zero in the accent colour. -->
    <a class="revbtn" id="revBtn" href="/review"
       title="Judge detections one by one — dog or not a dog">
      <span class="rvf">&#9873;</span>
      <span class="rvn"><b id="revN">&mdash;</b><em id="revL">to review</em>
      </span></a>
    <!-- Quiet by design: this is somewhere to go, not something to do, and
         one solid-filled control on the page is the whole rule. It sits in
         the header rather than in the training section because a dataset
         outlives the runs that used it, and the section can be collapsed. -->
    <a class="revbtn quiet" href="/datasets" title="Every dataset the logged runs trained on — open one and look inside">
      <span class="rvf">&#9638;</span>
      <span class="rvn"><b>Datasets</b><em>what runs trained on</em>
      </span></a>
__LLMNAV__
    <!-- The sentence is wrapped because a bare text node has nothing to
         style, and the scrolled header sheds the sentence while keeping the
         button that sits after it and the dot that says the page is live. -->
    <div class="upd"><span class="dot"></span><span class="updt">updated __NOW__ · auto-refreshes hourly</span><button id="refreshBtn" class="rbtn" title="Re-scan the catalog + image counts now">↻ Refresh now</button></div>
  </div>
</header>

<!-- Scrolled past this, a screenful of the page has gone by and the header
     sheds everything nobody acts on. A sentinel, not a scroll handler: the
     browser reports the crossing once instead of the page measuring itself
     on every frame.
     It rides UNDER the header, not above it, and that is the whole trick.
     Above the header it stays put while folding takes 54px (140px narrow)
     out of a sticky element, and the browser's scroll anchoring answers that
     by moving the viewport the same distance -- straight back over a
     sentinel that had only just been crossed. The header then asked to
     unfold, was refused as a flutter, and stayed folded at the very top of
     the page with no title and no tagline. Down here the sentinel moves with
     the fold, the anchoring moves the viewport with it, and the two cancel:
     measured, the crossing point holds to within a pixel at every width. -->
<i class="scrollcue" id="scrollcue" aria-hidden="true"></i>

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

<details class="fold sec" id="f-sys" open>
<summary class="sect">The machine <span id="syHint">cpu, memory and the card &mdash; every 2 s</span>
  <span class="syverdict" id="syVerdict"></span></summary>
<div class="panel">
  <div class="kpis sykpis">
    <div class="kpi spk"><div id="sySparkCpu" class="dspark"></div>
      <div class="kpi-label" title="busy share of all cores since the last reading, iowait counted as idle">cpu</div>
      <div class="kpi-val" id="syCpu">&mdash;</div>
      <div class="sysub" id="syLoad">&mdash;</div></div>
    <div class="kpi spk"><div id="sySparkMem" class="dspark"></div>
      <div class="kpi-label" title="MemTotal minus MemAvailable — the page cache is not counted as used">memory</div>
      <div class="kpi-val" id="syMem">&mdash;</div>
      <div class="sysub" id="sySwap">&mdash;</div></div>
    <div class="kpi spk"><div id="sySparkGpu" class="dspark"></div>
      <div class="kpi-label" title="share of the last 30 seconds the card had work in flight — sampled every second, because this workload runs in bursts a single reading lands between">gpu</div>
      <div class="kpi-val" id="syGpu">&mdash;</div>
      <div class="sysub" id="syVram">&mdash;</div></div>
    <div class="kpi spk"><div id="sySparkIo" class="dspark"></div>
      <!-- PSI `full`: the share of the last ten seconds in which EVERY task
           was stalled waiting on a disk. On this box it is the number that
           explains the others -- the gate runs with the GPU near zero
           because decoding 8000x4000 panoramas is what it is waiting for. -->
      <div class="kpi-label" title="share of the last 10 s in which every task on the box was stalled waiting on storage">io stall</div>
      <div class="kpi-val" id="syIo">&mdash;</div>
      <div class="sysub" id="syCpuStall">&mdash;</div></div>
  </div>
  <div class="symeta" id="syMeta"></div>
</div>
</details>

<details class="fold sec" id="f-detect" open>
<summary class="sect">Model sweeps <span id="stgHint">yolo26x @1280 — live, updates every 5 s while open</span>
  <!-- Two stages of one pipeline, so one section with a switch rather than
       two sections saying the same six things. The detector finds ground
       animals; the gate decides which of them are dogs. They run at different
       times over the same store, and the question "how far along is it" has
       the same shape for both. -->
  <span class="stagesw" id="stagesw" role="tablist">
    <button type="button" class="stagebtn on" data-stage="detect" role="tab" aria-selected="true">Detector</button>
    <button type="button" class="stagebtn" data-stage="gate" role="tab" aria-selected="false">Dog-bin gate</button>
    <!-- Same panel, next stage. The gate and the leash model differ in what
         they read and what they call the positive class, and in nothing the
         layout can see -- so one set of cards serves both and the labels are
         swapped, rather than a second copy that drifts. -->
    <button type="button" class="stagebtn" data-stage="leash" role="tab" aria-selected="false">Leash model</button>
  </span>
  <!-- one control slot, and the stage decides which run it drives -->
  <span class="swctl" id="sweepCtl"><span class="swpill" id="sweepState">checking</span><button id="sweepBtn" class="rbtn sw" disabled>Checking&hellip;</button></span>
  <span class="swctl" id="gateCtl" hidden><span class="swpill" id="gateState">checking</span><button id="gateBtn" class="rbtn sw" disabled>Checking&hellip;</button></span></summary>
<div class="panel">
  <!-- status line ABOVE the cards, never instead of them: the layout below is
       always present and goes to em-dashes when idle, so nothing jumps when
       the sweep starts. #detOn is kept (and never hidden) as the cards' box. -->
  __STOREPATH__
  <div id="detOff" class="dnone dstat">sweep idle</div>
  <div id="detOn">
    <div class="kpis" style="margin-bottom:12px">
      <div class="kpi lead"><div class="kpi-label">Complete</div><div class="kpi-val" id="dhPct">—</div></div>
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

  <!-- The gate, in the same shapes. Not every card carries over: the gate has
       no per-drive lanes of its own and no live crops, and inventing an empty
       one for symmetry would be worse than leaving it out. -->
  <div id="gateOn" hidden>
    <div class="kpis" style="margin-bottom:12px">
      <div class="kpi lead"><div class="kpi-label">Complete</div><div class="kpi-val" id="gPct">—</div></div>
      <div class="kpi"><div class="kpi-label" title="detections the gate has judged">Judged</div><div class="kpi-val" id="gDone" style="font-size:19px">—</div></div>
      <div class="kpi"><div class="kpi-label">ETA</div><div class="kpi-val" id="gEta" style="font-size:19px">—</div></div>
      <!-- the counts behind the share go in the card's own title, filled in
           as the numbers arrive; a percentage cannot say whether it is drawn
           from a thousand boxes or four million -->
      <div class="kpi" id="gDogCard"><div class="kpi-label" id="gDogLbl">Called dog</div><div class="kpi-val" id="gDog" style="font-size:19px">—</div></div>
      <div class="kpi ok spk"><div id="gateSpark" class="dspark"></div><div class="kpi-label" title="boxes per second, measured by the run itself">boxes/s (now)</div><div class="kpi-val" id="gNow" style="font-size:19px">—</div></div>
      <div class="kpi"><div class="kpi-label" title="boxes per second over the whole run — the ETA is computed from this">boxes/s (sustained)</div><div class="kpi-val" id="gSus" style="font-size:19px">—</div></div>
    </div>
    <div class="bar dmain"><div class="fill" id="gFill" style="background:var(--acc)"></div></div>
    <div class="dcount" id="gCount">—</div>
    <div class="drun" id="gRun">—</div>
    <div class="dmeta" id="gMeta"></div>
    <!-- The rejected pile is the one nothing downstream will ever look at
         again, so the way in has to be here, next to the number that made
         it. -->
    <div class="dcrophead" style="margin-top:14px">
      <div class="dsub">Rejected boxes</div>
      <span class="dcropsub" id="gAuditSub">sample what the gate threw away and
        count what it got wrong</span>
      <a href="/audit" class="rbtn nav rev" id="gAuditLink"
         title="judge a stratified sample of what this model decided">
        &#9873; Audit this model</a>
    </div>
  </div>
</div>
</details>

<details class="fold sec" id="f-training" open>
<summary class="sect">Training <span>what is training now, its curves, and the runs behind it</span></summary>
<div class="panel" id="trk"><!--TRK-->__TRAINING__<!--/TRK--></div>
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
  <summary class="phead"><i></i><b>Atlas</b><span class="phint">where the harvest went, where the detector called dogs, and each region's stage &mdash; click any country for its numbers</span></summary>
  <div class="mapbar">
    <button type="button" class="mchip on" data-l="harvest"
      title="Every Mapillary frame the harvest downloaded, binned by where it was taken. This is coverage — where you have looked, not what was found.">Harvest</button>
    <button type="button" class="mchip" data-l="dogs"
      title="Frames where the detection sweep called at least one dog with confidence 0.5 or better. Unreviewed model output, so some of these are goats, sheep and shadows.">Dogs found</button>
    <button type="button" class="mchip" data-l="rate"
      title="Dogs found ÷ harvest, per cell. Corrects for how hard each place was searched: a bright cell here means dogs were common in the frames, not just that many frames exist. Needs 30+ frames in a cell to show.">Hit rate</button>
    <label class="mtog" title="One marker per region, placed at the median of its frames. Its colour follows the layer; its stage and download progress are in the tooltip. Turn them off if they are in the way of the map."><input type="checkbox" id="mapRegions" checked> region markers</label>
    <label class="mtog" id="mapCleanWrap" title="Hides frames whose GPS cannot be right: points out at sea, and frames sitting a continent away from the rest of their own capture session. Untick to see them."><input type="checkbox" id="mapClean" checked> exclude GPS outliers</label>
    <input id="mapFind" list="cmdRegions" placeholder="fly to a region&hellip;" autocomplete="off"
      title="Jump the camera to a region">
    <button type="button" class="mreset" id="mapReset" title="Back to the whole world at the default zoom">Reset view</button>
  </div>
  <p class="maplede" id="mapLede">Every Mapillary frame the harvest downloaded, binned by where it was taken. This is coverage — where you have looked, not what was found.</p>
  <!-- The map roams on wheel, so hovering it while scrolling the page used to
       zoom the map instead. A scrim swallows wheel/drag until you opt in;
       echarts keeps roam:true and never has to be reconfigured. -->
  <div class="mapwrap">
    <div id="map" style="width:100%;height:520px"></div>
    <div class="mapgate" id="mapGate">
      <span class="mapgb">Click to interact &mdash; scroll to zoom, drag to pan</span>
    </div>
    <button class="rbtn maplock" id="mapLock" hidden>&#128274; Lock map</button>
    <div class="maphud" id="mapHud"></div>
    <div class="cpop" id="mapPop" hidden>
      <button type="button" class="cpx" id="mapPopX" aria-label="Close">&times;</button>
      <div class="cpbody" id="mapPopBody"></div>
    </div>
  </div>
  <div class="maplegend">
    <span class="mlmax" id="mapMin">1</span><i class="mramp" id="mapRamp"></i><span class="mlmax" id="mapMax"></span>
    <span id="mapRampLab">frames harvested per cell</span>
    <span class="mstats" id="mapStats"></span>
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

<details class="fold sec" id="f-drives" open>
<summary class="sect">Drive health</summary>
<div class="panel">__DRIVES__</div>
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
/* The stages are a SEQUENCE -- pending, extract, coverage, backfill,
   complete, downloading, downloaded -- so the colour says how far along a
   region is, not which of seven unrelated things it is. It used to be seven
   scattered hues, and two of them (#5b8fd6 coverage, #8b7fd6 extract) were
   6.9 OKLab units apart for an average reader and 0.2 apart for a red-green
   colourblind one: on the board that decides where every region stands, two
   columns were painted the same colour. Seven hues cannot be separated on a
   dark ground; a ramp does not have to be, because it climbs in lightness,
   which every kind of colour vision keeps.
   Grey is the absence of the ramp, which is what "not started" is. Green is
   kept for the end, because finished is a state and not a quantity. */
var STAGE_COLOR={pending:'#7d8893',extract:'#8a6529',coverage:'#a97c2e',
  backfill:'#c79536',complete:'#e0ae45',downloading:'#f5c570',
  downloaded:'#43b581'};
/* cards past this many stay in the column's scroll area (see .colbody max-height) */
/* Same ramp, same reason: how much of a thing is done is a magnitude. The
   old three-colour version reported 69% in the same red as 3% and 71% in the
   same amber as 98%, which is a claim about kind rather than degree. Full
   stays green -- done IS a state. */
function pctColor(p){return p>=100?'#43b581'
  :p>=75?'#e0ae45':p>=50?'#c79536':p>=25?'#a97c2e':'#8a6529'}
/* One sparkline, several cards. The detection panel grew its own and every
   later card that wanted the same thing would have copied the four details
   that make it read as a trend rather than a rendering fault: hold it back
   until there are enough points, resize on every draw because the KPI card
   is a grid track with no final width on first paint, run the line edge to
   edge, and pin the floor at zero so a flat series does not look like a
   cliff. Written once, they stay true everywhere.
   `cap` is the y-axis ceiling for a value with a known range: a percentage
   drawn against its own maximum shows noise at full scale, so 3% CPU would
   fill the card. Omit it and the axis follows the data, which is what a
   throughput wants. */
function mkSpark(el,color,cap){
  var ch=null;
  return function(vals){
    if(typeof echarts==='undefined'||!el)return;
    if(!vals||vals.length<4){el.style.display='none';return}
    el.style.display='';
    if(!ch)ch=echarts.init(el,null,{renderer:'canvas'});
    ch.resize();
    ch.setOption({backgroundColor:'transparent',animation:false,
      grid:{left:0,right:0,top:2,bottom:0},
      xAxis:{type:'category',show:false,boundaryGap:false,
             data:vals.map(function(_,i){return i})},
      yAxis:{type:'value',show:false,min:0,max:cap||null},
      tooltip:{show:false},
      series:[{type:'line',data:vals,symbol:'none',
        lineStyle:{width:1,color:color+'.55)'},
        areaStyle:{color:color+'.10)'}}]});
    return ch;
  };
}
var SPARK_ACC='rgba(232,166,69,',SPARK_OK='rgba(67,181,129,',
    SPARK_COOL='rgba(91,143,214,',SPARK_HOT='rgba(216,116,58,';
var COPY_SVG='<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
/* Says what happened, in the toast. This page's three copy buttons are all
   icon-only, so copyOnto() -- which writes "Copied" onto the button and puts
   its label back -- would eat the glyph; the toast is where they can speak.
   They used to have a copyText() of their own that toasted, declared in the
   same scope as the one in COPY_JS below and shadowed by it, which left all
   three silent on success AND on failure. */
function copySay(t,what){
  copyText(t).then(function(ok){toast(ok?'copied '+what:'copy failed')});
}
var boardEl=document.getElementById('board'),toastEl=document.getElementById('toast'),dragKey=null,toastT;
function toast(t){toastEl.textContent=t;toastEl.classList.add('show');clearTimeout(toastT);toastT=setTimeout(function(){toastEl.classList.remove('show')},1700)}
function bcard(r){
  var d=document.createElement('div');d.className='rc';d.draggable=true;d.dataset.key=r.key;
  d.innerHTML='<div class="rn"><span class="nm">'+r.name+'</span><button type="button" class="cp" title="Copy &quot;'+r.key+'&quot;" aria-label="Copy region name">'+COPY_SVG+'</button></div><div class="rs"><span>'+fmt(r.downloaded)+' / '+fmt(r.dogs)+'</span><span class="rpc">'+r.pct+'%</span></div><div class="mini"><i style="width:'+Math.min(r.pct,100)+'%;background:'+pctColor(r.pct)+'"></i></div>';
  var cp=d.querySelector('.cp');cp.draggable=false;
  cp.addEventListener('mousedown',function(e){e.stopPropagation()});
  cp.addEventListener('click',function(e){e.stopPropagation();e.preventDefault();copySay(r.key,r.key)});
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
    e.preventDefault(); copySay(p.textContent.trim(),'the database path');
  });
})();
/* ── command generator ── */
var cmdRegion=document.getElementById('cmdRegion'),cmdOut=document.getElementById('cmdOut'),cmdGen=document.getElementById('cmdGen');
function esc(s){return (''+s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
__COPY_JS__
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
  /* Controls that appear inside a swapped-in detail region. Called after every
     swap, because innerHTML threw the old nodes and their listeners away. */
  /* The chips and the matrix are two views of one fact, so they drive each
     other: picking a direction filters the grid, and clicking an off-diagonal
     cell of the matrix picks that direction. */
  function wireWrong(){
    var wrap=document.getElementById('wrong');
    if(!wrap)return;
    var chips=wrap.querySelectorAll('.wrchip'),
        tiles=[].slice.call(wrap.querySelectorAll('.wrtile')),
        grid=wrap.querySelector('.wrgrid'),
        at=wrap.querySelector('.wrat'),
        navs=wrap.querySelectorAll('.wrnav'),
        group='',page=0;
    /* rows x whatever the grid is currently laying out, so a page is always a
       whole number of rows and the panel never changes height mid-flick */
    function perPage(){
      var cols=getComputedStyle(grid).gridTemplateColumns.split(' ').length;
      return Math.max(1,cols)*2;
    }
    function draw(){
      var mine=tiles.filter(function(t){return !group||t.dataset.g===group});
      var per=perPage(),pages=Math.max(1,Math.ceil(mine.length/per));
      if(page>=pages)page=pages-1;
      if(page<0)page=0;
      var lo=page*per,hi=lo+per;
      tiles.forEach(function(t){t.hidden=true});
      mine.slice(lo,hi).forEach(function(t){t.hidden=false});
      at.textContent=mine.length?
        ((lo+1)+'\u2013'+Math.min(hi,mine.length)+' of '+mine.length.toLocaleString()):
        'none';
      navs[0].disabled=page<=0;
      navs[1].disabled=page>=pages-1;
      [].forEach.call(chips,function(c){c.classList.toggle('on',c.dataset.g===group)});
    }
    function show(g){group=g;page=0;draw()}
    [].forEach.call(chips,function(c){
      c.addEventListener('click',function(){show(c.dataset.g)});
    });
    [].forEach.call(navs,function(b){
      b.addEventListener('click',function(){page+=(+b.dataset.d);draw()});
    });
    /* the column count changes with the viewport, so the page size does too */
    addEventListener('resize',draw);

    /* Flagging says the DATASET is wrong here, not the model. It is the one
       control on a panel that is otherwise all report, so it looks like one
       and it is reversible -- this is a judgement about someone else's
       judgement and it will sometimes be the one that is wrong. */
    var tally=document.getElementById('wrflagn'),
        run=(document.querySelector('.wrwrap')||{}).dataset||{};
    function saytally(n){
      if(!tally)return;
      tally.hidden=!n;
      tally.textContent=n?(n.toLocaleString()+' flagged for removal'):'';
    }
    [].forEach.call(wrap.querySelectorAll('.wrflag'),function(b){
      b.addEventListener('click',function(e){
        e.stopPropagation();
        var had=b.classList.contains('on'),tile=b.closest('.wrtile');
        b.disabled=true;
        b.classList.toggle('on',!had);
        if(tile)tile.classList.toggle('flagged',!had);
        fetch('/api/training/relabel',{method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify(had?{file:b.dataset.f,remove:true}:{
            file:b.dataset.f,was:b.dataset.was,should:b.dataset.should,
            dataset:run.dataset||'',run:run.run||''})})
          .then(function(r){return r.json()})
          .catch(function(){return null})
          .then(function(j){
            b.disabled=false;
            if(!j||!j.ok){
              /* an optimistic mark the server refused is a lie about what is
                 recorded, so put it back */
              b.classList.toggle('on',had);
              if(tile)tile.classList.toggle('flagged',had);
              return;
            }
            saytally(j.total);
          });
      });
    });
    saytally(wrap.querySelectorAll('.wrflag.on').length);
    draw();
    /* an off-diagonal cell IS a direction: true class down the column,
       predicted across the row -- the same pair the chips are keyed on */
    var cx=document.querySelector('.cxscroll table.cx');
    if(cx){
      var rows=cx.querySelectorAll('tbody tr'),
          heads=cx.querySelectorAll('thead tr:last-child .cxt');
      [].forEach.call(rows,function(tr,i){
        var pred=tr.querySelector('.cxl'), cells=tr.querySelectorAll('.cxc');
        [].forEach.call(cells,function(td,j){
          if(i===j||!heads[j]||!pred)return;
          var g=heads[j].textContent.trim()+'|'+pred.textContent.trim();
          if(!wrap.querySelector('.wrchip[data-g="'+g.replace(/"/g,'\\"')+'"]'))return;
          td.classList.add('err');
          td.title=(td.title?td.title+' ':'')+'\u2014 click to see them';
          td.addEventListener('click',function(){
            show(g);
            wrap.scrollIntoView({block:'start',
              behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?
                'auto':'smooth'});
          });
        });
      });
    }
  }
  function wireDetail(){
    var back=document.getElementById('trkBack');
    if(back) back.addEventListener('click',function(){
      window.__trkSel=null;
      var d=document.getElementById('trkdet');
      if(d) delete d.dataset.run;
      refreshTracker(true);
    });
    var cmp=document.getElementById('trkCmp');
    if(cmp) cmp.addEventListener('change',function(){
      var b=cmp.value; if(!b) return;
      openDiff(cmp.getAttribute('data-a'),b);
    });
  }
  function openDiff(a,b){
    if(!det) return;
    fetch('/api/training/diff?a='+encodeURIComponent(a)
          +'&b='+encodeURIComponent(b))
      .then(function(r){return r.json()}).then(function(j){
        if(!j||!j.html) return;
        det.innerHTML=j.html;
        /* the comparison is a view of the run that is still selected, so the
           selection and the periodic refresh both stay on it */
        bindCharts();
        wireDetail();
        wireWrong();
      }).catch(function(){});
  }
  function openRun(key,tr,quiet){
    if(!det) return;
    /* A finished run's detail does not change while you look at it, so a
       refresh that finds the same one already open leaves the DOM alone --
       every rebuild reset the mistake grid to page one and dropped the
       filter, on a 30 second timer. */
    if(quiet&&det.dataset.run===key&&det.firstChild){
      [].forEach.call(document.querySelectorAll('.thist tbody tr'),
        function(x){ x.classList.toggle('sel',x===tr); });
      return;
    }
    fetch('/api/training/run?key='+encodeURIComponent(key))
      .then(function(r){return r.json()}).then(function(j){
        if(!j||!j.html) return;
        det.innerHTML=j.html;
        det.dataset.run=key;
        window.__trkSel=key;
        [].forEach.call(document.querySelectorAll('.thist tbody tr'),
          function(x){ x.classList.toggle('sel',x===tr); });
        bindCharts();
        wireDetail();
        wireWrong();
        if(!quiet)det.scrollIntoView({block:'nearest',
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
    /* quiet: this is the 30s refresh putting the panel back, not a click. It
       must not scroll the page -- the reader is somewhere, and moving them
       there every half minute is the whole complaint -- and it must not
       rebuild a detail that has not changed, because rebuilding it throws
       away the mistake grid's filter and page. */
    if(keep) openRun(window.__trkSel,keep,true);
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
    /* While a past run is open, leave the panel alone. Its detail does not
       change -- the run finished -- and rebuilding it every 30 seconds threw
       away the mistake grid's filter and page, and scrolled the reader back
       to it. The live card resumes the moment they go back to it, which is a
       forced refresh anyway. */
    if(window.__trkSel&&!force) return;
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
  /* once immediately: the server splices a fresh section into the page, but a
     tab restored from bfcache or left open across a run change starts from
     whatever it last held */
  refresh(true);
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
    if(j.error){cmdOut.innerHTML='<div style="color:var(--red);padding:8px 2px">unknown region: '+esc(region)+'</div>';return;}
    var labels=['① Extract','② Coverage audit','③ Backfill metadata (no download)','④ Download images','⑤ Consolidate data → one drive (dry-run; add --execute)'];
    cmdOut.innerHTML=j.commands.map(function(c,i){
      return '<div class="cmdblock"><div class="cmdhead"><span>'+labels[i]+'</span><button type="button" class="cp" data-i="'+i+'" title="Copy command">'+COPY_SVG+'</button></div><pre>'+esc(c)+'</pre></div>';
    }).join('');
    cmdOut.querySelectorAll('.cp').forEach(function(b){b.addEventListener('click',function(e){e.preventDefault();copySay(j.commands[+b.dataset.i],'the command')})});
  }).catch(function(){cmdOut.innerHTML='<div style="color:var(--red);padding:8px 2px">failed to generate</div>'});
}
if(cmdGen){cmdGen.addEventListener('click',genCommands);cmdRegion.addEventListener('keydown',function(e){if(e.key==='Enter')genCommands()});}
/* ── atlas (Equal Earth, three layers, zoom-adaptive raster) ── */
(function(){
  var mapEl=document.getElementById('map');
  if(!mapEl||typeof echarts==='undefined')return;
  /* click-to-interact gate: the scrim eats wheel and drag, so the page keeps
     scrolling past the map until the user asks for the map instead. Esc, the
     lock button, or scrolling the map out of view re-arms it. */
  var unlockMap=function(){};
  (function(){
    var gate=document.getElementById('mapGate'),lock=document.getElementById('mapLock'),
        wrap=gate&&gate.parentNode;
    if(!gate||!lock)return;
    function setLocked(on){
      gate.hidden=!on;lock.hidden=on;
    }
    unlockMap=function(){setLocked(false)};
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
  /* Equal Earth (Šavrič et al. 2018), the closed-form equal-area projection.
     Hand-rolled because no CDN is allowed at runtime; forward is the
     published polynomial, inverse is 12 Newton steps on theta. */
  var EA1=1.340264,EA2=-0.081106,EA3=0.000893,EA4=0.003796,EM=Math.sqrt(3)/2,RAD=Math.PI/180;
  /* Equal Earth is separable: y depends on latitude alone, and x is
     longitude times a factor that also depends on latitude alone. Both
     factors are tabulated once at 0.005° and read back with linear
     interpolation, because echarts re-projects the ENTIRE world geometry on
     every frame of a zoom -- 68,000 calls per wheel tick here -- and four
     trigonometric functions per call is a cost paid sixty times a second.
     Interpolating a smooth function at that step is accurate to ~1e-10 deg,
     nine orders of magnitude below a pixel. */
  var LATN=36001,LAT0=-90,LATD=180/(LATN-1),TK=new Float64Array(LATN),
      TY=new Float64Array(LATN);
  (function(){
    for(var i=0;i<LATN;i++){
      var t=Math.asin(EM*Math.sin((LAT0+i*LATD)*RAD)),t2=t*t,t6=t2*t2*t2;
      TK[i]=Math.cos(t)/(EM*(EA1+3*EA2*t2+t6*(7*EA3+9*EA4*t2)));
      TY[i]=t*(EA1+EA2*t2+t6*(EA3+EA4*t2));
    }
  })();
  /* screen y grows DOWN, the math's grows up: negate y both ways */
  function eeFwd(lp){
    var lat=lp[1];
    if(!(lat>-90))lat=-90; else if(lat>90)lat=90;   /* also catches NaN */
    var f=(lat-LAT0)/LATD,i=f|0;
    if(i>LATN-2)i=LATN-2;
    var w=f-i;
    return [lp[0]*RAD*(TK[i]+(TK[i+1]-TK[i])*w),-(TY[i]+(TY[i+1]-TY[i])*w)];
  }
  function eeInv(xy){
    var x=xy[0],y=-xy[1],t=y,i,t2,t6,f,fp;
    for(i=0;i<12;i++){
      t2=t*t;t6=t2*t2*t2;
      f=t*(EA1+EA2*t2+t6*(EA3+EA4*t2))-y;
      fp=EA1+3*EA2*t2+t6*(7*EA3+9*EA4*t2);
      t-=f/fp;
    }
    t2=t*t;t6=t2*t2*t2;
    var c=Math.cos(t);if(Math.abs(c)<1e-9)c=1e-9;
    var s=Math.sin(t)/EM;if(s>1)s=1;if(s<-1)s=-1;
    return [EM*x*(EA1+3*EA2*t2+t6*(7*EA3+9*EA4*t2))/c/RAD,Math.asin(s)/RAD];
  }
  /* graticule + projected globe edge, so the projection reads as a globe and
     not as a warped rectangle. Sampled: meridians and the frame curve. */
  function graticule(){
    var lines=[],lon,lat,seg;
    for(lon=-150;lon<=150;lon+=30){
      seg=[];for(lat=-88;lat<=88;lat+=2)seg.push([lon,lat]);
      lines.push({coords:seg});
    }
    for(lat=-60;lat<=60;lat+=30){
      seg=[];for(lon=-180;lon<=180;lon+=3)seg.push([lon,lat]);
      lines.push({coords:seg});
    }
    var edge=[];
    for(lon=-180;lon<=180;lon+=3)edge.push([lon,88.8]);
    for(lat=88;lat>=-88;lat-=2)edge.push([180,lat]);
    for(lon=180;lon>=-180;lon-=3)edge.push([lon,-88.8]);
    for(lat=-88;lat<=88;lat+=2)edge.push([-180,lat]);
    return {lines:lines,edge:edge};
  }
  var mchips=Array.prototype.slice.call(document.querySelectorAll('.mchip')),
      regTog=document.getElementById('mapRegions'),
      cleanTog=document.getElementById('mapClean'),
      findEl=document.getElementById('mapFind'),
      resetEl=document.getElementById('mapReset'),
      ledeEl=document.getElementById('mapLede'),
      hud=document.getElementById('mapHud'),
      rampEl=document.getElementById('mapRamp'),
      minEl=document.getElementById('mapMin'),maxEl=document.getElementById('mapMax'),
      labEl=document.getElementById('mapRampLab'),statsEl=document.getElementById('mapStats');
  /* the three layers wear the site's own inks: amber = the harvest, green =
     the detector's calls, cool blue = the rate between them */
  var RAMPS={
    harvest:['#191024','#3b1a4e','#7c2d59','#c15a41','#e89a4d','#f0b85f','#fdf0cd'],
    dogs:['#08211c','#0d3f31','#1a6b4c','#2f9a68','#43b581','#8ce8b6','#e2fbee'],
    rate:['#141d2b','#1e3a5c','#2f6296','#4a8fc2','#8cc3e8','#e8f6ff']
  };
  var RATE_MIN=30;  /* a rate needs a denominator: cells with fewer harvested
                       frames than this stay off the hit-rate layer */
  /* The cells are drawn as BANDS: one echarts series per colour step, each in
     large mode. A single scatter with a visualMap builds one graphic element
     per cell, and at 15,000 cells that is 150-190ms of work on every frame of
     a zoom -- the jank. Large mode batches a whole series into one element,
     but it paints every point in one colour, which would throw away the
     density ramp. Twelve large series, each a flat colour sampled from the
     ramp, keeps the ramp and pays the batched price: measured 1 slow frame
     per zoom instead of 11, worst frame 57ms instead of 192ms. The ramp is
     quantised rather than continuous, which on a log scale is invisible. */
  var BANDS=12;
  function rampAt(cols,t){
    t=t<0?0:(t>1?1:t);
    var x=t*(cols.length-1),i=x|0;
    if(i>cols.length-2)i=cols.length-2;
    var f=x-i,a=cols[i],b=cols[i+1],o='#',j,va,vb;
    for(j=1;j<7;j+=2){
      va=parseInt(a.substr(j,2),16);vb=parseInt(b.substr(j,2),16);
      o+=('0'+Math.round(va+(vb-va)*f).toString(16)).slice(-2);
    }
    return o;
  }
  Promise.all([
    fetch('world.json').then(function(r){return r.json()}),
    fetch('map_points.json').then(function(r){return r.json()}),
    fetch('/api/board').then(function(r){return r.json()}).catch(function(){return null})
  ]).then(function(res){
    var world=res[0],md=res[1],board=res[2];
    echarts.registerMap('world',world);
    var levels=md.levels||{},dogLevels=md.dog_levels||{},
        outLevels=md.out_levels||{},dogOutLevels=md.dog_out_levels||{};
    if(!Object.keys(levels).length&&md.points)levels[String(md.res)]={res:md.res,max:md.max,points:md.points};
    var hasDogs=!!Object.keys(dogLevels).length;
    if(!hasDogs)mchips.forEach(function(c){if(c.dataset.l!=='harvest')c.style.display='none'});
    /* an older map_points.json has no outlier grids; hide the control rather
       than offer a toggle that would do nothing */
    var hasOut=!!Object.keys(outLevels).length;
    if(!hasOut){
      var cw=document.getElementById('mapCleanWrap');
      if(cw)cw.style.display='none';
      if(cleanTog)cleanTog.checked=true;
    }
    var fineRes=md.fine_res||0,fine={state:'none'};   /* none|loading|ready|failed */
    var keys=Object.keys(levels).map(parseFloat).sort(function(a,b){return b-a}); // coarse→fine
    var layer='harvest',cache={};
    function cellKey(p){return p[0]+'|'+p[1]}
    function clean(){return !cleanTog||cleanTog.checked}
    /* The clean grid and the outlier grid are separate files of cells, and a
       cell can appear in both (a real street with one bad frame in it), so
       showing outliers means SUMMING per cell, not concatenating -- a
       concatenation would draw two rects on the same spot and report the
       smaller count on hover. */
    function pointsOf(base,extra,resKey){
      var L=base[resKey];
      if(clean()||!L)return L||{res:parseFloat(resKey),max:0,points:[]};
      var X=(extra||{})[resKey];
      if(!X||!X.points.length)return L;
      var by={},outp=[],mx=0;
      L.points.forEach(function(p){by[cellKey(p)]=[p[0],p[1],p[2]]});
      X.points.forEach(function(p){
        var k=cellKey(p),h=by[k];
        if(h)h[2]+=p[2]; else by[k]=[p[0],p[1],p[2]];
      });
      for(var k in by){outp.push(by[k]);if(by[k][2]>mx)mx=by[k][2];}
      return {res:L.res,max:mx,points:outp};
    }
    function density(lyr,resKey){
      var ck=lyr+'|'+resKey+'|'+(clean()?'c':'a');
      if(cache[ck])return cache[ck];
      var out;
      if(lyr==='rate'){
        var H=pointsOf(levels,outLevels,resKey).points,
            D=pointsOf(dogLevels,dogOutLevels,resKey).points,dd={};
        D.forEach(function(p){dd[cellKey(p)]=p[2]});
        var data=[],vals=[];
        H.forEach(function(p){
          if(p[2]<RATE_MIN)return;
          var r=(dd[cellKey(p)]||0)/p[2];
          data.push({value:[p[0],p[1],r],cnt:p[2],hits:dd[cellKey(p)]||0});
          vals.push(r);
        });
        vals.sort(function(a,b){return a-b});
        /* p99 cap: one 30-frame cell at 100% must not flatten the ramp */
        out={res:parseFloat(resKey),max:vals.length?vals[Math.floor(vals.length*0.99)]:1,data:data};
        if(!out.max)out.max=0.01;
      }else{
        var L=lyr==='dogs'?pointsOf(dogLevels,dogOutLevels,resKey)
                          :pointsOf(levels,outLevels,resKey);
        out={res:L.res,max:Math.log10((L.max||1)+1),
          data:L.points.map(function(p){return {value:[p[0],p[1],Math.log10(p[2]+1)],cnt:p[2]}})};
      }
      cache[ck]=out;
      return out;
    }
    /* region anchors, dressed with the board's stage + progress when it is
       reachable; without it they still show name + frame count */
    var byKey={},labels={};
    if(board&&board.regions){
      board.regions.forEach(function(r){byKey[r.key]=r});
      labels=board.labels||{};
    }
    var regData=(md.regions||[]).map(function(r){
      var b=byKey[r.key]||{};
      return {name:(b.name||r.key).replace(/_/g,' '),value:[r.lon,r.lat],
        key:r.key,n:r.n,nb:r.n_bad||0,stage:b.stage||'',pct:b.pct,
        downloaded:b.downloaded,dogs:b.dogs};
    });
    /* Markers borrow the ACTIVE layer's ink rather than carrying stage
       colour on the glyph. An anchor is a region's median point, so it
       always lands on that region's densest ground -- a green stage dot sat
       inside the green dogs layer and vanished, and stage green on the amber
       harvest read as a third data colour that meant nothing on that map.
       Stage stays where it can be read: the tooltip. The dark fill and halo
       keep the ring legible over both a bright cell and open ocean. */
    var RING={harvest:'#f7d9a0',dogs:'#bff0d6',rate:'#cfe6f8'};
    function rings(){
      var c=RING[layer];
      return regData.map(function(r){
        return Object.assign({},r,{itemStyle:{color:'rgba(15,18,23,.72)',
          borderColor:c,borderWidth:1.6,
          shadowBlur:5,shadowColor:'rgba(10,12,16,.95)'},
          label:{color:c}});
      });
    }
    var regRings=rings();
    var cur=density(layer,String(keys[0]));
    var ch=echarts.init(mapEl,null,{renderer:'canvas'});
    function cellPx(res){ /* pixel footprint of one res° cell at current zoom */
      try{
        var a=ch.convertToPixel({geoIndex:0},[0,0]),b=ch.convertToPixel({geoIndex:0},[res,res]);
        return [Math.max(Math.abs(b[0]-a[0]),1.1),Math.max(Math.abs(b[1]-a[1]),1.1)];
      }catch(e){return [3,3];}
    }
    /* Cell size is read from here at paint time instead of being pushed in
       with setOption. A cell is a geo-anchored rect, so it has to grow as you
       zoom; pushing the new size through setOption meant the size only caught
       up when the roam debounce fired, so mid-zoom the raster was drawn at the
       previous zoom's size -- gaps opening up, then snapping shut. Now a wheel
       tick just writes this variable and the very next frame is already
       right. */
    /* large mode takes a number, not a [w,h]; the larger side so that the
       raster stays gapless where the projection stretches a cell */
    var cellSize=3;
    function cellNum(res){var s=cellPx(res);return Math.max(s[0],s[1]);}
    var REG=2+BANDS;              /* the region markers sit after the bands */
    /* series array for a partial update: pass null to leave a slot alone */
    function upd(bandData,size,regionData){
      var a=[{},{}],i;
      for(i=0;i<BANDS;i++){
        var s={};
        /* large mode CANNOT hold an empty series: echarts throws inside its
           own afterBrush when the batched point buffer is undefined, the
           paint aborts, and the map stops redrawing entirely -- which looks
           exactly like a freeze. Culling empties whichever bands are off
           screen, so an empty band falls back to the ordinary path, where
           empty is fine and costs nothing to draw. */
        if(bandData){s.data=bandData[i];s.large=bandData[i].length>0;}
        if(size)s.symbolSize=size;
        a.push(s);
      }
      a.push(regionData?{data:regionData}:{});
      return a;
    }
    function bandSeries(){
      var out=[],i,b=bandsOf(cur.data,cur.max);
      for(i=0;i<BANDS;i++)out.push({
        name:'cells'+i,type:'scatter',coordinateSystem:'geo',symbol:'rect',z:2,
        large:b[i].length>0,largeThreshold:0,symbolSize:cellSize,data:b[i],
        itemStyle:{color:rampAt(RAMPS[layer],(i+0.5)/BANDS),opacity:.92},
        animation:false});
      return out;
    }
    function bandsOf(data,max){
      var out=[],i;
      for(i=0;i<BANDS;i++)out.push([]);
      for(i=0;i<data.length;i++){
        var b=Math.floor(data[i].value[2]/(max||1)*BANDS);
        out[b<0?0:(b>BANDS-1?BANDS-1:b)].push(data[i]);
      }
      return out;
    }
    /* Only draw the cells that are on screen. echarts runs every point of the
       series through the projection on every repaint, so at the 0.15° grid a
       wheel tick was projecting ~72,000 points and at 0.05° a quarter of a
       million -- while the viewport held a few hundred. Zoomed in, that is
       the whole frame budget spent on cells nobody can see. Sampling the
       viewport edge (rather than just the corners) because the projection
       curves: on Equal Earth a corner is not the extreme of its own edge. */
    function viewBox(){
      var r=mapEl.getBoundingClientRect(),i,c,
          x0=1e9,x1=-1e9,y0=1e9,y1=-1e9,n=0;
      if(!r.width||!r.height)return null;
      for(i=0;i<=8;i++){
        var f=i/8,probes=[[r.width*f,0],[r.width*f,r.height],
                          [0,r.height*f],[r.width,r.height*f]];
        for(var j=0;j<4;j++){
          c=pxToLL(probes[j][0],probes[j][1]);
          if(!c||!isFinite(c[0])||!isFinite(c[1]))continue;
          n++;
          if(c[0]<x0)x0=c[0];if(c[0]>x1)x1=c[0];
          if(c[1]<y0)y0=c[1];if(c[1]>y1)y1=c[1];
        }
      }
      if(n<4)return null;   /* edges off the globe: cannot bound, draw it all */
      var mx=(x1-x0)*0.3+1,my=(y1-y0)*0.3+1;   /* margin so a pan has cells
                                                  ready before the next update */
      return {x0:x0-mx,x1:x1+mx,y0:y0-my,y1:y1+my};
    }
    /* what the cell layer currently holds, and the box it was culled to */
    var painted={box:null,world:true,n:0};
    function cullTo(data,b){
      var out=[],i,v;
      for(i=0;i<data.length;i++){
        v=data[i].value;
        if(v[0]>=b.x0&&v[0]<=b.x1&&v[1]>=b.y0&&v[1]<=b.y1)out.push(data[i]);
      }
      return out;
    }
    /* Repaint only when it would change something. setOption reprocesses the
       whole series, so calling it on every roam settle costs a second full
       render on top of the one the roam already did -- which is worse than
       the culling saves. Skip while the view stays inside the box the cells
       were culled to and the set has not shrunk much. */
    function paint(force){
      var b=viewBox(),world=!b||(b.x1-b.x0>=350&&b.y1-b.y0>=170);
      var data=world?cur.data:cullTo(cur.data,b);
      if(!force){
        if(world&&painted.world)return;
        if(!world&&!painted.world&&painted.box
           &&b.x0>=painted.box.x0&&b.x1<=painted.box.x1
           &&b.y0>=painted.box.y0&&b.y1<=painted.box.y1
           &&data.length>painted.n*0.5)return;
      }
      painted={box:world?null:b,world:world,n:data.length};
      ch.setOption({series:upd(bandsOf(data,cur.max),cellSize,null)});
    }
    var g=graticule();
    /* kept as its own object so Reset view can rebuild geo from it verbatim */
    /* center:[0,0] is not cosmetic. Left to fit, echarts centres on the
       PROJECTED BOUNDING BOX OF THE DATA, and world.json carries no
       Antarctica -- so the resting centre sat 0.138 units north of the
       origin the clamp bounds are built around. The first drag then snapped
       the map ~25px to meet a bound it had always been outside of, and
       Reset put it back outside again. Fitting at the origin makes the
       resting view, Reset and the clamp agree on one point, and frames the
       globe symmetrically in the panel besides. */
    var geoOpt={map:'world',roam:true,center:[0,0],scaleLimit:{min:1,max:40},
      projection:{project:eeFwd,unproject:eeInv},
      itemStyle:{areaColor:'#1d232c',borderColor:'#323a44',borderWidth:.5},
      emphasis:{disabled:true},select:{disabled:true}};
    function tipDensity(p){
      if(!p.data)return '';
      if(layer==='rate')
        return '<b>'+(p.data.value[2]*100).toFixed(1)+'%</b> of '+fmt(p.data.cnt)+
          ' frames had a dog call<br><span style="color:#98a2ad">'+cur.res+'° cell</span>';
      return '<b>'+fmt(p.data.cnt)+'</b> '+(layer==='dogs'?'frames with a dog call':'frames harvested')+
        '<br><span style="color:#98a2ad">'+cur.res+'° cell</span>';
    }
    function tipRegion(p){
      var d=p.data,rows='<b>'+d.name+'</b>';
      if(d.stage)rows+='<br><span style="color:'+(STAGE_COLOR[d.stage]||'#7d8893')+'">&#9679;</span> '+(labels[d.stage]||d.stage);
      if(d.downloaded!=null)rows+='<br>'+fmt(d.downloaded)+' / '+fmt(d.dogs)+' downloaded ('+d.pct+'%)';
      rows+='<br>'+fmt(d.n+(clean()?0:d.nb))+' frames on the map';
      if(!clean()&&d.nb)rows+=' <span style="color:#98a2ad">('+fmt(d.nb)+' GPS outliers)</span>';
      rows+='<br><span style="color:#69727d">click for its numbers</span>';
      return rows;
    }
    ch.setOption({
      backgroundColor:'transparent',
      /* No transition on any update. The cells are a raster: when the grid
         swaps at a zoom threshold, tweening every rect from its old cell to
         its new one sends the whole layer sliding across the map and back,
         which is the "it flies away and returns" of a zoom. A raster should
         cut, not dissolve. */
      animation:false,
      tooltip:{trigger:'item',backgroundColor:'#21262d',borderColor:'#2c333b',borderWidth:1,
        textStyle:{color:'#eef1f4'},
        formatter:function(p){return p.seriesIndex===REG?tipRegion(p):tipDensity(p)}},
      geo:geoOpt,
      series:[
        {type:'lines',coordinateSystem:'geo',polyline:true,silent:true,z:1,
         data:g.lines,lineStyle:{color:'rgba(130,140,150,.10)',width:.7}},
        {type:'lines',coordinateSystem:'geo',polyline:true,silent:true,z:1,
         data:[{coords:g.edge}],lineStyle:{color:'rgba(130,140,150,.28)',width:1.1}}
      ].concat(bandSeries()).concat([
        /* rings, not dots: a filled dot in stage green disappears into the
           dogs layer's own green cells; a ring reads as a marker */
        {name:'regions',type:'scatter',coordinateSystem:'geo',z:3,
         symbolSize:9,data:regRings,
         /* The label sits ON the densest part of its own region -- the
            anchor is that region's median frame -- so plain grey text was
            being read straight through a field of bright cells. A dark
            halo separates the glyphs from whatever is behind them without
            adding a box the map has to carry. */
         label:{show:true,position:'bottom',distance:4,fontSize:9.5,
           fontWeight:600,color:'#e8edf2',formatter:'{b}',
           textBorderColor:'rgba(8,10,14,.92)',textBorderWidth:3},
         emphasis:{scale:1.6,label:{color:'#fff',fontSize:11}},
         cursor:'pointer'}
      ])
    });
    function legend(){
      rampEl.style.background='linear-gradient(90deg,'+RAMPS[layer].join(',')+')';
      if(layer==='rate'){
        minEl.textContent='0%';
        maxEl.textContent=(cur.max*100).toFixed(cur.max<0.1?1:0)+'%';
        labEl.textContent='share of harvested frames with a dog call ≥ '+(md.conf_min||0.5)+' — cells with ≥ '+RATE_MIN+' frames';
      }else{
        minEl.textContent='1';
        maxEl.textContent=fmt(Math.round(Math.pow(10,cur.max)-1));
        labEl.textContent=(layer==='dogs'?'frames with a dog call ≥ '+(md.conf_min||0.5):'frames harvested')+' per '+cur.res+'° cell';
      }
      var s=fmt(md.total+(clean()?0:(md.outlier_total||0)))+' frames';
      if(hasDogs)s+=' · '+fmt(md.dogs_total+(clean()?0:(md.dogs_outlier_total||0)))+' with a dog call';
      s+=' · '+cur.data.length.toLocaleString()+' cells @ '+cur.res+'°';
      if(hasOut)s+=clean()
        ? ' · '+fmt(md.outlier_total||0)+' GPS outliers hidden'
        : ' · showing '+fmt(md.outlier_total||0)+' GPS outliers';
      statsEl.textContent=s;
    }
    function apply(){
      cur=density(layer,String(cur.res));
      regRings=rings();
      cellSize=cellNum(cur.res);
      /* the band colours belong to the layer, so a layer switch replaces them */
      ch.setOption({series:[{},{}].concat(bandSeries()).concat(
        [{data:(!regTog||regTog.checked)?regRings:[]}])});
      paint(true);
      /* the lede is the chip's own tooltip text, so the two never drift */
      var src=mchips.filter(function(c){return c.dataset.l===layer})[0];
      if(ledeEl&&src)ledeEl.textContent=src.title;
      legend();
    }
    cellSize=cellNum(cur.res);  // sizeable only once the geo layout exists
    ch.setOption({series:upd(null,cellSize,null)});
    legend();
    /* Keep the world inside the frame. echarts roam has no pan bounds, so a
       drag could carry the whole map off the panel and leave an empty field
       with no way back but Reset. The rule is the ordinary one for a map:
       never show emptiness past an edge. Along an axis where the map is
       larger than the viewport, the centre may travel only as far as its own
       edge; where it is smaller, it stays centred, which locks panning at the
       resting zoom -- correct, since the whole world is already in view.
       geo.center is in PROJECTED units, so the bounds are the projected
       extent of the globe, not 180/90. */
    var WY=Math.abs(eeFwd([0,90])[1]);
    function pxPerUnit(){
      try{
        var a=ch.convertToPixel({geoIndex:0},[0,0]),
            b=ch.convertToPixel({geoIndex:0},[90,45]),f=eeFwd([90,45]);
        var s=(b[0]-a[0])/f[0];
        return isFinite(s)&&s>0?s:null;
      }catch(_){return null;}
    }
    function clampCenter(){
      var g;
      /* getOption() DEEP CLONES the whole option -- every one of the tens of
         thousands of cell objects -- to hand back two numbers, and this runs
         once per drag mousemove. Read the live component instead. */
      try{g=ch.getModel().getComponent('geo').option;}catch(_){return;}
      if(!g||!g.center)return;
      var s=pxPerUnit();
      if(!s)return;
      var r=mapEl.getBoundingClientRect();
      if(!r.width||!r.height)return;   /* fold shut: nothing to measure */
      var hw=(r.width/2)/s,hh=(r.height/2)/s;
      var my=Math.max(0,WY-hh),
          cy=Math.min(my,Math.max(-my,g.center[1]));
      /* Equal Earth is not a rectangle. Its half-width falls from 2.707 at
         the equator to 1.604 at the poles, so bounding x by the globe's
         bounding BOX let the viewport sit in a corner wedge that holds no
         map at all: at zoom 14 with the camera against that bound, 0.4% of
         the panel was inside the globe and the nearest ink was four panel
         widths away. Bound x by the NARROWEST row actually on screen, so
         every visible row spans the panel. */
      var latA=eeInv([0,cy-hh])[1],latB=eeInv([0,cy+hh])[1],
          lat=Math.max(Math.abs(latA),Math.abs(latB));
      if(!isFinite(lat)||lat>90)lat=90;
      var mx=Math.max(0,Math.abs(eeFwd([180,lat])[0])-hw),
          cx=Math.min(mx,Math.max(-mx,g.center[0]));
      if(cx!==g.center[0]||cy!==g.center[1])
        ch.setOption({geo:{center:[cx,cy]}});
    }
    /* pick the finest grid whose cells are big enough to see; the 0.05° grid
       lives in its own file and is fetched the first time zoom warrants it */
    function wantRes(){
      var want=keys[0],i;
      for(i=0;i<keys.length;i++)if(cellPx(keys[i])[0]>=1.9)want=keys[i];
      if(fineRes&&cellPx(fineRes)[0]>=1.9){
        if(fine.state==='ready')want=fineRes;
        else if(fine.state==='none'){
          fine.state='loading';
          fetch('map_points_fine.json').then(function(r){return r.json()}).then(function(f){
            var k=String(fineRes);
            if((f.levels||{})[k]){
              levels[k]=f.levels[k];
              if((f.dog_levels||{})[k])dogLevels[k]=f.dog_levels[k];
              /* without these the toggle would silently stop working at the
                 deepest zoom, which is exactly where a stray point shows */
              if((f.out_levels||{})[k])outLevels[k]=f.out_levels[k];
              if((f.dog_out_levels||{})[k])dogOutLevels[k]=f.dog_out_levels[k];
              keys.push(fineRes);fine.state='ready';roamed();
            }else fine.state='failed';
          }).catch(function(){fine.state='failed'});
        }
      }
      return want;
    }
    var t=null;
    function roamed(){
      var want=wantRes(),swap=want!==cur.res;
      if(swap){
        cur=density(layer,String(want));
        cellSize=cellNum(cur.res);
        legend();
      }
      paint(swap);
    }
    /* Two speeds, on purpose. Resizing the cells is two convertToPixel calls
       and must land on the very next frame, so it runs on every roam event.
       Swapping to a different grid replaces tens of thousands of points, so
       it waits for the wheel to stop. */
    ch.on('georoam',function(){
      clampCenter();
      cellSize=cellNum(cur.res);
      ch.setOption({series:upd(null,cellSize,null)});
      drawHud();
      if(t)clearTimeout(t);
      t=setTimeout(roamed,130);
    });
    /* layer chips */
    mchips.forEach(function(c){c.addEventListener('click',function(){
      if(c.dataset.l===layer)return;
      layer=c.dataset.l;
      mchips.forEach(function(x){x.classList.toggle('on',x===c)});
      apply();
      /* the card is a view of the active layer, so it follows the tab --
         without this, clicking the same country after a switch read as a
         second click on it and just closed the card */
      if(popName)showCountry(popName);
    })});
    /* region markers on/off */
    if(regTog)regTog.addEventListener('change',function(){
      ch.setOption({series:upd(null,null,regTog.checked?regRings:[])});
    });
    /* outliers in or out: same grid pipeline, so every layer and every zoom
       level follows without a second code path */
    if(cleanTog)cleanTog.addEventListener('change',function(){
      apply();
      if(popName)showCountry(popName);
    });
    /* ── country card ── click a country, read it in the terms of the layer
       you are looking at. The totals are joined per 0.05° cell at build
       time, so a count is exact and only its attribution is approximate
       where a cell straddles a border. */
    var CT=md.countries||{},cRes=md.country_res||0.05;
    var cRank={},cList=Object.keys(CT);
    (function(){
      ['n','d','rate'].forEach(function(kind){
        var arr=cList.filter(function(k){
          return kind==='rate'?CT[k][0]>=RATE_MIN:true;
        }).sort(function(a,b){
          var A=CT[a],B=CT[b];
          if(kind==='n')return B[0]-A[0];
          if(kind==='d')return B[2]-A[2];
          return (B[2]/Math.max(B[0],1))-(A[2]/Math.max(A[0],1));
        });
        var m={};arr.forEach(function(k,i){m[k]=[i+1,arr.length]});
        cRank[kind]=m;
      });
    })();
    function pct(a,b){return b?(100*a/b):0;}
    /* Bar scale: the leading country, not the world total. A share-of-world
       bar is a sliver for everyone (the biggest single country is under 40%
       of the harvest), which says nothing about how one country compares to
       another -- the only question the card is being asked. */
    var cMax={n:1,d:1,rate:0.0001};
    cList.forEach(function(k){
      var v=CT[k];
      if(v[0]>cMax.n)cMax.n=v[0];
      if(v[2]>cMax.d)cMax.d=v[2];
      if(v[0]>=RATE_MIN){var r=v[2]/v[0];if(r>cMax.rate)cMax.rate=r;}
    });
    function popRow(k,v){
      return '<div class="cprow"><span>'+k+'</span><span>'+v+'</span></div>';
    }
    function showCountry(name){
      var v=CT[name],ink=RING[layer];
      if(!v){
        popBody.innerHTML='<div class="cpname">'+esc(name)+'</div>'
          +'<div class="cpnote">No harvested frames here. The atlas only '
          +'covers ground the sweep actually walked.</div>';
        pop.hidden=false;return;
      }
      var frames=v[0]+(clean()?0:v[1]),dogs=v[2]+(clean()?0:v[3]),
          worldF=md.total+(clean()?0:(md.outlier_total||0)),
          worldD=md.dogs_total+(clean()?0:(md.dogs_outlier_total||0)),
          rate=pct(dogs,frames),worldRate=pct(worldD,worldF),h='';
      h+='<div class="cpname">'+esc(name)+'</div>';
      var rk=cRank[layer==='harvest'?'n':(layer==='dogs'?'d':'rate')][name];
      h+='<div class="cprank">'+(rk?('#'+rk[0]+' of '+rk[1]+' countries'
            +(layer==='rate'?' with enough frames to rate':''))
          :'not ranked')+'</div>';
      if(layer==='rate'){
        if(v[0]<RATE_MIN){
          h+='<div class="cpnote">Only '+fmt(frames)+' frames here &mdash; '
            +'below the '+RATE_MIN+' a rate needs to mean anything, so this '
            +'country is left off the hit-rate layer.</div>';
        }else{
          h+='<div class="cpbig" style="color:'+ink+'">'+rate.toFixed(1)+'%</div>'
            +'<div class="cpunit">of frames here had a dog call</div>'
            +'<div class="cpbar"><i style="width:'+Math.min(100,pct(rate/100,cMax.rate))
            +'%;background:'+ink+'"></i></div>'
            +popRow('Dog calls',fmt(dogs))
            +popRow('Frames',fmt(frames))
            +popRow('Worldwide rate',worldRate.toFixed(1)+'%')
            +'<div class="cpnote">'+(rate>=worldRate
              ?(rate/Math.max(worldRate,0.0001)).toFixed(1)+'&times; the worldwide rate.'
              :'Below the worldwide rate.')+'</div>';
        }
      }else if(layer==='dogs'){
        h+='<div class="cpbig" style="color:'+ink+'">'+fmt(dogs)+'</div>'
          +'<div class="cpunit">frames with a dog call &ge; '+(md.conf_min||0.5)+'</div>'
          +'<div class="cpbar"><i style="width:'+Math.min(100,pct(dogs,cMax.d))
          +'%;background:'+ink+'"></i></div>'
          +popRow('Share of all dog calls',pct(dogs,worldD).toFixed(2)+'%')
          +popRow('Frames harvested',fmt(frames))
          +popRow('Hit rate',frames?rate.toFixed(1)+'%':'&mdash;')
          +'<div class="cpnote">Unreviewed detector output &mdash; some of '
          +'these are goats, sheep and shadows.</div>';
      }else{
        h+='<div class="cpbig" style="color:'+ink+'">'+fmt(frames)+'</div>'
          +'<div class="cpunit">frames harvested</div>'
          +'<div class="cpbar"><i style="width:'+Math.min(100,pct(frames,cMax.n))
          +'%;background:'+ink+'"></i></div>'
          +popRow('Share of the harvest',pct(frames,worldF).toFixed(2)+'%')
          +popRow('Cells with frames',v[4].toLocaleString()+' @ '+cRes+'°')
          +popRow('Dog calls',fmt(dogs));
        if(!clean()&&v[1])h+=popRow('GPS outliers shown',fmt(v[1]));
        if(clean()&&v[1])h+=popRow('GPS outliers hidden',fmt(v[1]));
      }
      popBody.innerHTML=h;
      pop.hidden=false;
    }
    var pop=document.getElementById('mapPop'),
        popBody=document.getElementById('mapPopBody'),
        popX=document.getElementById('mapPopX'),popName=null;
    function closePop(){pop.hidden=true;popName=null;}
    if(popX)popX.addEventListener('click',closePop);
    document.addEventListener('keydown',function(e){
      if(e.key==='Escape'&&pop&&!pop.hidden)closePop();
    });
    /* Which country is under the pointer, worked out from the coordinates
       rather than from an echarts region event. Two reasons: the geo emits
       nothing for a region unless triggerEvent is set, and even then the
       cells are a series ON TOP of the map, so over land -- the only place
       worth clicking -- the click never reaches the country underneath.
       Resolving from lon/lat treats a cell, a bare country and open sea
       alike. world.json is already in the page; 217 bounding boxes reject
       almost everything before any ray casting happens. */
    var cIdx=null;
    function buildCIdx(){
      cIdx=[];
      (world.features||[]).forEach(function(f){
        var nm=(f.properties||{}).name;
        if(!nm||!f.geometry)return;
        var polys=f.geometry.type==='MultiPolygon'?f.geometry.coordinates
                                                 :[f.geometry.coordinates];
        var x0=1e9,x1=-1e9,y0=1e9,y1=-1e9,parts=[];
        polys.forEach(function(poly){
          if(!poly||!poly.length)return;
          parts.push(poly);
          poly.forEach(function(ring){
            (ring||[]).forEach(function(pt){
              if(pt[0]<x0)x0=pt[0];if(pt[0]>x1)x1=pt[0];
              if(pt[1]<y0)y0=pt[1];if(pt[1]>y1)y1=pt[1];
            });
          });
        });
        if(parts.length)cIdx.push({n:nm,b:[x0,y0,x1,y1],p:parts});
      });
    }
    function inRing(x,y,ring){
      var inside=false,i,j,xi,yi,xj,yj;
      for(i=0,j=ring.length-1;i<ring.length;j=i++){
        xi=ring[i][0];yi=ring[i][1];xj=ring[j][0];yj=ring[j][1];
        if(((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi)+xi))inside=!inside;
      }
      return inside;
    }
    function countryAt(x,y){
      if(!cIdx)buildCIdx();
      for(var k=0;k<cIdx.length;k++){
        var c=cIdx[k],b=c.b;
        if(x<b[0]||x>b[2]||y<b[1]||y>b[3])continue;
        for(var m=0;m<c.p.length;m++){
          var poly=c.p[m],hit=false;
          /* even-odd across every ring, so a hole punches back out */
          for(var r=0;r<poly.length;r++)if(inRing(x,y,poly[r]))hit=!hit;
          if(hit)return c.n;
        }
      }
      return null;
    }
    /* Every click on the map answers the same question -- which country is
       this -- so it is answered here and nowhere else. A click used to have
       to wait a tick first, to see whether a region marker had claimed it
       for the command generator; that deferral is what made the card feel
       slow, and a click that landed on a marker did nothing at all. Since
       the markers sit on their region's densest ground, that was much of the
       map. The generator is still there, above, driven by its own box. */
    buildCIdx();     /* ~26K vertices: built now so the first click is not
                        the one that pays for it */
    ch.getZr().on('click',function(e){
      var c=pxToLL(e.offsetX,e.offsetY),nm=c?countryAt(c[0],c[1]):null;
      if(!nm||nm===popName){closePop();return;}
      popName=nm;showCountry(nm);
    });
    /* Reset view: whole world at the resting zoom, harvest layer, markers on.
       At rest geo.center is NULL -- echarts fits the map to the container and
       only fills center in once you roam -- so there is no "home" pair to
       stash and put back. Rebuilding the geo component restores that fitted
       state exactly, at whatever size the panel is now. */
    if(resetEl)resetEl.addEventListener('click',function(){
      if(layer!=='harvest'){
        layer='harvest';
        mchips.forEach(function(x){x.classList.toggle('on',x.dataset.l==='harvest')});
      }
      if(regTog)regTog.checked=true;
      if(cleanTog&&hasOut)cleanTog.checked=true;
      if(findEl)findEl.value='';
      ch.setOption({geo:geoOpt},{replaceMerge:['geo']});
      ch.resize();          /* re-fit if the panel was resized while roaming */
      apply();
      roamed();
      toast('map reset');
    });
    /* fly to a region by name */
    if(findEl)findEl.addEventListener('change',function(){
      var q=findEl.value.trim().toLowerCase().replace(/ /g,'_');
      if(!q)return;
      var hit=regData.filter(function(r){return r.key.toLowerCase()===q||r.key.toLowerCase().indexOf(q)===0})[0];
      if(!hit){toast('no region called '+findEl.value);return;}
      unlockMap();
      /* with a custom projection, geo.center is in PROJECTED coordinates --
         lon/lat here pans the camera off the globe into blank space */
      ch.setOption({geo:{center:eeFwd(hit.value),zoom:5}});
      clampCenter();
      roamed();
      findEl.blur();
    });
    /* surveyor HUD. echarts 5.6 hands convertFromPixel's input to unproject
       BEFORE undoing the view transform, so that path returns garbage;
       invert it ourselves from two convertToPixel anchors instead. */
    function pxToLL(px,py){
      try{
        var a=ch.convertToPixel({geoIndex:0},[0,0]);        /* projects to (0,0) */
        var b=ch.convertToPixel({geoIndex:0},[90,45]);
        var f=eeFwd([90,45]);
        var s=(b[0]-a[0])/f[0];
        if(!isFinite(s)||!s)return null;
        return eeInv([(px-a[0])/s,(py-a[1])/s]);
      }catch(_){return null;}
    }
    /* Redrawn on roam as well as on move: zooming under a still cursor
       changes both the coordinate and the grid beneath it, and the readout
       was left asserting the grid it had a moment ago. */
    var hudPx=null;
    function drawHud(){
      if(!hudPx){hud.textContent='';return;}
      var c=pxToLL(hudPx[0],hudPx[1]);
      if(!c||!isFinite(c[0])||Math.abs(c[0])>180||Math.abs(c[1])>90){hud.textContent='';return;}
      hud.textContent=Math.abs(c[1]).toFixed(1)+'°'+(c[1]<0?'S':'N')+' '+
        Math.abs(c[0]).toFixed(1)+'°'+(c[0]<0?'W':'E')+' · '+cur.res+'° grid';
    }
    mapEl.addEventListener('mousemove',function(e){
      var r=mapEl.getBoundingClientRect();
      hudPx=[e.clientX-r.left,e.clientY-r.top];
      drawHud();
    });
    mapEl.addEventListener('mouseleave',function(){hudPx=null;hud.textContent=''});
    /* A resize refits the map, so a centre that was against an edge a moment
       ago can now sit outside the bounds -- narrowing the window while panned
       into a corner left a quarter of the panel empty. The clamp only ran on
       roam, and a resize is not one. Repaint too: the viewport the cells were
       culled to is the old one. */
    function refit(){
      var r=mapEl.getBoundingClientRect();
      if(!r.width||!r.height)return;   /* fold shut: nothing to measure */
      ch.resize();clampCenter();paint(true);drawHud();
    }
    window.addEventListener('resize',refit);
    /* Reopening the fold is a viewport change too. While it was shut the
       panel measured 0x0, so a resize in the meantime clamped against
       nothing; the generic fold handler only calls resize(), which refits
       the geo but leaves the centre where it was. */
    var mfold=document.getElementById('f-map');
    if(mfold)mfold.addEventListener('toggle',function(){
      if(mfold.open)refit();
    });
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
/* ── the machine: cpu, memory, the card, and what it is all waiting on ───── */
(function(){
  var el={}, ids=['syCpu','syLoad','syMem','sySwap','syGpu','syVram','syIo',
                  'syCpuStall','syMeta','syVerdict'];
  for(var i=0;i<ids.length;i++)el[ids[i]]=document.getElementById(ids[i]);
  if(!el.syCpu)return;
  var N=90,hist={cpu:[],mem:[],gpu:[],io:[]},DASH='\\u2014';
  /* percentages are drawn against 100, not against their own maximum: a card
     whose axis follows the data turns 3% jitter into a mountain range */
  var draw={
    cpu:mkSpark(document.getElementById('sySparkCpu'),SPARK_ACC,100),
    mem:mkSpark(document.getElementById('sySparkMem'),SPARK_COOL,100),
    gpu:mkSpark(document.getElementById('sySparkGpu'),SPARK_OK,100),
    io:mkSpark(document.getElementById('sySparkIo'),SPARK_HOT,100)};
  function gb(b){return (b/1073741824)}
  function push(k,v){
    if(v==null||v!==v)return;
    hist[k].push(+v.toFixed(1));
    if(hist[k].length>N)hist[k].shift();
    draw[k](hist[k]);
  }
  function pc(v,d){return v==null?DASH:v.toFixed(d==null?0:d)+'%'}
  function paint(j){
    if(!j)return;
    el.syCpu.textContent=pc(j.cpu);
    el.syLoad.textContent=(j.load==null?DASH:j.load.toFixed(2)+' load')+
      ' \\u00b7 '+j.cores+' cores';
    /* load is per-core: 18 on a 16-core box is a queue, not a busy machine,
       and the two readings disagreeing is the interesting case */
    el.syLoad.className='sysub'+(j.load!=null&&j.cores&&j.load>j.cores*1.25
      ?' warn':'');
    var mu=gb(j.mem_used||0),mt=gb(j.mem_total||0);
    el.syMem.textContent=mt?mu.toFixed(1)+' / '+mt.toFixed(0)+' GB':DASH;
    /* Swap in use is the line that matters on this box: the target is to sit
       just under the RAM ceiling, and the first sign of overshooting is not
       a memory figure but pages going to disk. */
    var su=gb(j.swap_used||0);
    el.sySwap.textContent=j.swap_total
      ?(su<0.05?'no swap in use':su.toFixed(1)+' GB swapped')
      :'no swap configured';
    el.sySwap.className='sysub'+(su>=8?' bad':su>=0.5?' warn':'');
    var g=j.gpu;
    /* one decimal: this workload's honest figure is single-digit, and a whole
       number would round the difference between an idle card and a working
       one away to nothing */
    el.syGpu.textContent=g?pc(g.util,1):'no card';
    el.syVram.textContent=g&&g.mem_total
      ?(g.util_peak?'peak '+g.util_peak.toFixed(0)+'% \\u00b7 ':'')+
       (g.mem_used/1024).toFixed(1)+' GB'+
       (g.temp!=null?' \\u00b7 '+g.temp.toFixed(0)+'\\u00b0C':'')+
       (g.power!=null?' \\u00b7 '+g.power.toFixed(0)+' W':'')
      :(g?DASH:'nvidia-smi not answering');
    el.syIo.textContent=j.io_stall==null?DASH:pc(j.io_stall,1);
    el.syCpuStall.textContent=j.cpu_stall==null
      ?'kernel reports no pressure'
      :'cpu stall '+j.cpu_stall.toFixed(1)+'%';
    el.syIo.parentNode.className='kpi spk'+(j.io_stall>=40?' hot':'');
    /* The one sentence worth putting in the header: on a box that spends its
       life on long batch jobs, WHICH resource is the ceiling decides whether
       any knob is worth turning. Storage wins here more often than not --
       decoding a 8000x4000 panorama is 98% of the gate's cost, so the card
       sits near zero while six disks are read flat out. */
    var v='';
    if(j.io_stall!=null&&j.io_stall>=40)v='waiting on disk';
    else if(g&&g.util!=null&&g.util>=70)v='gpu bound';
    else if(j.cpu!=null&&j.cpu>=85)v='cpu bound';
    el.syVerdict.textContent=v;
    el.syVerdict.className='syverdict'+(v==='waiting on disk'?' io':
      v==='gpu bound'?' gpu':'');
    el.syMeta.textContent=g&&g.name?g.name:'';
    push('cpu',j.cpu);
    push('mem',mt?100*mu/mt:null);
    push('gpu',g?g.util:null);
    push('io',j.io_stall);
  }
  /* A collapsed fold is not being read, and this one costs a live nvidia-smi
     to answer. The detection panel already stops when its fold shuts; this
     one polled on regardless, twice a second, at whatever the section was
     hiding. */
  var fold=document.getElementById('f-sys');
  function poll(){
    if(document.hidden||(fold&&!fold.open))return;
    fetch('/api/sys').then(function(r){return r.json()})
      .catch(function(){return null}).then(paint);
  }
  if(fold)fold.addEventListener('toggle',function(){
    if(fold.open){poll();for(var k in hist)draw[k](hist[k])}
  });
  document.addEventListener('visibilitychange',function(){
    if(!document.hidden)poll();
  });
  poll();setInterval(poll,2000);
  var rz=null;
  window.addEventListener('resize',function(){
    if(rz)clearTimeout(rz);
    rz=setTimeout(function(){
      for(var k in hist)draw[k](hist[k]);
    },150);
  });
})();

/* ── which stage the section is showing ─────────────────────────────────────
   Three passes over one store, so one section and one set of card shapes.
   Only one polls at a time: the others' numbers do not change while they are
   not running, and three pollers for one open fold is a cost with nothing
   behind it.
   The two classifier stages share the SAME panel. They differ in what they
   read and what they call the positive class, and in nothing the layout can
   see, so the labels are swapped rather than the markup duplicated. */
(function(){
  var sw=document.getElementById('stagesw');
  if(!sw)return;
  var STAGES={detect:1,gate:1,leash:1};
  var stage=STAGES[localStorage.getItem('sdStage')]
    ?localStorage.getItem('sdStage'):'detect';
  function paintStage(){
    var det=stage==='detect';
    var a=document.getElementById('detOn'), b=document.getElementById('gateOn'),
        off=document.getElementById('detOff');
    if(a)a.hidden=!det;
    if(off)off.hidden=!det;
    if(b)b.hidden=det;
    /* by id, not by class: querySelector('.swctl') meant "whichever control
       comes first in the document", which is only the detector's by accident
       of source order */
    var sc=document.getElementById('sweepCtl'), gc=document.getElementById('gateCtl');
    if(sc)sc.hidden=!det;
    if(gc)gc.hidden=det;
    var h=document.getElementById('stgHint');
    if(h)h.textContent=det
      ? 'yolo26x @1280 — live, updates every 5 s while open'
      : stage==='leash'
        ? 'leash_v2_001 over every box the gate called a dog'
        : 'dogbin_008 over every detection the sweep committed';
    /* the share card is the one label that is about the model rather than
       the run, so it is the one that has to follow the stage */
    var dl=document.getElementById('gDogLbl');
    if(dl)dl.textContent=stage==='leash'?'On a leash':'Called dog';
    /* each stage audits itself; the link follows the tab rather than always
       pointing at the gate's */
    var al=document.getElementById('gAuditLink');
    if(al)al.href=stage==='leash'?'/audit/leash':'/audit';
    var bs=sw.querySelectorAll('.stagebtn');
    for(var i=0;i<bs.length;i++){
      var on=bs[i].getAttribute('data-stage')===stage;
      bs[i].classList.toggle('on',on);
      bs[i].setAttribute('aria-selected',on?'true':'false');
    }
    window.__stage=stage;
    if(!det&&window.__gatePoll)window.__gatePoll();
  }
  sw.addEventListener('click',function(e){
    var b=e.target&&e.target.closest&&e.target.closest('.stagebtn');
    if(!b)return;
    stage=b.getAttribute('data-stage');
    localStorage.setItem('sdStage',stage);
    paintStage();
  });
  paintStage();
})();

/* ── the gate's progress, read from the shards it has written ────────────── */
(function(){
  var box=document.getElementById('gateOn');
  if(!box)return;
  function fmt(n){return (n||0).toLocaleString()}
  function dur(s){
    if(s==null||!isFinite(s))return '—';
    var h=Math.floor(s/3600), m=Math.round((s%3600)/60);
    return h?h+'h '+m+'m':m+'m';
  }
  /* Same trend line the detector's img/s card carries. It is worth more here
     than there: this rate is what the ETA is built from, and a run whose
     throughput is sagging as it moves onto a slower drive says so in the
     shape long before it says so in the number. */
  var gSpark=mkSpark(document.getElementById('gateSpark'),SPARK_OK),
      gHist=[],GN=90,histFor=null;
  function paint(j){
    if(!j)return;
    /* the trend belongs to ONE stage: carrying the gate's rate into the leash
       panel would draw a line from a run that is not this one */
    if(j.stage!==histFor){gHist=[];histFor=j.stage}
    pace(j.running?2000:5000);
    /* only while it runs: a stopped run's last rate repeated ninety times
       would draw a flat line that looks like a measurement */
    if(j.running&&j.rate!=null&&j.rate===j.rate){
      gHist.push(+(+j.rate).toFixed(1));
      if(gHist.length>GN)gHist.shift();
    }
    gSpark(gHist);
    /* A stage with no plan yet has no totals at all, and fmt(undefined) is
       the string "NaN" -- six of them across the cards, which reads as a
       broken panel rather than one that has not been asked to do anything.
       Absent is a dash. */
    /* `planned` is set on every payload this server builds, so a missing one
       means the response is not a progress document at all -- the error path
       returns {ever:false,error:...}, and treating that as "planned" put
       "0 of 0 detections judged" under the error message. Absent is unknown,
       and unknown is a dash. */
    var DASH='\u2014',known=j.planned===true;
    /* A stage with every planned shard on disk is FINISHED, which is a third
       thing next to running and stopped-part-way -- and the server is the one
       that knows it, because "done" is the shard test the runner itself uses
       and not rows>=total. Without it a gate that had judged all 4,688,510 of
       its detections said "paused", offered "Resume", and put "3,292,062
       frames to open" under a full bar reading 100.0%. */
    var done=known&&j.done===true;
    function K(v,f){return (!known||v==null||v!==v)?DASH:f(v)}
    var pct=(j.pct||0)*100;
    document.getElementById('gPct').textContent=
      (known&&j.total)?pct.toFixed(1)+'%':DASH;
    document.getElementById('gDone').textContent=K(j.rows,fmt);
    /* a finished stage has no ETA and does not need one -- it says so, the
       same word the detector's card uses for the same state */
    document.getElementById('gEta').textContent=
      j.running?dur(j.eta_s):(done?'complete':DASH);
    document.getElementById('gDog').textContent=
      j.dog_share==null?'—':(j.dog_share*100).toFixed(1)+'%';
    /* Hover gives the counts in full. fmt() abbreviates past a thousand
       everywhere on this page, which is right for a headline and wrong for
       the one place you went looking for the exact number -- so these are
       written out in full, with separators. */
    var dcard=document.getElementById('gDogCard');
    if(dcard)dcard.title=(!known||j.dogs==null)
      ?'no verdicts yet'
      :j.dogs.toLocaleString('en-US')+' of '+j.dogs_of.toLocaleString('en-US')+
       ' boxes judged so far were called a dog \\u2014 '+
       (j.dogs_of-j.dogs).toLocaleString('en-US')+' were not';
    document.getElementById('gNow').textContent=j.running?K(j.rate,fmt):DASH;
    document.getElementById('gSus').textContent=K(j.sustained,fmt);
    document.getElementById('gFill').style.width=Math.min(100,pct)+'%';
    document.getElementById('gCount').textContent=known
      ?fmt(j.rows)+' of '+fmt(j.total)+' detections judged'+
       (j.shards?' \u00b7 '+fmt(j.shards)+' shards written':'')
      :'nothing planned for this stage yet';
    /* while it runs, the honest headline is frames OPENED: a shard is 20,000
       of them, so the judged count above sits still for minutes at a time and
       a run that had just started read as a run that had done nothing */
    document.getElementById('gRun').textContent=
      j.images_done!=null
        ? fmt(j.images_done)+' of '+fmt(j.images)+' frames opened'+
          (j.img_s?' · '+j.img_s.toFixed(0)+' frames/s':'')+
          ' · '+(j.model||'')
        : (j.images?fmt(j.images)+(done?' frames opened · ':' frames to open · ')+
            (j.model||'')+
            (j.created?' · planned '+j.created:''):'—');
    /* Three things this line has to be able to say, and only one of them is
       an error: a stage waiting on the one upstream, a stage that has never
       been planned, and a run that has started but not yet closed a shard. */
    var up=j.upstream, waiting=up&&!up.ready;
    document.getElementById('gMeta').textContent=j.error||
      (waiting
        ? 'waiting on the '+up.title+' \u2014 '+fmt(up.rows)+' of '+
          fmt(up.total)+' boxes judged. This stage reads its verdicts, so it '+
          'cannot be planned until that finishes.'
        : !j.planned
          ? (j.planning
              ? 'building the work list for this stage \u2014 a join over '+
                'the whole store, a few minutes'
              : 'no work list for this stage yet \u2014 Plan builds one')
          : j.running&&j.images_done==null&&!j.rows
            ? 'running \u2014 nothing is written until the first shard '+
              'closes at 20,000 frames'
            : '');
    var pill=document.getElementById('gateState'),
        btn=document.getElementById('gateBtn');
    if(pill){
      pill.textContent=j.running?'running'
        :j.planning?'planning'
        :done?'complete'
        :waiting?'waiting':(j.rows?'paused':'not started');
      pill.className='swpill'+(j.running?' on':'');
    }
    if(btn&&btn.dataset.busy!=='1'){
      var what=j.stage==='leash'?'leash':'gate';
      /* A stage with no work list needs one built before it can run, and
         that is a different button doing a different thing -- saying Run and
         quietly planning instead would be a lie about what the click does. */
      var needsPlan=known===false&&!waiting;
      /* Nothing to press on a finished stage: the runner skips every shard
         already on disk, so a click would start a job that exits having done
         nothing -- and "Resume" said it would pick up work that does not
         exist. Unavailable with a sentence saying why, the way `waiting` is. */
      var over=done&&!j.running;
      btn.disabled=!j.can_run||j.planning||over;
      btn.textContent=j.planning?'Planning\u2026'
        :j.running?'Stop'
        :needsPlan?'Plan '+what
        :(j.rows&&!over?'Resume':'Run '+what);
      btn.title=waiting
        ? 'the '+up.title+' has to finish first \u2014 this stage judges its dogs'
        : j.planning
          ? 'building the work list \u2014 a few minutes over the whole store'
          : needsPlan
            ? 'build the work list for this stage, then run it'
            : over
              ? 'every planned shard is written \u2014 this stage has judged '+
                'all of it, and there is nothing left to resume'
              : j.can_run
                ? (j.running?'stop after the shard in flight \u2014 finished shards are kept'
                            :(what==='leash'
                              ?'judge every box the gate called a dog; resumes where it left off'
                              :'judge every detection the sweep committed; resumes where it left off'))
                : 'no interpreter configured for the '+what;
    }
  }
  var gen=0;
  var gfold=document.getElementById('f-detect');
  function poll(){
    var st=window.__stage;
    if((st!=='gate'&&st!=='leash')||document.hidden)return;
    /* same rule as the detector's: a shut fold is not being looked at, and
       this one now asks every two seconds */
    if(gfold&&!gfold.open)return;
    var mine=++gen;
    fetch('/api/gate?stage='+encodeURIComponent(st))
      .then(function(r){return r.json()})
      .catch(function(){return null})
      .then(function(j){if(mine===gen)paint(j)});
  }
  window.__gatePoll=poll;
  /* A running gate publishes every second, so ask at a rate you can see it
     at. Idle, nothing changes between shards or at all, and 5 s of a poll
     that returns the same document is 5 s of nothing. */
  var tick=null,every=0;
  function pace(ms){
    if(every===ms)return;
    every=ms;
    if(tick)clearInterval(tick);
    tick=setInterval(poll,ms);
  }
  pace(5000);
  var btn=document.getElementById('gateBtn');
  if(btn)btn.addEventListener('click',function(){
    var stopping=this.textContent.indexOf('Stop')===0;
    /* one line: the page template is a NON-raw Python string, so a \n
       written here arrives as a real newline and takes the whole script
       with it -- the same trap an escaped apostrophe sprang once already */
    if(stopping&&!confirm('Stop the gate? The shard in flight is lost; every '+
      'finished shard is kept, and Resume picks up from there.'))return;
    btn.dataset.busy='1';btn.disabled=true;
    btn.textContent=stopping?'Stopping…':'Starting…';
    fetch('/api/gate',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:stopping?'stop':'start',
                           stage:window.__stage==='leash'?'leash':'gate'})})
      .then(function(r){return r.json()})
      .catch(function(){return {ok:false,msg:'the dashboard did not answer'}})
      .then(function(j){
        btn.dataset.busy='';btn.disabled=false;
        if(j&&!j.ok&&j.msg)document.getElementById('gMeta').textContent=j.msg;
        poll();setTimeout(poll,1500);
      });
  });
})();

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
  /* What stays true after the run stops. A finished sweep's totals are facts
     about work that happened; only rates and an ETA describe a process, and
     only those go to a dash when there is no process. */
  function p(v,f){return (v==null||v!==v)?DASH:f(v)}
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
    /* Three states, not two. A fetch that failed and a status document that
       is not there both say NOTHING about the sweep, and painting either as
       "idle" redrew a finished 32.5M-image harvest as "0 of 17 complete":
       the remembered roster with every bar at zero and a header counting
       them as not started. Unknown holds the last good numbers instead, the
       way the machine and the gate panels beside it do with `if(!j)return`.
       Only the very first poll has nothing to hold, and that says so. */
    if(!j||j.ever===false){
      if(!lastJ){
        off.textContent='no sweep status yet';
        dEl.innerHTML='<div class="dnone">no drives reported</div>';
        rEl.innerHTML='<div class="dnone">no per-region data</div>';
      }
      return;
    }
    lastJ=j;
    live=!!j.running;
    /* idle is a STATUS LINE now — the cards below stay put and go to dashes */
    off.style.display=live?'none':'';
    if(!live)off.textContent='sweep idle'+(j.age_s!=null?' \\u2014 last run '+agoTxt(j.age_s):'')+
      (j.state==='failed'?' (failed)':'');
    /* headline: % complete, human ETA, now/sustained throughput. imgs_done is
       GLOBAL (all-time, across restarts) — the %, the bar and the ETA are all
       against imgs_total, never against the per-process run_imgs_done. */
    var ips=j.img_per_sec||{},rNow=+ips.w60||0,rSus=+ips.w900||0;
    /* NOT gated on live: how much of the store has been swept is a fact about
       work done. Gating it printed 0.00% and an empty bar under a finished
       sweep, which reads as "nothing happened" -- the exact opposite. */
    var tot=+j.imgs_total||0,pct=tot?100*(+j.imgs_done||0)/tot:0;
    /* always 2 decimals: at ~50 img/s the third digit is the only one that
       visibly moves, and a field that switches precision at 10% jitters */
    hPct.textContent=tot?(pct.toFixed(2)+'%'):DASH;
    hDone.textContent=p(j.imgs_done,hnum);
    /* a finished run has no ETA and does not need one -- it says so */
    hEta.textContent=live?etaTxt(j.eta_s):(j.finished?'complete':DASH);
    hNow.textContent=n(ips.w60,function(v){return (+v).toFixed(1)});
    hSus.textContent=n(ips.w900,function(v){return (+v).toFixed(1)});
    hFill.style.width=Math.min(pct||0,100).toFixed(2)+'%';
    if(hPos)hPos.textContent=p(j.positives,hnum);
    /* every figure on this line is ALL-TIME, so say so -- the same line used
       to mix a global image count with a per-process positives count and
       reported a 0.1% hit rate against a true ~2.8% */
    hCount.textContent='all-time: '+p(j.imgs_done,hnum)+' of '+
      (tot?hnum(tot):DASH)+' images \\u00b7 '+p(j.positives,hnum)+' positives'+
      (j.positive_rate!=null?' ('+j.positive_rate+'%)':'')+
      (j.boxes_total!=null?' \\u00b7 '+hnum(j.boxes_total)+' boxes':'');
    if(hRun)hRun.textContent=(live?'this run: ':'last run: ')+
      p(j.run_imgs_done,hnum)+' images'+
      (j.run_positives!=null?' \\u00b7 '+p(j.run_positives,hnum)+' positives':'')+
      (j.started_at?' \\u00b7 since '+j.started_at:'');
    if(live){samples.push(+rNow.toFixed(1));if(samples.length>SPARK_N)samples.shift();}
    drawSpark();
    /* per drive: name · bar · rate; queue only when nonzero, badge only when
       stalled. Idle keeps the rows (from the remembered roster) and dashes
       the numbers rather than dropping the block. */
    var dr=j.drives||{},dk=Object.keys(dr).sort();
    keep('drives',dk);
    dEl.innerHTML=(dk.length?dk:roster.drives).map(function(nm){
      /* the bar is progress (permanent); only the img/s beside it needs a
         running reader to mean anything */
      var d=dr[nm]||{},known=d.total!=null,p=known&&d.total?100*d.done/d.total:0;
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
        return {n:nm,p:(rg[nm]!=null)?+rg[nm]||0:null}})
      .sort(function(a,b){
        var ra=rank(a),rb=rank(b);
        if(ra!==rb) return ra-rb;
        if(ra===0) return (b.p||0)-(a.p||0)||a.n.localeCompare(b.n);
        return a.n.localeCompare(b.n);
      });
    var prog=all.filter(function(r){return r.p>0&&r.p<100});
    var doneN=all.filter(function(r){return r.p!=null&&r.p>=100}).length;
    /* the count is only a count of what this payload said. A remembered
       roster has no percentages in it at all, and counting those as "0 of 17
       complete" reports work not done from the absence of an answer. */
    rHead.textContent=(!all.length||!rk.length)?'Per region'
      : live?('Per region \\u2014 '+prog.length+' of '+all.length+' in progress')
      : ('Per region \\u2014 '+doneN+' of '+all.length+' complete');
    rEl.innerHTML=all.map(function(r){
      return '<div class="drow'+(r.p>0?'':' dmut')+'"><span class="dn">'+esc(r.n.replace(/_/g,' '))+'</span>'+
        bar(r.p||0,pctColor(r.p||0))+
        /* a region's completion is what was DONE, not what is happening:
           it survives the sweep stopping, so it is not gated on live */
        '<span class="dv">'+p(r.p,function(v){return v.toFixed(1)+'%'})+'</span></div>';
    }).join('')||'<div class="dnone">no per-region data</div>';
    /* classifier: gauge only once crops actually flow (A.5). What it measured
       is a share of crops already classified, so it outlives the run that
       measured it -- the gate on `live` said "unknown" about a number the
       panel was holding, which is the same mistake as the error line below. */
    if((j.crops_classified||0)>0){
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
    /* errors: one muted line; details expand on click; green zero state.
       Cumulative, like the positives above, and so NOT gated on live: these
       count frames the sweep already failed on, and the moment you want them
       is the postmortem -- a run killed mid-sweep left 64 decode errors with
       no state of the page in which they could be read. */
    var errN=errSum(j),errs=j.errors||{};
    if(!j.errors){
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
/* Fold the header once a screenful of the page has gone by. Guarded because
   the test harness has no IntersectionObserver, and a page that threw here
   would lose every handler bound after it. */
(function(){
  var cue=document.getElementById('scrollcue');
  if(!cue||typeof IntersectionObserver!=='function')return;
  /* hold is the timestamp of the last change, and null until there has been
     one. Not 0: entry.time counts from navigation start, so 0 is not "long
     ago", it is the load — and the callback that arrives at observe() time
     only records where the page started. Arming the window there spent it on
     the load, and a scroll in the first quarter-second was refused. */
  var at=null,hold=null,pend=null,tid=0;
  function set(want,now){
    at=want;hold=now;
    document.body.classList.toggle('compact',want);
  }
  new IntersectionObserver(function(es){
    var want=!es[0].isIntersecting;
    /* back where we already are: whatever was being held is stale */
    if(want===at){pend=null;return;}
    /* the observer's own timestamp, not the wall clock: it is what the
       browser measured the crossing at, and it can be driven in a test */
    var now=(typeof es[0].time==='number')?es[0].time:Date.now();
    /* Hysteresis. Compacting removes height from a sticky header, which moves
       everything below it — and that settling can cross the sentinel again
       and ask for the opposite. Refusing a reversal for a moment turns a
       flutter into one change. */
    if(hold!==null&&now-hold<260){
      /* Held, NOT dropped. An IntersectionObserver only reports changes, so a
         reversal thrown away here is never offered again: a wheel flicked
         down and straight back up folded the header and left it folded at
         the top of the page with nothing left to unfold it. pend carries the
         observer's latest reading, and a later crossing either replaces it or
         cancels it above. */
      pend=want;
      /* hold+260 is when the window closes, on the same clock the crossings
         are stamped with. Reading a fresh Date.now() here would put an epoch
         millisecond into hold and make every later comparison nonsense. */
      if(!tid)tid=setTimeout(function(){
        tid=0;
        var w=pend;pend=null;
        if(w!==null&&w!==at)set(w,hold+260);
      },260-(now-hold));
      return;
    }
    /* at===null is the callback observe() delivers straight away: it records
       where the page started and is nobody's scroll, so it leaves hold unset
       rather than spending the window on the load. */
    set(want,at===null?null:now);
  },{threshold:0}).observe(cue);
})();
</script></body></html>"""

if __name__ == '__main__':
    main()
