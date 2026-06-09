#!/usr/bin/env python3
"""Browse street-dogs pipeline outputs by region and data type.

Usage:
    python browse.py --dirs grid_runs /mnt/hdd/grid_runs
    python browse.py --dirs grid_runs --port 8080
"""

import argparse
import math
import os
import re
import sys

try:
    from flask import (Flask, jsonify, make_response, redirect, request,
                       send_file, session, url_for)
except ImportError:
    print("Flask is required. Install it with: pip install flask",
          file=sys.stderr)
    sys.exit(1)

import functools

app = Flask(__name__)

BASE_DIRS: list[str] = []
REGION_MAP: dict[str,
                 list[dict]] = {}  # parent_region -> [{folder, path}, ...]

ADMIN_USER = 'admin'
ADMIN_PASS = '#s6V9wjEKN2LT9'


def _check_auth(f):

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)

    return wrapper


# Matches the 4 coordinate values appended to every region folder name
# e.g. "_-150.0_7.0_-52.5_85.0" or "_-10_35_40_70"
_COORD_SUFFIX = re.compile(r'(?:_-?\d+(?:\.\d+)?){4}$')

PAGE_SIZE_PARQUET = 25
PAGE_SIZE_IMAGES = 50

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def human_size(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def extract_parent(folder_name: str) -> str:
    """Return the human-readable parent region from a safe region folder name."""
    m = _COORD_SUFFIX.search(folder_name)
    if m:
        raw = folder_name[:m.start()]
        return raw.replace('_', ' ').strip()
    # Fallback: everything before the first numeric token
    parts = folder_name.split('_')
    for i, part in enumerate(parts):
        if part and (part[0].isdigit() or
                     (part[0] == '-' and len(part) > 1 and part[1].isdigit())):
            candidate = ' '.join(parts[:i]).strip()
            return candidate if candidate else folder_name
    return folder_name


def scan_dirs(dirs: list[str]) -> dict[str, list[dict]]:
    region_map: dict[str, list[dict]] = {}
    for base in dirs:
        base = os.path.realpath(base)
        if not os.path.isdir(base):
            print(f"Warning: directory not found, skipping: {base}",
                  file=sys.stderr)
            continue
        try:
            for entry in os.scandir(base):
                if not entry.is_dir():
                    continue
                parent = extract_parent(entry.name)
                if not parent:
                    continue
                region_map.setdefault(parent, []).append({
                    'folder': entry.name,
                    'path': entry.path,
                })
        except OSError as exc:
            print(f"Warning: could not scan {base}: {exc}", file=sys.stderr)
    # Sort sub-regions within each parent
    for entries in region_map.values():
        entries.sort(key=lambda e: e['folder'])
    return region_map


def collect_files(folders: list[dict], data_type: str) -> list[dict]:
    files: list[dict] = []
    for fi in folders:
        folder_path = fi['path']
        folder_name = fi['folder']
        try:
            if data_type == 'animal':
                for entry in os.scandir(folder_path):
                    if (entry.is_file()
                            and entry.name.startswith('ground_animals_')
                            and entry.name.endswith('.parquet')):
                        st = entry.stat()
                        files.append({
                            'name': entry.name,
                            'path': entry.path,
                            'size': st.st_size,
                            'size_human': human_size(st.st_size),
                            'folder': folder_name,
                        })
            elif data_type == 'all_data':
                for entry in os.scandir(folder_path):
                    if (entry.is_file() and entry.name.startswith('all_data_')
                            and entry.name.endswith('.parquet')):
                        st = entry.stat()
                        files.append({
                            'name': entry.name,
                            'path': entry.path,
                            'size': st.st_size,
                            'size_human': human_size(st.st_size),
                            'folder': folder_name,
                        })
            elif data_type == 'images':
                img_dir = os.path.join(folder_path, 'ground_animal_images')
                if os.path.isdir(img_dir):
                    for entry in os.scandir(img_dir):
                        if entry.is_file() and entry.name.lower().endswith(
                                '.jpg'):
                            st = entry.stat()
                            files.append({
                                'name': entry.name,
                                'path': entry.path,
                                'size': st.st_size,
                                'size_human': human_size(st.st_size),
                                'mtime': int(st.st_mtime),
                                'folder': folder_name,
                            })
        except OSError:
            pass
    files.sort(key=lambda f: (f['folder'], f['name']))
    return files


# ---------------------------------------------------------------------------
# Location search helpers  (from find_location_folder.py)
# ---------------------------------------------------------------------------

_BBOX_RE = re.compile(
    r'_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$'
)


def _parse_folder_bbox(folder_name: str):
    m = _BBOX_RE.search(folder_name)
    if m:
        w, s, e, n = map(float, m.groups())
        return w, s, e, n
    return None


def _bboxes_intersect(w1, s1, e1, n1, w2, s2, e2, n2) -> bool:
    if e1 < w2 or w1 > e2:
        return False
    if n1 < s2 or s1 > n2:
        return False
    return True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if (request.form.get('username') == ADMIN_USER
                and request.form.get('password') == ADMIN_PASS):
            session['logged_in'] = True
            return redirect(request.args.get('next') or '/')
        return make_response(_render_login('Invalid username or password.'))
    return make_response(_render_login())


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@_check_auth
def index():
    resp = make_response(HTML)
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    return resp


@app.route('/api/regions')
@_check_auth
def api_regions():
    return jsonify({
        'regions': [r for r in sorted(REGION_MAP.keys()) if r != '__search__']
    })


@app.route('/api/search')
@_check_auth
def api_search():
    import requests as _req
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'error': 'No query provided'})

    try:
        resp = _req.get(
            'https://nominatim.openstreetmap.org/search',
            params={
                'q': q,
                'format': 'json',
                'limit': 1
            },
            headers={'User-Agent': 'street-dogs-browser/1.0'},
            timeout=10,
        )
        results = resp.json()
    except Exception as exc:
        return jsonify({'error': f'Geocoding failed: {exc}'})

    if not results:
        return jsonify({'error': f'No results found for "{q}"'})

    hit = results[0]
    display_name = hit.get('display_name', q)
    bbox = hit.get('boundingbox')  # [south, north, west, east]
    if not bbox:
        return jsonify(
            {'error': 'Location found but no bounding box returned'})

    loc_s, loc_n, loc_w, loc_e = map(float, bbox)

    matched = []
    for parent, folders in REGION_MAP.items():
        if parent == '__search__':
            continue
        for fi in folders:
            gb = _parse_folder_bbox(fi['folder'])
            if gb and _bboxes_intersect(loc_w, loc_s, loc_e, loc_n, *gb):
                matched.append(fi)

    if not matched:
        return jsonify({
            'error': f'No grid folders overlap with "{display_name}"',
            'display_name': display_name
        })

    REGION_MAP['__search__'] = matched
    return jsonify({
        'display_name': display_name,
        'total_folders': len(matched),
    })


@app.route('/api/files')
@_check_auth
def api_files():
    region = request.args.get('region', '')
    data_type = request.args.get('type', 'animal')
    try:
        page = max(1, int(request.args.get('page', 1)))
    except ValueError:
        page = 1

    if region not in REGION_MAP:
        return jsonify({
            'error': 'Unknown region',
            'files': [],
            'total': 0,
            'pages': 0,
            'page': 1
        })

    if data_type not in ('animal', 'all_data', 'images'):
        data_type = 'animal'

    sort = request.args.get('sort', 'name')
    per_page = PAGE_SIZE_IMAGES if data_type == 'images' else PAGE_SIZE_PARQUET
    all_files = collect_files(REGION_MAP[region], data_type)

    if data_type == 'images' and sort in ('date_asc', 'date_desc'):
        all_files.sort(key=lambda f: f.get('mtime', 0),
                       reverse=(sort == 'date_desc'))

    total = len(all_files)
    pages = max(1, math.ceil(total / per_page))
    page = min(page, pages)
    start = (page - 1) * per_page

    return jsonify({
        'files': all_files[start:start + per_page],
        'total': total,
        'page': page,
        'pages': pages,
        'per_page': per_page,
    })


@app.route('/api/serve')
@_check_auth
def api_serve():
    import mimetypes

    from flask import abort
    path = request.args.get('path', '')
    if not path:
        abort(400)
    real = os.path.realpath(path)
    # Security: only serve files that live inside one of the scanned base dirs
    allowed = any(
        real.startswith(os.path.realpath(d) + os.sep) for d in BASE_DIRS)
    if not allowed or not os.path.isfile(real):
        abort(403)
    as_attachment = request.args.get('download', '0') == '1'
    mime, _ = mimetypes.guess_type(real)
    return send_file(real,
                     mimetype=mime or 'application/octet-stream',
                     as_attachment=as_attachment,
                     download_name=os.path.basename(real))


@app.route('/api/map')
@_check_auth
def api_map():
    from flask import abort
    try:
        import polars as pl
    except ImportError:
        return jsonify({'error': 'polars not installed', 'points': []})

    path = request.args.get('path', '')
    region = request.args.get('region', '')
    dtype = request.args.get('type', 'animal')
    max_pts = 30000  # heatmap handles this easily

    if path:
        real = os.path.realpath(path)
        allowed = any(
            real.startswith(os.path.realpath(d) + os.sep) for d in BASE_DIRS)
        if not allowed or not os.path.isfile(real):
            abort(403)
        pq_files = [real]
    elif region and region in REGION_MAP:
        if dtype not in ('animal', 'all_data'):
            dtype = 'animal'
        pq_files = [
            f['path'] for f in collect_files(REGION_MAP[region], dtype)
        ]
    else:
        return jsonify({'points': [], 'total': 0, 'sampled': 0})

    if not pq_files:
        return jsonify({'points': [], 'total': 0, 'sampled': 0})

    def _coords(lf):
        return lf.select([
            pl.col('computed_geometry').str.json_path_match(
                '$.coordinates[0]').cast(pl.Float64).alias('lon'),
            pl.col('computed_geometry').str.json_path_match(
                '$.coordinates[1]').cast(pl.Float64).alias('lat'),
        ]).drop_nulls()

    def _read_one(f, per_file_max):
        lf = pl.scan_parquet(f)
        # Get row count from parquet footer metadata - fast, no data read
        n_rows = lf.select(pl.len()).collect().item()
        step = max(1, n_rows // per_file_max) if n_rows > per_file_max else 1
        lf = _coords(lf)
        if step > 1:
            lf = lf.gather_every(step)
        # streaming=True processes chunk-by-chunk; avoids loading the whole
        # file into RAM at once - essential for files with millions of rows
        return lf.collect(streaming=True)

    per_file_max = max(500, max_pts // max(len(pq_files), 1))
    frames, skipped = [], 0
    for f in pq_files:
        try:
            frames.append(_read_one(f, per_file_max))
        except Exception:
            skipped += 1

    if not frames:
        msg = f'All {skipped} file(s) were unreadable (corrupted or incomplete).'
        return jsonify({'error': msg, 'points': [], 'total': 0, 'sampled': 0})

    df = pl.concat(frames)
    # Final guard: if many files, total might still exceed max_pts
    total_sampled = len(df)
    if total_sampled > max_pts:
        step = max(1, total_sampled // max_pts)
        df = df.gather_every(step)

    points = df.select(['lat', 'lon']).to_numpy().tolist()
    warn = f' ({skipped} corrupted file(s) skipped)' if skipped else ''
    return jsonify({
        'points': points,
        'total': total_sampled,
        'sampled': len(points),
        'skipped': skipped,
        'warn': warn,
    })


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Street Dogs - Sign in</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0d0f18;
  color: #dde3f0;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
}
.card {
  width: 340px;
  background: #161927;
  border: 1px solid #252a3d;
  border-radius: 10px;
  padding: 40px 32px 36px;
}
.logo {
  font-size: 16px;
  font-weight: 700;
  text-align: center;
  margin-bottom: 28px;
  letter-spacing: -.01em;
}
.logo-dot { color: #5b8dee; }
.field { margin-bottom: 18px; }
label {
  display: block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: #6b7590;
  margin-bottom: 7px;
}
input {
  width: 100%;
  padding: 9px 12px;
  background: #1e2235;
  border: 1px solid #252a3d;
  border-radius: 7px;
  color: #dde3f0;
  font-size: 13.5px;
  font-family: inherit;
  outline: none;
  transition: border-color .15s;
}
input:focus { border-color: #5b8dee; box-shadow: 0 0 0 3px rgba(91,141,238,.15); }
.btn {
  width: 100%;
  padding: 10px;
  background: #5b8dee;
  border: none;
  border-radius: 7px;
  color: #fff;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  font-family: inherit;
  margin-top: 8px;
  transition: opacity .12s;
}
.btn:hover { opacity: .88; }
.error {
  background: rgba(252,129,129,.08);
  border: 1px solid rgba(252,129,129,.25);
  color: #fc8181;
  padding: 9px 12px;
  border-radius: 7px;
  font-size: 12.5px;
  margin-bottom: 20px;
  display: __ERROR_DISPLAY__;
}
</style>
</head>
<body>
<div class="card">
  <div class="logo">Street Dogs<span class="logo-dot"> •</span> Data Browser</div>
  <div class="error">__ERROR__</div>
  <form method="POST" autocomplete="on">
    <div class="field">
      <label>Username</label>
      <input type="text" name="username" autofocus autocomplete="username">
    </div>
    <div class="field">
      <label>Password</label>
      <input type="password" name="password" autocomplete="current-password">
    </div>
    <button class="btn" type="submit">Sign in</button>
  </form>
</div>
</body>
</html>
"""


def _render_login(error=''):
    display = 'block' if error else 'none'
    return (LOGIN_HTML.replace('__ERROR_DISPLAY__',
                               display).replace('__ERROR__', error))


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Street Dogs - Data Browser</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:          #0d0f18;
  --surface:     #161927;
  --surface2:    #1e2235;
  --border:      #252a3d;
  --accent:      #5b8dee;
  --accent-dim:  rgba(91,141,238,.12);
  --accent-glow: rgba(91,141,238,.25);
  --text:        #dde3f0;
  --muted:       #6b7590;
  --mono:        'SF Mono', 'Cascadia Mono', 'Consolas', monospace;
  --sans:        -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --radius:      7px;
}

body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  font-size: 14px;
}

/* ── Header ─────────────────────────────────────────── */
header {
  height: 54px;
  padding: 0 20px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
  z-index: 10;
}
.logo {
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -.01em;
}
.logo-dot { color: var(--accent); }
.logo-sub {
  font-size: 12px;
  font-weight: 400;
  color: var(--muted);
  margin-left: 2px;
}
.header-spacer { flex: 1; }
.dir-badge {
  font-size: 11px;
  color: var(--muted);
  background: var(--surface2);
  border: 1px solid var(--border);
  padding: 3px 10px;
  border-radius: 20px;
  white-space: nowrap;
  max-width: 340px;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── Layout ──────────────────────────────────────────── */
.layout {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Sidebar ─────────────────────────────────────────── */
.sidebar {
  width: 210px;
  flex-shrink: 0;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
/* Location search */
.loc-search-wrap {
  padding: 10px 10px 0;
  display: flex;
  gap: 6px;
}
.loc-search-input {
  flex: 1;
  padding: 7px 10px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  font-size: 12.5px;
  font-family: var(--sans);
  outline: none;
  transition: border-color .15s;
  min-width: 0;
}
.loc-search-input:focus { border-color: var(--accent); }
.loc-search-input::placeholder { color: var(--muted); }
.loc-search-btn {
  padding: 0 10px;
  background: var(--accent);
  border: none;
  border-radius: var(--radius);
  color: #fff;
  font-size: 14px;
  cursor: pointer;
  flex-shrink: 0;
  transition: opacity .12s;
}
.loc-search-btn:hover { opacity: .85; }
.loc-search-btn:disabled { opacity: .4; cursor: not-allowed; }
.loc-divider {
  margin: 10px 14px 4px;
  border: none;
  border-top: 1px solid var(--border);
}
/* Search result chip in sidebar */
.search-result-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 7px 14px;
  background: rgba(91,141,238,.08);
  border-left: 3px solid var(--accent);
  cursor: pointer;
  font-size: 12.5px;
  color: var(--text);
  transition: background .1s;
}
.search-result-item.active { background: var(--accent-dim); font-weight: 500; color: var(--accent); }
.search-result-item:hover { background: var(--accent-dim); }
.sr-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sr-count { font-size: 10.5px; color: var(--muted); background: var(--surface2); padding: 1px 6px; border-radius: 10px; flex-shrink: 0; }
.search-result-item.active .sr-count { background: var(--accent-glow); color: var(--accent); }
.sr-clear {
  background: transparent; border: none; color: var(--muted);
  cursor: pointer; font-size: 14px; padding: 0; line-height: 1;
  flex-shrink: 0; transition: color .12s;
}
.sr-clear:hover { color: var(--danger); }
/* Search result banner in content area */
.search-banner {
  padding: 8px 20px;
  background: rgba(91,141,238,.07);
  border-bottom: 1px solid var(--border);
  font-size: 12px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}
.search-banner strong { color: var(--text); }
.sidebar-head {
  padding: 14px 14px 10px;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: .09em;
  text-transform: uppercase;
  color: var(--muted);
}
.search-wrap {
  padding: 0 10px 10px;
}
.region-search {
  width: 100%;
  padding: 6px 10px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  font-size: 12.5px;
  font-family: var(--sans);
  outline: none;
  transition: border-color .15s;
}
.region-search:focus { border-color: var(--accent); }
.region-search::placeholder { color: var(--muted); }

.region-list {
  flex: 1;
  overflow-y: auto;
  padding: 2px 0 8px;
}
.region-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  cursor: pointer;
  font-size: 13px;
  color: var(--text);
  border-left: 3px solid transparent;
  transition: background .1s, color .1s, border-color .1s;
  user-select: none;
}
.region-item:hover { background: var(--surface2); }
.region-item.active {
  background: var(--accent-dim);
  border-left-color: var(--accent);
  color: var(--accent);
  font-weight: 500;
}
.ri-icon { font-size: 15px; line-height: 1; flex-shrink: 0; }
.ri-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ri-count {
  font-size: 10.5px;
  color: var(--muted);
  background: var(--surface2);
  padding: 1px 6px;
  border-radius: 10px;
  flex-shrink: 0;
}
.region-item.active .ri-count {
  background: var(--accent-glow);
  color: var(--accent);
}
.sidebar-empty {
  padding: 24px 14px;
  font-size: 12.5px;
  color: var(--muted);
  text-align: center;
}

/* ── Content ─────────────────────────────────────────── */
.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Type selector bar */
.type-bar {
  padding: 12px 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  background: var(--surface);
}
.type-btn {
  padding: 6px 15px;
  border-radius: 20px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  font-size: 12.5px;
  cursor: pointer;
  font-family: var(--sans);
  transition: all .14s;
  white-space: nowrap;
}
.type-btn:hover { border-color: var(--accent); color: var(--text); }
.type-btn.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
  font-weight: 600;
}

/* Stats bar */
.stats-bar {
  padding: 8px 20px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  background: var(--bg);
}
.stats-region {
  font-weight: 600;
  color: var(--text);
  font-size: 13px;
}
.stats-sep { color: var(--border); }
.count-badge {
  background: var(--accent-dim);
  color: var(--accent);
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 700;
}

/* Table */
.table-wrap {
  flex: 1;
  overflow-y: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
}
thead {
  position: sticky;
  top: 0;
  z-index: 2;
  background: var(--bg);
}
thead th {
  text-align: left;
  padding: 9px 16px;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
}
thead th.th-right { text-align: right; }
tbody tr {
  border-bottom: 1px solid var(--border);
  transition: background .08s;
}
tbody tr:last-child { border-bottom: none; }
tbody tr:hover { background: var(--surface2); }
tbody td { padding: 9px 16px; vertical-align: middle; }
.td-num {
  color: var(--muted);
  font-size: 11px;
  width: 44px;
  text-align: right;
  padding-right: 8px;
}
.td-name {
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--text);
  word-break: break-all;
}
.td-folder {
  color: var(--muted);
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
}
.td-size {
  text-align: right;
  color: var(--muted);
  white-space: nowrap;
  font-family: var(--mono);
  font-size: 11.5px;
}

/* Pagination */
.pagination {
  padding: 10px 20px;
  border-top: 1px solid var(--border);
  background: var(--surface);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  flex-shrink: 0;
}
.pg-btn {
  min-width: 30px;
  height: 30px;
  padding: 0 6px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text);
  font-size: 12.5px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: all .12s;
  font-family: var(--sans);
}
.pg-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.pg-btn.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
  font-weight: 700;
}
.pg-btn:disabled { opacity: .3; cursor: not-allowed; }
.pg-ellipsis {
  min-width: 30px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  font-size: 12.5px;
}
.pg-info {
  margin: 0 8px;
  font-size: 12px;
  color: var(--muted);
}

/* Empty / splash states */
.splash {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--muted);
  padding: 60px 40px;
  text-align: center;
}
.splash-icon { font-size: 44px; line-height: 1; }
.splash h2 { font-size: 15px; font-weight: 600; color: var(--text); }
.splash p { font-size: 13px; max-width: 300px; line-height: 1.6; }

/* Download button in table */
.dl-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  text-decoration: none;
  font-size: 14px;
  transition: all .12s;
  flex-shrink: 0;
}
.dl-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

/* Thumbnail grid */
.thumb-scroll { flex: 1; overflow-y: auto; }
.thumb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
  gap: 10px;
  padding: 16px 20px;
}
.thumb-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: border-color .12s, box-shadow .12s;
}
.thumb-card:hover {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent-dim);
}
.thumb-img-wrap {
  width: 100%;
  aspect-ratio: 4/3;
  background: var(--surface2);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  position: relative;
}
.thumb-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: opacity .2s;
}
.thumb-img.loading { opacity: 0; }
.thumb-info {
  padding: 7px 8px 8px;
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.thumb-name {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.thumb-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 4px;
}
.thumb-size { font-size: 10px; color: var(--muted); }
.thumb-date { font-size: 10px; color: var(--muted); margin-top: 2px; }
.thumb-dl {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 5px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  text-decoration: none;
  font-size: 12px;
  transition: all .12s;
  flex-shrink: 0;
}
.thumb-dl:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

/* Sort bar (images tab) */
.sort-bar {
  padding: 7px 20px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; color: var(--muted);
  flex-shrink: 0; background: var(--bg);
}
.sort-label { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }
.sort-btn {
  padding: 3px 11px; border-radius: 12px;
  border: 1px solid var(--border); background: transparent;
  color: var(--muted); font-size: 11.5px; cursor: pointer;
  font-family: var(--sans); transition: all .12s;
}
.sort-btn:hover { border-color: var(--accent); color: var(--text); }
.sort-btn.active { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); font-weight: 600; }

/* Map icon button (sidebar + table) */
.map-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 5px;
  border: 1px solid transparent;
  background: transparent;
  color: var(--muted);
  font-size: 13px;
  cursor: pointer;
  flex-shrink: 0;
  transition: all .12s;
  padding: 0;
  font-family: var(--sans);
}
.map-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

/* Lightbox */
.lightbox {
  position: fixed; inset: 0; z-index: 1000;
  background: rgba(0,0,0,.92);
  display: flex; align-items: center; justify-content: center;
  cursor: zoom-out;
}
.lightbox img {
  max-width: 92vw; max-height: 92vh;
  object-fit: contain;
  border-radius: 4px;
  box-shadow: 0 8px 40px rgba(0,0,0,.6);
  cursor: default;
}
.lightbox-close {
  position: absolute; top: 16px; right: 20px;
  font-size: 22px; color: #fff; cursor: pointer;
  background: rgba(255,255,255,.1); border: none;
  width: 36px; height: 36px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  transition: background .12s;
}
.lightbox-close:hover { background: rgba(255,255,255,.2); }
.lb-nav {
  position: absolute; top: 50%; transform: translateY(-50%);
  background: rgba(255,255,255,.1); border: none; color: #fff;
  font-size: 28px; width: 48px; height: 48px; border-radius: 50%;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  transition: background .12s; z-index: 1;
}
.lb-nav:hover:not(:disabled) { background: rgba(255,255,255,.22); }
.lb-nav:disabled { opacity: .18; cursor: default; }
.lb-prev { left: 20px; }
.lb-next { right: 20px; }
.lb-counter {
  position: absolute; top: 18px; left: 50%; transform: translateX(-50%);
  background: rgba(0,0,0,.6); color: rgba(255,255,255,.65);
  padding: 4px 14px; border-radius: 20px; font-size: 12px;
  pointer-events: none; white-space: nowrap;
}
.lightbox-info {
  position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
  background: rgba(0,0,0,.7); color: #e2e8f0;
  padding: 6px 16px; border-radius: 20px;
  font-size: 12px; font-family: var(--mono);
  white-space: nowrap; max-width: 80vw;
  overflow: hidden; text-overflow: ellipsis;
}

/* Map modal */
.map-overlay {
  position: fixed; inset: 0; z-index: 900;
  background: rgba(0,0,0,.7);
  display: flex; align-items: center; justify-content: center;
}
.map-box {
  width: min(860px, 94vw); height: min(600px, 88vh);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  display: flex; flex-direction: column;
  box-shadow: 0 20px 60px rgba(0,0,0,.5);
}
.map-header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
  flex-shrink: 0;
}
.map-header h3 { flex: 1; font-size: 14px; font-weight: 600; }
.map-header .map-meta { font-size: 11.5px; color: var(--muted); }
.map-close {
  background: transparent; border: 1px solid var(--border);
  color: var(--muted); font-size: 14px; cursor: pointer;
  width: 28px; height: 28px; border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--sans); transition: all .12s;
}
.map-close:hover { border-color: var(--danger); color: var(--danger); }
#leafletMap { flex: 1; background: var(--surface2); }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
</head>
<body>

<header>
  <div class="logo">Street Dogs<span class="logo-dot"> •</span><span class="logo-sub"> Data Browser</span></div>
  <div class="header-spacer"></div>
  <div class="dir-badge" id="dirBadge"></div>
  <a href="/logout" style="font-size:12px;color:var(--muted);text-decoration:none;padding:4px 10px;border:1px solid var(--border);border-radius:6px;transition:all .12s" onmouseover="this.style.color='var(--text)'" onmouseout="this.style.color='var(--muted)'">Sign out</a>
</header>

<div class="layout">
  <aside class="sidebar">
    <div class="loc-search-wrap">
      <input class="loc-search-input" id="locSearchInput" placeholder="City or country…"
             onkeydown="if(event.key==='Enter') runLocSearch()">
      <button class="loc-search-btn" id="locSearchBtn" onclick="runLocSearch()" title="Search">🔍</button>
    </div>
    <hr class="loc-divider">
    <div id="searchResultSlot"></div>
    <div class="sidebar-head">Regions</div>
    <div class="search-wrap">
      <input class="region-search" id="regionSearch" placeholder="Filter regions…" oninput="filterRegions(this.value)">
    </div>
    <div class="region-list" id="regionList">
      <div class="sidebar-empty">Loading…</div>
    </div>
  </aside>

  <main class="content" id="mainContent">
    <div class="splash">
      <div class="splash-icon">🗂️</div>
      <h2>Select a region</h2>
      <p>Pick a geographic region from the sidebar, then choose which type of data to browse.</p>
    </div>
  </main>
</div>

<!-- Lightbox -->
<div id="lightbox" class="lightbox" style="display:none" onclick="closeLightbox()">
  <button class="lightbox-close" onclick="event.stopPropagation(); closeLightbox()">✕</button>
  <button class="lb-nav lb-prev" id="lbPrev" onclick="event.stopPropagation(); lbNav(-1)">‹</button>
  <img id="lightboxImg" src="" alt="" onclick="event.stopPropagation()">
  <button class="lb-nav lb-next" id="lbNext" onclick="event.stopPropagation(); lbNav(1)">›</button>
  <div class="lb-counter" id="lbCounter"></div>
  <div class="lightbox-info" id="lightboxInfo"></div>
</div>

<!-- Map modal -->
<div id="mapOverlay" class="map-overlay" style="display:none">
  <div class="map-box">
    <div class="map-header">
      <h3 id="mapTitle">Map</h3>
      <span class="map-meta" id="mapMeta"></span>
      <div id="mapModeToggle" style="display:none;gap:4px;align-items:center">
        <button class="sort-btn active" id="btnHeat" onclick="setMapMode('heat')">🔥 Heat</button>
        <button class="sort-btn"        id="btnDots" onclick="setMapMode('dots')">● Dots</button>
      </div>
      <button class="map-close" onclick="closeMap()">✕</button>
    </div>
    <div id="leafletMap"></div>
  </div>
</div>

<script>
'use strict';

// ── Config ────────────────────────────────────────────
const DATA_TYPES = [
  { key: 'all_data', label: 'All Data',           hint: 'all_data_*.parquet'         },
  { key: 'animal',   label: 'Animal Detections',  hint: 'ground_animals_*.parquet'   },
  { key: 'images',   label: 'Downloaded Images',  hint: 'ground_animal_images/*.jpg' },
];

const GLOBES = ['🌍', '🌎', '🌏'];

function regionIcon(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  return GLOBES[hash % GLOBES.length];
}

// ── State ─────────────────────────────────────────────
const S = {
  allRegions: [],       // [{name}]
  filteredRegions: [],
  selected: null,
  type: 'all_data',
  sort: 'name',      // 'name' | 'date_asc' | 'date_desc'
  page: 1,
  result: null,
  loading: false,
  // Location search
  searchDisplayName: null,   // human name of active search result, e.g. "Paris, France"
  searchFolderCount: 0,
};

// ── Bootstrap ─────────────────────────────────────────
async function init() {
  const res = await fetch('/api/regions');
  const data = await res.json();
  S.allRegions = data.regions.map(r => ({ name: r }));
  S.filteredRegions = S.allRegions;

  document.getElementById('dirBadge').textContent =
    `${S.allRegions.length} region${S.allRegions.length !== 1 ? 's' : ''} loaded`;

  renderSidebar();
}

// ── Location search ───────────────────────────────────
async function runLocSearch() {
  const input = document.getElementById('locSearchInput');
  const btn   = document.getElementById('locSearchBtn');
  const q     = input.value.trim();
  if (!q) return;

  btn.disabled = true;
  btn.textContent = '⏳';
  S.searchDisplayName = null;

  const res  = await fetch('/api/search?' + new URLSearchParams({ q }));
  const data = await res.json();

  btn.disabled = false;
  btn.textContent = '🔍';

  if (data.error) {
    renderSearchSlot(null, 0, data.error);
    return;
  }
  S.searchDisplayName = data.display_name;
  S.searchFolderCount = data.total_folders;
  renderSearchSlot(data.display_name, data.total_folders, null);
  selectRegion('__search__');
}

function clearSearch() {
  S.searchDisplayName = null;
  S.searchFolderCount = 0;
  document.getElementById('locSearchInput').value = '';
  renderSearchSlot(null, 0, null);
  if (S.selected === '__search__') {
    S.selected = null;
    S.result = null;
    renderSidebar();
    renderContent();
  }
}

function renderSearchSlot(displayName, folderCount, errorMsg) {
  const slot = document.getElementById('searchResultSlot');
  if (errorMsg) {
    slot.innerHTML = `<div class="sidebar-empty" style="color:var(--danger);padding:8px 14px;font-size:12px">${esc(errorMsg)}</div>`;
    return;
  }
  if (!displayName) { slot.innerHTML = ''; return; }
  const active = S.selected === '__search__';
  slot.innerHTML = `
    <div class="search-result-item${active ? ' active' : ''}" onclick="selectRegion('__search__')">
      <span>🔍</span>
      <span class="sr-label" title="${esc(displayName)}">${esc(displayName)}</span>
      <span class="sr-count">${folderCount} folder${folderCount !== 1 ? 's' : ''}</span>
      <button class="sr-clear" onclick="event.stopPropagation();clearSearch()" title="Clear search">✕</button>
    </div>`;
}

// ── Sidebar ────────────────────────────────────────────
function filterRegions(q) {
  q = q.trim().toLowerCase();
  S.filteredRegions = q
    ? S.allRegions.filter(r => r.name.toLowerCase().includes(q))
    : S.allRegions;
  renderSidebar();
}

function renderSidebar() {
  const el = document.getElementById('regionList');
  if (!S.filteredRegions.length) {
    el.innerHTML = '<div class="sidebar-empty">No matching regions.</div>';
    return;
  }
  el.innerHTML = S.filteredRegions.map(r => `
    <div class="region-item${r.name === S.selected ? ' active' : ''}"
         data-region="${esc(r.name)}">
      <span class="ri-icon">${regionIcon(r.name)}</span>
      <span class="ri-name">${esc(r.name)}</span>
      <button class="map-btn" title="Map this region"
              onclick="event.stopPropagation(); openRegionMap(${esc(JSON.stringify(r.name))})">🗺</button>
    </div>`).join('');
}

function selectRegion(name) {
  S.selected = name;
  S.page = 1;
  S.result = null;
  renderSidebar();
  // Keep search slot active indicator in sync
  if (S.searchDisplayName) renderSearchSlot(S.searchDisplayName, S.searchFolderCount, null);
  loadFiles();
}

// ── Data loading ──────────────────────────────────────
async function loadFiles() {
  if (!S.selected) return;
  S.loading = true;
  renderContent();

  const params = new URLSearchParams({
    region: S.selected,
    type:   S.type,
    sort:   S.sort,
    page:   S.page,
  });
  const res = await fetch('/api/files?' + params);
  S.result = await res.json();
  S.page = S.result.page;
  S.loading = false;
  renderContent();
}

function setType(type) {
  S.type = type;
  S.sort = 'name';
  S.page = 1;
  S.result = null;
  loadFiles();
}

function setSort(sort) {
  S.sort = sort;
  S.page = 1;
  loadFiles();
}

function goPage(p) {
  S.page = p;
  loadFiles();
}

// ── Content rendering ─────────────────────────────────
function renderContent() {
  const el = document.getElementById('mainContent');

  if (!S.selected) {
    el.innerHTML = `
      <div class="splash">
        <div class="splash-icon">🗂️</div>
        <h2>Select a region</h2>
        <p>Pick a geographic region from the sidebar, then choose which type of data to browse.</p>
      </div>`;
    return;
  }

  const typeBar = `
    <div class="type-bar">
      ${DATA_TYPES.map(t => `
        <button class="type-btn${t.key === S.type ? ' active' : ''}"
                onclick="setType('${t.key}')">${t.label}</button>`).join('')}
    </div>`;

  if (S.loading || !S.result) {
    el.innerHTML = typeBar + `
      <div class="splash">
        <div class="splash-icon">⏳</div>
        <h2>Loading…</h2>
      </div>`;
    return;
  }

  const r = S.result;
  const typeInfo = DATA_TYPES.find(t => t.key === S.type);
  const isSearch = S.selected === '__search__';
  const regionLabel = isSearch ? (S.searchDisplayName || 'Search result') : S.selected;

  const searchBanner = isSearch ? `
    <div class="search-banner">
      🔍 <strong>${esc(regionLabel)}</strong>
      <span>•</span>
      <span>${S.searchFolderCount} overlapping folder${S.searchFolderCount !== 1 ? 's' : ''}</span>
    </div>` : '';

  const statsBar = `
    <div class="stats-bar">
      <span class="stats-region">${esc(regionLabel)}</span>
      <span class="stats-sep">•</span>
      <span>${esc(typeInfo.label)}</span>
      <span class="stats-sep">•</span>
      <span class="count-badge">${r.total.toLocaleString()} file${r.total !== 1 ? 's' : ''}</span>
      ${r.pages > 1 ? `<span class="stats-sep">•</span><span>page ${r.page} / ${r.pages}</span>` : ''}
    </div>`;

  if (!r.files.length) {
    el.innerHTML = typeBar + searchBanner + statsBar + `
      <div class="splash">
        <div class="splash-icon">📭</div>
        <h2>No files found</h2>
        <p>No <code style="font-size:12px;color:var(--accent)">${esc(typeInfo.hint)}</code>
           files were found in <strong>${esc(regionLabel)}</strong>.</p>
      </div>`;
    return;
  }

  if (S.type === 'images') {
    const sortBar = `
      <div class="sort-bar">
        <span class="sort-label">Sort</span>
        <button class="sort-btn${S.sort === 'name'      ? ' active' : ''}" onclick="setSort('name')">Name</button>
        <button class="sort-btn${S.sort === 'date_desc' ? ' active' : ''}" onclick="setSort('date_desc')">Newest first</button>
        <button class="sort-btn${S.sort === 'date_asc'  ? ' active' : ''}" onclick="setSort('date_asc')">Oldest first</button>
      </div>`;
    el.innerHTML = typeBar + searchBanner + sortBar + statsBar + renderThumbGrid(r) +
      (r.pages > 1 ? renderPagination(r.page, r.pages) : '');
    lazyLoadThumbs();
    return;
  }

  const offset = (r.page - 1) * r.per_page;
  const rows = r.files.map((f, i) => {
    const dlHref  = `/api/serve?path=${encodeURIComponent(f.path)}&download=1`;
    const mapPath = encodeURIComponent(f.path);
    return `
    <tr>
      <td class="td-num">${offset + i + 1}</td>
      <td class="td-name" title="${esc(f.path)}">${esc(f.name)}</td>
      <td class="td-folder" title="${esc(f.folder)}">${esc(f.folder)}</td>
      <td class="td-size">${esc(f.size_human)}</td>
      <td style="width:64px;text-align:center;white-space:nowrap">
        <button class="map-btn" title="View on map"
                onclick="openFileMap(${esc(JSON.stringify(f.path))}, ${esc(JSON.stringify(f.name))})">🗺</button>
        <a class="dl-btn" href="${dlHref}" title="Download" download>⬇</a>
      </td>
    </tr>`;
  }).join('');

  el.innerHTML = typeBar + searchBanner + statsBar + `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th class="th-right">#</th>
            <th>File Name</th>
            <th>Region Folder</th>
            <th class="th-right">Size</th>
            <th></th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${r.pages > 1 ? renderPagination(r.page, r.pages) : ''}`;
}

function renderThumbGrid(r) {
  // Store full image list so lightbox can navigate without re-encoding paths in HTML
  _lbImages = r.files.map(f => ({
    src:  `/api/serve?path=${encodeURIComponent(f.path)}`,
    name: f.name,
  }));

  const cards = r.files.map((f, i) => {
    const src  = _lbImages[i].src;
    const dl   = src + '&download=1';
    const date = f.mtime
      ? new Date(f.mtime * 1000).toLocaleDateString(undefined, {year:'numeric', month:'short', day:'numeric'})
      : '';
    return `
      <div class="thumb-card">
        <div class="thumb-img-wrap" style="cursor:zoom-in" onclick="openLightbox(${i})">
          <img class="thumb-img loading" data-src="${esc(src)}" alt="${esc(f.name)}"
               onload="this.classList.remove('loading')"
               onerror="this.style.display='none'">
        </div>
        <div class="thumb-info">
          <div class="thumb-name" title="${esc(f.name)}">${esc(f.name)}</div>
          ${date ? `<div class="thumb-date">${esc(date)}</div>` : ''}
          <div class="thumb-meta">
            <span class="thumb-size">${esc(f.size_human)}</span>
            <a class="thumb-dl" href="${esc(dl)}" title="Download" download>⬇</a>
          </div>
        </div>
      </div>`;
  }).join('');
  return `<div class="thumb-scroll"><div class="thumb-grid">${cards}</div></div>`;
}

function lazyLoadThumbs() {
  // Use IntersectionObserver to load thumbnails only when visible
  const imgs = document.querySelectorAll('.thumb-img[data-src]');
  if (!imgs.length) return;
  const obs = new IntersectionObserver((entries, o) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        const img = e.target;
        img.src = img.dataset.src;
        delete img.dataset.src;
        o.unobserve(img);
      }
    });
  }, { rootMargin: '200px' });
  imgs.forEach(img => obs.observe(img));
}

function renderPagination(cur, total) {
  const show = new Set([1, total]);
  for (let i = Math.max(1, cur - 2); i <= Math.min(total, cur + 2); i++) show.add(i);
  const sorted = [...show].sort((a, b) => a - b);

  const parts = [];
  let prev = 0;
  for (const p of sorted) {
    if (prev && p - prev > 1) parts.push('…');
    parts.push(p);
    prev = p;
  }

  const buttons = parts.map(p => {
    if (p === '…') return `<span class="pg-ellipsis">…</span>`;
    return `<button class="pg-btn${p === cur ? ' active' : ''}" onclick="goPage(${p})">${p}</button>`;
  }).join('');

  return `
    <div class="pagination">
      <button class="pg-btn" onclick="goPage(${cur - 1})" ${cur === 1 ? 'disabled' : ''}>‹</button>
      ${buttons}
      <button class="pg-btn" onclick="goPage(${cur + 1})" ${cur === total ? 'disabled' : ''}>›</button>
      <span class="pg-info">${cur} of ${total}</span>
    </div>`;
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Lightbox ──────────────────────────────────────────
let _lbImages = [];   // [{src, name}] - set by renderThumbGrid
let _lbIdx    = 0;

function openLightbox(idx) {
  _lbIdx = idx;
  _lbUpdate();
  document.getElementById('lightbox').style.display = 'flex';
  document.addEventListener('keydown', onLightboxKey);
}

function _lbUpdate() {
  const item = _lbImages[_lbIdx];
  const img  = document.getElementById('lightboxImg');
  img.src = item.src;
  img.alt = item.name;
  document.getElementById('lightboxInfo').textContent = item.name;
  document.getElementById('lbCounter').textContent    = `${_lbIdx + 1} / ${_lbImages.length}`;
  document.getElementById('lbPrev').disabled = (_lbIdx === 0);
  document.getElementById('lbNext').disabled = (_lbIdx === _lbImages.length - 1);
}

function lbNav(dir) {
  const next = _lbIdx + dir;
  if (next >= 0 && next < _lbImages.length) {
    _lbIdx = next;
    _lbUpdate();
  }
}

function closeLightbox() {
  document.getElementById('lightbox').style.display = 'none';
  document.getElementById('lightboxImg').src = '';
  document.removeEventListener('keydown', onLightboxKey);
}

function onLightboxKey(e) {
  if      (e.key === 'Escape')     closeLightbox();
  else if (e.key === 'ArrowLeft')  lbNav(-1);
  else if (e.key === 'ArrowRight') lbNav(1);
}

// ── Map ───────────────────────────────────────────────
let _leaflet   = null;
let _mapData   = null;   // last loaded {points, total, sampled, warn}
let _mapMode   = 'heat'; // 'heat' | 'dots'
let _mapLayers = [];     // current data layers (heat or dot)

function openRegionMap(region) {
  const dtype = (S.type === 'images') ? 'animal' : S.type;
  const label = DATA_TYPES.find(t => t.key === dtype)?.label || dtype;
  const display = (region === '__search__' && S.searchDisplayName) ? S.searchDisplayName : region;
  showMapModal(`${display} - ${label}`, `/api/map?region=${encodeURIComponent(region)}&type=${dtype}`);
}

function openFileMap(path, name) {
  showMapModal(name, `/api/map?path=${encodeURIComponent(path)}`);
}

function _clearMapLayers() {
  _mapLayers.forEach(l => _leaflet.removeLayer(l));
  _mapLayers = [];
}

function _renderMapData() {
  if (!_mapData || !_leaflet) return;
  _clearMapLayers();

  const pts = _mapData.points;  // [[lat, lon], ...]

  if (_mapMode === 'heat') {
    // leaflet-heat: expects [lat, lng, intensity]
    const heatPts = pts.map(([lat, lon]) => [lat, lon, 1]);
    const layer = L.heatLayer(heatPts, {
      radius: 18, blur: 15, maxZoom: 17, max: 1,
      gradient: { 0.2: '#3b82f6', 0.5: '#8b5cf6', 0.8: '#ef4444', 1.0: '#f97316' },
    }).addTo(_leaflet);
    _mapLayers.push(layer);
  } else {
    // Dots - CircleMarker (canvas-rendered via preferCanvas)
    pts.forEach(([lat, lon]) => {
      const m = L.circleMarker([lat, lon], {
        radius: 3, color: '#5b8dee', fillColor: '#5b8dee',
        fillOpacity: 0.7, weight: 0,
      }).addTo(_leaflet);
      _mapLayers.push(m);
    });
  }
}

function setMapMode(mode) {
  _mapMode = mode;
  document.getElementById('btnHeat').classList.toggle('active', mode === 'heat');
  document.getElementById('btnDots').classList.toggle('active', mode === 'dots');
  _renderMapData();
}

async function showMapModal(title, apiUrl) {
  document.getElementById('mapTitle').textContent = title;
  document.getElementById('mapMeta').textContent  = 'Loading…';
  document.getElementById('mapModeToggle').style.display = 'none';
  document.getElementById('mapOverlay').style.display = 'flex';

  if (!_leaflet) {
    _leaflet = L.map('leafletMap', { preferCanvas: true });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors', maxZoom: 19,
    }).addTo(_leaflet);
  } else {
    _clearMapLayers();
  }
  setTimeout(() => _leaflet.invalidateSize(), 120);

  const res  = await fetch(apiUrl);
  const data = await res.json();

  if (data.error) {
    document.getElementById('mapMeta').textContent = 'Error: ' + data.error;
    return;
  }
  if (!data.points.length) {
    document.getElementById('mapMeta').textContent = 'No coordinates found.';
    return;
  }

  _mapData = data;

  const label = data.sampled < data.total
    ? `${data.sampled.toLocaleString()} of ${data.total.toLocaleString()} points`
    : `${data.total.toLocaleString()} points`;
  document.getElementById('mapMeta').textContent = label + (data.warn || '');

  // Show mode toggle and keep current mode selection fresh
  const toggle = document.getElementById('mapModeToggle');
  toggle.style.display = 'flex';
  document.getElementById('btnHeat').classList.toggle('active', _mapMode === 'heat');
  document.getElementById('btnDots').classList.toggle('active', _mapMode === 'dots');

  _renderMapData();

  const bounds = L.latLngBounds(data.points.map(([lat, lon]) => [lat, lon]));
  _leaflet.fitBounds(bounds, { padding: [20, 20] });
}

function closeMap() {
  document.getElementById('mapOverlay').style.display = 'none';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMap(); });

// ── Boot ──────────────────────────────────────────────
document.getElementById('regionList').addEventListener('click', e => {
  const item = e.target.closest('.region-item');
  if (item) selectRegion(item.dataset.region);
});

init();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Street Dogs pipeline output browser',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dirs',
        nargs='+',
        required=True,
        help=
        'One or more base directories to scan for region folders (e.g. grid_runs)',
    )
    parser.add_argument('--port',
                        type=int,
                        default=54040,
                        help='Port to serve on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    app.secret_key = os.urandom(24)

    global BASE_DIRS, REGION_MAP
    BASE_DIRS = args.dirs

    print(f"Scanning: {', '.join(BASE_DIRS)}")
    REGION_MAP = scan_dirs(BASE_DIRS)

    total_folders = sum(len(v) for v in REGION_MAP.values())
    print(
        f"Found {len(REGION_MAP)} parent regions across {total_folders} sub-region folders"
    )
    print(f"Server: http://{args.host}:{args.port}/")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
