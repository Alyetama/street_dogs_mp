#!/usr/bin/env python3
"""Fetch the camera_parameters, altitude and computed_altitude fields
the harvest never asked for.

    python tools/detect/fetch_camera_params.py plan
    python tools/detect/fetch_camera_params.py run
    python tools/detect/fetch_camera_params.py run --limit 20
    python tools/detect/fetch_camera_params.py status
    python tools/detect/fetch_camera_params.py repair      # re-run what failed
    python tools/detect/fetch_camera_params.py merge --dry-run

WHAT IS MISSING. batch_chunks_mp_api.py asks Mapillary for seventeen fields
per image; camera_parameters, altitude and computed_altitude are not among
them, so no parquet on any drive has any of the three. None can be derived
from what we stored -- they have to be asked for.

    camera_parameters   the intrinsics. Undistort a crop, or turn a pixel
                        into a bearing.
    altitude            the altitude the camera reported.
    computed_altitude   Mapillary's own reconstruction of it, which is the
                        one to trust where the two disagree.

WHAT THIS WRITES, AND WHY NOT IN PLACE. By default this writes a SIDECAR
store under data/camera/, keyed by image_id, exactly the way data/gate/ and
data/leash/ hold their verdicts. Parquet is columnar and immutable: "adding a
column" to the harvest means rewriting all 33,170 ground_animals files (53
GiB) in place -- and `merge` will do exactly that if you want it. The sidecar
is the default because it is a few hundred MB, is resumable, is safe to run
while anything else reads the harvest, and can be thrown away and rebuilt
without touching the harvest at all. It joins in one line:

    select a.*, c.camera_parameters, c.altitude, c.computed_altitude
    from read_parquet('<cell parquets>') a
    left join read_parquet('data/camera/camera-*.parquet') c using (image_id)

`merge` does the in-place rewrite for anyone who wants the three columns
physically in the harvest files. It is opt-in, one file at a time, atomic per file, and
it refuses to replace a file whose row count changed. Fetch first, merge
later or never.

EVERY ID IS FETCHED, INCLUDING SPHERICAL. An earlier version of this file
skipped spherical images, because 300 of 300 sampled had no
camera_parameters -- 52% of the ids for a column of nulls. Adding the two
altitude fields killed that shortcut: measured on 100 ids per camera_type,

    camera_type       camera_parameters   altitude   computed_altitude
    spherical                     0/100    100/100              92/100
    perspective                 100/100    100/100             100/100
    fisheye / brown /
      fisheye62 / equirect      100/100    100/100             100/100

Spherical images have no intrinsics but they do have altitude, so skipping
them would leave a silent hole over half the harvest. The run is now
32.0M ids rather than 15.5M. Measured on a full cold shard: 200,000 ids at
~1,200 ids/s of actual fetching, so 161 shards is **~8 hours**.

That number used to be ~10 h. Writing each shard with executemany cost 48.4s
against 0.0s for the COPY that followed it -- 2.2 hours of the run spent
inserting rows one at a time with all 128 fetch workers idle. The shard is
now built as an Arrow table instead: same bytes, 0.55s.

ZERO IS A REAL ALTITUDE. Measured over a full 200,000-id shard: 35,040 rows
have altitude exactly 0. So every presence test here is `is not None`; a
truthiness test would silently discard a sixth of the harvest.

THE ALTITUDES ARE NOT CLEANED, AND SOME ARE NONSENSE. Over that same shard:

    altitude            199,744 plausible ... 10 above 9,000 m,
                        6 below -500 m, 1 exactly 4294967295 (2^32-1, an
                        unsigned sentinel that leaked through as a value)
    computed_altitude   193,619 plausible ... 1 above 9,000 m, 13 below -500 m

They are stored exactly as the API returned them, because a fetcher guessing
at which readings are real is how you get a store nobody can audit. Filter at
read time -- `altitude between -500 and 9000` covers it -- and prefer
computed_altitude, which is Mapillary's reconstruction rather than the
camera's own claim.

BATCH SIZE IS 100, NOT THE 300 THE API ALLOWS. The documented cap is 300
("must have at most 300 elements", error code 105), but a 300-id batch of
COLD ids times out server-side after ~30s and returns 500 "An unknown error
occurred". Measured on ids the API had not seen recently, 32 workers:

    batch 300      1 of 40 chunks succeeded      39 x HTTP 500
    batch 150     80 of 80
    batch 100    120 of 120                      <- default: fastest of
                                                    the sizes that all worked
    batch  50    240 of 240

The first draft of this file used 300 because a probe against ids fetched
minutes earlier succeeded -- warm ids answer fine at 300. That is a trap: any
throughput number measured on ids you just fetched is meaningless here.

THROUGHPUT, batch 100, cold ids, all chunks succeeding:

     64 workers     415 ids/s
     96 workers     571 ids/s
    128 workers     827 ids/s        <- default
    160 workers     868 ids/s

128 is where the curve flattens. Those were measured over the non-spherical
subset before the skip was removed; a full cold shard since then ran at
~926 ids/s, which is the number the 10-hour estimate above uses. One request per image -- the shape
batch_chunks_mp_api.py uses -- would be 100x more requests.

CREDENTIALS. Reads MLY_KEY_0..N from the repo-root .env and rotates across
all of them, one key per request. Keys are never logged, never written to the
store, and never put in a URL query string -- they go in the Authorization
header. The repo is public; nothing here hardcodes a path or a token.

EVERY FAILURE ON THIS API LOOKS LIKE HTTP 500. Probed directly:

    invalid token       500  {"code":190,"message":"Invalid OAuth 2.0 ..."}
    cold 300-id batch   500  {"code":2,  "message":"Service temporarily ..."}

So the status code alone cannot tell a dead credential from a batch the
server gave up on, and a run whose keys were all revoked would otherwise
retry, fail, and write "this id does not exist" over 15 million rows. The
error BODY carries the discriminator, so it is always read and branched on:
auth codes retire the key, rate codes back off hard, everything else retries.
If every key retires, the run aborts instead of writing verdicts it did not
earn. A 200 whose body has no `data` key is likewise treated as a failure,
not as an empty success.

CTRL-C DOES NOT COST YOU THE SHARD. Answered ids are flushed to
data/camera/camera-NNNNN.partial.parquet every 50 chunks (~5,000 ids) and
again on the way out of an interrupt, so stopping mid-shard -- or losing
power -- costs at most that window rather than the whole 200,000. A re-run
loads the checkpoint, fetches only what is left, writes the finished shard
and deletes the checkpoint. Measured: interrupting 6s into a shard salvaged
1,597 ids that would previously have been discarded, and the resumed shard
came out identical to an uninterrupted one.

The checkpoint name deliberately does not match SHARD_RE, so a part-fetched
shard is never counted as done, and it holds only ids the API actually
answered for -- fetch_failed rows are left out so a resume retries them.

PROGRESS. A tqdm bar per shard when stderr is a terminal, showing chunks
done, ids collected and any failures. Without a tty -- nohup, systemd, a
disconnected screen -- it is a no-op, because a bar rendered into a log file
is a wall of control characters; there the per-shard lines and
data/camera/progress.json carry the same information, and `status` prints
both the store's contents and any checkpoints waiting to resume.

AN ID IS NEVER RECORDED AS ABSENT BECAUSE A REQUEST FAILED. Three statuses,
and only three:

    ok            the API answered for this id. Any of the three columns may
                  still be NULL -- read the column, not the status.
    missing       the API answered the batch and did not include this id
                  (deleted, private).
    fetch_failed  we never got an answer. `repair` re-runs exactly these.

The distinction matters because a shard file is a completion marker: without
it, one network blip would permanently stamp up to 200,000 ids as having no
camera and no altitude.

CAMERA PARAMETERS ARE NOT ALWAYS THREE NUMBERS. Measured per camera_type:
perspective, fisheye and equirectangular return 3; brown returns 9;
fisheye62 returns 12. The column is DOUBLE[], so length varies by row --
read it as a list, not as a fixed triple.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / 'data' / 'camera'
WORK_FILE = OUT_DIR / 'work.parquet'
PLAN_FILE = OUT_DIR / 'plan.json'
BEAT_FILE = OUT_DIR / 'progress.json'
CATALOG = REPO / 'data' / 'catalog.duckdb'
ENV_FILE = REPO / '.env'

API = 'https://graph.mapillary.com/images'
# The columns this tool adds, in order. merge() treats a file as done only
# when all of them are present, and refuses one that has some but not all.
ADDED = ('camera_parameters', 'altitude', 'computed_altitude')
FIELDS = 'id,camera_type,camera_parameters,altitude,computed_altitude'

# Positions in a shard row. Named because the row grew from 4 columns to 6
# when the altitudes were added and a hardcoded r[3] silently began reading
# altitude where it meant status.
IMAGE_ID, CTYPE, PARAMS, ALT, CALT, STATUS = range(6)
# The API allows 300 per request but times out on cold batches that big. 150
# and 50 also answered every time; 100 was the fastest of them. See the
# header for the measurements.
MAX_IDS = 100
SHARD_IDS = 200_000
# Answered ids are flushed to a .partial file this often, so an interrupt or a
# power cut costs at most this many chunks rather than the whole shard.
CHECKPOINT_CHUNKS = 50

# Patience, not attempts. The old ladder was 5 tries totalling ~33s, so any
# outage longer than half a minute became "these ids do not exist".
RETRIES = 8
BACKOFF_CAP = 60

# Graph API error codes, read out of the response BODY because the HTTP status
# is 500 for all of them.
AUTH_CODES = {10, 102, 190, 200, 458, 463, 467}  # the key is bad -- retire it
RATE_CODES = {4, 17, 32, 613}  # slow down, do not retire
FATAL_MSG = ('invalid oauth', 'access token', 'permission')


class ApiError(Exception):
    """A Graph API failure with the body's error code attached."""

    def __init__(self, http, code, message):
        super().__init__(f'HTTP {http} code {code}: {message}')
        self.http, self.code, self.message = http, code, message

    @property
    def is_auth(self):
        if self.code in AUTH_CODES:
            return True
        low = (self.message or '').lower()
        return any(s in low for s in FATAL_MSG)

    @property
    def is_rate(self):
        return self.code in RATE_CODES


# Set when the run is giving up. Workers check it between attempts and wait
# on it instead of sleeping, so an interrupt during an API outage takes
# effect immediately instead of after the whole backoff ladder drains.
STOP = threading.Event()


class AllKeysDead(Exception):
    """Every credential was retired. Nothing further can be earned."""


def _num(v):
    """A float, or None. Never raises.

    The store column is DOUBLE, and DuckDB's executemany is not atomic -- one
    uncastable value aborts the insert partway and takes the whole shard's
    fetch down with it. Measured: 50 of 101 rows landed before it raised. So
    anything the API sends that is not a number becomes NULL here rather than
    a crash 3 minutes into a shard.
    """
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _params(v):
    """A list of floats, or None. Never raises. See _num."""
    if not isinstance(v, list):
        return None
    out = []
    for x in v:
        f = _num(x)
        if f is None:
            return None  # a partly-numeric vector is not intrinsics
        out.append(f)
    return out or None


def _err_from_body(http, raw):
    try:
        err = json.loads(raw).get('error') or {}
    except Exception:
        err = {}
    return ApiError(http, err.get('code'), err.get('message') or raw[:120])


class _NoBar:
    """Stand-in when there is no terminal to draw on.

    A detached run -- nohup, systemd, a disconnected screen -- has no tty, and
    a progress bar rendered into a log file is a wall of control characters.
    The per-shard lines and data/camera/progress.json carry the same
    information there.
    """

    def update(self, _n=1):
        pass

    def set_postfix_str(self, _s):
        pass

    def close(self):
        pass


def _bar(total, desc):
    if not sys.stderr.isatty():
        return _NoBar()
    try:
        from tqdm import tqdm
    except ImportError:
        return _NoBar()
    return tqdm(total=total,
                desc=desc,
                unit='chunk',
                leave=False,
                dynamic_ncols=True,
                file=sys.stderr)


def _pyarrow():
    """pyarrow, imported on first use so --help works without it."""
    try:
        import pyarrow
    except ImportError:
        sys.exit('pyarrow missing -- run this under the mp14 env '
                 '(it is in requirements.txt)')
    return pyarrow


def _duckdb():
    try:
        import duckdb
    except ImportError:
        sys.exit('duckdb missing -- run this under the mp14 env '
                 '(see data/duckdb_python.txt)')
    return duckdb


def keys():
    """Every MLY_KEY_N in the repo-root .env, in index order. Never logged."""
    if not ENV_FILE.exists():
        sys.exit(f'no {ENV_FILE.name} at the repo root -- cannot authenticate')
    found = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        name, val = line.split('=', 1)
        name, val = name.strip(), val.strip()
        if val[:1] in ('"', "'") and val[-1:] == val[:1] and len(val) > 1:
            val = val[1:-1]  # quoted: take it verbatim
        else:
            val = val.split('#', 1)[0].strip()  # unquoted: trailing comment
        if name.startswith('MLY_KEY_') and name[8:].isdigit() and val:
            found[int(name[8:])] = val
        elif name == 'MLY_KEY' and val:
            found.setdefault(-1, val)
    out = [found[i] for i in sorted(found)]
    if not out:
        sys.exit('no MLY_KEY / MLY_KEY_N found in .env')
    return out


def _catalog_paths(kind):
    """Harvest parquets to read ids from, straight out of the catalog.

    The catalog indexes 230 unfinished writes as kind='other' (the .temp_*
    files); asking for a real kind excludes them. Offline drives are excluded
    too -- their base_path no longer resolves.
    """
    duckdb = _duckdb()
    if not CATALOG.exists():
        sys.exit(f'no catalog at data/catalog.duckdb -- '
                 'run tools/catalog/catalog.py refresh first')
    con = duckdb.connect(str(CATALOG), read_only=True)
    rows = con.execute(
        "select f.path from files f join drives d "
        "  on d.drive = f.drive and d.online "
        "where f.kind = ? order by f.path", [kind]).fetchall()
    con.close()
    if not rows:
        sys.exit(f'catalog knows no {kind!r} files on an online drive')
    return [r[0] for r in rows]


def plan(args):
    duckdb = _duckdb()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if done_shards() and not args.force:
        sys.exit(f'{len(done_shards())} shard(s) already exist in '
                 'data/camera/. Shard boundaries are offsets into '
                 'work.parquet, so re-planning would re-cut them and the '
                 'existing shards would cover the wrong ids. Pass --force '
                 'only if you are also deleting those shards.')
    paths = _catalog_paths(args.kind)
    print(f'reading ids from {len(paths):,} {args.kind} parquets ...',
          flush=True)

    con = duckdb.connect()
    t = time.time()
    # One row per distinct image_id. camera_type rides along because it is
    # useful context in the store -- it is what tells a reader that a NULL
    # camera_parameters on a spherical image is expected rather than a gap.
    con.execute(
        """create or replace table work as
           select image_id,
                  any_value(camera_type) as camera_type
           from read_parquet($paths)
           where image_id is not null
           group by image_id""", {'paths': paths})
    n = con.execute('select count(*) from work').fetchone()[0]
    tmp = Path(str(WORK_FILE) + '.tmp')
    con.execute('copy (select * from work order by image_id) to '
                f'{_sql_path(tmp)} (format parquet)')
    os.replace(tmp, WORK_FILE)

    by_type = con.execute(
        'select camera_type, count(*) as n from work group by 1 order by 2 desc'
    ).fetchall()
    con.close()

    meta = {
        'kind': args.kind,
        'sources': len(paths),
        'ids': n,
        'shard_ids': SHARD_IDS,
        'shards': (n + SHARD_IDS - 1) // SHARD_IDS,
        'work_rows': n,
        'planned_at': int(time.time()),
    }
    PLAN_FILE.write_text(json.dumps(meta, indent=2) + '\n')

    print(f'{n:,} distinct image_ids in {time.time() - t:.0f}s')
    for ct, cnt in by_type:
        print(f'  {str(ct):<16} {cnt:>12,}')
    print(f'\nto fetch: {n:,}   shards: {meta["shards"]}')
    print(
        f'wrote {WORK_FILE.relative_to(REPO)} and {PLAN_FILE.relative_to(REPO)}'
    )


def shard_name(i):
    return f'camera-{i:05d}.parquet'


SHARD_RE = re.compile(r'^camera-(\d{5})\.parquet$')


def partial_name(i):
    """Checkpoint for a shard still in flight.

    Deliberately does NOT match SHARD_RE, so a half-finished shard is never
    counted as done. It holds only ids the API actually answered for --
    fetch_failed rows are left out so a resume retries them.
    """
    return f'camera-{i:05d}.partial.parquet'


def done_shards():
    """What is on disk is what is done. Dot-names are partial writes.

    Matched by an exact pattern: any other camera-*.parquet in the directory
    (a backup, a hand-cut export) is ignored rather than crashing the parse or
    being counted as a shard that was never fetched.
    """
    if not OUT_DIR.exists():
        return set()
    return {
        int(m.group(1))
        for f in OUT_DIR.iterdir()
        for m in [SHARD_RE.match(f.name)] if m
    }


def _beat(**kw):
    tmp = str(BEAT_FILE) + '.tmp'
    Path(tmp).write_text(json.dumps(kw))
    os.replace(tmp, BEAT_FILE)


def fetch_chunk(chunk, key, timeout):
    """One bulk request.

    Returns {image_id: (camera_parameters, altitude, computed_altitude)}, any
    element of which may be None. An id absent from the mapping is one the API
    did not answer for.

    Raises ApiError for anything that is not a well-formed success, including
    an HTTP 200 whose body carries an error envelope instead of `data` --
    "Application request limit reached" arrives that way, and treating it as
    an empty result would record every id in the batch as absent.
    """
    q = urllib.parse.urlencode({
        'image_ids': ','.join(chunk),
        'fields': FIELDS
    })
    req = urllib.request.Request(f'{API}?{q}',
                                 headers={'Authorization': f'OAuth {key}'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as e:
        raise _err_from_body(e.code, e.read().decode(errors='replace'))
    body = json.loads(raw)
    if 'data' not in body:
        raise _err_from_body(200, raw)
    out = {}
    for d in body.get('data', []):
        image_id = d.get('id')
        if image_id is None:
            continue  # a row with no id names nothing; skipping it
        out[str(image_id)] = (_params(d.get('camera_parameters')),
                              _num(d.get('altitude')),
                              _num(d.get('computed_altitude')))
    return out


class KeyPool:
    """Live credentials, with dead ones retired.

    Every failure mode on this API is an HTTP 500, so a revoked key is only
    distinguishable by the error code in the body. Retiring it keeps the run
    from spending its whole retry budget on a credential that will never work
    again; emptying the pool aborts the run rather than manufacturing
    verdicts.
    """

    def __init__(self, keys):
        self._all = list(keys)
        self._live = list(range(len(keys)))
        self._lock = threading.Lock()

    def pick(self, n):
        with self._lock:
            if not self._live:
                raise AllKeysDead(
                    f'all {len(self._all)} credentials were rejected')
            return self._live[n % len(self._live)]

    def key(self, i):
        return self._all[i]

    def retire(self, i):
        with self._lock:
            if i in self._live:
                self._live.remove(i)
            return len(self._live)

    def n_live(self):
        with self._lock:
            return len(self._live)


def fetch_with_retry(chunk, pool, timeout, stats, lock, idx=0):
    """Fetch one chunk, or report why it could not be fetched.

    Returns (rows, None) on success and (None, reason) on failure -- never a
    bare None that the caller could confuse with "the API returned nothing".
    Auth failures retire the offending key and immediately try another, so one
    revoked credential costs one attempt rather than the whole budget.
    """

    def bump(name):
        with lock:
            stats[name] = stats.get(name, 0) + 1

    last = 'unknown'
    attempt = 0
    while attempt < RETRIES:
        if STOP.is_set():
            return None, 'interrupted'
        ki = pool.pick(idx + attempt)
        try:
            return fetch_chunk(chunk, pool.key(ki), timeout), None
        except ApiError as e:
            last = f'code {e.code}'
            if e.is_auth:
                # Retiring a dead credential is not this chunk's failure, so
                # it does not spend an attempt -- otherwise a handful of
                # revoked keys would exhaust every chunk's budget and the run
                # would record real ids as unreachable.
                left = pool.retire(ki)
                bump(f'auth_rejected_code_{e.code}')
                if left == 0:
                    raise AllKeysDead(e.message)
                continue  # another key, no sleep, no attempt spent
            attempt += 1
            bump(f'http_{e.http}_code_{e.code}')
            nap = min(2**attempt + random.random(), BACKOFF_CAP)
            if e.is_rate:
                bump('rate_limited')
                nap = min(nap * 4, BACKOFF_CAP)
            if STOP.wait(nap):  # returns True the moment STOP is set
                return None, 'interrupted'
        except Exception as e:  # timeouts, resets, DNS
            attempt += 1
            last = type(e).__name__
            bump(last)
            if STOP.wait(min(2**attempt + random.random(), BACKOFF_CAP)):
                return None, 'interrupted'
    bump('failed_chunks')
    return None, last


def _classify(want, by_id, failed_ids):
    """Turn one shard's fetch results into rows, keeping the reasons apart.

    'ok'           the API answered for this id; any column may still be NULL
    'missing'      the API answered the batch and did not include this id
    'fetch_failed' we never got an answer -- `repair` re-runs these

    Membership is tested with `in` and values pass through untouched:
    altitude is legitimately 0 for a sixth of the harvest, so nothing here
    may branch on truthiness.
    """
    out = []
    for image_id, ctype in want:
        if image_id in by_id:
            cp, alt, calt = by_id[image_id]
            out.append((image_id, ctype, cp, alt, calt, 'ok'))
        elif image_id in failed_ids:
            out.append((image_id, ctype, None, None, None, 'fetch_failed'))
        else:
            out.append((image_id, ctype, None, None, None, 'missing'))
    return out


def _write_shard(con, out, si, now):
    """Write one shard atomically.

    Rows may be 6-tuples (freshly fetched -- stamped `now`) or 7-tuples
    (carried over by `repair`, keeping the timestamp they already had). A
    repair touching 12 rows must not restamp the other 199,988 as if they
    were fetched today.
    """
    rows = [r if len(r) == 7 else r + (now, ) for r in out]
    col = list(zip(*rows)) if rows else [()] * 7
    # Columns, not 200,000 prepared-statement executions. Measured on a real
    # shard: executemany 48.4s and the COPY 0.0s, so the insert WAS the write
    # -- 2.2 hours across 161 shards with every fetch worker idle. Building an
    # Arrow table instead does the same write in 0.3s. Types are stated rather
    # than inferred so an all-NULL altitude column cannot land as null-typed.
    pa = _pyarrow()
    tb = pa.table({
        'image_id':
        pa.array(col[0], type=pa.string()),
        'camera_type':
        pa.array(col[1], type=pa.string()),
        'camera_parameters':
        pa.array(col[2], type=pa.list_(pa.float64())),
        'altitude':
        pa.array(col[3], type=pa.float64()),
        'computed_altitude':
        pa.array(col[4], type=pa.float64()),
        'status':
        pa.array(col[5], type=pa.string()),
        'fetched_at':
        pa.array(col[6], type=pa.int64()),
    })
    con.register('shard', tb)
    tmp = OUT_DIR / ('.' + shard_name(si))
    con.execute('copy (select image_id, camera_type, camera_parameters, '
                'altitude, computed_altitude, status, fetched_at from shard) '
                f'to {_sql_path(tmp)} (format parquet)')
    os.replace(tmp, OUT_DIR / shard_name(si))


def _save_partial(con, si, answered):
    """Flush what the API has answered so far. Atomic; safe to lose."""
    if not answered:
        return
    pa = _pyarrow()
    ids, cps, alts, calts = [], [], [], []
    for image_id, (cp, alt, calt) in answered.items():
        ids.append(image_id)
        cps.append(cp)
        alts.append(alt)
        calts.append(calt)
    tb = pa.table({
        'image_id':
        pa.array(ids, type=pa.string()),
        'camera_parameters':
        pa.array(cps, type=pa.list_(pa.float64())),
        'altitude':
        pa.array(alts, type=pa.float64()),
        'computed_altitude':
        pa.array(calts, type=pa.float64()),
    })
    con.register('partial', tb)
    tmp = OUT_DIR / ('.' + partial_name(si))
    con.execute('copy (select image_id, camera_parameters, altitude, '
                f'computed_altitude from partial) to {_sql_path(tmp)} '
                '(format parquet)')
    os.replace(tmp, OUT_DIR / partial_name(si))
    con.unregister('partial')


def _load_partial(con, si):
    """Ids a previous attempt at this shard already got answers for."""
    path = OUT_DIR / partial_name(si)
    if not path.exists():
        return {}
    try:
        rows = con.execute(
            'select image_id, camera_parameters, altitude, computed_altitude '
            'from read_parquet(?)', [str(path)]).fetchall()
    except Exception:
        path.unlink()  # unreadable checkpoint: drop it, refetch
        return {}
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def _fetch_ids(want, pool, args, label='', checkpoint=None, desc=None):
    """Fetch a list of (image_id, camera_type). Returns (by_id, failed, stats).

    Raises AllKeysDead upward -- the caller must not write a shard after that,
    because every remaining row would be a verdict nothing earned.
    """
    by_id, failed, stats = {}, set(), {}
    lock = threading.Lock()
    save = checkpoint or (lambda answered: None)
    chunks = [[i for i, _ in want[j:j + args.batch]]
              for j in range(0, len(want), args.batch)]
    if want and not chunks:
        # Only reachable with a nonsense --batch. Without this, zero requests
        # would be issued and _classify would read the silence as "the API
        # answered and omitted every id", stamping them all the terminal
        # 'missing' that `repair` will never revisit.
        raise ValueError(f'--batch {args.batch} produced no chunks for '
                         f'{len(want):,} ids')
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {
            ex.submit(fetch_with_retry, c, pool, args.timeout, stats, lock, k):
            c
            for k, c in enumerate(chunks)
        }
        bar = _bar(len(chunks), desc or f'shard {label}')
        try:
            for k, fut in enumerate(as_completed(futures)):
                got, why = fut.result()
                if got is None:
                    failed.update(futures[fut])
                else:
                    by_id.update(got)
                bar.update(1)
                bar.set_postfix_str(
                    f'{len(by_id):,} ids' +
                    (f' {len(failed):,} failed' if failed else '') +
                    (f' {pool.n_live()} keys' if pool.n_live() < 5 else ''))
                if k % 25 == 0:
                    _beat(shard=label,
                          chunk=k,
                          of=len(chunks),
                          ids=len(by_id),
                          failed=len(failed),
                          keys_live=pool.n_live())
                if k and k % CHECKPOINT_CHUNKS == 0:
                    save(by_id)
        except (KeyboardInterrupt, AllKeysDead):
            # Tell the workers first: the `with` block's __exit__ joins them
            # unconditionally, and a worker parked in the backoff ladder would
            # otherwise hold the process for minutes.
            STOP.set()
            for f in futures:
                f.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
            save(by_id)  # keep what was already answered
            bar.close()
            raise
        finally:
            bar.close()
    return by_id, failed, stats


def _check_fetch_args(args):
    """Validate the knobs `run` and `repair` share. Both fetch; both need it."""
    if not 1 <= args.batch <= 300:
        sys.exit(f'--batch {args.batch}: the API accepts 1..300 per request '
                 '(error code 105 above that); 100 is the default because it '
                 'was the fastest size that answered every time on cold ids')
    if args.workers < 1:
        sys.exit('--workers must be at least 1')
    if args.timeout < 5:
        sys.exit(f'--timeout {args.timeout}: too short to complete a request, '
                 'so every chunk would fail and every id would be stored as '
                 'fetch_failed')


def run(args):
    duckdb = _duckdb()
    if not WORK_FILE.exists():
        sys.exit('no work list -- run `plan` first')
    _check_fetch_args(args)
    pool = KeyPool(keys())
    meta = json.loads(PLAN_FILE.read_text()) if PLAN_FILE.exists() else {}
    shard_ids = meta.get('shard_ids', SHARD_IDS)

    # A shard's boundaries are an offset into work.parquet, so a work list
    # rewritten under a half-finished store would silently re-cut the shards.
    stamp = meta.get('work_rows')
    con = duckdb.connect()
    total = con.execute("select count(*) from read_parquet(?)",
                        [str(WORK_FILE)]).fetchone()[0]
    if stamp is not None and stamp != total:
        sys.exit(f'work.parquet holds {total:,} ids but plan.json recorded '
                 f'{stamp:,} -- the work list changed under the store. '
                 'Re-plan into an empty data/camera/ or restore the old list.')
    n_shards = (total + shard_ids - 1) // shard_ids
    done = done_shards()
    todo = [i for i in range(n_shards) if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(
        f'{total:,} ids  ->  {n_shards} shards, {len(done)} done, '
        f'{len(todo)} to do now  ({pool.n_live()} keys, {args.workers} workers)'
    )
    if not todo:
        print('nothing to do')
        return

    if PLAN_FILE.exists():
        meta['work_rows'] = total
        PLAN_FILE.write_text(json.dumps(meta, indent=2) + '\n')

    started = time.time()
    fetched = total_failed = 0
    for nth, si in enumerate(todo, 1):
        rows = con.execute(
            "select image_id, camera_type from read_parquet(?) "
            "order by image_id limit ? offset ?",
            [str(WORK_FILE), shard_ids, si * shard_ids]).fetchall()

        want = [(str(image_id), ctype) for image_id, ctype in rows]

        # Anything a previous attempt at this shard already got an answer for.
        resumed = _load_partial(con, si)
        todo_ids = [(i, c) for i, c in want if i not in resumed]
        if resumed:
            print(f'  shard {si:05d}: resuming, {len(resumed):,} ids already '
                  f'answered, {len(todo_ids):,} to go')

        t0 = time.time()
        try:
            by_id, failed, stats = _fetch_ids(
                todo_ids,
                pool,
                args,
                label=si,
                desc=f'shard {si:05d} ({nth}/{len(todo)})',
                checkpoint=lambda answered: _save_partial(
                    con, si, {
                        **resumed,
                        **answered
                    }))
        except AllKeysDead as e:
            sys.exit(f'\nABORTED before writing shard {si:05d}: {e}. '
                     'No rows were written -- every credential was rejected, '
                     'so nothing here would be a real answer. Fix the keys in '
                     '.env and re-run; this shard is untouched.')
        except KeyboardInterrupt:
            sys.exit(f'\ninterrupted during shard {si:05d}. The ids already '
                     'answered are checkpointed in '
                     f'{partial_name(si)} -- a re-run picks up from there, '
                     'not from the start.')

        by_id = {**resumed, **by_id}
        out = _classify(want, by_id, failed)
        _write_shard(con, out, si, int(time.time()))
        part = OUT_DIR / partial_name(si)
        if part.exists():
            part.unlink()  # the shard supersedes it

        fetched += len(out)
        total_failed += len(failed)
        rate = len(todo_ids) / max(time.time() - t0, 1e-9)
        answered = sum(1 for r in out if r[STATUS] == 'ok')
        with_params = sum(1 for r in out if r[PARAMS] is not None)
        with_alt = sum(1 for r in out if r[ALT] is not None)
        line = (f'  shard {si:05d}: {len(out):>7,} rows  '
                f'{answered:>7,} answered  {with_params:>7,} params  '
                f'{with_alt:>7,} alt  @ {rate:,.0f} ids/s')
        if failed:
            line += f'  !! {len(failed):,} FETCH_FAILED'
        print(line + (f'  {stats}' if stats else ''))

    if BEAT_FILE.exists():
        BEAT_FILE.unlink()
    print(
        f'\n{fetched:,} rows written in {(time.time() - started) / 60:.1f} min'
    )
    if total_failed:
        print(
            f'\n!! {total_failed:,} ids could not be fetched and are stored as '
            "'fetch_failed', NOT as absent.\n"
            '   Re-run them with:  python tools/detect/fetch_camera_params.py repair'
        )


def repair(args):
    """Re-fetch every row a previous run could not reach.

    Rewrites only the shards that hold fetch_failed rows, in place and
    atomically, so a repair is itself resumable and never touches a shard that
    was already clean.
    """
    duckdb = _duckdb()
    _check_fetch_args(args)
    pool = KeyPool(keys())
    con = duckdb.connect()
    shards = sorted(done_shards())
    if not shards:
        sys.exit('no shards to repair')
    dirty = []
    for si in shards:
        n = con.execute(
            "select count(*) from read_parquet(?) where status = 'fetch_failed'",
            [str(OUT_DIR / shard_name(si))]).fetchone()[0]
        if n:
            dirty.append((si, n))
    if not dirty:
        print('no fetch_failed rows -- nothing to repair')
        return
    print(f'{len(dirty)} shard(s) hold {sum(n for _, n in dirty):,} '
          'fetch_failed rows')

    for si, n in dirty:
        path = str(OUT_DIR / shard_name(si))
        rows = con.execute(
            "select image_id, camera_type from read_parquet(?) "
            "where status = 'fetch_failed'", [path]).fetchall()
        want = [(str(i), c) for i, c in rows]
        try:
            by_id, failed, stats = _fetch_ids(want,
                                              pool,
                                              args,
                                              label=f'{si}r',
                                              desc=f'repair {si:05d}')
        except AllKeysDead as e:
            sys.exit(f'\nABORTED: {e}. Shard {si:05d} left as it was.')
        except KeyboardInterrupt:
            sys.exit(f'\ninterrupted -- shard {si:05d} left as it was')

        keep = con.execute(
            "select image_id, camera_type, camera_parameters, altitude, "
            "computed_altitude, status, fetched_at from read_parquet(?) "
            "where status <> 'fetch_failed'", [path]).fetchall()
        out = [tuple(r) for r in keep] + _classify(want, by_id, failed)
        _write_shard(con, out, si, int(time.time()))
        fixed = n - len(failed)
        print(f'  shard {si:05d}: {fixed:,} of {n:,} recovered' +
              (f', {len(failed):,} still failing' if failed else '') +
              (f'  {stats}' if stats else ''))

    if BEAT_FILE.exists():
        BEAT_FILE.unlink()  # or `status` reports a live run forever after


def status(args):
    duckdb = _duckdb()
    if not PLAN_FILE.exists():
        sys.exit('no plan -- run `plan` first')
    meta = json.loads(PLAN_FILE.read_text())
    done = done_shards()
    print(f'plan: {meta["ids"]:,} ids from {meta["sources"]:,} {meta["kind"]} '
          f'parquets, {meta["shards"]} shards')
    print(f'shards on disk: {len(done)} / {meta["shards"]}')
    if not done:
        return
    con = duckdb.connect()
    rows = con.execute(
        "select status, count(*) as n from read_parquet(?) group by 1 order by 2 desc",
        [str(OUT_DIR / 'camera-*.parquet')]).fetchall()
    tot = sum(n for _, n in rows)
    print(f'rows: {tot:,}')
    for st, n in rows:
        print(f'  {st:<20} {n:>12,}  {100 * n / tot:5.1f}%')
    parts = sorted(
        int(SHARD_RE.match(f.name.replace('.partial', '')).group(1))
        for f in OUT_DIR.iterdir() if f.name.endswith('.partial.parquet')
        and SHARD_RE.match(f.name.replace('.partial', '')))
    if parts:
        held = con.execute(
            "select count(*) from read_parquet(?)",
            [str(OUT_DIR / 'camera-*.partial.parquet')]).fetchone()[0]
        print(f'\ncheckpoints: {len(parts)} shard(s) part-fetched '
              f'({held:,} ids already answered) -- shard(s) '
              f'{", ".join(f"{i:05d}" for i in parts[:5])}'
              f'{" ..." if len(parts) > 5 else ""}\n'
              '   a re-run resumes from these rather than refetching them')
    bad = dict(rows).get('fetch_failed', 0)
    if bad:
        print(
            f'\n!! {bad:,} rows are fetch_failed -- the API was never reached '
            'for them.\n   They are NOT evidence the ids lack a camera. '
            'Re-run with:  python tools/detect/fetch_camera_params.py repair')
    if BEAT_FILE.exists():
        print(f'live: {BEAT_FILE.read_text()}')


def backup_name(path):
    """Where the original is parked while its replacement is checked."""
    return Path(path).with_name(Path(path).name + '.premerge')


def recover_backups(paths):
    """Put back anything a previous run died in the middle of.

    A .premerge beside a file means that file was mid-swap. The backup is the
    original, so restoring it is always the safe direction: at worst the merge
    is redone, and merge is idempotent because a file that already has all
    three columns is skipped.
    """
    restored = 0
    for path in paths:
        bak = backup_name(path)
        if not bak.exists():
            continue
        os.replace(bak, path)
        restored += 1
        print(f'  recovered {Path(path).name} from an interrupted merge')
    return restored


def verify_merged(con, bak, new, cols):
    """Is `new` the original plus three columns, and nothing else changed?

    Row count, column order, and -- the check that matters -- every value in
    every ORIGINAL column, compared both directions. A row-count match alone
    would not notice a join that silently reordered or rewrote a column.
    Returns None when it is sound, or the reason it is not.
    """
    got = [
        c[0] for c in con.execute("select * from read_parquet(?) limit 0",
                                  [str(new)]).description
    ]
    if got != list(cols) + list(ADDED):
        return f'columns are {got}, expected {list(cols) + list(ADDED)}'
    a = con.execute("select count(*) from read_parquet(?)",
                    [str(bak)]).fetchone()[0]
    b = con.execute("select count(*) from read_parquet(?)",
                    [str(new)]).fetchone()[0]
    if a != b:
        return f'{a:,} rows became {b:,}'
    sel = ', '.join(f'"{c}"' for c in cols)
    drift = con.execute(
        f'select (select count(*) from ((select {sel} from read_parquet(?)) '
        f'   except (select {sel} from read_parquet(?)))) as lost, '
        f'       (select count(*) from ((select {sel} from read_parquet(?)) '
        f'   except (select {sel} from read_parquet(?)))) as gained',
        [str(bak), str(new), str(new), str(bak)]).fetchone()
    if drift[0] or drift[1]:
        return (f'{drift[0]} original row(s) missing and {drift[1]} altered '
                'in the rewrite')
    return None


def _sql_path(s):
    """A path as a SQL string literal.

    The COPY target CANNOT be a bound parameter: DuckDB binds the target
    first, so `copy (... read_parquet(?) ...) to ?` with [src, dst] makes SRC
    the output file. Proven by probe -- with markers [READ1, READ2, TARGET]
    the error names READ2 as the missing input, i.e. the target ate READ1.
    Getting this wrong would have written the join over the source harvest
    file, so the target is always inlined and only reads are parameters.
    """
    return "'" + str(s).replace("'", "''") + "'"


def merge(args):
    """Rewrite the harvest parquets with the three fetched columns joined on.

    A file counts as done only when all three are present; one that has some
    but not all is refused, because merge touches a file once and skipping it
    would strand the missing columns forever.

    NOTHING IS OVERWRITTEN UNTIL ITS REPLACEMENT HAS BEEN CHECKED. Per file:

        os.link(path, path.premerge)   a hardlink -- instant, no extra space,
                                       a second name for the original inode
        write the rewrite to a dot-name
        verify it against the backup   columns, order, row count, and every
                                       value in every ORIGINAL column, both
                                       directions
        os.replace(tmp, path)          atomic
        verify the file now on disk    the one a reader will actually open
        unlink the backup              only now is the original let go

    Any failure puts the original back with os.replace(bak, path). A run
    killed between the swap and the unlink leaves a .premerge behind, and the
    next run restores from it before doing anything else -- restoring is
    always the safe direction, since merge skips a file that is already done.

    Opt-in, and the slow path by design: it rewrites every source file. Each
    file is written to a dot-name and os.replace()d, so a crash leaves the
    original intact, and a file whose row count changed under the join is
    refused rather than replaced.

    It is also ONE-SHOT per file -- a file that already has the column is
    skipped forever -- so it refuses to run against an incomplete store. A
    merge at 4 of 161 shards would bake NULL into 98% of the rows and lock
    them that way.
    """
    duckdb = _duckdb()
    if not PLAN_FILE.exists():
        sys.exit('no plan -- run `plan` first')
    meta = json.loads(PLAN_FILE.read_text())
    have, want_shards = len(done_shards()), meta.get('shards', 0)
    if have < want_shards and not args.allow_partial:
        sys.exit(
            f'the store holds {have} of {want_shards} shards. Merging now '
            f'would write NULL into ~{100 * (1 - have / max(want_shards, 1)):.0f}% '
            'of rows AND skip those files forever, because merge only '
            'touches a file that lacks the column. Finish `run` (and '
            '`repair`) first, or pass --allow-partial if you truly mean it.')

    con = duckdb.connect()
    bad = con.execute(
        "select count(*) from read_parquet(?) where status = 'fetch_failed'",
        [str(OUT_DIR / 'camera-*.parquet')]).fetchone()[0] if have else 0
    if bad and not args.allow_partial:
        sys.exit(
            f'{bad:,} rows are fetch_failed -- run `repair` first, or the '
            'harvest permanently records them as having no camera.')

    if args.kind == 'all_data' and not args.allow_partial:
        sys.exit('--kind all_data rewrites 35,220 files and 3.66 TiB, and the '
                 'store only covers ground-animal ids, so almost every row '
                 'would get NULL. Pass --allow-partial if that is really the '
                 'intent.')

    paths = _catalog_paths(args.kind)
    shards = str(OUT_DIR / 'camera-*.parquet')

    # A .premerge left behind means a previous run was killed mid-swap. Put
    # those originals back before deciding what still needs merging.
    n_rec = recover_backups(paths)
    if n_rec:
        print(f'recovered {n_rec} file(s) from an interrupted merge\n')

    print(f'{len(paths):,} {args.kind} files to rewrite' +
          ('  (DRY RUN -- nothing is written)' if args.dry_run else ''))
    if not args.dry_run:
        print('NOTE: a half-merged directory has two schemas in it, so a glob '
              'read over these\n      files will drop the three new columns '
              'until the merge finishes.')
    changed = skipped = refused = 0
    for i, path in enumerate(paths):
        cols = [
            c[0] for c in con.execute("select * from read_parquet(?) limit 0",
                                      [path]).description
        ]
        present = [c for c in ADDED if c in cols]
        if len(present) == len(ADDED):
            skipped += 1
            continue
        if present:
            # Adding the rest would collide with the columns already there,
            # and skipping would strand this file forever -- merge only ever
            # touches a file once.
            refused += 1
            print(
                f'  REFUSED {Path(path).name}: already has {present} but not '
                f'{[c for c in ADDED if c not in cols]}. Restore it from a '
                'file that has none of them, or drop the partial columns.')
            continue
        before = con.execute("select count(*) from read_parquet(?)",
                             [path]).fetchone()[0]
        if args.dry_run:
            changed += 1
            continue
        tmp = Path(path).with_name('.' + Path(path).name + '.merge')
        bak = backup_name(path)
        swapped = False
        try:
            # A HARDLINK, not a copy: a second name for the same inode, made
            # instantly and costing no space. os.replace then repoints `path`
            # at the rewrite while `bak` still names the untouched original,
            # so the original survives until it has been checked and is one
            # rename away from coming back.
            os.link(path, bak)
            # The harvest is ZSTD; DuckDB's COPY default is SNAPPY, which
            # would inflate these files ~1.9x across terabytes.
            con.execute(
                "copy (select a.*, c.camera_parameters, c.altitude, "
                "       c.computed_altitude from read_parquet(?) a "
                "      left join read_parquet(?) c using (image_id)) "
                f"to {_sql_path(tmp)} (format parquet, compression zstd)",
                [path, shards])
            why = verify_merged(con, bak, tmp, cols)
            if why:
                refused += 1
                print(f'  REFUSED {Path(path).name}: {why} -- original kept')
                continue
            os.replace(tmp, path)
            swapped = True
            # Read the file that is actually on disk now, not the temp it came
            # from: the check is worth nothing if it never looks at what a
            # reader will open.
            why = verify_merged(con, bak, path, cols)
            if why:
                os.replace(bak, path)  # put the original back
                refused += 1
                print(f'  REVERTED {Path(path).name}: {why}')
                continue
            changed += 1
        except Exception as e:
            if swapped and bak.exists():
                os.replace(bak, path)
                print(f'  REVERTED {Path(path).name}: {type(e).__name__}: {e}')
            raise
        finally:
            if tmp.exists():
                tmp.unlink()  # never leave a dot-file the catalog would index
            if bak.exists():
                bak.unlink()  # verified: the original is no longer needed
        if i % 500 == 0:
            print(f'  {i:,}/{len(paths):,} ...', flush=True)
    print(f'\nrewritten: {changed:,}   already had the column: {skipped:,}   '
          f'refused: {refused:,}')
    if changed and not args.dry_run:
        print('run tools/catalog/catalog.py refresh -- file sizes changed')


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('plan',
                       help='build the work list of distinct image_ids')
    p.add_argument('--kind',
                   default='ground_animals',
                   choices=('ground_animals', 'all_data'),
                   help='which harvest parquets to read ids from '
                   '(default: ground_animals, 32.0M ids; all_data is 3.0B '
                   'rows and months of fetching)')
    p.add_argument('--force',
                   action='store_true',
                   help='re-plan even though shards exist (they will no '
                   'longer line up -- delete them too)')
    p.set_defaults(fn=plan)

    r = sub.add_parser('run',
                       help='fetch camera_parameters + altitudes, resumably')
    r.add_argument('--workers',
                   type=int,
                   default=128,
                   help='concurrent requests (measured: 128 -> ~827 ids/s, '
                   'where the curve flattens)')
    r.add_argument('--batch',
                   type=int,
                   default=MAX_IDS,
                   help=f'ids per request (default {MAX_IDS}; the API caps at '
                   '300 but times out on cold batches that large)')
    r.add_argument('--timeout', type=int, default=90)
    r.add_argument('--limit',
                   type=int,
                   default=0,
                   help='stop after N shards this run')
    r.set_defaults(fn=run)

    s = sub.add_parser('status', help='what is done')
    s.set_defaults(fn=status)

    rp = sub.add_parser('repair',
                        help="re-fetch rows a previous run recorded as "
                        "fetch_failed")
    rp.add_argument('--workers', type=int, default=128)
    rp.add_argument('--batch', type=int, default=MAX_IDS)
    rp.add_argument('--timeout', type=int, default=90)
    rp.set_defaults(fn=repair)

    m = sub.add_parser('merge',
                       help='rewrite the harvest parquets with the column')
    m.add_argument('--kind',
                   default='ground_animals',
                   choices=('ground_animals', 'all_data'))
    m.add_argument('--dry-run', action='store_true')
    m.add_argument('--allow-partial',
                   action='store_true',
                   help='merge even though the store is incomplete, has '
                   'fetch_failed rows, or is being applied to all_data '
                   '-- the affected rows are then NULL forever')
    m.set_defaults(fn=merge)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
