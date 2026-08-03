"""
Live status publisher for the detection sweep (spec §7, Addendum A.5).

The engine owns all counters in memory; this module only *publishes* them.
Every ``interval`` seconds a daemon thread snapshots the counters the engine
pushed via ``StatusWriter.update()`` and atomically rewrites
``$DETECT_ROOT/status.json`` (``.tmp`` + ``os.replace`` in the same directory,
so the rename can never cross a filesystem and a reader can never observe a
half-written file — §7.2). The publisher NEVER opens DuckDB: the catalog
writer lock must not be contended by a status tick (§7.2), and every number it
needs is already an in-memory accumulator.

The dashboard side uses ``read_status()``, which collapses "file missing",
"file unparsable" and "file stale" into ``{'running': False}`` so the client
has exactly one degraded state to render (§7.4).

Payload correctness rules from §7.3 baked in here:
  - ``json.dumps(..., allow_nan=False)`` — bare ``NaN``/``Infinity`` is
    invalid JSON and would break the client at exactly the moment a drive
    parks; unknown ETA is JSON ``null`` instead.
  - every division is guarded (``rate > 1e-9``) so the first minutes of a run
    cannot raise ``ZeroDivisionError`` inside the publisher thread.
  - serialize to a string first, write only on success — a serialization
    error must never leave a truncated ``.tmp`` behind; failures are counted
    in ``publish_errors`` so a broken publisher is distinguishable from a
    dead pipeline.
  - every float is rounded (1 dp rates/percentages, whole seconds) to keep
    the blob small.

Drive labels only, never absolute roots or image paths — the dashboard is
published on the tailnet with no auth (§7.4 security note).
"""

import copy
import json
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# gitignored pointer to the sweep's output root (repo convention: no
# env-specific drive paths in tracked files)
DETECT_ROOT_FILE = os.path.join(REPO, 'data', 'detect_root.txt')

# not_a_dog healthy band, percent (Addendum A.5): the labelled prior is
# 7–16%; a sustained reading outside it means the detector behaves
# differently on unseen geography than on the labelled sample.
NOT_A_DOG_BAND = (7.0, 16.0)

# rate windows (seconds): a fast one for "what is it doing right now" and a
# slow one that the ETA is computed from, so a 30 s stall does not swing the
# finish estimate of a 3.5-day run.
WINDOW_FAST = 60.0
WINDOW_SLOW = 900.0

# a drive whose done-counter has not moved for this long (while unfinished)
# is flagged stalled — the spin-down failure mode on the 1058:25a3 bridges
# shows up exactly as one lane silently freezing (§7.4 drive table).
STALL_AFTER = 30.0

ERROR_KEYS = ('read', 'decode', 'missing', 'infer', 'mount_lost')


def detect_root():
    """Sweep output root from the gitignored pointer file (with fallback)."""
    try:
        with open(DETECT_ROOT_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return line.rstrip('/')
    except OSError:
        pass
    return os.path.join(REPO, 'data', 'detect')


def default_status_path():
    return os.path.join(detect_root(), 'status.json')


def _r1(v):
    """Round to 1 dp for the wire; None passes through (JSON null)."""
    return None if v is None else round(float(v), 1)


def _win_rate(samples, now, window):
    """Rate over the trailing ``window`` from (ts, count) samples.

    Uses the oldest sample still inside the window; returns 0.0 until two
    samples span a measurable interval (guarded division, §7.3).
    """
    if len(samples) < 2:
        return 0.0
    old = None
    for ts, n in samples:
        if ts >= now - window:
            old = (ts, n)
            break
    if old is None:
        old = samples[-2]
    dt = samples[-1][0] - old[0]
    if dt > 1e-9:
        return max(0.0, (samples[-1][1] - old[1]) / dt)
    return 0.0


def sample_gpu():
    """One nvidia-smi --query-gpu sample; None when unavailable.

    Absence (no binary / no driver) is remembered so a GPU-less box does not
    fork a failing child every tick; transient errors are retried next tick.
    """
    if sample_gpu._absent:
        return None
    try:
        out = subprocess.run([
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,'
            'memory.total,temperature.gpu', '--format=csv,noheader,nounits'
        ],
                             capture_output=True,
                             text=True,
                             timeout=3)
        if out.returncode != 0:
            return None
        parts = [
            p.strip() for p in out.stdout.strip().splitlines()[0].split(',')
        ]

        def num(s):
            try:
                return int(float(s))
            except ValueError:
                return None  # "[N/A]" on some boards

        return {
            'util': num(parts[0]),
            'mem_used_mb': num(parts[1]),
            'mem_total_mb': num(parts[2]),
            'temp': num(parts[3])
        }
    except FileNotFoundError:
        sample_gpu._absent = True
        return None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


sample_gpu._absent = False


class StatusWriter:
    """Engine-side publisher of ``status.json``.

    The engine pushes *cumulative* counters with ``update()`` (cheap, just a
    dict merge under a lock); a daemon thread started by ``start()`` samples
    them every ``interval`` seconds, derives windowed rates / ETA / health
    flags, and atomically rewrites the status file. ``stop()`` publishes one
    final frame with a terminal state.

    All counters are absolute since run start (rehydrated by the engine on
    resume, §7.3), never deltas — so a missed tick loses nothing.
    """

    def __init__(self,
                 run_id,
                 gen,
                 imgs_total,
                 status_path=None,
                 interval=5.0,
                 drive_totals=None,
                 region_totals=None,
                 gpu_fn=sample_gpu,
                 time_fn=time.time):
        self.path = status_path or default_status_path()
        self.interval = float(interval)
        self._now = time_fn
        self._gpu_fn = gpu_fn
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.publish_errors = 0
        # window sample history: (ts, imgs_done); sized for WINDOW_SLOW at the
        # publish cadence plus slack
        n = int(WINDOW_SLOW / max(self.interval, 0.1)) + 8
        self._samples = deque(maxlen=n)
        self._drv_samples = {}  # drive -> deque[(ts, done)]
        self._drv_last_move = {}  # drive -> ts of last progress
        self._maxlen = n
        started = self._now()
        # denominators come from the deduped worklist (§7.3): totals are
        # fixed at plan-build time and passed in once, never re-derived.
        self._c = {
            'run_id': run_id,
            'gen': gen,
            'state': 'running',
            'started_at': started,
            'imgs_done': 0,
            'imgs_total': int(imgs_total),
            # images this PROCESS did; imgs_done is global across restarts
            'run_imgs_done': 0,
            'boxes_total': 0,
            # positives is ALL-TIME (seeded from the store at startup, like
            # imgs_done); run_positives is what this process found
            'positives': 0,
            'run_positives': 0,
            'crops_classified': 0,
            'class_counts': {
                'leashed': 0,
                'unleashed': 0,
                'not_a_dog': 0
            },
            'drive_done': {},
            'drive_queue': {},
            'drive_totals': dict(drive_totals or {}),
            'region_done': {},
            'region_totals': dict(region_totals or {}),
            'errors': {
                k: 0
                for k in ERROR_KEYS
            },
            'last_error': None,
        }

    # ── engine-facing API ───────────────────────────────────────────────
    def update(self, **counters):
        """Merge cumulative counters (dict-valued keys merge per entry)."""
        with self._lock:
            for k, v in counters.items():
                if isinstance(v, dict) and isinstance(self._c.get(k), dict):
                    self._c[k].update(v)
                else:
                    self._c[k] = v

    def start(self):
        self._thread = threading.Thread(target=self._run,
                                        daemon=True,
                                        name='status-writer')
        self._thread.start()
        return self

    def stop(self, state='stopped', timeout=10.0):
        """Publish a final frame and join the publisher thread."""
        self.update(state=state)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self.publish_now()

    # ── publishing ──────────────────────────────────────────────────────
    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.publish_now()
            except Exception:
                # never let the publisher thread die silently mid-run; the
                # error is surfaced through publish_errors (§7.3)
                self.publish_errors += 1

    def publish_now(self):
        """Snapshot, derive, serialize, and atomically replace the file."""
        try:
            # derivation AND serialization inside the guard: a malformed
            # engine push (or a raising gpu_fn) must fail this one frame —
            # counted in publish_errors — not propagate into stop() during
            # engine shutdown or bypass the accounting (§7.3: a broken
            # publisher must be distinguishable from a dead pipeline).
            payload = self._payload()
            # serialize FIRST; only a fully-built string is ever written, so
            # a serialization error cannot truncate the published file (§7.3)
            blob = json.dumps(payload, allow_nan=False, separators=(',', ':'))
        except Exception:
            self.publish_errors += 1
            return False
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + '.tmp'
            with open(tmp, 'w') as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)  # atomic, same dir → same fs
        except OSError:
            self.publish_errors += 1
            return False
        return True

    def _payload(self):
        now = self._now()
        with self._lock:
            # deep copy so derivation runs lock-free; serializability is
            # checked later at the guarded json.dumps, never here
            c = copy.deepcopy(self._c)
        self._samples.append((now, c['imgs_done']))
        r60 = _win_rate(self._samples, now, WINDOW_FAST)
        r900 = _win_rate(self._samples, now, WINDOW_SLOW)

        remaining = max(0, c['imgs_total'] - c['imgs_done'])
        # unknown ETA is null, not inf/NaN (§7.3) — the client renders "—"
        eta = int(remaining / r900) if r900 > 1e-9 else None
        if remaining == 0 and c['imgs_total'] > 0:
            eta = 0

        drives = {}
        for d in sorted(set(c['drive_done']) | set(c['drive_totals'])):
            done = int(c['drive_done'].get(d, 0))
            dq = self._drv_samples.setdefault(d, deque(maxlen=self._maxlen))
            if not dq or dq[-1][1] != done:
                self._drv_last_move[d] = now
            dq.append((now, done))
            total = int(c['drive_totals'].get(d, 0))
            unfinished = total == 0 or done < total
            stalled = (unfinished and c['state'] == 'running'
                       and now - self._drv_last_move.get(d, now) > STALL_AFTER)
            drives[d] = {
                'done': done,
                'total': total,
                'rate': _r1(_win_rate(dq, now, WINDOW_FAST)),
                'queue_depth': int(c['drive_queue'].get(d, 0)),
                'stalled': bool(stalled),
            }

        regions = {}
        for r in sorted(set(c['region_done']) | set(c['region_totals'])):
            tot = c['region_totals'].get(r, 0)
            pct = (100.0 * c['region_done'].get(r, 0) / tot) if tot else 0.0
            regions[r] = _r1(min(pct, 100.0))

        done = c['imgs_done']
        positive_rate = _r1(100.0 * c['positives'] / done) if done else None
        boxes_per_img = self._trailing_boxes(c, now)

        cls = c['class_counts']
        n_cls = c['crops_classified']
        nad = _r1(100.0 * cls.get('not_a_dog', 0) / n_cls) if n_cls else None
        band_lo, band_hi = NOT_A_DOG_BAND
        in_band = None if nad is None else bool(band_lo <= nad <= band_hi)

        return {
            'run_id': c['run_id'],
            'gen': c['gen'],
            'state': c['state'],
            'pid': os.getpid(),
            'ts': round(now, 1),
            'started_at': _iso(c['started_at']),
            'updated_at': _iso(now),
            'imgs_done': done,
            'imgs_total': c['imgs_total'],
            'run_imgs_done': int(c.get('run_imgs_done') or 0),
            'img_per_sec': {
                'w60': _r1(r60),
                'w900': _r1(r900)
            },
            'eta_s': eta,
            'drives': drives,
            'regions': regions,
            'positives': c['positives'],
            'run_positives': c.get('run_positives', 0),
            'positive_rate': positive_rate,
            'boxes_total': c['boxes_total'],
            'boxes_per_img': boxes_per_img,
            'crops_classified': n_cls,
            'class_split': {
                k: int(cls.get(k, 0))
                for k in ('leashed', 'unleashed', 'not_a_dog')
            },
            'not_a_dog_rate': nad,
            'not_a_dog_band': {
                'lo': band_lo,
                'hi': band_hi,
                'in_band': in_band
            },
            'gpu': self._gpu_fn(),  # once per write; None if absent
            'errors': {
                k: int(c['errors'].get(k, 0))
                for k in ERROR_KEYS
            },
            'last_error': c['last_error'],
            'publish_errors': self.publish_errors,
        }

    def _trailing_boxes(self, c, now):
        """boxes/img over the slow window (falls back to cumulative).

        Trailing, not cumulative, because a drift in box density mid-run is a
        health signal (§8.3) and a 3-day cumulative average would bury it.
        """
        dq = getattr(self, '_box_samples', None)
        if dq is None:
            dq = self._box_samples = deque(maxlen=self._maxlen)
        dq.append((now, c['imgs_done'], c['boxes_total']))
        old = next((s for s in dq if s[0] >= now - WINDOW_SLOW), dq[0])
        d_img = dq[-1][1] - old[1]
        d_box = dq[-1][2] - old[2]
        if d_img > 0:
            return round(d_box / d_img, 3)
        if c['imgs_done'] > 0:
            return round(c['boxes_total'] / c['imgs_done'], 3)
        return None


def _iso(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


# ── dashboard-facing reader ─────────────────────────────────────────────────
def read_status(path=None, stale_after=120.0, now_fn=time.time):
    """Read the published status; degrade to ``{'running': False}``.

    Missing file, unparsable file, and a payload older than ``stale_after``
    seconds all collapse to the same shape, so the dashboard has exactly one
    "sweep not running" state to render (§7.4). A fresh but terminal frame
    (state stopped/failed/done) is also not-running — but keeps its state so
    the client can say *why*.
    """
    path = path or default_status_path()
    try:
        with open(path) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError('payload is not an object')
    except (OSError, ValueError):
        return {'running': False}
    ts = d.get('ts')
    if not isinstance(ts, (int, float)):
        try:  # tolerate a foreign payload
            ts = os.stat(path).st_mtime
        except OSError:
            return {'running': False}
    age = now_fn() - ts
    if age > stale_after:
        return {
            'running': False,
            'stale': True,
            'age_s': round(age),
            'state': d.get('state'),
            'run_id': d.get('run_id'),
            'pid': d.get('pid')
        }
    d['age_s'] = round(max(age, 0))
    d['running'] = d.get('state') not in ('stopped', 'failed', 'done')
    return d
