"""
Predictions store: immutable parquet part-pairs written straight from the
inference process, queried with ``duckdb.connect()`` (no db file), per
DETECTION_RUN_STRATEGY.md §5.

The one architectural decision (§5.1): NO hot binary log, NO compactor
process, NO ``predictions.duckdb`` anywhere -- a foreign writer lock on a
.duckdb blocks even READ_ONLY attaches [M], while duckdb over a parquet tree
answered the full-store rollup in 0.04 s during that same lock. Every file
here is written once and never mutated (§5.2); a resumed partial shard gets a
new ``p<start>_<end>`` filename instead of overwriting its own committed
prefix, and a shard is DONE iff its committed parts tile ``[0, shard_len)``
exactly -- no gaps, no overlaps (§5.2, §6.3).

What this module owns:
  * ``Writer`` / ``ShardWriter``     -- buffered per-shard-part writes with the
    exact §5.3 schemas + encodings and the §5.6 durable-commit sequence
  * ``tiling_resume``                -- committed parts on disk are the truth;
    footer-verified adoption + exact-tiling DONE test (§6.3)
  * ``invariants``                   -- §5.4 canonical progress query + the
    three accounting assertions over the parquet glob
  * ``verify``                       -- read every parquet footer so a
    truncated (glob-poisoning, §5.6) file is found by a scan
  * ``compact``                      -- §5.7 post-run per-cell rewrite
  * ``ensure_bootstrap``             -- the 0-row full-schema partition without
    which every query on an empty store raises IOException (§5.2)

Environment: needs pyarrow (installed into ``dnd`` per §5.1). duckdb is only
needed by ``invariants``; when it is not importable in-process (dnd has no
duckdb) the queries run in a helper interpreter resolved from
``$DETECT_DUCKDB_PYTHON`` or the gitignored ``data/duckdb_python.txt`` --
never a hardcoded env path (repo convention).
"""

import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter, deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger('detect.store')

# --------------------------------------------------------------------------
# Constants pinned by the spec
# --------------------------------------------------------------------------

# §6.1: shard = contiguous inode-ordered slice of 4,000 ids from one
# (cell, drive) pair. The writer does not enforce this length -- shard
# geometry belongs to sweep.py -- but it is the reference part size.
SHARD_LEN = 4000

# §5.3 encoding block, all measured: zstd level 9 was 0.7% LARGER (pure CPU
# waste), DELTA_BINARY_PACKED on image_id+ts_off took 11.63 -> 7.56 B/row.
COMPRESSION = 'zstd'
COMPRESSION_LEVEL = 3
ROW_GROUP_SIZE = 131072
DELTA_COLS = ('image_id', 'ts_off')  # only where present in the table

# §5.5 guards: the hard abort thresholds that can actually fire. The NMS
# ceiling (max_det=300 x 32.54M x 26 B = 254 GB) genuinely exceeds free
# space, so "it can't get that big" is false.
FREE_MIN_BYTES = 20 * 2**30  # hard abort: DETECT_ROOT free < 20 GB
DET_ROWS_MAX = 200_000_000  # hard abort: cumulative detection rows
SOFT_WINDOW_IMAGES = 100_000  # soft alarm window, GLOBAL not per-worker
SOFT_BOXES_PER_IMG = 0.60  # 2.4x the pessimistic 0.2514 boxes/img
MAX_DET = 300  # §4.6 inference max_det; n_det==MAX_DET flag

# §5.3 images.status codes. Never NULL.
STATUS_OK = 0
STATUS_READ_ERROR = 1
STATUS_DECODE_ERROR = 2
STATUS_MISSING = 3
STATUS_INFER_ERROR = 4
STATUS_MOUNT_LOST = 5

# §8.3 per-image guard tier -> fixed-width bitmask in img.guards (UINT16,
# "not a comma string"). Bit n-1 <=> guard Gn; G3 is the drive-level stderr
# tier and never sets a per-image bit, its slot is reserved so numbering
# stays 1:1 with the spec table. Unlisted G6/G8/G10/G11 are reserved too.
GUARD_BITS = {
    'G1_SOI_EOI': 1 << 0,  # SOI/EOI markers on the raw bytes
    'G2_DECODE_NONE': 1 << 1,  # imdecode returned None
    'G3_RESERVED': 1 << 2,  # drive-level libjpeg-stderr tier, never set
    'G4_DIMS_VS_SOF':
    1 << 3,  # decoded dims x reduction vs SOF (never parquet)
    'G5_SHAPE': 1 << 4,  # ndim==3 and shape[2]==3 failed
    'G6_RESERVED': 1 << 5,
    'G7_LOW_STD': 1 << 6,  # arr.std() < 2.0
    'G8_RESERVED': 1 << 7,
    'G9_CMYK': 1 << 8,  # SOFn component count == 4
    'G10_RESERVED': 1 << 9,
    'G11_RESERVED': 1 << 10,
    'G12_TINY_FILE': 1 << 11,  # file_bytes < 1024
}
# §5.5 "flag every image where n_det == max_det" -- set by the writer itself,
# high bit so it can never collide with a future per-image decode guard.
GUARD_NDET_MAXED = 1 << 15

# §5.2 immutable part naming: s<shard 5d>.p<start 6d>_<end 6d>.{img,det}.
# \d{5,}/\d{6,} so zero-padding can grow without breaking old files.
_PART_RE = re.compile(
    r'^s(?P<shard>\d{5,})\.p(?P<start>\d{6,})_(?P<end>\d{6,})'
    r'\.(?P<kind>img|det)\.parquet$')

_STATE_NAME = '_state.json'

# --------------------------------------------------------------------------
# Schemas -- §5.3, byte-exact. Hive partition columns gen/region/cell/drive
# come from the path; region is an explicit *catalog* column upstream and is
# never parsed from the cell name, but it lives in the path here, not in the
# file. leash_class/leash_conf extend the detection row per Addendum A.4 and
# are NULL until the classifier stage runs.
# --------------------------------------------------------------------------

DET_SCHEMA = pa.schema([
    # 15-17 digit Mapillary ids: max 9.99e16 vs UBIGINT 1.8e19, 0 cast
    # failures in 674,398 sampled rows [M].
    pa.field('image_id', pa.uint64(), nullable=False),
    pa.field('det_idx', pa.uint8(), nullable=False),  # 0..n-1, desc conf
    # Full precision, never a 3-decimal string (§5.3).
    pa.field('conf', pa.float32(), nullable=False),
    # Original full-res pixels (§4.5 transform). Float, not uint16: the
    # float->unsigned cast is formally undefined and 66 MB saved is noise.
    pa.field('x1', pa.float32(), nullable=False),
    pa.field('y1', pa.float32(), nullable=False),
    pa.field('x2', pa.float32(), nullable=False),
    pa.field('y2', pa.float32(), nullable=False),
    pa.field('run_id', pa.uint16(), nullable=False),
    # Ties the row to _state.json (§5.3).
    pa.field('shard_idx', pa.uint32(), nullable=False),
    # Addendum A.4: keep every row even when not_a_dog -- that IS the
    # measured non-dog rate. 0=leashed 1=unleashed 2=not_a_dog, NULL until
    # classified.
    pa.field('leash_class', pa.uint8(), nullable=True),
    pa.field('leash_conf', pa.float32(), nullable=True),
    # ── provenance ────────────────────────────────────────────────────────
    # Which model produced this box. Eight hex of the engine file's sha256 --
    # the same digest data/best_models.json records as sha256_engine, so a row
    # joins straight to the model registry.
    #
    # A hash, not a filename: engine files get overwritten in place, and a
    # path that means one model today can mean another tomorrow. NULL on every
    # row written before this column existed (see _sql_src on union_by_name),
    # which is honest -- those rows really are unattributable except through
    # the run manifest.
    pa.field('model_sha8', pa.string(), nullable=True),
])

LEASH_CLASSES = {'leashed': 0, 'unleashed': 1, 'not_a_dog': 2}

IMG_SCHEMA = pa.schema([
    # UNIQUE across the whole store -- asserted by invariants() (§5.4).
    pa.field('image_id', pa.uint64(), nullable=False),
    pa.field('drive', pa.uint8(), nullable=False),  # copy actually read
    pa.field('status', pa.uint8(), nullable=False),  # 0..5, never NULL
    # UINT16, never SQL NULL: UINT8 cannot hold max_det=300 (300 -> 44
    # silently) (§5.3).
    pa.field('n_det', pa.uint16(), nullable=False),
    # SQL NULL when n_det=0, NEVER NaN: 'nan'::FLOAT >= 0.25 is TRUE in
    # duckdb and max() propagates it -- 29,995,876 vs the correct 1,393,911
    # (§5.4). numpy has no NULL so the writer maps NaN -> NULL explicitly.
    pa.field('max_conf', pa.float32(), nullable=True),
    pa.field('orig_w', pa.uint16(), nullable=False),  # true SOF dims
    pa.field('orig_h', pa.uint16(), nullable=False),
    pa.field('reduce', pa.uint8(), nullable=False),  # 1|2|4|8, audit trail
    # Fixed-width bitmask of fired §8.3 guards, not a comma string.
    pa.field('guards', pa.uint16(), nullable=False),
    # Seconds since run epoch; DELTA_BINARY_PACKED (§5.3).
    pa.field('ts_off', pa.uint32(), nullable=False),
    pa.field('run_id', pa.uint16(), nullable=False),
    pa.field('shard_idx', pa.uint32(), nullable=False),
    # ── provenance ────────────────────────────────────────────────────────
    # Which model produced this box. Eight hex of the engine file's sha256 --
    # the same digest data/best_models.json records as sha256_engine, so a row
    # joins straight to the model registry.
    #
    # A hash, not a filename: engine files get overwritten in place, and a
    # path that means one model today can mean another tomorrow. NULL on every
    # row written before this column existed (see _sql_src on union_by_name),
    # which is honest -- those rows really are unattributable except through
    # the run manifest.
    pa.field('model_sha8', pa.string(), nullable=True),
])

# §5.3 errors: narrow, keeps fat strings out of the 32.5M-row images table.
ERR_SCHEMA = pa.schema([
    pa.field('image_id', pa.uint64(), nullable=False),
    pa.field('status', pa.uint8(), nullable=False),
    pa.field('drive', pa.uint8(), nullable=False),
    pa.field('path', pa.string(), nullable=False),
    pa.field('exc_type', pa.string(), nullable=False),
    pa.field('msg', pa.string(), nullable=False),
    pa.field('ts_off', pa.uint32(), nullable=False),
    pa.field('run_id', pa.uint16(), nullable=False),
])


class StoreError(Exception):
    """Any store-level failure that is a bug or corruption, not a guard."""


class CommitError(StoreError):
    """A shard part failed pre-commit validation; nothing was written."""


class StoreGuardAbort(StoreError):
    """§5.5 hard abort: disk nearly full or detection rows past the ceiling.

    The engine must stop the run when it sees this -- both conditions mean
    the store is about to damage the host volume or has left the design
    envelope by >8x.
    """


class InvariantError(StoreError):
    """One of the §5.4 accounting invariants failed over the store."""


# --------------------------------------------------------------------------
# Paths / config (no hardcoded drive paths in tracked files -- repo rule)
# --------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_detect_root(repo_root=None):
    """DETECT_ROOT from the gitignored ``data/detect_root.txt`` (§5.2).

    Returns:
        Absolute path string.

    Raises:
        StoreError: if the config file is missing.
    """
    root = Path(repo_root) if repo_root else _REPO_ROOT
    cfg = root / 'data' / 'detect_root.txt'
    if not cfg.is_file():
        raise StoreError(f'missing {cfg}; DETECT_ROOT is configured there')
    return cfg.read_text().strip()


def _gen_dir(gen):
    """Normalise a generation to its ``gen=NNNN`` path component."""
    if isinstance(gen, str) and gen.startswith('gen='):
        gen = gen[4:]
    return f'gen={int(gen):04d}'


def pair_dir(detect_root, gen, region, cell, drive):
    """Shard-pair directory ``shards/gen=/region=/cell=/drive=/`` (§5.2)."""
    return os.path.join(detect_root, 'shards', _gen_dir(gen),
                        f'region={region}', f'cell={cell}', f'drive={drive}')


def part_basename(shard_idx, start, end, kind):
    """§5.2 immutable part name ``s00007.p000000_004000.img.parquet``."""
    return f's{shard_idx:05d}.p{start:06d}_{end:06d}.{kind}.parquet'


# --------------------------------------------------------------------------
# Durable commit primitives -- §5.6, exactly
# --------------------------------------------------------------------------


def _fsync_dir(dirpath):
    """fsync the directory so the rename itself is durable (§5.6)."""
    dirfd = os.open(dirpath, os.O_DIRECTORY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def _durable_write_bytes(data, final_path):
    """``.tmp`` write -> flush -> fsync -> rename. No directory fsync here;
    §6.3 fsyncs the dir once per commit, after all renames of the commit.

    rename(2) makes the directory *entry* atomic, not the data durable:
    without the fsyncs a power loss can land metadata while data blocks do
    not, leaving a truncated parquet at the live path -- which poisons the
    entire ``**/*.parquet`` glob [M]. A sibling ``*.parquet.tmp`` of
    identically corrupt content is ignored by the glob [M], which is why
    .tmp + rename is the only safe commit primitive (§5.6). ``f.flush()``
    BEFORE ``os.fsync`` is load-bearing: fsync without it leaves CPython's
    userspace buffer unwritten while f.tell() reports success (§5.1).
    """
    tmp = final_path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final_path)


def _table_bytes(table, schema):
    """Serialise a table to parquet bytes with the §5.3 encoding block.

    Serialising to memory first (parts are <=4,000 images, well under the
    64 GB budget) lets the on-disk write be a single write+flush+fsync,
    matching §5.6 verbatim instead of trusting pyarrow's own file handling
    to flush before we fsync.
    """
    table = table.cast(schema)
    table.validate(full=True)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL,
        # Dictionary off store-wide (§5.3); also required for the DELTA
        # column encodings below.
        use_dictionary=False,
        row_group_size=ROW_GROUP_SIZE,
        column_encoding={
            c: 'DELTA_BINARY_PACKED'
            for c in DELTA_COLS if c in schema.names
        },
    )
    return sink.getvalue().to_pybytes()


def _durable_write_table(table, schema, final_path):
    _durable_write_bytes(_table_bytes(table, schema), final_path)


# --------------------------------------------------------------------------
# _state.json -- a fast index over the committed parts, ALWAYS
# reconstructible by a listdir; committed parts on disk are the truth and
# there is no second resume path (§6.3).
# --------------------------------------------------------------------------


def _scan_parts(dirpath, shard_idx=None):
    """List committed (both-files-present) parts in a pair dir.

    Returns:
        dict shard_idx -> sorted list of dicts
        ``{start, end, img, det}`` (paths). Parts with only one file of the
        pair present are returned under the ``orphans`` key of the second
        element so callers can repair them.
    """
    seen = {}
    try:
        names = os.listdir(dirpath)
    except FileNotFoundError:
        return {}, []
    for name in names:
        m = _PART_RE.match(name)
        if not m:
            continue
        sid = int(m.group('shard'))
        if shard_idx is not None and sid != shard_idx:
            continue
        key = (sid, int(m.group('start')), int(m.group('end')))
        seen.setdefault(key, {})[m.group('kind')] = os.path.join(dirpath, name)
    shards, orphans = {}, []
    for (sid, start, end), files in sorted(seen.items()):
        if 'img' in files and 'det' in files:
            shards.setdefault(sid, []).append({
                'start': start,
                'end': end,
                'img': files['img'],
                'det': files['det'],
            })
        else:
            # A crash between the det and img renames (§6.3 order: det
            # first) leaves a det-only part: not committed, redo the range.
            orphans.extend(files.values())
    return shards, orphans


def _rebuild_state(dirpath, known_lens=None):
    """Rewrite ``_state.json`` from a listdir of the pair dir (§6.3).

    Rebuilding from disk on every commit (a pair dir holds at most a few
    hundred small names) is what keeps the sidecar incapable of disagreeing
    with the truth: entries whose files are gone drop out automatically.
    """
    shards, _ = _scan_parts(dirpath)
    # Preserve shard lens recorded by earlier commits in this dir -- the
    # rebuild is from listdir, which cannot know them. Best effort: the
    # sidecar stays a reconstructible index either way (§6.3).
    lens = {}
    try:
        with open(os.path.join(dirpath, _STATE_NAME)) as f:
            for sid, entry in json.load(f).get('shards', {}).items():
                if 'len' in entry:
                    lens[int(sid)] = entry['len']
    except (FileNotFoundError, ValueError, KeyError):
        pass
    if known_lens:
        lens.update(known_lens)
    state = {
        'shards': {
            str(sid): {
                'parts': [[p['start'], p['end']] for p in parts],
                **({
                    'len': lens[sid]
                } if sid in lens else {}),
            }
            for sid, parts in sorted(shards.items())
        },
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    _durable_write_bytes(
        json.dumps(state, indent=1).encode(),
        os.path.join(dirpath, _STATE_NAME))


# --------------------------------------------------------------------------
# Writer / ShardWriter
# --------------------------------------------------------------------------


class Writer:
    """Process-wide store writer: opens shards, owns the §5.5 disk guards.

    One Writer per engine process. The guard counters are process-global on
    purpose -- the soft alarm is a *global* trailing window, not per-worker
    (§5.5: at lynx's paced rate a per-worker window takes 2.1 h to fill).
    """

    def __init__(self,
                 detect_root=None,
                 max_det=MAX_DET,
                 free_min_bytes=FREE_MIN_BYTES,
                 det_rows_max=DET_ROWS_MAX,
                 soft_window=SOFT_WINDOW_IMAGES,
                 soft_ratio=SOFT_BOXES_PER_IMG,
                 model_sha8=None):
        self.detect_root = detect_root or get_detect_root()
        self.max_det = max_det
        # Stamped onto every row this writer commits. None is allowed and
        # means the caller did not say -- the rows then read NULL, which is
        # accurate rather than a guess, and run_manifest.py show() reports
        # them as unattributable.
        self.model_sha8 = model_sha8
        self.free_min_bytes = free_min_bytes
        self.det_rows_max = det_rows_max
        self.soft_window = soft_window
        self.soft_ratio = soft_ratio
        # Cumulative detection rows committed (seed with the store total on
        # resume via seed_det_rows so the 200M ceiling covers the whole run,
        # not just this process).
        self.total_det_rows = 0
        self.total_img_rows = 0
        # Trailing (n_images, n_boxes) per commit for the soft alarm.
        self._trail = deque()
        self._trail_imgs = 0
        self._trail_boxes = 0
        self.soft_alarm = False

    def seed_det_rows(self, n):
        """Seed the cumulative detection-row counter (resume path)."""
        self.total_det_rows = int(n)

    def open_shard(self, gen, region, cell, drive, shard_idx, start, end):
        """Open a ShardWriter for rows covering ``[start, end)`` of a shard.

        ``start`` > 0 is the continuation of a partial commit (§5.2: the
        continuation gets its own p<start>_<end> file, never the same name).
        """
        if not 0 <= start < end:
            raise ValueError(f'bad shard range [{start}, {end})')
        dirpath = pair_dir(self.detect_root, gen, region, cell, drive)
        os.makedirs(dirpath, exist_ok=True)
        return ShardWriter(self, dirpath, shard_idx, start, end)

    # -- §5.5 guards ------------------------------------------------------

    def _check_hard_guards(self, n_new_det):
        """Hard aborts, checked every shard commit (§5.5), BEFORE writing."""
        free = shutil.disk_usage(self.detect_root).free
        if free < self.free_min_bytes:
            raise StoreGuardAbort(
                f'DETECT_ROOT free space {free / 2**30:.1f} GiB < '
                f'{self.free_min_bytes / 2**30:.0f} GiB hard floor (§5.5)')
        if self.total_det_rows + n_new_det > self.det_rows_max:
            raise StoreGuardAbort(
                f'cumulative detection rows {self.total_det_rows} + '
                f'{n_new_det} would exceed the {self.det_rows_max} hard '
                f'ceiling (§5.5)')

    def _account_commit(self, n_img, n_det):
        """Update counters + trailing-window soft alarm after a commit."""
        self.total_img_rows += n_img
        self.total_det_rows += n_det
        self._trail.append((n_img, n_det))
        self._trail_imgs += n_img
        self._trail_boxes += n_det
        # Trim so the window holds the most recent >= soft_window images
        # (never trims below the window size).
        while self._trail and (self._trail_imgs -
                               self._trail[0][0]) >= self.soft_window:
            i, b = self._trail.popleft()
            self._trail_imgs -= i
            self._trail_boxes -= b
        if self._trail_imgs >= self.soft_window:
            ratio = self._trail_boxes / self._trail_imgs
            alarm = ratio > self.soft_ratio
            if alarm and not self.soft_alarm:
                log.warning(
                    'SOFT ALARM (§5.5): trailing %d-image boxes/img %.3f > '
                    '%.2f', self._trail_imgs, ratio, self.soft_ratio)
            self.soft_alarm = alarm

    # -- bootstrap + errors ----------------------------------------------

    def ensure_bootstrap(self):
        """Write the 0-row full-schema ``_bootstrap`` partition (§5.2).

        Not cosmetic: read_parquet over an existing-but-empty tree raises
        ``IOException: No files found that match the pattern`` [M], so
        without this every query, the dashboard rollup and every smoke test
        fail until the first shard commits.
        """
        return ensure_bootstrap(self.detect_root)

    def write_errors(self, gen, region, cell, drive, shard_idx, rows):
        """Durably write the §5.3 errors table for one shard.

        Layout (§5.2): ``errors/gen=/region=/cell=/drive=/e<shard>.parquet``.
        A shard can commit in several parts, so an existing file is merged
        (read + concat + atomic replace) rather than clobbered -- the
        replace is still a §5.6 tmp+fsync+rename, so a crash leaves either
        the old or the new file, never a torn one.
        """
        dirpath = os.path.join(self.detect_root, 'errors', _gen_dir(gen),
                               f'region={region}', f'cell={cell}',
                               f'drive={drive}')
        os.makedirs(dirpath, exist_ok=True)
        final = os.path.join(dirpath, f'e{shard_idx:05d}.parquet')
        new = pa.Table.from_pylist(list(rows), schema=ERR_SCHEMA)
        if os.path.exists(final):
            new = pa.concat_tables(
                [pq.read_table(final, schema=ERR_SCHEMA), new])
        new = new.sort_by('image_id')
        _durable_write_table(new, ERR_SCHEMA, final)
        _fsync_dir(dirpath)
        return final


class ShardWriter:
    """Buffers one shard part's rows and commits them durably (§5.6, §6.3).

    Usage by the engine: ``add_image``/``add_detections`` as GPU results
    arrive (order does not matter; rows are sorted by image_id at commit,
    §5.3), then ``commit(part_end)`` when the shard's outstanding counter
    hits 0 -- or at graceful shutdown with the exact ``[start, part_end)``
    actually covered (§6.3). After a partial commit the same ShardWriter can
    keep buffering; the next commit becomes the continuation part.
    """

    def __init__(self, writer, dirpath, shard_idx, start, end):
        self._writer = writer
        self.dirpath = dirpath
        self.shard_idx = shard_idx
        self.start = start  # immutable: opening range
        self.end = end
        self.cur_start = start  # advances past each committed part
        self._img_rows = []
        self._det_rows = []
        self._img_ids = set()

    # -- row intake -------------------------------------------------------

    def add_image(self, row):
        """Buffer one images row (dict of §5.3 columns).

        Enforces at intake, where the bug would be introduced:
          * status / n_det present and never None (§5.3);
          * NaN -> NULL on max_conf, and NULL iff n_det==0 (§5.4);
          * writer-owned columns: shard_idx stamped, GUARD_NDET_MAXED
            flagged when n_det == max_det (§5.5).
        """
        row = dict(row)
        if row.get('status') is None:
            raise CommitError(f'status is None for image {row.get("image_id")}'
                              ' -- §5.3 forbids NULL status')
        n_det = row.get('n_det')
        if n_det is None:
            raise CommitError(f'n_det is None for image {row.get("image_id")}'
                              ' -- §5.3 forbids NULL n_det')
        if not 0 <= int(n_det) <= 65535:
            raise CommitError(f'n_det={n_det} out of UINT16 range')
        # §5.4: numpy has no NULL, so the NaN negative-sentinel is mapped to
        # SQL NULL here, explicitly. NaN with n_det>0 is a corrupt row, not
        # something to paper over.
        mc = row.get('max_conf')
        if mc is not None and math.isnan(mc):
            mc = None
        if int(n_det) == 0:
            if mc is not None:
                raise CommitError(
                    f'image {row.get("image_id")}: max_conf={mc} with '
                    f'n_det=0 -- must be NULL (§5.4)')
        elif mc is None or not math.isfinite(mc):
            raise CommitError(
                f'image {row.get("image_id")}: n_det={n_det} but max_conf '
                f'is {mc!r} -- must be finite (§5.4)')
        row['max_conf'] = mc
        row['guards'] = int(row.get('guards', 0))
        if int(n_det) == self._writer.max_det:
            row['guards'] |= GUARD_NDET_MAXED  # §5.5 flag
        row['shard_idx'] = self.shard_idx
        row.setdefault('model_sha8', self._writer.model_sha8)
        iid = int(row['image_id'])
        if iid in self._img_ids:
            raise CommitError(f'duplicate image_id {iid} in shard part')
        self._img_ids.add(iid)
        self._img_rows.append(row)

    def add_detections(self, rows):
        """Buffer detections rows (dicts of §5.3 det columns).

        Enforces det_idx in 0..255 at intake: §5.3 pins det_idx UINT8 while
        §4.6 pins max_det=300, so boxes 256..299 of an NMS-maxed image
        CANNOT be stored under the pinned schema. Failing here is loud and
        attributable to one image; without it the whole shard part dies at
        commit with an unattributed ArrowInvalid and can never commit.
        """
        for row in rows:
            row = dict(row)
            di = row.get('det_idx')
            if di is None or not 0 <= int(di) <= 255:
                raise CommitError(
                    f'det_idx={di!r} for image {row.get("image_id")} does '
                    f'not fit UINT8 (§5.3) -- the §4.6 max_det=300 config '
                    f'exceeds the det_idx ceiling; cap boxes/image at 256 '
                    f'or amend the spec')
            row['shard_idx'] = self.shard_idx
            row.setdefault('model_sha8', self._writer.model_sha8)
            self._det_rows.append(row)

    # -- commit -----------------------------------------------------------

    def commit(self, part_end):
        """Durably commit the buffered rows as part ``[cur_start, part_end)``.

        §6.3 order, exactly: (1) det .tmp -> flush -> fsync -> replace;
        (2) img likewise; (3) _state.json likewise; (4) fsync the dir fd.
        Crash between (1) and (2) leaves a det-only orphan that
        ``tiling_resume`` treats as uncommitted and deletes.

        Returns:
            (img_path, det_path) of the committed part.
        """
        if not self.cur_start < part_end <= self.end:
            raise CommitError(
                f'part_end {part_end} outside ({self.cur_start}, {self.end}]')
        n_expected = part_end - self.cur_start
        if len(self._img_rows) != n_expected:
            raise CommitError(
                f'part [{self.cur_start},{part_end}) declares {n_expected} '
                f'images but {len(self._img_rows)} are buffered -- the part '
                f'range is positional truth (§6.1) and must match exactly')
        # det <-> img accounting must hold WITHIN the part being committed;
        # this is the local version of the three global §5.4 invariants and
        # the only place a mismatch is still attributable to one shard.
        det_counts = Counter(int(r['image_id']) for r in self._det_rows)
        if not set(det_counts) <= self._img_ids:
            stray = sorted(set(det_counts) - self._img_ids)[:5]
            raise CommitError(f'detections for images not in part: {stray}')
        for row in self._img_rows:
            n = det_counts.get(int(row['image_id']), 0)
            if n != int(row['n_det']):
                raise CommitError(
                    f'image {row["image_id"]}: n_det={row["n_det"]} but '
                    f'{n} detection rows buffered')

        # §5.5 hard guards, every commit, before any bytes hit disk.
        self._writer._check_hard_guards(len(self._det_rows))

        # §5.3: rows sorted by image_id within each part.
        img_rows = sorted(self._img_rows, key=lambda r: int(r['image_id']))
        det_rows = sorted(self._det_rows,
                          key=lambda r:
                          (int(r['image_id']), int(r['det_idx'])))
        img_tbl = pa.Table.from_pylist(img_rows, schema=IMG_SCHEMA)
        det_tbl = pa.Table.from_pylist(det_rows, schema=DET_SCHEMA)

        det_path = os.path.join(
            self.dirpath,
            part_basename(self.shard_idx, self.cur_start, part_end, 'det'))
        img_path = os.path.join(
            self.dirpath,
            part_basename(self.shard_idx, self.cur_start, part_end, 'img'))
        # Immutability check (§5.2: every file written once, never mutated).
        for p in (det_path, img_path):
            if os.path.exists(p):
                raise CommitError(f'refusing to overwrite committed part {p}')

        _durable_write_table(det_tbl, DET_SCHEMA, det_path)  # (1)
        _durable_write_table(img_tbl, IMG_SCHEMA, img_path)  # (2)
        _rebuild_state(
            self.dirpath,  # (3)
            known_lens={self.shard_idx: self.end})
        _fsync_dir(self.dirpath)  # (4)

        self._writer._account_commit(len(img_rows), len(det_rows))
        # Advance: a later commit from this writer is the continuation part.
        self.cur_start = part_end
        self._img_rows, self._det_rows, self._img_ids = [], [], set()
        return img_path, det_path


# --------------------------------------------------------------------------
# Bootstrap partition -- §5.2
# --------------------------------------------------------------------------


def ensure_bootstrap(detect_root):
    """Write the 0-row, full-schema ``_bootstrap`` partition if absent.

    Path (§5.2): ``_bootstrap/region=_bootstrap/cell=_bootstrap/
    drive=_bootstrap/{img,det}.parquet``. Idempotent; files are written with
    the same §5.6 durable sequence as real parts.
    """
    # gen= level included so the hive schema matches the real shards glob --
    # with hive_partitioning=1 a depth mismatch is a BinderException.
    dirpath = os.path.join(detect_root, '_bootstrap', 'gen=_bootstrap',
                           'region=_bootstrap', 'cell=_bootstrap',
                           'drive=_bootstrap')
    os.makedirs(dirpath, exist_ok=True)
    wrote = False
    for name, schema in (('img.parquet', IMG_SCHEMA), ('det.parquet',
                                                       DET_SCHEMA)):
        final = os.path.join(dirpath, name)
        if os.path.exists(final):
            continue
        empty = pa.Table.from_pylist([], schema=schema)
        _durable_write_table(empty, schema, final)
        wrote = True
    if wrote:
        _fsync_dir(dirpath)
    return dirpath


# --------------------------------------------------------------------------
# Resume tiling -- §6.3
# --------------------------------------------------------------------------


def tiling_resume(shard_dir, shard_len, shard_idx=None, repair=True):
    """Reconcile one shard's committed parts from disk (§6.3).

    Committed parts on disk are the truth; this is the ONE resume algorithm.
    Per part, in start order:
      * both img and det present, img footer ``num_rows`` equals the
        declared ``[start,end)`` length, det footer readable -> adopt;
      * short, unreadable, out-of-range or overlapping part -> delete the
        pair and redo that range (<= 4,000 images);
      * a lone .tmp or a single file of a pair is never adopted (crash
        window between the two renames) and is deleted under ``repair``.

    Args:
        shard_dir: the pair directory (``.../cell=C/drive=D``).
        shard_len: frozen length of this shard from the worklist.
        shard_idx: which shard to reconcile; may be omitted only when the
            directory holds parts of a single shard.
        repair: actually delete invalid parts (§6.3 "delete it and redo").

    Returns:
        (parts, done): sorted list of adopted ``(start, end)`` tuples and
        whether they tile ``[0, shard_len)`` exactly -- no gaps, no
        overlaps (§5.2).
    """
    shards, orphans = _scan_parts(shard_dir, shard_idx=shard_idx)
    if shard_idx is None:
        if len(shards) > 1:
            raise ValueError(
                f'{shard_dir} holds parts of shards {sorted(shards)}; pass '
                f'shard_idx')
        shard_idx = next(iter(shards), None)
    candidates = shards.get(shard_idx, [])

    def _drop(paths, why):
        log.warning('tiling_resume: dropping %s (%s)', paths, why)
        if repair:
            for p in paths:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

    # Half-committed pairs are never adoptable regardless of shard.
    if orphans:
        _drop(orphans, 'single file of an img/det pair')

    adopted = []
    prev_end = 0
    for part in sorted(candidates, key=lambda p: (p['start'], p['end'])):
        start, end = part['start'], part['end']
        pair = (part['img'], part['det'])
        if not 0 <= start < end <= shard_len:
            _drop(pair, f'range [{start},{end}) outside [0,{shard_len})')
            continue
        if start < prev_end:
            # Overlap with an already-adopted part: the earlier adoption
            # stands (it was committed first); the overlapper is redone.
            _drop(pair, f'overlaps committed coverage ending at {prev_end}')
            continue
        try:
            # Footer num_rows must equal the declared range length -- this
            # is what stops "both files exist and read cleanly -> adopt"
            # from blessing a 1,536-row file as a 4,000-image shard (§5.2).
            n = pq.ParquetFile(part['img']).metadata.num_rows
            if n != end - start:
                _drop(pair, f'img footer num_rows {n} != {end - start}')
                continue
            pq.ParquetFile(part['det']).metadata  # readable footer
        except Exception as exc:  # torn/truncated/garbage parquet
            _drop(pair, f'unreadable footer: {exc}')
            continue
        adopted.append((start, end))
        prev_end = end

    # Stale .tmp files from a crash mid-write: harmless to the glob [M] but
    # removed under repair so they cannot accumulate.
    if repair:
        for name in os.listdir(shard_dir) if os.path.isdir(shard_dir) else ():
            if name.endswith('.tmp') and _PART_RE.match(name[:-4]):
                try:
                    os.remove(os.path.join(shard_dir, name))
                except FileNotFoundError:
                    pass

    done = (bool(adopted) and adopted[0][0] == 0
            and adopted[-1][1] == shard_len
            and all(a[1] == b[0] for a, b in zip(adopted, adopted[1:])))
    return adopted, done


# --------------------------------------------------------------------------
# Store-wide queries: invariants + verify -- §5.4, §5.6
# --------------------------------------------------------------------------


def _store_globs(detect_root, kind):
    """duckdb glob list for one table, .tmp-safe by construction.

    ``*.{kind}.parquet`` cannot match ``*.tmp`` names, and the bootstrap
    files (named plain ``img.parquet``/``det.parquet``, §5.2) need their own
    pattern because ``*`` cannot swallow the leading dot. The shards glob is
    only included when it matches something -- duckdb raises on a pattern
    with zero matches, and a fresh store legitimately has only _bootstrap.
    """
    globs = []
    shard_glob = os.path.join(detect_root, 'shards', '**', f'*.{kind}.parquet')
    if glob.glob(shard_glob, recursive=True):
        globs.append(shard_glob)
    boot = os.path.join(detect_root, '_bootstrap', '**', f'{kind}.parquet')
    if not glob.glob(boot, recursive=True):
        raise StoreError(
            f'no _bootstrap {kind} partition under {detect_root} -- run '
            f'ensure_bootstrap() first (§5.2)')
    globs.append(boot)
    return globs


def _sql_src(globs):
    paths = ', '.join("'" + g.replace("'", "''") + "'" for g in globs)
    # hive_partitioning exposes gen/region/cell/drive from the path -- the
    # invariants key on (image_id, cell, drive) because cell twins (the same
    # image stored under more than one cell) are corpus-legitimate. See
    # unique_src() before computing any PER-IMAGE statistic off this.
    # union_by_name is NOT optional. Without it duckdb takes the FIRST file's
    # schema and silently drops any column the later files added: selecting it
    # then fails with "Referenced column not found" even though the data is
    # right there on disk. That makes every schema addition a breaking change
    # for readers, which is how a provenance column would have been added and
    # then quietly lost. With it, files written before a column existed read
    # back NULL for it and filters work across the whole store.
    return (f'read_parquet([{paths}], hive_partitioning=1, '
            f'union_by_name=1)')


def unique_src(detect_root=None, kind='img'):
    """SQL source with exactly ONE row per image_id -- cell twins collapsed.

    The harvest wrote some images into several cells, so the worklist (which
    dedups per cell across drives, never ACROSS cells) hands the same jpg to
    the detector once per cell it landed in, and the store keeps one row per
    (image_id, cell, drive).

    Measured on the live sweep at ~2% progress: 10,254 images duplicated
    across 2-6 cells each, ~1.5% of rows. Three properties were checked
    before writing this helper, and they are what make collapsing safe:

      * the repeated passes are bit-identical -- same n_det, same orig_w/h,
        and identical box geometry for every image that had detections, so
        picking any one row loses nothing;
      * every duplicated image stays inside a SINGLE region, so region
        attribution is unaffected by which row survives;
      * the twins are NOT adjacent-cell neighbours (observed spanning 20 deg
        of longitude), so this is harvest-side cell attribution, not a
        boundary rounding effect -- do not "fix" it by nudging cell bounds.

    Use this for any per-image count, rate or join. Use _sql_src() when you
    genuinely want per-(image, cell) rows, e.g. drive/cell throughput.
    """
    detect_root = detect_root or get_detect_root()
    src = _sql_src(_store_globs(detect_root, kind))
    # deterministic survivor so repeated runs agree: lowest (cell, drive)
    return (f'(SELECT * FROM {src} QUALIFY row_number() OVER '
            f'(PARTITION BY image_id ORDER BY cell, drive) = 1)')


# Helper program for environments without an importable duckdb (dnd has
# pyarrow but no duckdb; mp14 has duckdb). Reads {"queries": {name: sql}}
# on stdin, emits {name: rows} JSON on stdout.
_DUCKDB_PROG = ('import json,sys\n'
                'import duckdb\n'
                'spec = json.load(sys.stdin)\n'
                'con = duckdb.connect()\n'
                'out = {n: [list(r) for r in con.execute(q).fetchall()]\n'
                '       for n, q in spec["queries"].items()}\n'
                'print(json.dumps(out))\n')


def _duckdb_helper_python(repo_root=None):
    """Interpreter with duckdb, from env or gitignored config -- never a
    hardcoded env path in this tracked file (repo convention)."""
    p = os.environ.get('DETECT_DUCKDB_PYTHON')
    if p:
        return p
    cfg = (Path(repo_root)
           if repo_root else _REPO_ROOT) / 'data' / 'duckdb_python.txt'
    if cfg.is_file():
        return cfg.read_text().strip()
    return None


def _run_queries(queries):
    """Run named SQL queries via in-process duckdb or the helper python."""
    try:
        import duckdb
    except ImportError:
        duckdb = None
    if duckdb is not None:
        con = duckdb.connect()  # no path -- §5.1, never a .duckdb file
        try:
            return {n: con.execute(q).fetchall() for n, q in queries.items()}
        finally:
            con.close()
    helper = _duckdb_helper_python()
    if not helper:
        raise StoreError(
            'duckdb not importable and no helper interpreter configured; '
            'set $DETECT_DUCKDB_PYTHON or data/duckdb_python.txt')
    proc = subprocess.run([helper, '-c', _DUCKDB_PROG],
                          input=json.dumps({'queries': queries}),
                          capture_output=True,
                          text=True)
    if proc.returncode != 0:
        raise StoreError(f'duckdb helper failed: {proc.stderr.strip()}')
    return {
        n: [tuple(r) for r in rows]
        for n, rows in json.loads(proc.stdout).items()
    }


def invariants(detect_root=None):
    """§5.4: canonical progress query + the accounting assertions.

    Runs over the .tmp-skipping parquet glob (committed files only). This
    is the only detector for double-processing -- the single biggest
    silent-waste risk in the project -- and costs ~0.5 s over the full
    store [M].

    Returns:
        dict(scanned, positive, negative, errored, boxes).

    Raises:
        InvariantError: any assertion non-zero, or the progress identity
            positive + negative + errored != scanned.
    """
    detect_root = detect_root or get_detect_root()
    img = _sql_src(_store_globs(detect_root, 'img'))
    det = _sql_src(_store_globs(detect_root, 'det'))
    queries = {
        # Canonical progress query, double-count fixed: error rows have
        # n_det=0, so the naive sum(n_det=0) counts every error as a
        # negative (§5.4).
        'progress':
        f'''
            SELECT count(*)                                AS scanned,
                   sum(status=0 AND n_det>0)::BIGINT       AS positive,
                   sum(status=0 AND n_det=0)::BIGINT       AS negative,
                   sum(status<>0)::BIGINT                  AS errored,
                   sum(n_det)::BIGINT                      AS boxes
            FROM {img} img''',
        # The three per-commit / hourly assertions (§5.4). Keyed on
        # (image_id, cell, drive), not image_id alone: ~0.05% of the corpus
        # is the SAME image legitimately stored in two ADJACENT cells (cell-
        # boundary twins, e.g. Australia_110_-35 / Australia_115_-35) and
        # both copies are processed by design. A duplicate within one
        # (cell, drive) partition is the real double-processing signal.
        'dup_image_ids':
        f'''
            SELECT count(*) - count(DISTINCT (image_id, cell, drive))
            FROM {img} img''',
        'cross_cell_twins':
        f'''
            SELECT count(*) - count(DISTINCT image_id) FROM {img} img''',
        'orphan_dets':
        f'''
            SELECT count(*) FROM {det} d ANTI JOIN {img} i
            USING (image_id, cell, drive)''',
        'n_det_mismatch':
        f'''
            SELECT count(*) FROM {img} img JOIN
              (SELECT image_id, cell, drive, count(*) n
               FROM {det} det GROUP BY 1, 2, 3)
            USING (image_id, cell, drive) WHERE img.n_det <> n''',
        # §5.4 unit test: the NULL-not-NaN mapping is observable.
        'nan_null_mapping':
        f'''
            SELECT (SELECT count(*) FROM {img} img
                    WHERE max_conf IS NULL AND status = 0)
                 - (SELECT count(*) FROM {img} img
                    WHERE n_det = 0 AND status = 0)''',
    }
    res = _run_queries(queries)
    scanned, positive, negative, errored, boxes = res['progress'][0]
    scanned = int(scanned or 0)
    positive, negative = int(positive or 0), int(negative or 0)
    errored, boxes = int(errored or 0), int(boxes or 0)
    failures = []
    if positive + negative + errored != scanned:
        failures.append(
            f'progress identity: {positive}+{negative}+{errored} != '
            f'{scanned}')
    # cross_cell_twins is reported but never a failure (corpus property).
    for name in ('dup_image_ids', 'orphan_dets', 'n_det_mismatch',
                 'nan_null_mapping'):
        v = int(res[name][0][0] or 0)
        if v != 0:
            failures.append(f'{name} = {v} (must be 0)')
    if failures:
        raise InvariantError('; '.join(failures))
    return {
        'scanned': scanned,
        'positive': positive,
        'negative': negative,
        'errored': errored,
        'boxes': boxes,
    }


def verify(detect_root=None):
    """Read every committed parquet footer under the store (§5.6).

    A parquet truncated by even 200 bytes is unreadable AND poisons the
    whole read_parquet glob [M]; this scan (~30 s over 18k files [E]) finds
    it proactively instead of via a failing dashboard query on day 3.
    ``*.tmp`` files are skipped -- the glob ignores them too [M].

    Returns:
        (n_ok, bad): count of clean files and a list of (path, error).
    """
    detect_root = detect_root or get_detect_root()
    n_ok, bad = 0, []
    for dirpath, _dirnames, filenames in os.walk(detect_root):
        for name in filenames:
            if not name.endswith('.parquet'):  # skips *.tmp by suffix
                continue
            path = os.path.join(dirpath, name)
            try:
                md = pq.ParquetFile(path).metadata
                # Touch the footer fields so a lazily-parsed footer cannot
                # defer the failure past this scan.
                md.num_rows, md.num_row_groups
                n_ok += 1
            except Exception as exc:
                bad.append((path, f'{type(exc).__name__}: {exc}'))
    return n_ok, bad


# --------------------------------------------------------------------------
# Post-run compaction -- §5.7
# --------------------------------------------------------------------------


def compact(gen, cell, detect_root=None):
    """Rewrite one finished cell into a single img + det pair (§5.7).

    18k part files are fine for the run but not for analytics. The rewrite
    uses the same .tmp + fsync + rename commit and leaves the parts in
    place until the compacted files verify (footer read + row-count
    equality against the sum of the parts); only then are the parts and
    their ``_state.json`` sidecars removed. Run this AFTER the run
    completes for the cell -- shard-level resume is over by then, and the
    caller (sweep.py) is responsible for the cell-is-done check against the
    frozen worklist.

    Idempotent across crashes: rerunning after a crash between the rename
    and the part deletion -- including one MID-deletion, when only some
    parts survive -- verifies the surviving parts are contained in the
    compacted pair and finishes the deletion (the brief both-present window
    double-counts in queries, which is why compaction is post-run only).

    Returns:
        (img_path, det_path) of the compacted pair, or None if the cell has
        no parts (already compacted).
    """
    detect_root = detect_root or get_detect_root()
    cell_dirs = glob.glob(
        os.path.join(detect_root, 'shards', _gen_dir(gen), 'region=*',
                     f'cell={cell}'))
    if len(cell_dirs) != 1:
        raise StoreError(
            f'expected exactly one cell dir for {cell}, got {cell_dirs}')
    cell_dir = cell_dirs[0]
    # Full hive depth (region=/cell=/drive=_merged): a file one level short
    # makes every read_parquet(..., hive_partitioning=1) over the store raise
    # a Binder 'Hive partition mismatch'. The physical uint8 ``drive`` column
    # inside the rows is untouched (duckdb lets the file column win); the
    # _merged path key is only there to keep the tree depth uniform.
    merged_dir = os.path.join(cell_dir, 'drive=_merged')
    os.makedirs(merged_dir, exist_ok=True)
    out_img = os.path.join(merged_dir, 'compact.img.parquet')
    out_det = os.path.join(merged_dir, 'compact.det.parquet')

    # Gather committed parts across the per-drive dirs. Half-pairs are a
    # crash artifact and must not survive into a compacted file.
    part_pairs, drive_dirs = [], []
    for drive_dir in sorted(glob.glob(os.path.join(cell_dir, 'drive=*'))):
        drive_dirs.append(drive_dir)
        shards, orphans = _scan_parts(drive_dir)
        if orphans:
            raise StoreError(f'half-committed parts in {drive_dir}: '
                             f'{orphans}; run tiling_resume first')
        for parts in shards.values():
            part_pairs.extend(parts)
    if not part_pairs:
        if os.path.exists(out_img) and os.path.exists(out_det):
            # Already compacted; finish any sidecar cleanup a crash between
            # the part deletions and the _state.json removals left behind.
            for drive_dir in drive_dirs:
                state = os.path.join(drive_dir, _STATE_NAME)
                if os.path.exists(state):
                    os.remove(state)
                    _fsync_dir(drive_dir)
            return out_img, out_det
        raise StoreError(f'no parts to compact under {cell_dir}')

    def _read_all(kind, schema):
        tables = [pq.read_table(p[kind], schema=schema) for p in part_pairs]
        return pa.concat_tables(tables)

    img_tbl = _read_all('img', IMG_SCHEMA).sort_by('image_id')
    det_tbl = _read_all('det', DET_SCHEMA).sort_by([('image_id', 'ascending'),
                                                    ('det_idx', 'ascending')])

    # A both-present compacted pair means a previous compact() got past its
    # §5.6 renames and crashed somewhere in the deletion loop below -- the
    # surviving parts are then a SUBSET of the compacted data, not its
    # equal, so the two cases verify differently.
    recovery = os.path.exists(out_img) and os.path.exists(out_det)
    if not recovery:
        _durable_write_table(det_tbl, DET_SCHEMA, out_det)
        _durable_write_table(img_tbl, IMG_SCHEMA, out_img)
        _fsync_dir(cell_dir)

    # Verify before touching the parts (§5.7): fresh footer reads of the
    # files at their FINAL paths.
    comp_img_n = pq.ParquetFile(out_img).metadata.num_rows
    comp_det_n = pq.ParquetFile(out_det).metadata.num_rows
    if not recovery:
        # Fresh write: row counts must equal the sum of the parts exactly.
        for path, n, expect in ((out_img, comp_img_n, img_tbl.num_rows),
                                (out_det, comp_det_n, det_tbl.num_rows)):
            if n != expect:
                raise StoreError(
                    f'compacted {path} has {n} rows, expected {expect}; '
                    f'parts left in place')
    else:
        # Crash recovery: verify containment by image_id, not by count --
        # every surviving part row must already be in the compacted pair.
        # Anything NOT contained means new parts landed after compaction
        # (the compacted pair is stale); refuse and leave everything.
        comp_ids = set(
            pq.read_table(out_img,
                          columns=['image_id']).column('image_id').to_pylist())
        part_ids = set(img_tbl.column('image_id').to_pylist())
        if not (part_ids <= comp_ids and comp_img_n >= img_tbl.num_rows
                and comp_det_n >= det_tbl.num_rows):
            raise StoreError(
                f'compacted pair in {cell_dir} does not contain the '
                f'surviving parts ({len(part_ids - comp_ids)} unknown '
                f'image_ids); refusing to delete them')

    # Only now drop the parts + sidecars; fsync each dir so the deletions
    # are durable before anyone trusts the compacted view.
    for p in part_pairs:
        os.remove(p['img'])
        os.remove(p['det'])
    for drive_dir in drive_dirs:
        state = os.path.join(drive_dir, _STATE_NAME)
        if os.path.exists(state):
            os.remove(state)
        _fsync_dir(drive_dir)
    return out_img, out_det
