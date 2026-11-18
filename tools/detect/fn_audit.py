#!/usr/bin/env python3
"""Measure how often the dog-bin gate is right, across its whole score range.

    python tools/detect/fn_audit.py build --stage gate --gen 2 --tmp DIR
                                    # once per generation: the candidate pool
    python tools/detect/fn_audit.py stats [--gen N]  # what the verdicts say
    python tools/detect/fn_audit.py export --gen N   # the dataset, re-filed
    python tools/detect/fn_audit.py verify --gen N   # the pool is its record

WHY. The gate judged 4,688,510 boxes and its accuracy was measured on a few
hundred held-out crops before it ran. This measures it on the thing it
actually did, at the scale it did it, in BOTH directions:

    a dog it rejected   is gone -- nothing downstream will ever see it
    a not-dog it kept   is work, and noise in whatever is trained next

Both are found the same way: show a person the crop, ask "is this a dog", and
compare the answer to what the model said. So the pool is every box the gate
judged, not only the ones it threw away -- bands under 0.5 are its rejections
and bands over 0.5 are its acceptances, and the same question is asked of all
of them.

WHY THE SAMPLE IS STRATIFIED. Three quarters of the boxes score under 0.1, so
a uniform sample is mostly obvious-not-a-dog and tells you almost nothing
about where the model's edge is. The pool is banded by p_dog and drawn from
evenly; each band's rate is reported on its own, and the headline weights
those bands by how many boxes each really holds. That is the difference
between "we looked at 500 crops" and a number.

AND WHAT IS NOT PART OF THAT SAMPLE. The sheet can also be drawn at one END
of a band -- the dashboard's 'least confident' and 'most confident' walks --
which is a good way to FIND errors and not a way to measure them: those crops
are chosen on the very property being estimated, and every rate here is
multiplied by its band's whole population. Each page document records how it
was drawn, so summarise() holds those answers out of the rates and reports
them separately as `aimed`. The finds are kept; only the arithmetic refuses
them.

AND WHY IT IS STILL A CEILING. Which crops are DRAWN is even; which of them
get answered is not. A sheet of twenty-five can be scrolled past as easily as
it can be clicked, and the one worth stopping on is the one that already looks
like the answer -- measured on the gate's own store, 63 of 1,175 crops shown
came back, band by band between 1.7% and 10.2%. So every share here is an
upper bound until a sheet is answered in full, and the answer rate is printed
beside each band so the bound can be read for what it is.

WHY SEQUENCES. Mapillary frames come in sequences -- one car, one road, one
second apart. Twenty crops from one sequence are twenty photographs of the
same dog from twenty metres, and scoring them as twenty independent samples
would state a confidence the data does not have. One box per sequence, and a
sequence is never drawn twice.

WHAT THIS WRITES. A pool of candidates under data/fn_audit_gen<N>/ (gate) or
data/leash_audit_gen<N>/ (leash), one directory per DETECTOR GENERATION, and
nothing else. The human verdicts it collects live in their own ledger beside
that pool, and are the only thing here that is not a model's opinion.

WHY ONE DIRECTORY PER GENERATION. A key is '<image_id>#<det_idx>' and det_idx
is numbered per detector generation: gen 2's box 0 of a frame is not gen 1's
box 0 -- of 35,876 gen-1 gate verdict keys, 27,156 exist in the gen-2 store
and only 7,748 of those are the same box. So a gen-1 verdict must never be
read against a gen-2 pool, crop or export, and the other way round. Gen 1
keeps its legacy names (data/fn_audit, data/leash_audit, data/audit_finds*)
byte for byte -- scripts outside this repo read them -- and is FROZEN: no
build, no export, no new page. Every later generation gets `_gen<N>` on each
name (audit_name), a pool.json that names the pool's identity (pool_meta),
and verdict rows that carry that identity (read_verdicts refuses any that do
not). Nothing here falls back from one generation to another.
"""

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import sys
import time
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One row per model this can audit. The two are the same exercise -- show a
# person the crop the model saw, ask the question the model was asked, compare
# -- so they are one code path with a table, not two files that drift.
#
# `asymmetric` is the difference that matters to how results are read. A dog
# the gate throws away is unrecoverable, so its two errors are not worth the
# same and the headline is about misses. The leash model was promoted on
# BALANCED accuracy precisely because its two errors DO cost the same, so its
# headline reports both directions side by side and neither is "the" error.
STAGES = {
    'gate': {
        'dir': 'gate', 'p_col': 'p_dog',
        'answers': ('dog', 'not_dog', 'unsure'),
        'positive': 'dog', 'negative': 'not_dog',
        'audit_dir': 'fn_audit', 'dataset': 'audit_finds',
        # 'model' is the GEN-1 pool's model, which its pool.json never
        # recorded. A pool of any later generation records the model its
        # store was judged by (pool_meta()['model']) and that is the one
        # every export and page names.
        'title': 'dog-bin gate', 'model': 'dogbin_008',
        'asks': 'Is this a dog?',
        'yes': 'it\u2019s a dog', 'no': 'not a dog',
        'asymmetric': True,
        # what the score below the threshold means, in one phrase
        'below': 'what it rejected', 'above': 'what it kept',
        'miss': 'dogs it threw away',
        'legacy': {'missed': 'dog', 'correct': 'not_dog'},
    },
    'leash': {
        'dir': 'leash', 'p_col': 'p_leashed',
        'answers': ('leashed', 'unleashed', 'unsure'),
        'positive': 'leashed', 'negative': 'unleashed',
        'audit_dir': 'leash_audit', 'dataset': 'audit_finds_leash',
        'title': 'leash model', 'model': 'leash_v2_001',
        'asks': 'Is this dog on a leash?',
        # the two words the model itself uses, so the button, the store and
        # every count on the page say the same thing
        'yes': 'leashed', 'no': 'unleashed',
        'asymmetric': False,
        'below': 'called it unleashed', 'above': 'called it leashed',
        'miss': 'leashed dogs called unleashed',
        'legacy': {},
    },
}
DEFAULT_STAGE = 'gate'


def spec(stage=DEFAULT_STAGE):
    if stage not in STAGES:
        raise KeyError(f'unknown audit stage {stage!r}')
    return STAGES[stage]


class GenError(Exception):
    """A record, plan or store of another detector generation -- or of none
    that can be told -- where exactly one generation was required.

    Raised, never skipped: an audit ledger holds the rows of one pool only,
    so a foreign row in one is damage to look at, not noise to filter."""


class PoolIdentityError(GenError):
    """An audit pool that cannot prove which pool it is: no pool.json, a
    pool.json that names another stage or generation, or a pool.parquet that
    is not the file pool.json describes (a build that stopped between its
    two renames leaves exactly that). Such a pool is NOT READY -- never gen 1,
    never "probably fine"."""


def _want_gen(gen):
    """`gen` as a generation number >= 1; ValueError for anything else.

    No default, on purpose. Before generations had their own directories the
    one audit dir per stage WAS gen 1, and a caller that forgets the argument
    after the switch would read or write the wrong generation without an
    error. So the forgotten argument is a TypeError (it is keyword-only and
    required everywhere) and a wrong one -- None, '2', True, 0 -- is this."""
    if gen is None or gen_of(gen) is None or gen < 1:
        raise ValueError(f'not a detector generation: {gen!r} (an int >= 1 '
                         f'is required, and there is no default)')
    return gen


def audit_name(base, gen):
    """The name generation `gen` of an audit dir or dataset dir goes by.

    Gen 1 keeps the bare name -- data/fn_audit, data/leash_audit,
    data/audit_finds, data/audit_finds_leash -- byte for byte: thirty-six
    thousand human verdicts are keyed against those pools, and scripts outside
    this repo (dogs_detection/leash90) open data/leash_audit/{pool.parquet,
    full} by name. Every later generation is `<base>_gen<N>`. None of these
    match crop_sources.FORBIDDEN_PATH_RE (data/gate_*, data/leash), which
    refuses a model's store; never name an audit dir that way."""
    g = _want_gen(gen)
    return base if g == 1 else f'{base}_gen{g}'


def paths(stage=DEFAULT_STAGE, *, gen):
    """Every file one generation of this stage's audit owns.

    Computed, never module state: one dashboard process serves both audits and
    two requests can be in flight at once, so a `use(stage)` global -- which is
    what the runner does, where a process is one stage from start to exit --
    would have one page writing into the other's ledger. And a pure function
    of (stage, gen) for the same reason one level down: gen 1 and gen 2 of one
    stage are served side by side, and nothing about a request may move which
    directory it lands in except the generation it names.

    `gen` is keyword-only and REQUIRED (see _want_gen). `store`, `shards` and
    `work` are the stage's live model store, which is not per generation: only
    build() and pool_info() read them, and only for the live generation.
    """
    sp, g = spec(stage), _want_gen(gen)
    out = os.path.join(REPO, 'data', audit_name(sp['audit_dir'], g))
    return {
        'out': out,
        'gen': g,
        'pool': os.path.join(out, 'pool.parquet'),
        'pool_json': os.path.join(out, 'pool.json'),
        'verdicts': os.path.join(out, 'verdicts.jsonl'),
        'drawn': os.path.join(out, 'drawn.jsonl'),
        'crops': os.path.join(out, 'crops'),
        'full': os.path.join(out, 'full'),
        # a person's redraw: the crop, and its contact-sheet thumbnail. full/
        # and crops/ are what the model was judged on and are never rewritten
        'edited': os.path.join(out, 'edited'),
        'edited_thumbs': os.path.join(out, 'edited_thumbs'),
        'pages': os.path.join(out, 'pages'),
        'dataset': os.path.join(REPO, 'data', audit_name(sp['dataset'], g)),
        'store': os.path.join(REPO, 'data', sp['dir']),
        'shards': os.path.join(REPO, 'data', sp['dir'],
                               f"{sp['dir']}-*.parquet"),
        'work': os.path.join(REPO, 'data', sp['dir'], 'work.parquet'),
    }


# No module-level OUT_DIR/POOL/VERDICTS here. They existed for a moment as
# "the older names still resolve", and every one of them was the GATE's path
# wearing a stage-neutral name -- so any later code reaching for one would
# quietly read the wrong audit. paths(stage, gen=N) is the only way in.

# The identity every gen-1 artifact has. Gen-1 pool.json files predate pool
# identity (neither has one), so the constant stands for "the one gen-1 pool
# of this stage"; a gen-1 verdict or redraw row may carry it and may not
# carry any other.
G1_POOL_ID = 'g1-legacy'
POOL_FIELD = 'pool'
# The crop each protocol's model judged, by the stage store's protocol tag
# (gate_store.protocol_tag). `pad012` is gate_store.cut_crops' framing
# (PAD_FRAC 0.12 + PAD_PX 4), `pad050` is cut_vit's (leash_vit.rect_pad at
# 0.5 of the long side). A protocol not listed is one the audit page cannot
# reproduce, and build() refuses it. This is only the THUMBNAIL a tile shows:
# full/ and edited/ are cut_crops framing for every stage and generation,
# because crop_sources reuses them as production crops and re-cuts a sample
# with cut_crops to prove it (framing_check_audit).
VIEW_FRAMES = {'full@640': 'pad012', 'pad050@448': 'pad050'}
# A dataset dir's claim about which pool filled it (export, audit.place).
DATASET_MARKER = '.audit_pool.json'

# The whole score range, in tenths. It used to stop at 0.5 because the pool
# held only rejections, and a rejected box cannot score above the threshold
# that rejected it -- so half the axis was missing and there was no way to ask
# whether what the model KEPT was any good either.
BANDS = [(round(i / 10, 1), round((i + 1) / 10, 1)) for i in range(10)]
BAND_W = BANDS[0][1] - BANDS[0][0]
# Where the model itself draws the line.
THRESHOLD = 0.5


def band_of(p):
    """The band a score falls in.

    Compared against the edges rather than divided by the width. `int(0.3 /
    0.1)` is 2, not 3, because 0.3 is not 0.3 -- and a score sitting exactly
    on a band edge is the one place a rate gets attributed to the wrong band.
    The pool's SQL reaches the same answer by adding an epsilon before the
    floor, on float32 where the error goes the other way; a guard holds the
    two together."""
    for i, (lo, hi) in enumerate(BANDS):
        if p < hi:
            return i
    return len(BANDS) - 1


def _roots():
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import dashboard as dash
    return dash._grid_roots()


def _q(path):
    """A path as a duckdb string literal's body."""
    return str(path).replace("'", "''")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _store_gen_of(root):
    """crop_sources.store_gen_of -- the generation a detect store's det_idx
    and confidences are, or None -- imported late (crop_sources imports this
    module, and the dashboard pays for neither until it asks)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import crop_sources
    return crop_sources.store_gen_of(root)


def build(args):
    """Every box this stage judged, with the geometry to cut it and the
    sequence it belongs to -- ONCE per detector generation, into an empty dir.

    One pass, kept on disk, because the alternative is re-joining 6.2M rows
    against a 32.5M-row manifest on every page of the audit.

    WHAT IT PROVES BEFORE THE POOL IS VISIBLE. The stage store's plan.json is
    the only thing that says which detector its (image_id, det_idx) belong to,
    and the store's shards and work.parquet carry no generation at all. So
    before anything is swapped in, every pool row is joined back to the
    detect store the plan names, and every box and conf must be that store's
    -- not a sample, and not a comparison of counts: the det shards hold some
    keys twice, and every one of those copies has to agree. A pool that
    passes is gen N's boxes because it was checked against gen N's boxes.

    WRITE ORDER. pool.parquet.tmp, then its size, mtime and sha256 into
    pool.json.tmp, then pool.json is renamed into place, THEN pool.parquet.
    A stop between the two renames leaves a pool.json whose parquet is
    missing, which pool_meta() reads as "not ready" -- never as a pool, and
    never as gen 1. (The order was the other way round, and in a fresh dir a
    stop in between left a pool with no pool.json, which then read as gen 1.)
    """
    stage = getattr(args, 'stage', DEFAULT_STAGE)
    gen = getattr(args, 'gen', None)
    sp = spec(stage)
    # Refused before anything is read or written (beyond plan.json, the
    # store's status and a listing of the out dir).
    why = _build_refusal(stage, gen)
    if why:
        raise SystemExit(why)
    P = paths(stage, gen=gen)
    plan = _plan_doc(P)
    detect_root = os.path.realpath(plan['detect_root'])
    protocol = _protocol_now(P)
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{_q(args.tmp)}'")
    shards, work = P['shards'], P['work']
    if not glob.glob(shards):
        raise SystemExit(f"no {sp['title']} shards -- has it run?")

    # ONE model judged this store, and it is the one the pool will name. Read
    # rather than dropped: a store half re-run with another model is two
    # audits' worth of scores under one band axis.
    try:
        models = [m for (m,) in con.execute(
            f"SELECT DISTINCT model FROM read_parquet('{_q(shards)}', "
            f"union_by_name=true)").fetchall()]
    except duckdb.Error as e:
        raise SystemExit(f"refusing to build: the {sp['title']} shards have "
                         f"no readable model column ({e})")
    if len(models) != 1 or not models[0]:
        raise SystemExit(f"refusing to build: the {sp['title']} shards name "
                         f"{len(models)} model value(s) {sorted(map(str, models))}"
                         f" -- a pool is one model's verdicts on one "
                         f"generation's boxes")
    model = str(models[0])
    # Every judged box, exactly once, and every one of them in work.parquet.
    # The COPY below is an inner join on work, so a missing key would just
    # not be in the pool, and a doubled one would be in it twice.
    n_shard, n_keys = con.execute(
        f"SELECT count(*), count(DISTINCT (image_id, det_idx)) "
        f"FROM read_parquet('{_q(shards)}')").fetchone()
    if n_shard != plan.get('rows'):
        raise SystemExit(f"refusing to build: the {sp['title']} shards hold "
                         f"{n_shard:,} rows and {P['store']}/plan.json "
                         f"planned {plan.get('rows')!r} -- the run is not "
                         f"complete, or not this plan's")
    if n_keys != n_shard:
        raise SystemExit(f"refusing to build: {n_shard - n_keys:,} "
                         f"(image_id, det_idx) key(s) are judged more than "
                         f"once in the {sp['title']} shards")
    lost = con.execute(
        f"SELECT count(*) FROM read_parquet('{_q(shards)}') s "
        f"ANTI JOIN read_parquet('{_q(work)}') w "
        f"ON w.image_id = s.image_id AND w.det_idx = s.det_idx").fetchone()[0]
    if lost:
        raise SystemExit(f"refusing to build: {lost:,} judged key(s) have no "
                         f"row in {work} -- their boxes cannot be cut")
    os.makedirs(P['out'], exist_ok=True)

    # image_id -> sequence, for the images that carry a judged box. Read off
    # the harvest manifests, which is where the field lives; the detection
    # store never carried it. The roots are the plan's own -- the same ones
    # recorded in pool.json for the page to cut from.
    roots = plan.get('roots') or _roots()
    mans = []
    for root in roots.values():
        mans += glob.glob(os.path.join(root, '*', 'all_data_*.parquet'))
    if not mans:
        raise SystemExit('no harvest manifests found under the grid roots')
    print(f'{len(mans):,} manifests', flush=True)

    # The manifests hold 32.5M rows and only a few million of them matter.
    # Grouping the whole thing by image_id first spilled 150 GB of temp and
    # died: the keys we need are known up front, so they go in first and the
    # scan is filtered against them rather than aggregated and joined after.
    # Every box the gate judged, not only the ones it rejected: the question
    # "is this a dog" is worth asking of what it kept as well, and that is the
    # only way the bands above 0.5 exist at all.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE rejected AS
        SELECT s.image_id, s.det_idx, s.{sp['p_col']} AS p_dog
        FROM read_parquet('{_q(shards)}') s
    """)
    con.execute("CREATE OR REPLACE TEMP TABLE need AS "
                "SELECT DISTINCT image_id FROM rejected")
    need = con.execute("SELECT count(*) FROM need").fetchone()[0]
    print(f'{need:,} frames carry a judged box', flush=True)

    # SEMI JOIN, not GROUP BY: one row per wanted image, nothing else kept.
    # A frame harvested into two cells appears twice with the same sequence,
    # so it is still deduplicated -- just over the wanted rows, not 32.5M.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE seqs AS
        SELECT image_id, any_value(seq) AS seq FROM (
            SELECT CAST(m.image_id AS VARCHAR) AS image_id,
                   m."sequence" AS seq
            FROM read_parquet({mans!r}, union_by_name=true) m
            SEMI JOIN need n ON n.image_id = CAST(m.image_id AS VARCHAR)
            WHERE m."sequence" IS NOT NULL
        ) GROUP BY 1
    """)
    got = con.execute("SELECT count(*) FROM seqs").fetchone()[0]
    print(f'{got:,} of them resolved to a sequence', flush=True)

    tmp = P['pool'] + '.tmp'
    con.execute(f"""
        COPY (
            SELECT r.image_id, r.det_idx, r.p_dog,
                   w.x1, w.y1, w.x2, w.y2, w.conf, w.cell, w.drive,
                   COALESCE(q.seq, 'img:' || r.image_id) AS seq,
                   least({len(BANDS)} - 1,
                         floor((r.p_dog + 1e-9) / {BAND_W})::INT) AS band
            FROM rejected r
            JOIN read_parquet('{_q(work)}') w
              ON w.image_id = r.image_id AND w.det_idx = r.det_idx
            LEFT JOIN seqs q ON q.image_id = r.image_id
        ) TO '{_q(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n, seqs, unmatched = con.execute(f"""
        SELECT count(*), count(DISTINCT seq),
               sum(CASE WHEN seq LIKE 'img:%' THEN 1 ELSE 0 END)
        FROM read_parquet('{_q(tmp)}')""").fetchone()
    if n != n_shard:
        raise SystemExit(f'refusing to swap the pool in: it holds {n:,} rows '
                         f'for {n_shard:,} judged boxes (work.parquet holds a '
                         f'key more than once?) -- left at {tmp}')
    _geometry_check(con, tmp, detect_root, gen)
    print(f'{n:,} judged boxes across {seqs:,} sequences '
          f'({unmatched or 0:,} had no sequence and stand alone)')

    # What this pool IS, recorded before it is visible. Size and mtime are
    # checked on every request (cheap); the sha256 by `verify` and the tests.
    st = os.stat(tmp)
    doc = {'stage': stage, 'gen': gen, 'pool_id': uuid.uuid4().hex,
           # WHICH detector's boxes: det_idx means nothing without it
           'detect_root': detect_root,
           # WHICH model judged them, read off the shards above
           'model': model,
           # HOW those verdicts were made: a re-run at another imgsz keeps the
           # model name and can keep the shard count
           'protocol': protocol, 'view_frame': VIEW_FRAMES[protocol],
           # where the page finds each drive's frames, as planned
           'roots': roots,
           'rows': int(n), 'sequences': int(seqs),
           # a stage still running has more shards every few minutes; the
           # page can say how far behind the pool is rather than present a
           # stretch of the job as "the store"
           'shards': len(glob.glob(shards)),
           'plan_created': plan.get('created'),
           'built': time.time(),
           'parquet_size': st.st_size, 'parquet_mtime_ns': st.st_mtime_ns,
           'sha256': _sha256(tmp), 'state': 'ready'}
    jtmp = P['pool_json'] + '.tmp'
    with open(jtmp, 'w') as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    # pool.json FIRST, then the parquet (see the docstring); os.replace keeps
    # the mtime, so the stat recorded above is the file's from here on
    os.replace(jtmp, P['pool_json'])
    os.replace(tmp, P['pool'])
    for i, (lo, hi) in enumerate(BANDS):
        c = con.execute(f"SELECT count(*), count(DISTINCT seq) "
                        f"FROM read_parquet('{_q(P['pool'])}') WHERE band = {i}"
                        ).fetchone()
        print(f"  {sp['p_col']} {lo:.1f}-{hi:.1f}  {c[0]:>10,} boxes  "
              f"{c[1]:>9,} sequences")
    print(f"pool {doc['pool_id']} (gen {gen}, {model}, {protocol}) -> "
          f"{os.path.relpath(P['out'], REPO)}")
    return 0


def _geometry_check(con, pool_path, detect_root, gen):
    """SystemExit unless every pool row IS a box of `detect_root`.

    LEFT JOIN of the pool against gen `gen`'s det shards on (image_id,
    det_idx), per joined row: no key may be missing, and no joined row may
    differ by 0.5 px or more on any edge, or by 1e-4 or more in conf. The det
    shards hold some keys twice (4,061 joined rows for 4,000 keys in one
    sample), so it is every COPY that must agree -- comparing counts would
    pass a moved box hidden behind a duplicate. A NULL anywhere is a
    disagreement. union_by_name because the det files' schemas differ (some
    lack cell/drive); hive_partitioning off because the files carry their own
    `gen` column."""
    det = os.path.join(detect_root, 'shards', 'gen=%04d' % gen, '**',
                       '*.det.parquet')
    try:
        unmatched, moved, joined, keys = con.execute(f"""
            SELECT count(*) FILTER (WHERE d.di IS NULL),
                   count(*) FILTER (WHERE d.di IS NOT NULL AND NOT coalesce(
                       abs(p.x1 - d.x1) < 0.5 AND abs(p.y1 - d.y1) < 0.5
                       AND abs(p.x2 - d.x2) < 0.5 AND abs(p.y2 - d.y2) < 0.5
                       AND abs(p.conf - d.conf) < 1e-4, false)),
                   count(*),
                   count(DISTINCT (p.image_id, p.det_idx))
            FROM read_parquet('{_q(pool_path)}') p
            LEFT JOIN (
                SELECT CAST(image_id AS VARCHAR) AS iid,
                       CAST(det_idx AS INTEGER) AS di, x1, y1, x2, y2, conf
                FROM read_parquet('{_q(det)}', hive_partitioning=false,
                                  union_by_name=true)) d
              ON d.iid = CAST(p.image_id AS VARCHAR)
             AND d.di = p.det_idx
        """).fetchone()
    except Exception as e:                    # noqa: BLE001 - say what, then stop
        raise SystemExit(f'refusing to swap the pool in: its boxes cannot be '
                         f'checked against {det} ({type(e).__name__}: {e}) '
                         f'-- left at {pool_path}')
    print(f'geometry: {keys:,} pool keys joined to {joined:,} gen-{gen} det '
          f'rows; {unmatched:,} unmatched, {moved:,} disagree', flush=True)
    if unmatched or moved:
        raise SystemExit(
            f'refusing to swap the pool in: {unmatched:,} pool key(s) are not '
            f'in {detect_root} and {moved:,} joined row(s) disagree with it '
            f'(>= 0.5 px or conf >= 1e-4) -- these are not gen-{gen} boxes. '
            f'Left at {pool_path}')


def _protocol_now(P):
    """The stage store's protocol tag, the runner's own reading of it."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gate_store
    return gate_store.protocol_tag(P['store'])


def _plan_doc(P):
    """The stage store's plan.json, or None when it has none."""
    try:
        with open(os.path.join(P['store'], 'plan.json')) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _plan_gen(plan):
    """The detector generation a stage's plan was cut from; None unknown."""
    if not plan:
        return None
    try:
        return int(plan.get('gen') or 1)
    except (TypeError, ValueError):
        return None


def _pool_gen(doc):
    """The generation a GEN-1 DIR's pool says it was cut from. A pool.json
    written before the key existed was cut from gen 1 -- the only generation
    there was. Read for gen 1 only: a pool of any later generation must say
    so itself (pool_meta)."""
    try:
        return int((doc or {}).get('gen') or 1)
    except (TypeError, ValueError):
        return 1


def _ledger_pin():
    """data/ledger_detect_root.txt -- the gen-1 store, which the two gen-1
    pools were cut from and never recorded -- or None."""
    try:
        with open(os.path.join(REPO, 'data', 'ledger_detect_root.txt')) as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _plan_detect_root(plan):
    """The detect store whose det_idx the plan's keys are. A gen-1 plan did
    not record it; that store is the pinned ledger root. A plan of any later
    generation names its own or names none -- never the pin, which is gen 1
    by definition -- and build() refuses one that names none."""
    root = (plan or {}).get('detect_root')
    if root or (_plan_gen(plan) or 1) != 1:
        return root or None
    return _ledger_pin()


def _pool_doc(P):
    """(pool.json as a dict, whether a pool exists at all). Gen 1 only."""
    try:
        with open(P['pool_json']) as fh:
            doc = json.load(fh)
        doc = doc if isinstance(doc, dict) else {}
        return doc, True
    except (OSError, ValueError):
        return {}, os.path.exists(P['pool'])


def _build_refusal(stage, gen):
    """Why `build --stage stage --gen gen` must not run, or None.

    A pool is cut ONCE, from the live store of its generation, into a
    directory that holds nothing yet:

      gen 1        frozen. Its two pools hold every gen-1 human verdict,
                   redraw and crop, keyed against the boxes they were cut
                   from; nothing re-cuts them.
      the plan     must be this generation's: plan.gen == gen, plan.json
                   names its detect_root, and that store says it IS gen
                   `gen` (crop_sources.store_gen_of). The ledger pin is
                   never consulted -- it is gen 1 by definition.
      the out dir  must be empty (stray *.tmp of a stopped build aside). A
                   rebuild re-keys every verdict, crop and redraw cut from
                   the old pool; --overwrite-judged is gone with it. A new
                   pool of the same generation needs a new audit dir.
      protocol     must be one the audit page can frame (VIEW_FRAMES).

    The model count, the row count and the key coverage need the shards
    read, and build() refuses those itself, before writing anything.
    """
    if gen is None:
        return ('refusing to build: --gen is required. A pool is cut from one '
                'detector generation\'s boxes and there is no default.')
    try:
        gen = _want_gen(gen)
    except ValueError as e:
        return f'refusing to build: {e}'
    sp = spec(stage)
    if gen == 1:
        return (f"refusing to build a gen-1 {sp['title']} pool: gen 1 is "
                f"frozen. data/{sp['audit_dir']} holds the human verdicts, "
                f"redraws and crops keyed against the boxes it was cut from, "
                f"and re-cutting it would re-key every one of them.")
    P = paths(stage, gen=gen)
    plan = _plan_doc(P)
    if plan is None:
        return (f"refusing to build: {P['store']}/plan.json is missing or "
                f"unreadable, and it is the only record of which detector "
                f"the store's det_idx belong to")
    if plan.get('gen') != gen:
        return (f"refusing to build a gen-{gen} pool: {P['store']}/plan.json "
                f"is gen {plan.get('gen')!r}. det_idx numbers different boxes "
                f"in each generation, so a pool cut from it would key gen-"
                f"{gen} verdicts onto other boxes.")
    root = plan.get('detect_root')
    if not root:
        return (f"refusing to build: {P['store']}/plan.json names no "
                f"detect_root, so the pool's boxes cannot be checked against "
                f"the gen-{gen} store (the gen-1 ledger pin is never a "
                f"stand-in)")
    got = _store_gen_of(root)
    if got != gen:
        return (f"refusing to build: {root} says it is gen {got!r}, and the "
                f"plan says gen {gen}")
    try:
        names = os.listdir(P['out'])
    except FileNotFoundError:
        names = []
    except OSError as e:
        return f"refusing to build: {P['out']} cannot be listed ({e})"
    other = sorted(n for n in names if not n.endswith('.tmp'))
    if other:
        return (f"refusing to build into {P['out']}: it already holds "
                f"{', '.join(other[:6])}{' ...' if len(other) > 6 else ''}. "
                f"A pool is cut once into an empty dir: verdicts, pages, "
                f"crops and redraws are keyed against the pool that is "
                f"there, and a re-cut would re-key them. If a build stopped "
                f"between its two renames (pool.json and no pool.parquet), "
                f"nothing was ever served from it; remove that pool.json by "
                f"hand and build again.")
    proto = _protocol_now(P)
    if proto not in VIEW_FRAMES:
        return (f"refusing to build: {P['store']} was judged under protocol "
                f"{proto!r}, which the audit page cannot frame "
                f"(known: {', '.join(sorted(VIEW_FRAMES))})")
    return None


# ── which pool, which generation ────────────────────────────────────────────

def live_gen(stage):
    """(gen, detect_root) of the stage's LIVE store -- the generation new
    audit pages are drawn from.

    Read from data/<dir>/plan.json and nothing else: it must name `gen` and
    `detect_root`, and that detect store must say it is that generation. Any
    gap is GenError, never a guess -- data/detect_root.txt names the live
    DETECTOR store and the ledger pin the gen-1 one, and neither says which
    generation the stage store's verdicts were made on.
    """
    P = paths(stage, gen=1)
    plan = _plan_doc(P)
    if plan is None:
        raise GenError(f"{P['store']}/plan.json is missing or unreadable: "
                       f"which generation the {spec(stage)['title']} store "
                       f"is cannot be told")
    g, root = plan.get('gen'), plan.get('detect_root')
    if g is None or gen_of(g) is None or g < 1:
        raise GenError(f"{P['store']}/plan.json names no generation "
                       f"(gen {g!r})")
    if not root:
        raise GenError(f"{P['store']}/plan.json (gen {g}) names no "
                       f"detect_root")
    got = _store_gen_of(root)
    if got != g:
        raise GenError(f"{P['store']}/plan.json says gen {g}, and its "
                       f"detect_root {root} says gen {got!r}")
    return g, root


def is_frozen(stage, gen):
    """Whether generation `gen` of this stage's audit is closed for new pages:
    it is not the live store's generation -- or the live generation cannot be
    told, which closes every generation rather than guess one open."""
    try:
        return _want_gen(gen) != live_gen(stage)[0]
    except GenError:
        return True


# pool.json documents already parsed and checked, by path -> ((inode,
# mtime_ns, ctime_ns, size), doc or None, why). A request reads pool_meta
# several times; the file changes once per build (a rename: a new inode),
# and the inode and ctime are in the key so a same-size rewrite inside one
# mtime tick is not served from the cache.
_META = {}


def _meta_problems(doc, stage, gen):
    """Why a gen >= 2 pool.json does not identify a ready pool, or ''."""
    if not isinstance(doc, dict):
        return 'it is not a JSON object'
    why = []
    if doc.get('state') != 'ready':
        why.append(f"state {doc.get('state')!r}, not 'ready'")
    if doc.get('stage') != stage:
        why.append(f"stage {doc.get('stage')!r}, not {stage!r}")
    if doc.get('gen') != gen or gen_of(doc.get('gen')) != gen:
        why.append(f"gen {doc.get('gen')!r}, not {gen}")
    for k in ('pool_id', 'detect_root', 'model', 'protocol', 'view_frame',
              'sha256'):
        if not isinstance(doc.get(k), str) or not doc.get(k):
            why.append(f'no {k}')
    for k in ('parquet_size', 'parquet_mtime_ns'):
        if gen_of(doc.get(k)) is None or doc.get(k) is None:
            why.append(f'no {k}')
    if doc.get('protocol') in VIEW_FRAMES and \
            doc.get('view_frame') != VIEW_FRAMES[doc['protocol']]:
        why.append(f"view_frame {doc.get('view_frame')!r} is not protocol "
                   f"{doc.get('protocol')!r}'s")
    elif isinstance(doc.get('protocol'), str) \
            and doc['protocol'] not in VIEW_FRAMES:
        why.append(f"unknown protocol {doc['protocol']!r}")
    if not isinstance(doc.get('roots'), dict) or not doc.get('roots'):
        why.append('no roots')
    return '; '.join(why)


def pool_meta(stage, gen):
    """The identity of generation `gen`'s pool for this stage, or
    PoolIdentityError. A copy: callers may keep it, not change the cache.

      gen 1    pool.json may be missing (a pool.parquet with none is how the
               first pool was cut) or carry no `gen`; it may not say any
               generation other than 1, and may not be unreadable. The
               identity is G1_POOL_ID; detect_root is the ledger pin (what
               both gen-1 pools were cut from, and never recorded); the view
               is pad012; the model is STAGES' legacy name.
      gen >= 2 pool.json must exist and name state 'ready', this stage, this
               gen, a pool_id, the parquet's size, mtime and sha256, the
               detect_root, model, protocol, view_frame and roots. The
               parquet's size and mtime are compared on every call (the sha
               by `verify`). A missing pool.json is NOT gen 1 here: it is a
               pool that does not exist, or one whose build stopped.

    Includes `gen`, `pool_id`, `detect_root`, `model`, `view_frame`, `roots`
    (gen >= 2) and whatever else pool.json records (rows, shards, built,
    protocol, sequences).
    """
    sp, g = spec(stage), _want_gen(gen)
    P = paths(stage, gen=g)
    f = P['pool_json']
    try:
        st = os.stat(f)
    except FileNotFoundError:
        if g == 1:
            return _gen1_meta(stage, {})
        raise PoolIdentityError(
            f"{f} does not exist: there is no gen-{g} {sp['title']} pool "
            f"(fn_audit.py build --stage {stage} --gen {g} --tmp DIR)")
    except OSError as e:
        raise PoolIdentityError(f'{f} cannot be read ({e})')
    key = (st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)
    got = _META.get(f)
    if got is None or got[0] != key:
        try:
            with open(f) as fh:
                doc = json.load(fh)
            why = ''
        except (OSError, ValueError) as e:
            doc, why = None, f'{f} is unreadable ({type(e).__name__})'
        if not why:
            if g == 1:
                if not isinstance(doc, dict):
                    why = f'{f} is not a JSON object'
                elif gen_of(doc.get('gen')) != 1:
                    why = (f"{f} says gen {doc.get('gen')!r}, and this is the "
                           f"gen-1 audit dir")
            else:
                why = _meta_problems(doc, stage, g)
                why = f'{f}: {why}' if why else ''
        got = (key, doc, why)
        _META[f] = got
    _, doc, why = got
    if why:
        raise PoolIdentityError(why)
    if g == 1:
        return _gen1_meta(stage, doc)
    try:
        pst = os.stat(P['pool'])
    except OSError:
        raise PoolIdentityError(
            f"{P['pool']} is missing: the gen-{g} pool is not ready (a build "
            f"that stopped between its two renames leaves exactly this)")
    if (pst.st_size, pst.st_mtime_ns) != (doc['parquet_size'],
                                          doc['parquet_mtime_ns']):
        raise PoolIdentityError(
            f"{P['pool']} is not the file {f} describes (size/mtime "
            f"{pst.st_size}/{pst.st_mtime_ns}, recorded "
            f"{doc['parquet_size']}/{doc['parquet_mtime_ns']})")
    return json.loads(json.dumps(doc))


def _gen1_meta(stage, doc):
    meta = dict(doc)
    meta.update({'gen': 1, 'pool_id': G1_POOL_ID, 'stage': stage,
                 'detect_root': doc.get('detect_root') or _ledger_pin(),
                 'view_frame': 'pad012',
                 'model': doc.get('model') or spec(stage)['model'],
                 'state': 'ready'})
    return meta


def verify_pool(stage, gen):
    """None when generation `gen`'s pool is exactly the file its pool.json
    describes, sha256 included; else why not. Gen 1 recorded no sha."""
    try:
        meta = pool_meta(stage, gen)
    except (PoolIdentityError, ValueError) as e:
        return str(e)
    if gen == 1:
        return None
    P = paths(stage, gen=gen)
    got = _sha256(P['pool'])
    if got != meta['sha256']:
        return (f"{P['pool']} sha256 {got[:12]} is not the recorded "
                f"{meta['sha256'][:12]}")
    return None


def audit_gens(stage):
    """[1, ...]: every generation of this stage's audit there is to read --
    gen 1 always, then each data/<audit_dir>_gen<N> whose pool passes
    pool_meta. A directory that does not (a build that stopped) is not
    listed: it has no pool to have answered against."""
    sp = spec(stage)
    base = os.path.join(REPO, 'data', sp['audit_dir'] + '_gen')
    out = {1}
    for d in glob.glob(glob.escape(base) + '*'):
        n = d[len(base):]
        if not (n.isascii() and n.isdigit()) or n != str(int(n)) \
                or int(n) < 2:
            continue
        try:
            pool_meta(stage, int(n))
        except PoolIdentityError:
            continue
        out.add(int(n))
    return sorted(out)


@dataclasses.dataclass(frozen=True)
class AuditCtx:
    """ONE generation of one stage's audit, resolved once per request.

    Every read and write in a request goes through the one ctx it resolved:
    `P` (paths), `meta` (pool_meta), `pool_id` and whether the generation is
    `frozen` (closed for new pages). A request that re-resolved paths() at
    each step could straddle two directories if anything moved in between.
    """
    stage: str
    gen: int
    pool_id: str
    P: dict
    meta: dict
    frozen: bool


def ctx(stage, gen):
    """AuditCtx for (stage, gen); PoolIdentityError when that pool is not
    ready, KeyError for an unknown stage, ValueError for a bad gen."""
    spec(stage)
    g = _want_gen(gen)
    meta = pool_meta(stage, g)
    return AuditCtx(stage=stage, gen=g, pool_id=meta['pool_id'],
                    P=paths(stage, gen=g), meta=meta,
                    frozen=is_frozen(stage, g))


def pool_gen(stage=DEFAULT_STAGE, *, gen):
    """`gen` when that generation's pool of this stage is identifiable
    (pool_meta), else None -- writers refuse then and readers keep nothing.

    A thin wrapper now: which generation an audit belongs to is its
    directory's, so this only says whether that directory can vouch for it.
    A missing pool.json is gen 1 for the gen-1 dir and nothing for any
    other."""
    try:
        return pool_meta(stage, gen)['gen']
    except PoolIdentityError:
        return None


# The stage store's shard models, by store -> (stat key, [model, ...] or
# None). Read once per change of the shards: pool_info is polled.
_MODELS = {}


def _store_models(P):
    """Sorted distinct `model` values in the live stage store's shards ([]
    for none, None when they cannot be read)."""
    names = sorted(glob.glob(P['shards']))
    if not names:
        return []
    try:
        key = tuple((n, os.stat(n).st_mtime_ns, os.stat(n).st_size)
                    for n in names)
    except OSError:
        return None
    got = _MODELS.get(P['store'])
    if got is not None and got[0] == key:
        return got[1]
    try:
        import duckdb
        con = duckdb.connect()
        try:
            con.execute("SET memory_limit='2GB'")
            rows = con.execute(f"SELECT DISTINCT model FROM read_parquet("
                               f"{names!r}, union_by_name=true)").fetchall()
        finally:
            con.close()
    except Exception:                         # noqa: BLE001 - unknown, said so
        return None
    models = sorted((None if m is None else str(m) for (m,) in rows),
                    key=lambda m: (m is None, m or ''))
    _MODELS[P['store']] = (key, models)
    return models


def draw_refusal(c):
    """Why no new page may be drawn for AuditCtx `c`, or None.

    A frozen generation (not the live store's) takes no new pages -- its
    answers can still be changed on keys already drawn. A live pool whose
    store has since been judged by another model is closed too: the sheets
    would show one model's scores as another's, and the cure is a new audit
    dir, not a re-cut (build refuses a non-empty dir)."""
    title = spec(c.stage)['title']
    if c.frozen:
        return (f'the gen-{c.gen} {title} audit is closed for new pages; its '
                f'answers can still be changed')
    if c.gen == 1:
        return None
    models = _store_models(c.P)
    if models is None:
        return (f"the {title} store's model cannot be read, so whether it "
                f"still matches this pool's ({c.meta['model']}) cannot be "
                f"told; no new pages until it can")
    if models != [c.meta['model']]:
        return (f"the {title} store is now judged by "
                f"{', '.join(map(str, models)) or 'nothing'}, and this pool "
                f"by {c.meta['model']}: no new pages here. A new model's "
                f"audit needs a new audit dir.")
    return None


def pool_info(stage=DEFAULT_STAGE, *, gen):
    """When generation `gen`'s pool was cut, and whether the stage has moved
    on since.

    A pool is a snapshot. The leash model was 7% through its run when its
    first pool was built, so every band count in it was a seventh of the
    truth -- and the column is headed "in the store".

    A FROZEN generation (not the live store's) is not behind anything: the
    store it was cut from is archived, and the live one numbers other boxes.
    So it reports `frozen` and a message, stale False, behind 0, and reads no
    shards at all. A live gen >= 2 pool also reports `model_moved`: the store
    has since been judged by another model (draw_refusal closes it then).

    Gen >= 2 raises PoolIdentityError for a pool that is not ready; gen 1
    reads a missing or damaged pool.json as it always did (no provenance).
    """
    g = _want_gen(gen)
    P = paths(stage, gen=g)
    title = spec(stage)['title']
    try:
        live, live_why = live_gen(stage)[0], ''
    except GenError as e:
        live, live_why = None, str(e)
    if g == 1:
        doc, exists = _pool_doc(P)
        pool_g = _pool_gen(doc)
    else:
        doc, exists, pool_g = pool_meta(stage, g), True, g
        # the page's business is the pool's state, not where the drives are
        doc.pop('roots', None)
    doc['gen'] = g
    doc['pool_gen'] = pool_g
    doc['plan_gen'] = _plan_gen(_plan_doc(P))
    doc['live_gen'] = live
    if live != g:
        doc.update({'frozen': True, 'stale': False, 'behind': 0,
                    'unknown': False, 'gen_mismatch': False,
                    'model_moved': False,
                    'message': f'gen-{g} audit — closed for new pages; '
                               f'answers can still be changed'})
        if live_why:
            doc['live_error'] = live_why
        return doc
    doc['frozen'] = False
    now = len(glob.glob(P['shards']))
    doc['shards_now'] = now
    # A pool cut before this file existed carries no provenance, and "behind
    # by 165 shards" is a worse answer than "cannot tell" -- rebuilding is
    # what establishes it either way.
    doc['unknown'] = 'shards' not in doc
    then = int(doc.get('shards') or 0)
    # a pool cut before the key existed was cut from the legacy 224 verdicts
    doc['protocol_now'] = _protocol_now(P)
    proto_moved = doc.get('protocol', 'legacy') != doc['protocol_now'] \
        and doc['protocol_now'] != ''
    doc['stale'] = (not doc['unknown']) and (now > then or proto_moved)
    doc['behind'] = max(0, now - then) if not doc['unknown'] else 0
    # A store of ANOTHER detector generation is not this pool's future: its
    # det_idx numbers other boxes, so "rebuild to catch up" would re-key every
    # verdict here onto them. Not stale -- a different audit.
    plan_gen = doc['plan_gen']
    doc['gen_mismatch'] = bool(exists and plan_gen is not None
                               and plan_gen != pool_g)
    if doc['gen_mismatch']:
        doc['stale'] = False
        doc['behind'] = 0
        doc['message'] = (f"this pool is gen-{pool_g} and the {title} store "
                          f"is now gen-{plan_gen}; start a new audit dir, do "
                          f"not rebuild")
    doc['model_moved'] = False
    if g >= 2:
        models = _store_models(P)
        doc['models_now'] = models
        doc['model_moved'] = models is None or models != [doc['model']]
        if doc['model_moved']:
            doc['message'] = (f"the {title} store is now judged by "
                              f"{', '.join(map(str, models or [])) or 'an unreadable model'}"
                              f", this pool by {doc['model']}: no new pages "
                              f"here; a new model's audit needs a new audit "
                              f"dir")
    return doc


def band_totals(stage=DEFAULT_STAGE, *, gen):
    """[(lo, hi, boxes)] -- the weights the headline rate needs.

    Gen 1 reads a missing pool as no weights, as it always has. A gen >= 2
    pool that is not ready, or cannot be read, raises: a page of rates with
    every weight silently zero is a wrong answer, not an empty one."""
    g = _want_gen(gen)
    if g >= 2:
        pool_meta(stage, g)
    try:
        import duckdb
        con = duckdb.connect()
        try:
            rows = con.execute(
                f"SELECT band, count(*) FROM read_parquet("
                f"'{_q(paths(stage, gen=g)['pool'])}') "
                f"GROUP BY 1 ORDER BY 1").fetchall()
        finally:
            con.close()
    except Exception:
        if g >= 2:
            raise
        return []
    got = dict(rows)
    return [(lo, hi, int(got.get(i, 0))) for i, (lo, hi) in enumerate(BANDS)]


# Page documents already parsed, by (path, mtime). A page is written under an
# index that is the count of the pages before it and never touched again, so a
# stats poll only ever has a newly drawn sheet to read; the mtime is in the key
# because a rebuilt store starts over at 00000.
_SHEETS = {}


# How a sheet was drawn, written into the page document's own band value as
# '<band>~<mode>'. The vocabulary lives HERE because this file is what has to
# tell a stratified draw from a targeted one; audit.py, which serves the
# control, reads it back off this constant so there is one spelling of it.
DRAW_MODES = ('least', 'most')


def draw_mode_of(band):
    """The draw mode a page document's band value carries, or None.

    A spread draw -- 'rejected', 9, None -- has no mode and IS the sample the
    estimator assumes. Anything else was aimed somewhere.
    """
    if isinstance(band, str) and '~' in band:
        m = band.partition('~')[2]
        return m if m in DRAW_MODES else None
    return None


def _read_pages(stage=DEFAULT_STAGE, *, gen):
    """[(draw mode or None, [(band, key), ...]), ...] -- one entry per sheet
    of generation `gen`'s audit.

    Every reader of pages/ comes through here so a page document is parsed
    once per (path, mtime) however many questions are asked of it.
    """
    P = paths(stage, gen=gen)
    out = []
    for f in glob.glob(os.path.join(P['pages'], '*.json')):
        try:
            ck = (f, os.path.getmtime(f))
        except OSError:
            continue
        got = _SHEETS.get(ck)
        if got is None:
            try:
                with open(f) as fh:
                    doc = json.load(fh)
            except (OSError, ValueError):
                doc = {}
            got = (draw_mode_of(doc.get('band')),
                   [(it.get('band'), it['key'])
                    for it in (doc.get('items') or []) if it.get('key')])
            _SHEETS[ck] = got
        out.append(got)
    return out


def sheets(stage=DEFAULT_STAGE, *, gen):
    """{band: {key, ...}} -- what was actually put in front of a person.

    Every draw is written to pages/ before it is served and the document names
    its crops, so the sheets are the record of what was SHOWN and the ledger is
    the record of what came back. Without them a band's answers were counted as
    a sample OF the band, when they are a sample of what someone chose to click
    on a sheet where every other tile could be scrolled past.
    """
    out = {}
    for _, items in _read_pages(stage, gen=gen):
        for band, key in items:
            out.setdefault(band, set()).add(key)
    return out


def targeted(stage=DEFAULT_STAGE, *, gen):
    """{key, ...} -- every crop handed out by a TARGETED draw.

    'least confident' and 'most confident' hand back one end of a band on
    purpose: measured on the live gate store, a 'most confident' page below
    the line came back 25 crops all scoring exactly 0.000, out of a band
    holding 3,530,147 boxes. Those are finds, and worth drawing -- but they
    are not a sample of the band, and summarise() multiplies a band's rate by
    the band's whole population. So they are held out of the rate, and the
    page says how many were held.
    """
    out = set()
    for mode, items in _read_pages(stage, gen=gen):
        if mode:
            out.update(k for _, k in items)
    return out


# ── who judged it ───────────────────────────────────────────────────────────
# Every store a person's DECISION lands in names its annotator in a field
# called `by`: the two flag ledgers, both verdict ledgers here, the box
# corrections, the reviewed-and-kept ledger, and the two databases' column of
# the same name -- leash verdicts and wrong-label flags. The exported dataset
# manifests carry the same name as `judged_by`, spelled out because a manifest
# row is read next to `judged_at` and `judged_model` and `by` alone would say
# nothing there. What has no annotator is what nobody decided: the two
# drawn.jsonl ledgers record which crops were SHOWN, which is the denominator
# of the rates the audit reports and not an answer anybody gave.
#
# The word for a row that has none is spelled ONCE, here, and imported by
# every module that writes or reads one -- a second spelling is the day one
# reader says admin and another says None, over the same 3,247 lines.
AUTHOR_FIELD = 'by'
LEGACY_AUTHOR = 'malyeta'
"""Who an annotation with no author on it was made by.

An absent author does not mean "nobody" and it does not mean "unknown": it
means the row was written before the dashboard had accounts, when there was
exactly one person judging crops and that person is the person who runs this
deployment. So it reads as them at the point of reading, everywhere, and it
reads that way from the file exactly as it stands.

It is the DASHBOARD_USER name, and it moves when that account is renamed --
three thousand rows carry no author at all, and pointing this at a name no
account holds would orphan every one of them.

WHY THE LEDGERS WERE NOT REWRITTEN. Adding `by` to 3,247 existing lines would
put a script in front of the only copy of work nobody can reproduce -- crops
cut from a pool that rotates hourly, judged one at a time by hand -- to add a
fact that is already known and that nothing was missing. The stores are
append-only because that is the property that makes them safe; a migration
that rewrites them spends it. A default costs one function call per read and
cannot lose a line.

AND IT IS NEVER WRITTEN. Nothing puts this value into a new record: a write
that cannot name its annotator is refused instead (see the write paths in
dashboard.py, audit.py and leash_store.py). Defaulting on the way OUT states
a fact about rows that predate accounts; defaulting on the way IN would forge
attribution for rows that do not, which is the one thing this field exists to
prevent.
"""


def author_of(who):
    """The annotator to show for one stored value -- LEGACY_AUTHOR if absent.

    Takes the field as it came off disk (``rec.get('by')``, a NULL column),
    so a missing key, a None and an empty string all resolve the same way.
    """
    return str(who) if who else LEGACY_AUTHOR


# WHICH DETECTOR a human ledger row's det_idx, conf and crop belong to. The
# second word every ledger shares, owned here for the same reason as `by`:
# the flag ledgers, the box corrections, the reviewed-and-kept ledger and the
# leash database's column of the same name. A det_idx (and a conf to two
# decimals) names a box only within one detector generation -- gen 2 numbers
# the same frame's boxes differently -- so a gen-1 row joined to a gen-2 store
# lands on a box nobody judged. Every new row records it; every reader keeps
# only rows whose generation is the one of the store it joins against.
#
# The audit ledgers sit one per generation, each in the audit dir beside the
# pool its keys were cut from (paths(stage, gen=N)), and build refuses to cut
# a pool into a dir that holds anything. The gen-1 ledgers' rows carry no
# tag (they read as LEGACY_GEN); every row of a gen >= 2 ledger carries `gen`
# AND `pool` (POOL_FIELD, the pool_id of pool_meta), and read_verdicts raises
# on one that does not -- a row is never trusted to be in the right file just
# because it is in it. Box corrections are one file for the review page and
# both audits of every generation, so there the tags are what select a row.
GEN_FIELD = 'gen'
LEGACY_GEN = 1
"""The generation a ledger row with no `gen` on it was recorded against.

Every row written before the tag existed was made on the gen-1 detector's
boxes -- the only one there was -- so an absent tag reads as 1, on the way
out, exactly as an absent `by` reads as LEGACY_AUTHOR. Never written: a new
row takes its generation from the store (or pool) it was judged on.
"""


def gen_of(tag):
    """The detector generation one stored `gen` value names, as an int.

    Takes the value as it came off disk (``rec.get('gen')``, a NULL column):
    absent or None is LEGACY_GEN. A value that is not a non-negative int --
    a string, a float, a bool -- names no generation and comes back None,
    which matches no store, so a reader skips that row rather than guess.
    """
    if tag is None:
        return LEGACY_GEN
    if isinstance(tag, bool) or not isinstance(tag, int) or tag < 0:
        return None
    return tag


def read_verdicts(path=None, *, stage, gen):
    """[{key, verdict, band, p_dog, ts, by, ...}] -- generation `gen`'s
    ledger of this stage (or `path`, read AS that ledger), append-only, last
    write wins.

    Keyed on image_id#det_idx so a crop judged twice (reloaded page, changed
    mind) counts once, as its latest answer.

    THE ONE READER of every verdict ledger -- the audit page, the statistics
    and the dataset export all come through here, as does everything outside
    this module that reads a verdict at all -- so the legacy author is applied
    once, on the way out, and no caller downstream can see a row with no
    annotator on it.

    AND THE ONE PLACE A ROW'S GENERATION IS CHECKED. A row whose `gen` (absent:
    gen 1) is not `gen` raises GenError naming the file and line; so does, in
    a gen >= 2 ledger, a row with no `gen` or with a `pool` that is not the
    pool's pool_id, and in the gen-1 ledger a `pool` other than G1_POOL_ID.
    Raised, not skipped: a ledger holds one pool's answers, so a foreign row
    in it is a writer's bug, and skipping it would hide the bug while keeping
    the rows it wrote beside it.
    """
    g = _want_gen(gen)
    f = path or paths(stage, gen=g)['verdicts']
    want_pool = G1_POOL_ID if g == 1 else None
    out = {}
    try:
        fh = open(f)
    except OSError:
        return []
    with fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not (isinstance(d, dict) and d.get('key')):
                continue
            tag = d.get(GEN_FIELD)
            if g >= 2 and tag is None:
                raise GenError(f'{f}:{n}: a row with no {GEN_FIELD} in the '
                               f'gen-{g} ledger')
            if gen_of(tag) != g:
                raise GenError(f'{f}:{n}: a {GEN_FIELD} {tag!r} row in the '
                               f'gen-{g} ledger')
            if g >= 2 and want_pool is None:
                want_pool = pool_meta(stage, g)['pool_id']
            got = d.get(POOL_FIELD)
            if (g >= 2 or got is not None) and got != want_pool:
                raise GenError(f'{f}:{n}: a row of pool {got!r} in the ledger '
                               f'of pool {want_pool}')
            # a null verdict is a withdrawal -- the box goes back to
            # unjudged rather than becoming a third kind of answer
            if d.get('verdict') is None:
                out.pop(d['key'], None)
            else:
                d[AUTHOR_FIELD] = author_of(d.get(AUTHOR_FIELD))
                out[d['key']] = d
    return list(out.values())


def wilson(k, n, z=1.96):
    """95% interval for a proportion. Not k/n +- 1.96*sqrt(...): at the rates
    this audit is looking for -- a handful of misses in a few hundred crops --
    the normal approximation runs off the end of the scale and reports
    negative lower bounds."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


# What a person can answer, and what it means as a label. "missed"/"correct"
# only made sense while the pool was rejections -- once it holds what the gate
# KEPT as well, "missed a dog" is meaningless on a box the gate called a dog.
# The question is the same everywhere now: is this a dog. Whether that is an
# error is derived from the model's own score, not from the wording.
# NOT `VERDICTS` -- that name is already the path to the ledger a few lines
# up, and reusing it made read_verdicts() open a tuple.
def answers(stage=DEFAULT_STAGE):
    return spec(stage)['answers']


def verdict_of(v, stage=DEFAULT_STAGE):
    sp = spec(stage)
    # the gate's ledger predates the current wording; a verdict written as
    # 'missed' still reads rather than being dropped
    v = sp['legacy'].get(v, v)
    return v if v in sp['answers'] else None


def summarise(verdicts=None, totals=None, shown=None, stage=DEFAULT_STAGE,
              aimed=None, *, gen):
    """Per band and overall, for generation `gen`'s audit of this stage.

    Each band reports the share of its crops a person called a dog. Below the
    threshold that share IS the false-negative rate -- the gate rejected them
    all. Above it, the share that are NOT dogs is the false-positive rate. One
    measurement, read from either end.

    The two headline rates weight bands by how many boxes the gate actually
    put in each; a flat mean over the bands would report the near-threshold
    error rate as if it were the whole store's.

    WHAT A BAND'S RATE IS NOT. `shown` counts the crops the sheets put in front
    of a person; `answered` counts the ones that came back, and they are a long
    way apart -- 63 of 1,175 when this was first measured, band by band between
    1.7% and 10.2%. The crops that come back are the ones someone chose to
    click, and on a contact sheet the one worth stopping on is the one that
    already looks like the answer. So a band's share is a CEILING on the band's
    real share until its sheets are answered in full, the bias does not cancel
    across bands because the answer rate is not the same in any two of them,
    and the answer rate travels beside every rate here for that reason.

    AND WHAT IS NOT IN IT AT ALL. A sheet can also be drawn at one END of a
    band -- 'least confident' walks out from the threshold, 'most confident'
    in from the edges -- which selects crops on exactly the property being
    measured. Every rate here is multiplied by its band's whole population,
    so folding a targeted page in would state the surest corner of a band as
    the band: measured on the live gate store, one 'most confident' page
    below the line moved the headline from 95.5% to anywhere between 52.6%
    and 97.3% depending on the answers. Those crops are held out of the rates
    and counted on their own as `aimed` -- the finds are still finds, and
    still land in the dataset.
    """
    sp, g = spec(stage), _want_gen(gen)
    vs = read_verdicts(stage=stage, gen=g) if verdicts is None else verdicts
    totals = band_totals(stage, gen=g) if totals is None else totals
    shown = sheets(stage, gen=g) if shown is None else shown
    aimed = targeted(stage, gen=g) if aimed is None else set(aimed)
    pos, neg = sp['positive'], sp['negative']
    back = {v.get('key') for v in vs}
    per = []
    for i, (lo, hi) in enumerate(BANDS):
        answered = [v for v in vs if v.get('band') == i
                    and verdict_of(v.get('verdict'), stage) in (pos, neg)]
        # the targeted ones are answers, and they are not the sample
        seen = [v for v in answered if v.get('key') not in aimed]
        aim = [v for v in answered if v.get('key') in aimed]
        k = sum(1 for v in seen if verdict_of(v['verdict'], stage) == pos)
        ak = sum(1 for v in aim if verdict_of(v['verdict'], stage) == pos)
        p, a, b = wilson(k, len(seen))
        boxes = totals[i][2] if i < len(totals) else 0
        # the sheets those crops came off are not a denominator for the rate
        # above either -- a targeted page answered in full would otherwise
        # read as this band's ceiling being reached
        put = [key for key in (shown.get(i) or ()) if key not in aimed]
        per.append({'lo': lo, 'hi': hi, 'judged': len(seen), 'dogs': k,
                    'rate': p, 'lo95': a, 'hi95': b, 'boxes': boxes,
                    # answered off a least/most-confident sheet: held out of
                    # every number on this row, and stated so it is not lost
                    'aimed': len(aim), 'aimed_dogs': ak,
                    # the errors those crops turned up. They are real finds
                    # and the page counts them as finds -- what they are not
                    # is a share of anything, which is why they sit in their
                    # own field rather than in 'wrong'.
                    'aimed_wrong': (len(aim) - ak) if lo >= THRESHOLD else ak,
                    # the denominator the rate above is missing: crops served
                    # on a sheet, and how many of them anyone answered
                    'shown': len(put),
                    'answered': sum(1 for key in put if key in back),
                    # what the gate said about this whole band
                    'kept': lo >= THRESHOLD,
                    # crops where the person and the model disagree
                    'wrong': (len(seen) - k) if lo >= THRESHOLD else k})
    def side(kept):
        rows = [b for b in per if b['kept'] == kept and b['judged']]
        pop = sum(b['boxes'] for b in rows)
        allpop = sum(b['boxes'] for b in per if b['kept'] == kept)
        # over EVERY band on this side, not only the ones with an answer in
        # them: a band that was shown and never answered is exactly what the
        # rate beside it needs to be read against
        was = [b for b in per if b['kept'] == kept]
        put = sum(b['shown'] for b in was)
        got = sum(b['answered'] for b in was)
        if not pop:
            # nothing judged on this side yet -- but how many boxes are
            # waiting is still a fact, and reporting it as zero read as "the
            # gate kept nothing"
            return {'rate': 0.0, 'judged': 0, 'wrong': 0, 'boxes': allpop,
                    'covered': 0.0, 'shown': put, 'answered': got,
                    'aimed': sum(b['aimed'] for b in was),
                    'aimed_wrong': sum(b['aimed_wrong'] for b in was)}
        rate = sum((b['wrong'] / b['judged']) * b['boxes']
                   for b in rows) / pop
        return {'rate': rate, 'judged': sum(b['judged'] for b in rows),
                'wrong': sum(b['wrong'] for b in rows), 'boxes': allpop,
                'covered': pop / allpop if allpop else 0.0,
                'shown': put, 'answered': got,
                # answered on this side off a targeted sheet, and therefore
                # in none of the rates beside it
                'aimed': sum(b['aimed'] for b in was),
                'aimed_wrong': sum(b['aimed_wrong'] for b in was)}
    pop = sum(b['boxes'] for b in per) or 1
    seen_pop = sum(b['boxes'] for b in per if b['judged'])
    # 'unrecorded' is the ledger's own past: the first sessions were served
    # before pages/ was written, so their crops have an answer and no sheet to
    # be a share OF. They are still verdicts, they are just not a rate.
    on_sheet = set().union(*shown.values()) if shown else set()
    return {'bands': per,
            # dogs it threw away, and not-dogs it kept
            'rejected': side(False), 'kept': side(True),
            'sheets': {'shown': sum(b['shown'] for b in per),
                       'answered': sum(b['answered'] for b in per),
                       'unrecorded': len(back - on_sheet)},
            # NOT the share of the audit that has been done: every band holding
            # one answer reads as 1.0 here. 'sheets' above is the one that says
            # how much of what was shown came back.
            'covered': seen_pop / pop if pop else 0.0,
            'judged': sum(b['judged'] for b in per),
            'wrong': sum(b['wrong'] for b in per),
            # answered off a least/most-confident sheet: in no rate above,
            # and never silently -- a page that holds answers out has to say
            # how many it is holding
            'aimed': sum(b['aimed'] for b in per),
            'aimed_dogs': sum(b['aimed_dogs'] for b in per),
            'aimed_wrong': sum(b['aimed_wrong'] for b in per),
            'threshold': THRESHOLD, 'pool': pop, 'stage': stage, 'gen': g,
            'pool_info': pool_info(stage, gen=g),
            'asymmetric': sp['asymmetric'], 'title': sp['title'],
            'asks': sp['asks'], 'positive': pos, 'negative': neg,
            'below': sp['below'], 'above': sp['above'], 'miss': sp['miss']}


README = """# audit crops from the {title}

Laid out for `yolo classify` so they can be folded into a training set the
same way `data/hard_negatives` and `data/hard_positives` are. The question
asked of every crop was: **{asks}**

    {pos}/<image_id>_<det_idx>.jpg     a person answered yes
    {neg}/<image_id>_<det_idx>.jpg     a person answered no
    manifest.jsonl                     one line per crop, with its sequence
    .audit_pool.json                   the one pool every crop here is from

Every file here carries a HUMAN verdict. The model's own label is what put the
box in front of someone; it is never what is written down. `p_dog` on each row
is what {model} thought, kept so the disagreements can be found again --
`disagrees: true` marks every row where the person and the model differ.

## Which boxes these are

Detector generation {gen}, pool `{pool_id}`, boxes of `{detect_root}`. A key
`<image_id>_<det_idx>` names a box only within one detector generation --
another generation numbers the same frame's boxes differently -- so never mix
these with another generation's audit crops by file name or key. Each
manifest row carries `gen`, `pool`, `detect_root` and the detector `box`.

## Read this before training on it

**This is not a random sample of anything.** The audit draws evenly from ten
p_dog bands rather than in proportion to how many boxes each holds, so the
near-threshold cases are massively over-represented relative to the store.
That is deliberate -- it is where the model is wrong, and it is what makes
these worth training on. It also means class balance here says nothing about
the store's, and accuracy measured on a split of these says nothing about
accuracy in production.

**And within a band, the crops that were answered chose themselves.** A sheet
of twenty-five can be scrolled past as easily as it can be clicked, and the
crop worth stopping on is the one that already looks like the answer. So a
band's measured share is an upper bound on its real one until that band's
sheets are answered in full -- `pages/` records what was shown, and
`fn_audit.py stats` prints the answer rate beside every band.

**Split on `sequence`, never on `image_id`.** The manifest carries it for
every crop. Splitting per image looked clean once before and put 70.8% of a
val set in a sequence that was also in train.

Rebuild at any time with:

    python tools/detect/fn_audit.py export --stage {stage} --gen {gen}
"""


def dataset_marker(c):
    """What a dataset dir filled from AuditCtx `c` says about itself."""
    return {'stage': c.stage, 'gen': c.gen, 'pool_id': c.pool_id,
            'detect_root': c.meta.get('detect_root')}


def dataset_marker_refusal(c):
    """Why nothing may be filed into `c`'s dataset dir, or None.

    The dir's DATASET_MARKER must name exactly `c`'s pool. A dir that holds
    files and no marker is refused for gen >= 2 -- whatever is in it came
    from somewhere else, and export deletes what its ledger does not list --
    and accepted for gen 1 only: data/audit_finds and data/audit_finds_leash
    predate the marker and are never given one (gen-1 paths are byte for
    byte what they were). An absent or empty dir is fine: the first export
    writes the marker (write_dataset_marker)."""
    d = c.P['dataset']
    f = os.path.join(d, DATASET_MARKER)
    try:
        with open(f) as fh:
            got = json.load(fh)
    except FileNotFoundError:
        try:
            names = os.listdir(d)
        except FileNotFoundError:
            return None
        except OSError as e:
            return f'{d} cannot be listed ({e})'
        if not names or c.gen == 1:
            return None
        return (f'{d} holds {len(names)} entr(ies) and no {DATASET_MARKER}: '
                f'it was not filled from gen-{c.gen} pool {c.pool_id}, and '
                f'nothing is filed or deleted there')
    except (OSError, ValueError) as e:
        return f'{f} is unreadable ({type(e).__name__})'
    want = dataset_marker(c)
    if got != want:
        return (f'{d} belongs to {got!r}, not to {want!r}: nothing is filed '
                f'or deleted there')
    return None


def write_dataset_marker(c):
    """Create `c`'s dataset dir and its marker if it has none (gen >= 2
    only). Call after dataset_marker_refusal(c) said None."""
    if c.gen == 1:
        return
    d = c.P['dataset']
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, DATASET_MARKER)
    if os.path.exists(f):
        return
    with open(f + '.tmp', 'w') as fh:
        json.dump(dataset_marker(c), fh, sort_keys=True)
    os.replace(f + '.tmp', f)


def pool_boxes(c, keys):
    """{key: [x1, y1, x2, y2]} for the '<image_id>#<det_idx>' keys found in
    `c`'s pool -- the detector box each one names in THIS generation."""
    pairs = []
    for k in keys:
        iid, _, di = str(k).partition('#')
        if iid and di.isdigit():
            pairs.append((iid, int(di)))
    if not pairs:
        return {}
    import duckdb
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit='2GB'")
        con.execute('CREATE TEMP TABLE want(i VARCHAR, d INTEGER)')
        con.executemany('INSERT INTO want VALUES (?, ?)', sorted(set(pairs)))
        rows = con.execute(
            f"SELECT p.image_id, p.det_idx, p.x1, p.y1, p.x2, p.y2 FROM "
            f"read_parquet('{_q(c.P['pool'])}') p JOIN want w "
            f"ON CAST(p.image_id AS VARCHAR) = w.i AND p.det_idx = w.d"
        ).fetchall()
    finally:
        con.close()
    return {f'{i}#{d}': [float(x1), float(y1), float(x2), float(y2)]
            for i, d, x1, y1, x2, y2 in rows}


def export(args):
    """Write generation `--gen`'s dataset from its ledger. Idempotent.

    The dashboard already files each crop as it is judged; this rebuilds the
    whole thing from the record -- after the ledger is edited by hand, after
    crops are re-cut, or just to be sure the two agree. Anything in a class
    directory that the ledger does not still say belongs there is removed,
    because a dataset that keeps a label nobody stands behind is worse than
    one that is a rebuild out of date.

    ONLY THE LIVE GENERATION, ONLY ITS OWN DIR. A frozen generation's dataset
    is not rebuilt -- gen 1's is what it is, and its files are read by name
    elsewhere -- and the deletion above happens only in a dir whose marker
    names this exact pool (dataset_marker_refusal): run against the wrong
    dir, it would delete every crop the wrong ledger does not list. Every
    manifest row names the generation, pool, detect store and detector box
    it is a label of, and the model that judged it comes from pool.json.
    """
    stage = getattr(args, 'stage', DEFAULT_STAGE)
    gen = getattr(args, 'gen', None)
    if gen is None:
        raise SystemExit('refusing to export: --gen is required')
    try:
        c = ctx(stage, gen)
    except (GenError, ValueError) as e:
        raise SystemExit(f'refusing to export: {e}')
    sp, P = spec(stage), c.P
    if c.frozen:
        raise SystemExit(f"refusing to export the gen-{c.gen} "
                         f"{sp['title']} audit: it is frozen (the live store "
                         f"is another generation), and its dataset is left "
                         f"exactly as it is")
    why = dataset_marker_refusal(c)
    if why:
        raise SystemExit(f'refusing to export: {why}')
    dataset, classes = P['dataset'], (sp['positive'], sp['negative'])
    import shutil
    # A hand-drawn box wins for TRAINING and only for training: full/ is the
    # picture the model was judged on and never changes, edited/ is the same
    # box redrawn by a person. What gets exported is the best framing
    # available; what was measured is untouched.
    full, edited = P['full'], P['edited']
    keep = {}
    for v in read_verdicts(stage=stage, gen=c.gen):
        cls = verdict_of(v.get('verdict'), stage)
        if cls in classes:
            keep[str(v['key']).replace('#', '_') + '.jpg'] = (cls, v)
    # every answer names a box of THIS pool, and the box it recorded (if it
    # recorded one) is that box -- checked before anything is deleted
    boxes = pool_boxes(c, [v['key'] for _, v in keep.values()])
    lost = sorted(str(v['key']) for _, v in keep.values()
                  if str(v['key']) not in boxes)
    if lost:
        raise SystemExit(f'refusing to export: {len(lost)} verdict key(s) '
                         f'are not in pool {c.pool_id} (e.g. {lost[:3]})')
    off = sorted(str(v['key']) for _, v in keep.values()
                 if isinstance(v.get('box'), list) and len(v['box']) == 4
                 and any(abs(float(a) - b) >= 0.5 for a, b in
                         zip(v['box'], boxes[str(v['key'])])))
    if off:
        raise SystemExit(f'refusing to export: {len(off)} verdict(s) recorded '
                         f'a box that is not their pool box (e.g. {off[:3]})')
    write_dataset_marker(c)
    placed = missing = removed = 0
    for cls in classes:
        d = os.path.join(dataset, cls)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if keep.get(f, (None,))[0] != cls:
                os.remove(os.path.join(d, f))
                removed += 1
    redrawn = 0
    for name, (cls, v) in sorted(keep.items()):
        src, dst = os.path.join(full, name), os.path.join(dataset, cls, name)
        fixed = os.path.join(edited, name)
        if os.path.exists(fixed):
            src = fixed
            redrawn += 1
        if not os.path.exists(src):
            missing += 1
            continue
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        placed += 1
    model = c.meta['model']
    with open(os.path.join(dataset, 'manifest.jsonl'), 'w') as fh:
        for name, (cls, v) in sorted(keep.items()):
            iid, _, di = name[:-4].rpartition('_')
            fh.write(json.dumps({
                'file': f'{cls}/{name}', 'label': cls,
                'image_id': iid, 'det_idx': int(di) if di.isdigit() else None,
                # the split column. Never image_id -- see the README.
                'sequence': v.get('seq'),
                'verdict': verdict_of(v.get('verdict'), stage),
                'p_dog': v.get('p_dog'), 'band': v.get('band'),
                # the row's reason for existing: the model and the person
                # disagreed, and this is the direction of the disagreement
                'disagrees': bool(
                    v.get('p_dog') is not None
                    and ((float(v['p_dog']) >= THRESHOLD)
                         != (cls == sp['positive']))),
                # whether a person redrew this box before it was exported
                'corrected': os.path.exists(os.path.join(edited, name)),
                'judged_at': v.get('ts'),
                # WHO said so, beside WHEN. read_verdicts() has already put
                # the legacy author on the rows that predate accounts, so a
                # manifest row can name a person even when its ledger line
                # could not.
                'judged_by': author_of(v.get(AUTHOR_FIELD)),
                'judged_model': model,
                # WHICH box: det_idx means nothing without the generation
                'gen': c.gen, 'pool': c.pool_id,
                'detect_root': c.meta.get('detect_root'),
                'box': boxes[str(v['key'])],
                'stage': stage}) + '\n')
    with open(os.path.join(dataset, 'README.md'), 'w') as fh:
        fh.write(README.format(model=model, title=sp['title'],
                               pos=sp['positive'], neg=sp['negative'],
                               asks=sp['asks'], stage=stage, gen=c.gen,
                               pool_id=c.pool_id,
                               detect_root=c.meta.get('detect_root')))
    n_pos = sum(1 for name, (c_, _) in keep.items() if c_ == sp['positive']
                and os.path.exists(os.path.join(dataset, sp['positive'], name)))
    wrong = sum(1 for c_, v in keep.values()
                if v.get('p_dog') is not None
                and ((float(v['p_dog']) >= THRESHOLD) != (c_ == sp['positive'])))
    print(f'{placed:,} crops in {os.path.relpath(dataset, REPO)} '
          f"({n_pos:,} {sp['positive']}, {placed - n_pos:,} {sp['negative']})"
          f' -- gen {c.gen}, pool {c.pool_id}')
    print(f"  {wrong:,} are cases the {sp['title']} got WRONG -- a person "
          f"disagreed with the side of {THRESHOLD} it put them on")
    if redrawn:
        print(f'  {redrawn:,} were exported with a hand-drawn box rather than '
              f"the detector's")
    if removed:
        print(f'  {removed:,} removed -- the ledger no longer says they '
              f'belong there')
    if missing:
        print(f'  {missing:,} judged but the full-resolution crop is gone; '
              f'they are in the manifest of neither')
    seqs = {v.get('seq') for _, v in keep.values() if v.get('seq')}
    print(f'  {len(seqs):,} distinct sequences -- split on those, not on '
          f'image_id')
    return 0


def stats(args):
    stage = getattr(args, 'stage', DEFAULT_STAGE)
    gen = getattr(args, 'gen', None)
    asked = gen is not None
    if gen is None:
        try:
            gen = live_gen(stage)[0]
        except GenError as e:
            raise SystemExit(f'no live generation to report on ({e}); pass '
                             f'--gen (--gen 1 for the gen-1 audit)')
    try:
        s = summarise(stage=stage, gen=gen)
    except (GenError, ValueError) as e:
        # with no --gen this is the LIVE generation, whose pool may not be
        # built yet -- say where the gen-1 numbers are
        raise SystemExit(str(e) + ('' if asked or gen == 1 else
                                   f' -- the live generation is {gen}; pass '
                                   f'--gen 1 for the gen-1 audit'))
    sp = spec(stage)
    sh = s['sheets']
    print(f"{sp['title']}, gen-{gen} audit "
          f"(data/{audit_name(sp['audit_dir'], gen)})")
    pi = s.get('pool_info') or {}
    if pi.get('frozen'):
        print(f"  NOTE: {pi['message']}.")
    if not s['judged']:
        # the sheets have still been drawn and cut, and how many crops are
        # sitting on them unanswered is the first thing worth knowing
        print('nothing judged yet'
              + (f" -- 0 of the {sh['shown']:,} crops already on a sheet"
                 if sh['shown'] else ''))
        return 0
    print(f"{sp['title']}: {s['judged']:,} boxes judged; the model and a "
          f"person disagreed on {s['wrong']:,}")
    if pi.get('frozen'):
        pass                    # said above, before the counts
    elif pi.get('gen_mismatch') or pi.get('model_moved'):
        print(f"  NOTE: {pi['message']}.")
    elif pi.get('unknown'):
        print('  NOTE: this pool predates provenance tracking -- it cannot '
              'say what part of the store it covers')
    elif pi.get('stale'):
        print(f"  NOTE: cut from {pi['shards']} shards; the run has written "
              f"{pi['shards_now']} since, so the counts below are a snapshot "
              f"and not the store. A pool is cut once: a new one needs a new "
              f"audit dir.")
    print(f"\n  band        boxes in store   answered   said dog   "
          f"share [95% interval]")
    for b in s['bands']:
        side = 'kept' if b['kept'] else 'threw away'
        # how many of the crops served in this band anyone answered. The share
        # to its right is a share of THIS, not of the band.
        ans = (f"{b['answered']:>4}/{b['shown']:<5}" if b['shown']
               else '     -   ')
        if not b['judged']:
            print(f"  {b['lo']:.1f}-{b['hi']:.1f}  {b['boxes']:>14,}  "
                  f"{ans}  not sampled ({side})")
            continue
        print(f"  {b['lo']:.1f}-{b['hi']:.1f}  {b['boxes']:>14,}  "
              f"{ans}  {b['dogs']:>4}/{b['judged']:<4}  {b['rate']:6.1%} "
              f"[{b['lo95']:.1%}, {b['hi95']:.1%}]  ({side})")
    r, k = s['rejected'], s['kept']
    # A rate off a sheet nobody finished is an upper bound, and saying so is
    # the whole difference between a measurement and a number.
    def cap(d):
        return 'at most ' if d['answered'] < d['shown'] else ''
    print(f"\nwrong below {THRESHOLD}: {cap(r)}{r['rate']:.2%} of "
          f"{r['boxes']:,} ({sp['below']})  ~{int(r['rate'] * r['boxes']):,}, "
          f"from {r['judged']:,} judged")
    print(f"wrong above {THRESHOLD}: {cap(k)}{k['rate']:.2%} of "
          f"{k['boxes']:,} ({sp['above']})  ~{int(k['rate'] * k['boxes']):,}, "
          f"from {k['judged']:,} judged")
    if sh['shown']:
        print(f"\nanswered {sh['answered']:,} of the {sh['shown']:,} crops "
              f"the sheets put in front of someone "
              f"({sh['answered'] / sh['shown']:.1%})")
        print('  the ones that came back are the ones someone chose to click, '
              'which is not a\n  random sample of what was shown -- every '
              'share above is a ceiling until a\n  sheet is answered in full')
    if sh['unrecorded']:
        print(f"  {sh['unrecorded']:,} more verdicts predate pages/ and have "
              f"no sheet to be a share of")
    if s['aimed']:
        print(f"\n{s['aimed']:,} more were answered off a least/most-confident "
              f"sheet ({s['aimed_dogs']:,} called {sp['positive']})")
        print('  those sheets are drawn at one END of a band, so they are '
              'finds and not a\n  sample of it -- they are in none of the '
              'shares above')
    return 0


def verify(args):
    """`verify --stage S --gen N`: 0 when the pool is exactly the file its
    pool.json describes, sha256 included; 1 with the reason otherwise."""
    why = verify_pool(args.stage, args.gen)
    if why:
        print(f'FAIL {why}')
        return 1
    meta = pool_meta(args.stage, args.gen)
    print(f"ok   gen-{args.gen} {spec(args.stage)['title']} pool "
          f"{meta['pool_id']}: {meta.get('rows', '?')} rows, "
          f"{'sha256 matches' if args.gen != 1 else 'gen 1 records no sha'}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.set_defaults(fn=build)
    b.add_argument('--memory', default='8GB')
    # required, with no default: the old one, /tmp/fn_audit_spill, is on
    # tmpfs here, so a full-store build spilled into RAM beside its own
    # --memory and pushed the machine into swap
    b.add_argument('--tmp', required=True,
                   help='where duckdb may spill: a dir on a large disk (not '
                        '/tmp, which is tmpfs here and counts against RAM)')
    e = sub.add_parser('export'); e.set_defaults(fn=export)
    st = sub.add_parser('stats'); st.set_defaults(fn=stats)
    v = sub.add_parser('verify'); v.set_defaults(fn=verify)
    for sub_p in (b, e, st, v):
        sub_p.add_argument('--stage', default=DEFAULT_STAGE,
                           choices=sorted(STAGES),
                           help='which model to audit')
    # Required wherever something is written or proved: there is no default
    # generation to fall back to. stats alone defaults to the live one.
    for sub_p in (b, e, v):
        sub_p.add_argument('--gen', type=int, required=True,
                           help='the detector generation of the pool')
    st.add_argument('--gen', type=int, default=None,
                    help='which generation\'s audit (default: the live '
                         'store\'s)')
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
