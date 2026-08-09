#!/usr/bin/env python3
"""Measure how often the dog-bin gate is right, across its whole score range.

    python tools/detect/fn_audit.py build     # one-off: the candidate pool
    python tools/detect/fn_audit.py stats     # what the verdicts say so far

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

WHY SEQUENCES. Mapillary frames come in sequences -- one car, one road, one
second apart. Twenty crops from one sequence are twenty photographs of the
same dog from twenty metres, and scoring them as twenty independent samples
would state a confidence the data does not have. One box per sequence, and a
sequence is never drawn twice.

WHAT THIS WRITES. A pool of candidates under data/fn_audit/, and nothing else.
The human verdicts it collects live in their own ledger, are never read by
anything that assembles training data, and are the only thing here that is not
a model's opinion.
"""

import argparse
import glob
import json
import os
import sys

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
        'yes': 'on a leash', 'no': 'loose',
        'asymmetric': False,
        'below': 'called it loose', 'above': 'called it leashed',
        'miss': 'leashed dogs called loose',
        'legacy': {},
    },
}
DEFAULT_STAGE = 'gate'


def spec(stage=DEFAULT_STAGE):
    if stage not in STAGES:
        raise KeyError(f'unknown audit stage {stage!r}')
    return STAGES[stage]


def paths(stage=DEFAULT_STAGE):
    """Every file this stage owns.

    Computed, never module state: one dashboard process serves both audits and
    two requests can be in flight at once, so a `use(stage)` global -- which is
    what the runner does, where a process is one stage from start to exit --
    would have one page writing into the other's ledger.
    """
    sp = spec(stage)
    out = os.path.join(REPO, 'data', sp['audit_dir'])
    return {
        'out': out,
        'pool': os.path.join(out, 'pool.parquet'),
        'verdicts': os.path.join(out, 'verdicts.jsonl'),
        'drawn': os.path.join(out, 'drawn.jsonl'),
        'crops': os.path.join(out, 'crops'),
        'full': os.path.join(out, 'full'),
        'pages': os.path.join(out, 'pages'),
        'dataset': os.path.join(REPO, 'data', sp['dataset']),
        'store': os.path.join(REPO, 'data', sp['dir']),
        'shards': os.path.join(REPO, 'data', sp['dir'],
                               f"{sp['dir']}-*.parquet"),
        'work': os.path.join(REPO, 'data', sp['dir'], 'work.parquet'),
    }


# kept so the older module-level names still resolve; both name the gate
OUT_DIR = paths()['out']
POOL = paths()['pool']
VERDICTS = paths()['verdicts']
GATE = paths()['store']

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


def build(args):
    stage = getattr(args, 'stage', DEFAULT_STAGE)
    sp, P = spec(stage), paths(stage)
    """Every rejected box, with the geometry to cut it and the sequence it
    belongs to.

    One pass, kept on disk, because the alternative is re-joining 4.7M rows
    against a 32.5M-row manifest on every page of the audit.
    """
    import duckdb
    os.makedirs(P['out'], exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory}'")
    shards, work = P['shards'], P['work']
    if not glob.glob(shards):
        raise SystemExit(f"no {sp['title']} shards -- has it run?")

    # image_id -> sequence, for the images that carry a rejected box. Read off
    # the harvest manifests, which is where the field lives; the detection
    # store never carried it.
    mans = []
    for root in _roots().values():
        mans += glob.glob(os.path.join(root, '*', 'all_data_*.parquet'))
    if not mans:
        raise SystemExit('no harvest manifests found under the grid roots')
    print(f'{len(mans):,} manifests', flush=True)

    # The manifests hold 32.5M rows and only 2.9M of them matter. Grouping
    # the whole thing by image_id first spilled 150 GB of temp and died: the
    # keys we need are known up front, so they go in first and the scan is
    # filtered against them rather than aggregated and joined afterwards.
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{args.tmp}'")
    # Every box the gate judged, not only the ones it rejected: the question
    # "is this a dog" is worth asking of what it kept as well, and that is the
    # only way the bands above 0.5 exist at all.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE rejected AS
        SELECT s.image_id, s.det_idx, s.{sp['p_col']} AS p_dog
        FROM read_parquet('{shards}') s
    """)
    con.execute("CREATE OR REPLACE TEMP TABLE need AS "
                "SELECT DISTINCT image_id FROM rejected")
    need = con.execute("SELECT count(*) FROM need").fetchone()[0]
    print(f'{need:,} frames carry a judged box', flush=True)

    # SEMI JOIN, not GROUP BY: one row per wanted image, nothing else kept.
    # A frame harvested into two cells appears twice with the same sequence,
    # so it is still deduplicated -- just over 2.9M rows instead of 32.5M.
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

    con.execute(f"""
        COPY (
            SELECT r.image_id, r.det_idx, r.p_dog,
                   w.x1, w.y1, w.x2, w.y2, w.conf, w.cell, w.drive,
                   COALESCE(q.seq, 'img:' || r.image_id) AS seq,
                   least({len(BANDS)} - 1,
                         floor((r.p_dog + 1e-9) / {BAND_W})::INT) AS band
            FROM rejected r
            JOIN read_parquet('{work}') w
              ON w.image_id = r.image_id AND w.det_idx = r.det_idx
            LEFT JOIN seqs q ON q.image_id = r.image_id
        ) TO '{P['pool']}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n, seqs, unmatched = con.execute(f"""
        SELECT count(*), count(DISTINCT seq),
               sum(CASE WHEN seq LIKE 'img:%' THEN 1 ELSE 0 END)
        FROM read_parquet('{P['pool']}')""").fetchone()
    print(f'{n:,} judged boxes across {seqs:,} sequences '
          f'({unmatched:,} had no sequence and stand alone)')
    for i, (lo, hi) in enumerate(BANDS):
        c = con.execute(f"SELECT count(*), count(DISTINCT seq) "
                        f"FROM read_parquet('{P['pool']}') WHERE band = {i}"
                        ).fetchone()
        print(f"  {sp['p_col']} {lo:.1f}-{hi:.1f}  {c[0]:>10,} boxes  "
              f"{c[1]:>9,} sequences")
    return 0


def band_totals(stage=DEFAULT_STAGE):
    """[(lo, hi, boxes)] -- the weights the headline rate needs."""
    try:
        import duckdb
        rows = duckdb.connect().execute(
            f"SELECT band, count(*) FROM read_parquet('{paths(stage)['pool']}') "
            f"GROUP BY 1 ORDER BY 1").fetchall()
    except Exception:
        return []
    got = dict(rows)
    return [(lo, hi, int(got.get(i, 0))) for i, (lo, hi) in enumerate(BANDS)]


def read_verdicts(path=None, stage=DEFAULT_STAGE):
    """[{key, verdict, band, p_dog, ts}] -- append-only, last write wins.

    Keyed on image_id#det_idx so a crop judged twice (reloaded page, changed
    mind) counts once, as its latest answer.
    """
    out = {}
    try:
        with open(path or paths(stage)['verdicts']) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict) and d.get('key'):
                    # a null verdict is a withdrawal -- the box goes back to
                    # unjudged rather than becoming a third kind of answer
                    if d.get('verdict') is None:
                        out.pop(d['key'], None)
                    else:
                        out[d['key']] = d
    except OSError:
        pass
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
ANSWERS = STAGES['gate']['answers']
CLASS_OF = {'dog': 'dog', 'not_dog': 'not_dog'}


def answers(stage=DEFAULT_STAGE):
    return spec(stage)['answers']


def verdict_of(v, stage=DEFAULT_STAGE):
    sp = spec(stage)
    # the gate's ledger predates the current wording; a verdict written as
    # 'missed' still reads rather than being dropped
    v = sp['legacy'].get(v, v)
    return v if v in sp['answers'] else None


def summarise(verdicts=None, totals=None, stage=DEFAULT_STAGE):
    """Per band and overall.

    Each band reports the share of its crops a person called a dog. Below the
    threshold that share IS the false-negative rate -- the gate rejected them
    all. Above it, the share that are NOT dogs is the false-positive rate. One
    measurement, read from either end.

    The two headline rates weight bands by how many boxes the gate actually
    put in each; a flat mean over the bands would report the near-threshold
    error rate as if it were the whole store's.
    """
    sp = spec(stage)
    vs = read_verdicts(stage=stage) if verdicts is None else verdicts
    totals = band_totals(stage) if totals is None else totals
    pos, neg = sp['positive'], sp['negative']
    per = []
    for i, (lo, hi) in enumerate(BANDS):
        seen = [v for v in vs if v.get('band') == i
                and verdict_of(v.get('verdict'), stage) in (pos, neg)]
        k = sum(1 for v in seen if verdict_of(v['verdict'], stage) == pos)
        p, a, b = wilson(k, len(seen))
        boxes = totals[i][2] if i < len(totals) else 0
        per.append({'lo': lo, 'hi': hi, 'judged': len(seen), 'dogs': k,
                    'rate': p, 'lo95': a, 'hi95': b, 'boxes': boxes,
                    # what the gate said about this whole band
                    'kept': lo >= THRESHOLD,
                    # crops where the person and the model disagree
                    'wrong': (len(seen) - k) if lo >= THRESHOLD else k})
    def side(kept):
        rows = [b for b in per if b['kept'] == kept and b['judged']]
        pop = sum(b['boxes'] for b in rows)
        allpop = sum(b['boxes'] for b in per if b['kept'] == kept)
        if not pop:
            # nothing judged on this side yet -- but how many boxes are
            # waiting is still a fact, and reporting it as zero read as "the
            # gate kept nothing"
            return {'rate': 0.0, 'judged': 0, 'wrong': 0, 'boxes': allpop,
                    'covered': 0.0}
        rate = sum((b['wrong'] / b['judged']) * b['boxes']
                   for b in rows) / pop
        return {'rate': rate, 'judged': sum(b['judged'] for b in rows),
                'wrong': sum(b['wrong'] for b in rows), 'boxes': allpop,
                'covered': pop / allpop if allpop else 0.0}
    pop = sum(b['boxes'] for b in per) or 1
    seen_pop = sum(b['boxes'] for b in per if b['judged'])
    return {'bands': per,
            # dogs it threw away, and not-dogs it kept
            'rejected': side(False), 'kept': side(True),
            'covered': seen_pop / pop if pop else 0.0,
            'judged': sum(b['judged'] for b in per),
            'wrong': sum(b['wrong'] for b in per),
            'threshold': THRESHOLD, 'pool': pop, 'stage': stage,
            'asymmetric': sp['asymmetric'], 'title': sp['title'],
            'asks': sp['asks'], 'positive': pos, 'negative': neg,
            'below': sp['below'], 'above': sp['above'], 'miss': sp['miss']}


DATASET = os.path.join(REPO, 'data', 'audit_finds')
README = """# audit crops from the {title}

Laid out for `yolo classify` so they can be folded into a training set the
same way `data/hard_negatives` and `data/hard_positives` are. The question
asked of every crop was: **{asks}**

    {pos}/<image_id>_<det_idx>.jpg     a person answered yes
    {neg}/<image_id>_<det_idx>.jpg     a person answered no
    manifest.jsonl                     one line per crop, with its sequence

Every file here carries a HUMAN verdict. The model's own label is what put the
box in front of someone; it is never what is written down. `p_dog` on each row
is what {model} thought, kept so the disagreements can be found again --
`disagrees: true` marks every row where the person and the model differ.

## Read this before training on it

**This is not a random sample of anything.** The audit draws evenly from ten
p_dog bands rather than in proportion to how many boxes each holds, so the
near-threshold cases are massively over-represented relative to the store.
That is deliberate -- it is where the model is wrong, and it is what makes
these worth training on. It also means class balance here says nothing about
the store's, and accuracy measured on a split of these says nothing about
accuracy in production.

**Split on `sequence`, never on `image_id`.** The manifest carries it for
every crop. Splitting per image looked clean once before and put 70.8% of a
val set in a sequence that was also in train.

Rebuild at any time with:

    python tools/detect/fn_audit.py export
"""


def export(args):
    stage = getattr(args, 'stage', DEFAULT_STAGE)
    sp, P = spec(stage), paths(stage)
    dataset, classes = P['dataset'], (sp['positive'], sp['negative'])
    """Write the dataset from the ledger. Idempotent.

    The dashboard already files each crop as it is judged; this rebuilds the
    whole thing from the record -- after the ledger is edited by hand, after
    crops are re-cut, or just to be sure the two agree. Anything in a class
    directory that the ledger does not still say belongs there is removed,
    because a dataset that keeps a label nobody stands behind is worse than
    one that is a rebuild out of date.
    """
    import shutil
    full = P['full']
    keep = {}
    for v in read_verdicts(stage=stage):
        cls = verdict_of(v.get('verdict'), stage)
        if cls in classes:
            keep[str(v['key']).replace('#', '_') + '.jpg'] = (cls, v)
    placed = missing = removed = 0
    for cls in classes:
        d = os.path.join(dataset, cls)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if keep.get(f, (None,))[0] != cls:
                os.remove(os.path.join(d, f))
                removed += 1
    for name, (cls, v) in sorted(keep.items()):
        src, dst = os.path.join(full, name), os.path.join(dataset, cls, name)
        if not os.path.exists(src):
            missing += 1
            continue
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        placed += 1
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
                # the row's reason for existing: a dog the gate said no to.
                # Everything downstream wants to weight or filter on this.
                # the row's reason for existing: the model and the person
                # disagreed, and this is the direction of the disagreement
                'disagrees': bool(
                    v.get('p_dog') is not None
                    and ((float(v['p_dog']) >= THRESHOLD)
                         != (cls == sp['positive']))),
                'judged_at': v.get('ts'),
                'judged_model': getattr(args, 'model', None) or sp['model'],
                'stage': stage}) + '\n')
    model = getattr(args, 'model', None) or sp['model']
    with open(os.path.join(dataset, 'README.md'), 'w') as fh:
        fh.write(README.format(model=model, title=sp['title'],
                               pos=sp['positive'], neg=sp['negative'],
                               asks=sp['asks']))
    n_pos = sum(1 for name, (c, _) in keep.items() if c == sp['positive']
                and os.path.exists(os.path.join(dataset, sp['positive'], name)))
    wrong = sum(1 for c, v in keep.values()
                if v.get('p_dog') is not None
                and ((float(v['p_dog']) >= THRESHOLD) != (c == sp['positive'])))
    print(f'{placed:,} crops in {os.path.relpath(dataset, REPO)} '
          f"({n_pos:,} {sp['positive']}, {placed - n_pos:,} {sp['negative']})")
    print(f"  {wrong:,} are cases the {sp['title']} got WRONG -- a person "
          f"disagreed with the side of {THRESHOLD} it put them on")
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
    s = summarise(stage=stage)
    sp = spec(stage)
    if not s['judged']:
        print('nothing judged yet')
        return 0
    print(f"{sp['title']}: {s['judged']:,} boxes judged; the model and a "
          f"person disagreed on {s['wrong']:,}")
    print(f"\n  band        boxes in store   said dog   share [95% interval]")
    for b in s['bands']:
        side = 'kept' if b['kept'] else 'threw away'
        if not b['judged']:
            print(f"  {b['lo']:.1f}-{b['hi']:.1f}  {b['boxes']:>14,}   "
                  f"not sampled ({side})")
            continue
        print(f"  {b['lo']:.1f}-{b['hi']:.1f}  {b['boxes']:>14,}   "
              f"{b['dogs']:>4}/{b['judged']:<4}  {b['rate']:6.1%} "
              f"[{b['lo95']:.1%}, {b['hi95']:.1%}]  ({side})")
    r, k = s['rejected'], s['kept']
    print(f"\nwrong below {THRESHOLD}: {r['rate']:.2%} of {r['boxes']:,} "
          f"({sp['below']})  ~{int(r['rate'] * r['boxes']):,}, "
          f"from {r['judged']:,} judged")
    print(f"wrong above {THRESHOLD}: {k['rate']:.2%} of {k['boxes']:,} "
          f"({sp['above']})  ~{int(k['rate'] * k['boxes']):,}, "
          f"from {k['judged']:,} judged")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.set_defaults(fn=build)
    b.add_argument('--memory', default='8GB')
    b.add_argument('--tmp', default='/tmp/fn_audit_spill',
                   help='where duckdb may spill; needs room')
    e = sub.add_parser('export'); e.set_defaults(fn=export)
    e.add_argument('--model', default=None,
                   help='which gate these crops were rejected by; recorded '
                        'on every manifest row')
    st = sub.add_parser('stats'); st.set_defaults(fn=stats)
    for sub_p in (b, e, st):
        sub_p.add_argument('--stage', default=DEFAULT_STAGE,
                           choices=sorted(STAGES),
                           help='which model to audit')
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
