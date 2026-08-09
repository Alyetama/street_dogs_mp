#!/usr/bin/env python3
"""Measure what the dog-bin gate threw away.

    python tools/detect/fn_audit.py build     # one-off: the candidate pool
    python tools/detect/fn_audit.py stats     # what the verdicts say so far

WHY. The gate rejected 3,945,390 of 4,688,510 boxes. Its accuracy was measured
on a few hundred held-out crops before it ran; this measures it on the thing
it actually did, at the scale it did it. The question is not "is the model
good" but "how many dogs are in the 3.9M boxes it discarded, and what do they
look like" -- because a discarded dog is unrecoverable and nothing downstream
will ever see it again.

WHY THE SAMPLE IS STRATIFIED. 89.5% of the rejected boxes score under 0.1, so
a uniform sample is 9 parts obvious-not-a-dog to 1 part everything else, and
tells you almost nothing about where the model's edge actually is. The pool is
banded by p_dog and drawn from evenly; each band's rate is reported on its own,
and the headline rate weights those bands by how many boxes each really holds.
That is the difference between "we looked at 500 crops" and a number.

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
OUT_DIR = os.path.join(REPO, 'data', 'fn_audit')
POOL = os.path.join(OUT_DIR, 'pool.parquet')
VERDICTS = os.path.join(OUT_DIR, 'verdicts.jsonl')
GATE = os.path.join(REPO, 'data', 'gate')

# Edges in p_dog. Not equal-width by accident: the gate's own threshold is
# 0.5, so the bands nearest it are the ones that decide whether the threshold
# is in the right place, and the 0.0-0.1 band -- nine tenths of the pool --
# only has to be shown to be empty of dogs, not characterised.
BANDS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
# The bands are 0.1 wide and stop at the gate's own threshold, so the index of
# a score is p/0.1 -- NOT p*len(BANDS), which is the same thing only when the
# bands happen to span the whole 0..1 and silently buckets by 0.2 here.
BAND_W = BANDS[0][1] - BANDS[0][0]


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
    """Every rejected box, with the geometry to cut it and the sequence it
    belongs to.

    One pass, kept on disk, because the alternative is re-joining 4.7M rows
    against a 32.5M-row manifest on every page of the audit.
    """
    import duckdb
    os.makedirs(OUT_DIR, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory}'")
    shards = os.path.join(GATE, 'gate-*.parquet')
    work = os.path.join(GATE, 'work.parquet')
    if not glob.glob(shards):
        raise SystemExit('no gate shards -- has the gate run?')

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
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE rejected AS
        SELECT s.image_id, s.det_idx, s.p_dog
        FROM read_parquet('{shards}') s
        WHERE s.label = 'not_dog'
    """)
    con.execute("CREATE OR REPLACE TEMP TABLE need AS "
                "SELECT DISTINCT image_id FROM rejected")
    need = con.execute("SELECT count(*) FROM need").fetchone()[0]
    print(f'{need:,} frames carry a rejected box', flush=True)

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
        ) TO '{POOL}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n, seqs, unmatched = con.execute(f"""
        SELECT count(*), count(DISTINCT seq),
               sum(CASE WHEN seq LIKE 'img:%' THEN 1 ELSE 0 END)
        FROM read_parquet('{POOL}')""").fetchone()
    print(f'{n:,} rejected boxes across {seqs:,} sequences '
          f'({unmatched:,} had no sequence and stand alone)')
    for i, (lo, hi) in enumerate(BANDS):
        c = con.execute(f"SELECT count(*), count(DISTINCT seq) "
                        f"FROM read_parquet('{POOL}') WHERE band = {i}"
                        ).fetchone()
        print(f'  p_dog {lo:.1f}-{hi:.1f}  {c[0]:>10,} boxes  '
              f'{c[1]:>9,} sequences')
    return 0


def band_totals():
    """[(lo, hi, boxes)] -- the weights the headline rate needs."""
    try:
        import duckdb
        rows = duckdb.connect().execute(
            f"SELECT band, count(*) FROM read_parquet('{POOL}') "
            f"GROUP BY 1 ORDER BY 1").fetchall()
    except Exception:
        return []
    got = dict(rows)
    return [(lo, hi, int(got.get(i, 0))) for i, (lo, hi) in enumerate(BANDS)]


def read_verdicts(path=None):
    """[{key, verdict, band, p_dog, ts}] -- append-only, last write wins.

    Keyed on image_id#det_idx so a crop judged twice (reloaded page, changed
    mind) counts once, as its latest answer.
    """
    out = {}
    try:
        with open(path or VERDICTS) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict) and d.get('key'):
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


def summarise(verdicts=None, totals=None):
    """Per band and overall. The overall rate weights each band by how many
    boxes the gate actually put in it -- a flat mean over the bands would
    report the near-threshold error rate as if it were the whole store's."""
    vs = read_verdicts() if verdicts is None else verdicts
    totals = band_totals() if totals is None else totals
    per = []
    for i, (lo, hi) in enumerate(BANDS):
        seen = [v for v in vs if v.get('band') == i
                and v.get('verdict') in ('missed', 'correct')]
        k = sum(1 for v in seen if v['verdict'] == 'missed')
        p, a, b = wilson(k, len(seen))
        boxes = totals[i][2] if i < len(totals) else 0
        per.append({'lo': lo, 'hi': hi, 'judged': len(seen), 'missed': k,
                    'rate': p, 'lo95': a, 'hi95': b, 'boxes': boxes})
    pop = sum(b['boxes'] for b in per) or 1
    # bands with nothing judged contribute nothing and are excluded from the
    # weight, so the headline says what HAS been measured rather than assuming
    # zero for what has not
    seen_pop = sum(b['boxes'] for b in per if b['judged'])
    rate = (sum(b['rate'] * b['boxes'] for b in per if b['judged'])
            / seen_pop) if seen_pop else 0.0
    return {'bands': per, 'weighted_rate': rate,
            'covered': seen_pop / pop if pop else 0.0,
            'judged': sum(b['judged'] for b in per),
            'missed': sum(b['missed'] for b in per),
            'pool': pop}


DATASET = os.path.join(REPO, 'data', 'audit_finds')
CLASS_OF = {'missed': 'dog', 'correct': 'not_dog'}
README = """# audit_finds

Crops from the false-negative audit of the dog-bin gate, laid out for
`yolo classify` so they can be folded into a dog-bin dataset the same way
`data/hard_negatives` and `data/hard_positives` are:

    dog/<image_id>_<det_idx>.jpg       a human said the gate threw a dog away
    not_dog/<image_id>_<det_idx>.jpg   a human confirmed the rejection
    manifest.jsonl                     one line per crop, with its sequence

Every file here carries a HUMAN verdict. The model's own label is what put
the box in front of someone; it is never what is written down.

## Read this before training on it

**This is not a random sample of anything.** Every crop was rejected by
{model}, and the audit draws evenly from five p_dog bands rather than in
proportion to how many boxes each holds -- so the near-threshold cases are
massively over-represented relative to the store. That is deliberate: it is
where the model is wrong, and it is what makes these worth training on. It
also means class balance here says nothing about the store's, and accuracy
measured on a split of these says nothing about accuracy in production.

**Split on `sequence`, never on `image_id`.** The manifest carries it for
every crop. Splitting per image looked clean once before and put 70.8% of a
val set in a sequence that was also in train.

Rebuild at any time with:

    python tools/detect/fn_audit.py export
"""


def export(args):
    """Write the dataset from the ledger. Idempotent.

    The dashboard already files each crop as it is judged; this rebuilds the
    whole thing from the record -- after the ledger is edited by hand, after
    crops are re-cut, or just to be sure the two agree. Anything in a class
    directory that the ledger does not still say belongs there is removed,
    because a dataset that keeps a label nobody stands behind is worse than
    one that is a rebuild out of date.
    """
    import shutil
    full = os.path.join(OUT_DIR, 'full')
    vs = read_verdicts()
    keep = {}
    for v in vs:
        cls = CLASS_OF.get(v.get('verdict'))
        if cls:
            keep[str(v['key']).replace('#', '_') + '.jpg'] = (cls, v)
    placed = missing = removed = 0
    for cls in ('dog', 'not_dog'):
        d = os.path.join(DATASET, cls)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if keep.get(f, (None,))[0] != cls:
                os.remove(os.path.join(d, f))
                removed += 1
    for name, (cls, v) in sorted(keep.items()):
        src, dst = os.path.join(full, name), os.path.join(DATASET, cls, name)
        if not os.path.exists(src):
            missing += 1
            continue
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        placed += 1
    with open(os.path.join(DATASET, 'manifest.jsonl'), 'w') as fh:
        for name, (cls, v) in sorted(keep.items()):
            iid, _, di = name[:-4].rpartition('_')
            fh.write(json.dumps({
                'file': f'{cls}/{name}', 'label': cls,
                'image_id': iid, 'det_idx': int(di) if di.isdigit() else None,
                # the split column. Never image_id -- see the README.
                'sequence': v.get('seq'),
                'verdict': v.get('verdict'), 'p_dog': v.get('p_dog'),
                'band': v.get('band'), 'judged_at': v.get('ts'),
                'rejected_by': args.model}) + '\n')
    with open(os.path.join(DATASET, 'README.md'), 'w') as fh:
        fh.write(README.format(model=args.model))
    n_dog = sum(1 for name, (c, _) in keep.items() if c == 'dog'
                and os.path.exists(os.path.join(DATASET, 'dog', name)))
    print(f'{placed:,} crops in {os.path.relpath(DATASET, REPO)} '
          f'({n_dog:,} dog, {placed - n_dog:,} not_dog)')
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
    s = summarise()
    if not s['judged']:
        print('nothing judged yet')
        return 0
    print(f"{s['judged']:,} boxes judged, {s['missed']:,} were dogs the gate "
          f"threw away")
    for b in s['bands']:
        if not b['judged']:
            print(f"  p_dog {b['lo']:.1f}-{b['hi']:.1f}  "
                  f"{b['boxes']:>10,} boxes   not sampled")
            continue
        print(f"  p_dog {b['lo']:.1f}-{b['hi']:.1f}  {b['boxes']:>10,} boxes   "
              f"{b['missed']:>4}/{b['judged']:<4} missed  "
              f"{b['rate']:6.1%}  [{b['lo95']:.1%}, {b['hi95']:.1%}]")
    print(f"\nweighted false-negative rate: {s['weighted_rate']:.2%} "
          f"of the {s['pool']:,} rejected boxes "
          f"(~{int(s['weighted_rate'] * s['pool']):,} dogs)")
    print(f"bands covering {s['covered']:.0%} of the pool have been sampled")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.set_defaults(fn=build)
    b.add_argument('--memory', default='8GB')
    b.add_argument('--tmp', default='/tmp/fn_audit_spill',
                   help='where duckdb may spill; needs room')
    e = sub.add_parser('export'); e.set_defaults(fn=export)
    e.add_argument('--model', default='dogbin_008',
                   help='which gate these crops were rejected by; recorded '
                        'on every manifest row')
    s = sub.add_parser('stats'); s.set_defaults(fn=stats)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
