#!/usr/bin/env python3
"""Pick street frames for a person to label whole, aimed at the dogs the
detector does not see.

    python tools/detect/pick_label_frames.py plan  --batch small_dogs_001 --n 300
    python tools/detect/pick_label_frames.py stage --batch small_dogs_001
    python tools/detect/pick_label_frames.py list

WHY. Every positive the detector dataset has gained since dogdet_v2 is a dog
the detector already found -- the review page, the hard positives and the
gate audit all start from its own boxes. So the dogs it MISSES never get a
label, and they are nearly all small: of the 166 val dogs the Sep 23 run
missed, the median is 20 px and 72% are under 32 px, while the training set's
median dog is 86 px. The only way a missed dog gets a label is a person
drawing every dog in a whole frame. This picks the frames worth that time.

HOW. Mapillary frames come about a second apart down one road, so a dog the
gate is sure of in one frame is often still in the frame beside it -- farther
off, smaller, and very often undetected. The candidates are the frames within
--window places and --seconds of a confident dog (gate p_dog >= --p-dog) in
the SAME SEQUENCE, on which the detector drew nothing at all. Checked by eye
on the first batch: about a third show a dog where the seed stood, where a
random undetected frame almost never does. The rest come back as frames a
person has confirmed empty, which are worth having too.

WHAT IS KEPT OUT, and why:
  * any image already in a training dataset, already in Label Studio, or
    already picked by an earlier batch
  * panoramas and frames wider than --max-width: the detector trains at 1280,
    so a small dog in a squashed panorama is a label it cannot learn from
  * every SEQUENCE holding a val, holdout or acceptance-set frame of the
    newest detector dataset. The neighbours of a val frame are near-copies of
    it; labelling them for training would leak val into train -- the defect
    measured at 70.8% in leash_binary_v1.

WHAT IT WRITES. data/label_batches/<batch>/manifest.jsonl, one line per
frame: why it was chosen, the seed dog it was chosen for, and where its jpg
is; summary.json, what was asked and what was excluded; and seeds.json, where
each frame's seed dog stood in its own frame, which the dashboard's Annotate
page draws as a hint and never as a label. A batch is never rebuilt over.
`seeds` writes seeds.json for a batch planned before it existed. `stage` copies the jpgs into the Label Studio intake directory,
<label_studio_dir>/new_images, with each file's mtime set to its capture time
-- which is what create_new_tasks.ipynb turns into image_timestamp. Nothing
here uploads anything or touches Label Studio; those steps stay manual.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BATCHES = os.path.join(REPO, 'data', 'label_batches')
CATALOG = os.path.join(REPO, 'data', 'catalog.duckdb')
GATE = os.path.join(REPO, 'data', 'gate', 'gate-*.parquet')
DET = os.path.join(REPO, 'data', 'detect', 'shards', '**', '*.det.parquet')
IMG = os.path.join(REPO, 'data', 'detect', 'shards', '**', '*.img.parquet')

# a Mapillary image id, wherever it sits in a file name
ID_RE = re.compile(r'(?<!\d)(\d{12,17})(?!\d)')
BATCH_RE = re.compile(r'^[a-z0-9][a-z0-9_]{2,60}$')


def _config():
    try:
        with open(
                os.path.join(REPO, 'tools', 'dashboard',
                             'dashboard.config.json')) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def training_root():
    """Where the datasets live -- a separate repo, never assumed."""
    got = os.environ.get('TRAINING_ROOT') or _config().get('training_root')
    if not got:
        raise SystemExit('no training root: set $TRAINING_ROOT, or '
                         'training_root in tools/dashboard/'
                         'dashboard.config.json')
    return got


def label_studio_dir(override=None):
    """The Label Studio intake directory: new_images/, images_to_add.csv,
    uploaded_to_wasabi/ and the notebook that joins them."""
    got = (override or os.environ.get('LABEL_STUDIO_DIR')
           or _config().get('label_studio_dir'))
    if not got:
        raise SystemExit('no Label Studio directory: pass --ls-dir, set '
                         '$LABEL_STUDIO_DIR, or label_studio_dir in '
                         'tools/dashboard/dashboard.config.json')
    if not os.path.isdir(got):
        raise SystemExit('Label Studio directory %s does not exist' % (got, ))
    return got


def _gate_is_gen1():
    """Refuse a gate store of another detector generation. Every join here
    pairs its det_idx with DET/IMG, the gen-1 detect store, and det_idx
    numbers different boxes in each generation -- a gen-2 gate would hand
    back wrong seed widths and hint boxes, and nothing would error."""
    try:
        with open(os.path.join(os.path.dirname(GATE), 'plan.json')) as fh:
            gen = int(json.load(fh).get('gen') or 1)
    except FileNotFoundError:
        return
    except (OSError, ValueError, TypeError, AttributeError):
        gen = None
    if gen != 1:
        raise SystemExit(
            '%s/plan.json is gen %s, not 1: its det_idx do not match the '
            'gen-1 detect store this joins to -- refusing' %
            (os.path.dirname(GATE), gen))


def _duckdb():
    try:
        import duckdb
    except ImportError:
        raise SystemExit('duckdb missing -- run this under the mp14 env')
    return duckdb


def newest_detector_dataset(root):
    """The newest built detector dataset: the one whose val is the yardstick
    every candidate run is measured on, so its sequences must stay out."""
    got = [
        d for d in glob.glob(os.path.join(root, 'dogdet_*'))
        if os.path.isfile(os.path.join(d, 'dataset.yaml'))
    ]
    if not got:
        raise SystemExit('no built detector dataset under %s' % (root, ))
    return max(got, key=os.path.getmtime)


def ids_in(paths):
    out = set()
    for p in paths:
        out.update(ID_RE.findall(os.path.basename(p)))
    return out


def excluded_ids(root, ls_dir):
    """Every image that must not be offered again, by where it already is."""
    got = {}
    got['datasets'] = ids_in(
        glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True))
    ls = ids_in(
        os.listdir(os.path.join(ls_dir, 'uploaded_to_wasabi')) if os.path.
        isdir(os.path.join(ls_dir, 'uploaded_to_wasabi')) else [])
    ls |= ids_in(
        os.listdir(os.path.join(ls_dir, 'new_images')) if os.path.
        isdir(os.path.join(ls_dir, 'new_images')) else [])
    csv_path = os.path.join(ls_dir, 'images_to_add.csv')
    if os.path.isfile(csv_path):
        with open(csv_path) as fh:
            ls |= set(ID_RE.findall(fh.read()))
    for f in glob.glob(os.path.join(root, 'project-*-at-*.json')):
        with open(f) as fh:
            ls |= set(ID_RE.findall(fh.read()))
    got['label_studio'] = ls
    prev = set()
    for m in glob.glob(os.path.join(BATCHES, '*', 'manifest.jsonl')):
        with open(m) as fh:
            prev |= {json.loads(l)['image_id'] for l in fh if l.strip()}
    got['earlier_batches'] = prev
    acc = set()
    for f in glob.glob(os.path.join(REPO, 'data', '*acceptance_set.json')):
        with open(f) as fh:
            acc |= set(ID_RE.findall(fh.read()))
    got['acceptance_sets'] = acc
    return got


def select(con, n, batch, window=3, seconds=5.0, max_width=4096, per_cell=1):
    """The pick, from tables already loaded on `con`:

      meta(image_id, seq, ts, pano, ctype)       every harvested frame
      img(image_id, n_det, w, h, cell, drive)    what the sweep saw
      seeds(image_id, bw)                        frames holding a sure dog,
                                                 widest such dog in pixels
      excl(image_id)                             never offer these
      held_seq(seq)                              never offer these sequences

    Kept pure -- nothing read from disk -- so a guard can build the tables by
    hand and know exactly what should come back. The order within a cell and
    across the batch is a hash of the image id salted with the batch name:
    the same batch name always picks the same frames, a new one picks afresh,
    and no id range or drive is favoured.
    """
    con.execute("""
        create or replace temp table ranked as
        select image_id, seq, ts,
               row_number() over (partition by seq order by ts, image_id) rk
        from meta where seq is not null and ts is not null""")
    con.execute(
        """
        create or replace temp table cand as
        select n.image_id, n.seq, s.image_id seed_id,
               abs(n.ts - s.ts) dt, abs(n.rk - s.rk) drank,
               sd.bw seed_px, sd.bw / nullif(i2.w, 0) seed_rel,
               i.w, i.h, i.cell, i.drive
        from seeds sd
        join ranked s on s.image_id = sd.image_id
        join ranked n on n.seq = s.seq
                     and n.rk between s.rk - $win and s.rk + $win
                     and n.rk <> s.rk
        join img i on i.image_id = n.image_id and i.n_det = 0
        join img i2 on i2.image_id = s.image_id
        join meta mn on mn.image_id = n.image_id
        where abs(n.ts - s.ts) <= $sec
          and i.w <= $maxw
          and not coalesce(mn.pano, false)
          and coalesce(mn.ctype, '') not in ('spherical', 'equirectangular')
          and n.image_id not in (select image_id from excl)
          and n.seq not in (select seq from held_seq)""", {
            'win': int(window),
            'sec': float(seconds),
            'maxw': int(max_width)
        })
    rows = con.execute(
        """
        with one as (
            select * from (select *, row_number() over (
                partition by seq order by dt, drank,
                hash(image_id || $salt)) r from cand) where r = 1),
        spread as (
            select *, row_number() over (
                partition by cell order by hash(image_id || $salt)) rc
            from one)
        select image_id, seq, seed_id, dt, drank, seed_px, seed_rel,
               w, h, cell, drive
        from spread where rc <= $per
        order by hash(image_id || $salt) limit $n""", {
            'salt': batch,
            'per': int(per_cell),
            'n': int(n)
        }).fetchall()
    total = con.execute(
        'select count(distinct image_id), count(distinct seq) from cand'
    ).fetchone()
    cols = ('image_id', 'seq', 'seed_id', 'dt', 'drank', 'seed_px', 'seed_rel',
            'w', 'h', 'cell', 'drive')
    return [dict(zip(cols, r)) for r in rows], {
        'candidate_frames': total[0],
        'candidate_sequences': total[1]
    }


def _load(con, p_dog):
    _gate_is_gen1()
    cat = _duckdb().connect(CATALOG, read_only=True)
    ga = [
        r[0] for r in cat.execute(
            "select f.path from files f join drives d on d.drive = f.drive "
            "and d.online where f.kind = 'ground_animals'").fetchall()
    ]
    roots = dict(
        cat.execute(
            'select drive, base_path from drives where online').fetchall())
    cat.close()
    if not ga:
        raise SystemExit('the catalog knows no ground_animals parquets')
    con.execute(
        """create table meta as select image_id::varchar image_id,
             any_value(sequence) seq,
             any_value(epoch(try_cast(captured_at as timestamp))) ts,
             any_value(is_pano) pano, any_value(camera_type) ctype
           from read_parquet($p) where image_id is not null group by 1""",
        {'p': ga})
    con.execute(f"""create table img as select image_id::varchar image_id,
             any_value(n_det) n_det, any_value(orig_w) w,
             any_value(orig_h) h, any_value(cell) cell,
             any_value(drive) drive
           from read_parquet('{IMG}', hive_partitioning = true)
           group by 1""")
    con.execute(
        f"""create table seeds as
           select g.image_id::varchar image_id,
                  max((d.x2 - d.x1)::double) bw
           from read_parquet('{GATE}') g
           join read_parquet('{DET}', hive_partitioning = true) d
             on d.image_id::varchar = g.image_id::varchar
            and d.det_idx = g.det_idx
           where g.label = 'dog' and g.p_dog >= $p
           group by 1""", {'p': float(p_dog)})
    return roots


def _git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'],
                              cwd=REPO,
                              capture_output=True,
                              text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ''


def plan(args):
    if not BATCH_RE.match(args.batch):
        raise SystemExit('--batch: lowercase letters, digits and _ only')
    out = os.path.join(BATCHES, args.batch)
    if os.path.exists(out):
        raise SystemExit('%s already exists -- a batch is never rebuilt '
                         'over; pick a new name' % (out, ))
    root = training_root()
    ls_dir = label_studio_dir(args.ls_dir)
    val_ds = args.val_dataset or newest_detector_dataset(root)
    t0 = time.time()
    duckdb = _duckdb()
    con = duckdb.connect()
    con.execute('set threads = 16')
    roots = _load(con, args.p_dog)
    print('loaded %s frames, %s sure dogs in %.0fs' %
          (f"{con.execute('select count(*) from meta').fetchone()[0]:,}",
           f"{con.execute('select count(*) from seeds').fetchone()[0]:,}",
           time.time() - t0),
          flush=True)

    ex = excluded_ids(root, ls_dir)
    every = set().union(*ex.values())
    con.execute('create table excl as select unnest($e) image_id',
                {'e': sorted(every)})
    held = ids_in(glob.glob(os.path.join(val_ds, 'images', 'val', '*')))
    held |= ex['acceptance_sets']
    con.execute(
        'create table held_seq as select distinct seq from meta '
        'where seq is not null and image_id in (select unnest($h))',
        {'h': sorted(held)})
    n_held = con.execute('select count(*) from held_seq').fetchone()[0]

    picks, stats = select(con, args.n, args.batch, args.window, args.seconds,
                          args.max_width, args.per_cell)
    ts = dict(
        con.execute(
            'select image_id, ts from meta where image_id in '
            '(select unnest($i))', {
                'i': [p['image_id'] for p in picks]
            }).fetchall())
    for p in picks:
        root_dir = roots.get(p['drive'])
        p['jpg'] = (os.path.join(root_dir, p['cell'], 'ground_animal_images',
                                 p['image_id'] + '.jpg') if root_dir else '')
        p['captured'] = ts.get(p['image_id'])
        p['why'] = 'undetected neighbour of a sure dog'
    missing = [p for p in picks if not os.path.isfile(p['jpg'])]
    if missing:
        raise SystemExit('%d picked frame(s) have no jpg on disk (first: %s) '
                         '-- is a drive unmounted?' %
                         (len(missing), missing[0]['image_id']))

    os.makedirs(out)
    with open(os.path.join(out, 'manifest.jsonl'), 'w') as fh:
        for p in picks:
            fh.write(json.dumps(p) + '\n')
    summary = {
        'kind': 'neighbours',
        'batch': args.batch,
        'planned_at': int(time.time()),
        'asked': args.n,
        'picked': len(picks),
        'params': {
            'p_dog': args.p_dog,
            'window': args.window,
            'seconds': args.seconds,
            'max_width': args.max_width,
            'per_cell': args.per_cell
        },
        'val_dataset': os.path.basename(val_ds),
        'held_out_sequences': n_held,
        'excluded_ids': {
            k: len(v)
            for k, v in ex.items()
        },
        'candidates': stats,
        'git': _git_head(),
        'command': sys.argv
    }
    with open(os.path.join(out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    try:
        write_seeds(out, picks, args.p_dog)
    except Exception as e:  # noqa: BLE001 - the batch stands without hints
        print('no seeds.json (%s) -- the Annotate page will show no hints; '
              'retry with: pick_label_frames.py seeds --batch %s' %
              (e, args.batch))
    cells = len({p['cell'] for p in picks})
    print('picked %d of %s candidate frames (%s sequences), across %d cells' %
          (len(picks), f"{stats['candidate_frames']:,}",
           f"{stats['candidate_sequences']:,}", cells))
    print('held out %d sequences touching %s val / acceptance frames' %
          (n_held, os.path.basename(val_ds)))
    print('wrote %s' % (out, ))
    print('next: python tools/detect/pick_label_frames.py stage --batch %s' %
          (args.batch, ))


def seed_boxes(picks, p_dog):
    """{picked image_id: {'seed_id', 'box', 'w', 'h'}} -- the seed dog each
    frame was chosen for, as a box in the SEED frame's pixels with that
    frame's size. The widest sure dog, the one select() measured."""
    ids = sorted({str(p['seed_id']) for p in picks if p.get('seed_id')})
    if not ids:
        return {}
    _gate_is_gen1()
    con = _duckdb().connect()
    box = dict(
        con.execute(
            f"""select g.image_id::varchar,
                      arg_max([d.x1, d.y1, d.x2, d.y2], d.x2 - d.x1)
               from read_parquet('{GATE}') g
               join read_parquet('{DET}', hive_partitioning = true) d
                 on d.image_id::varchar = g.image_id::varchar
                and d.det_idx = g.det_idx
               where g.label = 'dog' and g.p_dog >= $p
                 and g.image_id::varchar in (select unnest($i))
               group by 1""", {
                'p': float(p_dog),
                'i': ids
            }).fetchall())
    size = {
        r[0]: (r[1], r[2])
        for r in con.execute(
            f"""select image_id::varchar, any_value(orig_w),
                      any_value(orig_h)
               from read_parquet('{IMG}', hive_partitioning = true)
               where image_id::varchar in (select unnest($i))
               group by 1""", {
                'i': ids
            }).fetchall()
    }
    con.close()
    out = {}
    for p in picks:
        sid = str(p.get('seed_id') or '')
        b, wh = box.get(sid), size.get(sid)
        if b and wh and wh[0] and wh[1]:
            out[p['image_id']] = {
                'seed_id': sid,
                'box': [round(float(v), 1) for v in b],
                'w': int(wh[0]),
                'h': int(wh[1])
            }
    return out


def write_seeds(out, picks, p_dog):
    got = seed_boxes(picks, p_dog)
    path = os.path.join(out, 'seeds.json')
    with open(path + '.part', 'w') as fh:
        json.dump(got, fh)
    os.replace(path + '.part', path)
    return len(got)


def seeds(args):
    """seeds.json for a batch that has none, or has a stale one. A hint, not
    a label, so rewriting it changes nothing anybody decided."""
    out = os.path.join(BATCHES, args.batch)
    man = os.path.join(out, 'manifest.jsonl')
    if not os.path.isfile(man):
        raise SystemExit('no batch %s -- run plan first' % (args.batch, ))
    with open(man) as fh:
        picks = [json.loads(l) for l in fh if l.strip()]
    p_dog = args.p_dog
    if p_dog is None:
        try:
            with open(os.path.join(out, 'summary.json')) as fh:
                p_dog = json.load(fh)['params']['p_dog']
        except (OSError, ValueError, KeyError):
            p_dog = 0.9
    n = write_seeds(out, picks, p_dog)
    print('wrote seeds.json: a hint for %d of %d frames' % (n, len(picks)))


def stage(args):
    out = os.path.join(BATCHES, args.batch)
    man = os.path.join(out, 'manifest.jsonl')
    if not os.path.isfile(man):
        raise SystemExit('no batch %s -- run plan first' % (args.batch, ))
    try:
        with open(os.path.join(out, 'summary.json')) as fh:
            use = (json.load(fh) or {}).get('use')
    except (OSError, ValueError, AttributeError):
        use = None
    if use == 'test':
        raise SystemExit('%s is a test set (summary.json use: test) -- its '
                         'frames are for measuring the detector and must '
                         'never reach Label Studio, where they would become '
                         'training data' % (args.batch, ))
    with open(man) as fh:
        picks = [json.loads(l) for l in fh if l.strip()]
    dst = os.path.join(label_studio_dir(args.ls_dir), 'new_images')
    os.makedirs(dst, exist_ok=True)
    clash = [
        p for p in picks
        if os.path.exists(os.path.join(dst, p['image_id'] + '.jpg'))
    ]
    if clash:
        raise SystemExit('%d of these are already in %s (first: %s) -- '
                         'nothing was copied' %
                         (len(clash), dst, clash[0]['image_id']))
    need = sum(os.path.getsize(p['jpg']) for p in picks)
    free = shutil.disk_usage(dst).free
    if need * 1.2 > free:
        raise SystemExit('need %.1f GB in %s, %.1f GB free' %
                         (need / 1e9, dst, free / 1e9))
    done = 0
    for p in picks:
        to = os.path.join(dst, p['image_id'] + '.jpg')
        tmp = to + '.part'
        shutil.copyfile(p['jpg'], tmp)
        # the notebook reads image_timestamp from mtime, and every earlier
        # task carries the CAPTURE time there, not the copy time
        if p.get('captured'):
            os.utime(tmp, (p['captured'], p['captured']))
        os.replace(tmp, to)
        done += 1
    with open(os.path.join(out, 'staged.json'), 'w') as fh:
        json.dump({'staged_at': int(time.time()), 'to': dst, 'n': done}, fh)
    print('staged %d frames (%.1f MB) into %s' % (done, need / 1e6, dst))
    print('next, by hand: run create_new_tasks.ipynb there, upload '
          'uploaded_to_wasabi/, import images_to_add.csv into Label Studio')


def list_batches(args):
    for d in sorted(glob.glob(os.path.join(BATCHES, '*'))):
        try:
            with open(os.path.join(d, 'summary.json')) as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            continue
        staged = os.path.isfile(os.path.join(d, 'staged.json'))
        print('%-24s %4d frames  %s' %
              (s['batch'], s['picked'], 'staged' if staged else 'not staged'))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan', help='choose the frames and write a manifest')
    p.add_argument('--batch', required=True)
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--p-dog',
                   type=float,
                   default=0.9,
                   help='how sure the gate must be of a seed dog')
    p.add_argument('--window',
                   type=int,
                   default=3,
                   help='frames either side of the seed')
    p.add_argument('--seconds',
                   type=float,
                   default=5.0,
                   help='and no further apart in capture time than this')
    p.add_argument('--max-width', type=int, default=4096)
    p.add_argument('--per-cell', type=int, default=1)
    p.add_argument('--ls-dir')
    p.add_argument('--val-dataset',
                   help='the dataset whose val sequences stay out '
                   '(default: the newest dogdet_*)')
    p.set_defaults(fn=plan)
    h = sub.add_parser('seeds',
                       help="write a batch's seeds.json (the Annotate "
                       "page's hints)")
    h.add_argument('--batch', required=True)
    h.add_argument('--p-dog',
                   type=float,
                   default=None,
                   help="default: the batch's own, from summary.json")
    h.set_defaults(fn=seeds)
    s = sub.add_parser('stage', help='copy a batch into new_images/')
    s.add_argument('--batch', required=True)
    s.add_argument('--ls-dir')
    s.set_defaults(fn=stage)
    l = sub.add_parser('list', help='the batches so far')
    l.set_defaults(fn=list_batches)
    args = ap.parse_args(argv)
    if getattr(args, 'n', 1) < 1:
        raise SystemExit('--n must be at least 1')
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
