#!/usr/bin/env python3
"""Record which model produced a sweep run, and where its training lives.

The predictions store had no answer to "which detector drew this box". It
carries run_id, but that is a uint16 assigned per PROCESS -- gen=0001 already
holds eleven of them from restarts, all assumed to be yolo26x_train30 because
nothing says otherwise. The moment a second detector exists, every row in the
store becomes ambiguous retroactively.

Two records fix that, and they answer different questions:

  model_sha8 on each row  -- "filter this store to one model" (store.py)
  a run manifest here     -- "what WAS that model, and where did it come from"

The manifest is per (gen, run_id) and holds the engine's full sha256, its
path, the inference settings that change what a box means (conf, iou,
max_det, imgsz), the library versions, and the COMET EXPERIMENT KEY of the
training run behind the engine -- so a box in the store links back to the
curve, the dataset and the hyperparameters that produced the model.

    # write one for a run that is starting (or already running)
    python tools/detect/run_manifest.py write --run-id 56381 --gen 1

    # what produced these rows?
    python tools/detect/run_manifest.py show

Read-only on the store. Writes only data/detect/runs/gen=NNNN/.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def runs_dir(repo, gen):
    sys.path.insert(0, os.path.join(repo, 'tools', 'detect'))
    import store as _store
    return os.path.join(_store.get_detect_root(), 'runs', f'gen={gen:04d}')


def comet_key_for(engine_sha256, repo):
    """The training run behind this engine, from data/best_models.json.

    Matched on the engine digest rather than the run name: a name can be
    reused, a digest cannot. Returns (key, run, project) or (None, None, None)
    when the registry has never seen this engine -- which is itself worth
    knowing, because it means an unregistered model is writing to the store.
    """
    try:
        with open(os.path.join(repo, 'data', 'best_models.json')) as fh:
            reg = json.load(fh)
    except (OSError, ValueError):
        return (None, None, None)
    for proj, d in (reg.get('projects') or {}).items():
        for rec in [d.get('best')] + list(d.get('candidates') or []):
            if rec and rec.get('sha256_engine') == engine_sha256:
                return (rec.get('key'), rec.get('run'), proj)
    return (None, None, None)


def write_for_run(repo, gen, run_id, cfg, engine=None, force=True,
                  started=None):
    """Entry point for sweep.py -- same record, no argparse round trip."""
    ns = argparse.Namespace(repo=repo, gen=gen, run_id=run_id,
                            engine=engine or cfg.get('engine'), force=force,
                            started=started, _cfg=cfg)
    return cmd_write(ns)


def manifests_for(root, gen=None):
    """{(gen, run_id): [doc, ...]} -- a LIST, because run_id is not unique.

    run_id is ``time.time() & 0xFFFF``, so it wraps every 18.2 hours. Two runs
    a day apart can share one. Keeping a single doc per key would let the
    second run silently claim the first run's rows, which is worse than
    admitting the ambiguity.
    """
    pat = os.path.join(root, 'runs',
                       f'gen={gen:04d}' if gen else 'gen=*', 'run_*.json')
    out = {}
    for f in sorted(glob.glob(pat)):
        try:
            with open(f) as fh:
                d = json.load(fh)
            out.setdefault((int(d['gen']), int(d['run_id'])), []).append(d)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return out


def cmd_write(args):
    cfg = getattr(args, '_cfg', None)
    if cfg is None:
        sys.path.insert(0, os.path.join(args.repo, 'tools', 'detect'))
        import sweep as _sweep
        cfg = _sweep.load_cfg()
    engine = args.engine or cfg.get('engine')
    if not engine or not os.path.exists(engine):
        raise SystemExit(f'engine not found: {engine}')
    digest = sha256_of(engine)
    key, run, proj = comet_key_for(digest, args.repo)
    if not key:
        print(f'WARNING: no entry in data/best_models.json has '
              f'sha256_engine={digest[:16]}... -- this run is writing boxes '
              f'from a model the registry does not know about', file=sys.stderr)

    vers = {}
    for mod in ('ultralytics', 'torch', 'tensorrt'):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = None

    # ts_off on every store row is seconds since the sweep's own start, and
    # until now nothing recorded what that start WAS -- making every timestamp
    # in the store uninterpretable in absolute terms. It also separates two
    # runs that share a wrapped run_id.
    started = float(args.started) if args.started else time.time()
    doc = {
        'gen': args.gen,
        'run_id': args.run_id,
        'run_started_epoch': int(started),
        'run_started': time.strftime('%Y-%m-%d %H:%M:%S',
                                     time.localtime(started)),
        'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': {
            # relative, so the record is not machine-specific
            'engine': os.path.relpath(engine, args.repo),
            'sha256': digest,
            'sha8': digest[:8],
            'comet_key': key,
            'comet_run': run,
            'comet_project': proj,
        },
        # these change what a box MEANS, so they belong with the model
        'inference': {k: cfg.get(k) for k in
                      ('conf', 'iou', 'max_det', 'imgsz')},
        'versions': vers,
    }
    d = runs_dir(args.repo, args.gen)
    os.makedirs(d, exist_ok=True)
    # The start time is in the NAME, not just the body: two runs 18.2 hours
    # apart share a run_id, and a bare run_<id>.json would have the second
    # overwrite the first -- retroactively reassigning millions of already
    # written rows to a model that never saw them.
    stamp = time.strftime('%Y%m%dT%H%M%S', time.localtime(started))
    out = os.path.join(d, f'run_{args.run_id}_{stamp}.json')
    if os.path.exists(out) and not args.force:
        raise SystemExit(f'{out} exists; pass --force to overwrite')
    tmp = out + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, out)
    print(f'gen={args.gen} run_id={args.run_id}')
    print(f'  started   {doc["run_started"]}')
    print(f'  engine    {doc["model"]["engine"]}')
    print(f'  sha8      {doc["model"]["sha8"]}')
    print(f'  comet     {run or "(unregistered)"} '
          f'{("[" + key + "]") if key else ""}')
    print(f'  inference {doc["inference"]}')
    print(f'-> {out}')
    return 0


def cmd_show(args):
    import duckdb
    sys.path.insert(0, os.path.join(args.repo, 'tools', 'detect'))
    import store as _store
    root = _store.get_detect_root()
    src = _store._sql_src(_store._store_globs(root, 'img'))
    con = duckdb.connect()
    # The column exists in the SCHEMA but not necessarily in any FILE yet:
    # union_by_name can only union what has been written, so until a run
    # writes with the new schema, selecting it is a binder error rather than
    # a column of NULLs. Probe instead of assuming.
    cols = {r[0] for r in con.execute(
        f'DESCRIBE SELECT * FROM {src}').fetchall()}
    has = 'model_sha8' in cols
    sha = 'model_sha8' if has else "CAST(NULL AS VARCHAR)"
    # Grouped by the DIGEST as well as the run_id, so a run_id that wrapped
    # onto two different models shows as two lines instead of one wrong one.
    rows = con.execute(
        f'SELECT gen, run_id, {sha} AS sha8, count(*) n '
        f'FROM {src} GROUP BY 1, 2, 3 ORDER BY n DESC').fetchall()
    if not has:
        print('note: no file carries model_sha8 yet -- these rows predate the '
              'column and are attributable only through a manifest\n')

    man = manifests_for(root)
    print(f"{'gen':>5}{'run_id':>8}{'sha8':>10}{'rows':>12}  model")
    amb = 0
    for gen, rid, sha8, n in rows:
        docs = man.get((int(gen), int(rid)), [])
        if sha8:  # the row says which model; the manifest only adds the name
            docs = [d for d in docs if d['model']['sha8'] == sha8] or docs
        if len(docs) > 1:
            amb += 1
            who = (f'AMBIGUOUS -- {len(docs)} runs share run_id={int(rid)}: '
                   + ', '.join(f'{d["model"]["sha8"]}@{d["run_started"]}'
                               for d in docs))
        elif docs:
            m = docs[0]['model']
            who = (f'{m.get("comet_run") or "?"}'
                   f'{"  [" + m["comet_key"][:8] + "]" if m.get("comet_key") else ""}'
                   f'   started {docs[0].get("run_started", "?")}')
        else:
            who = '(no manifest -- provenance unknown)'
        print(f'{int(gen):>5}{int(rid):>8}{(sha8 or "-"):>10}{n:>12,}  {who}')
    if amb:
        print(f'\n{amb} run_id(s) are shared by more than one run. run_id is '
              f'time.time() & 0xFFFF and wraps every 18.2h; rows written with '
              f'a model_sha8 resolve themselves, older ones cannot.')
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=REPO)
    sub = ap.add_subparsers(dest='cmd', required=True)

    w = sub.add_parser('write', help='record the model behind a run')
    w.add_argument('--run-id', type=int, required=True)
    w.add_argument('--gen', type=int, default=1)
    w.add_argument('--engine', help='defaults to the sweep config engine')
    w.add_argument('--started', type=float,
                   help='unix time the run began (ts_off is relative to it). '
                        'Defaults to now, which is only right if the run is '
                        'starting; for one already running, pass its real '
                        'start: ps -o lstart= -p <pid>')
    w.add_argument('--force', action='store_true')
    w.set_defaults(func=cmd_write)

    s = sub.add_parser('show', help='what produced the rows in the store')
    s.set_defaults(func=cmd_show)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
