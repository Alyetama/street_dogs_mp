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


SCHEMA = 2

# How a manifest came to say what it says. Required, no default: a reader must
# never have to assume, and an ABSENT class is "unknown", never "measured".
MEASURED_AT_START = 'measured_at_start'
# the sweep hashed the engine it was about to load, before its first box
MEASURED_LATE = 'measured_after_the_fact'
# something hashed whatever engine was on disk later -- that measures the file
# on the day of writing, not the process that ran
ATTESTED = 'attested'
# a person states which model ran. Nothing about the run was measured.

BASIS = {MEASURED_AT_START: 'measured', MEASURED_LATE: 'hashed-late',
         ATTESTED: 'ATTESTED', None: 'unknown'}


def write_for_run(repo, gen, run_id, cfg, engine=None, force=True,
                  started=None):
    """Entry point for sweep.py -- same record, no argparse round trip.

    Called at run start, before a box exists, so the digest it takes is of the
    engine this run is about to load: measured_at_start.
    """
    ns = argparse.Namespace(repo=repo, gen=gen, run_id=run_id,
                            engine=engine or cfg.get('engine'), force=force,
                            started=started, at_start=True, _cfg=cfg)
    return cmd_write(ns)


def manifests_for(root, gen=None):
    """{(gen, run_id): [doc, ...]} -- a LIST, because run_id is not unique.

    run_id is ``time.time() & 0xFFFF``, so it wraps every 18.2 hours. Two runs
    a day apart can share one. Keeping a single doc per key would let the
    second run silently claim the first run's rows, which is worse than
    admitting the ambiguity.
    """
    # `if gen` would read generation 0 as "no generation given" and silently
    # widen the glob to every generation.
    pat = os.path.join(root, 'runs',
                       'gen=*' if gen is None else f'gen={gen:04d}',
                       'run_*.json')
    out = {}
    for f in sorted(glob.glob(pat)):
        try:
            with open(f) as fh:
                d = json.load(fh)
            out.setdefault((int(d['gen']), int(d['run_id'])), []).append(d)
        except (OSError, ValueError, KeyError, TypeError) as e:
            # NOT silent. A manifest truncated by a power loss would otherwise
            # vanish, and a run_id shared by two runs would go from AMBIGUOUS
            # to confidently-single with nothing on screen to say so.
            print(f'WARNING: unreadable manifest {f}: {e}', file=sys.stderr)
    return out


def derive_start(root, gen, run_id):
    """(epoch, uncertainty_s, method) for a run that recorded no start.

    run_id IS int(start) & 0xFFFF, so the start is recoverable to the second
    once it is bracketed to better than the 18.2 h wrap. Each part file was
    written at commit, and its newest row's ts_off is seconds since the run
    began, so file_mtime - max(ts_off in that file) estimates the start; the
    median over every part narrows it further, and the congruence then snaps
    it to the exact second.

    Validated against the one run whose start is known independently: run
    56381 derives to 20:09:33 against ps's 20:09:32 -- and unlike the operator
    -entered value it satisfies the run's own run_id congruence.

    Returns (None, None, reason) when it cannot be derived, never a guess.
    """
    try:
        import duckdb
        import statistics
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import store as _store
        src = _store._sql_src(_store._store_globs(root, 'img')).replace(
            'hive_partitioning=1', 'hive_partitioning=1, filename=true')
        con = duckdb.connect()
        rows = con.execute(
            f'SELECT filename, max(ts_off) FROM {src} WHERE run_id = ? '
            f'AND gen = ? GROUP BY 1', [run_id, f'{gen:04d}']).fetchall()
        con.close()
    except Exception as e:
        return (None, None, f'not derivable: {e}')
    est = []
    for f, mx in rows:
        try:
            est.append(os.path.getmtime(f) - float(mx))
        except OSError:
            pass
    if not est:
        return (None, None, 'no part files carry this run_id')
    med = statistics.median(est)
    base = int(med) - (int(med) & 0xFFFF) + run_id
    t = min((base - 65536, base, base + 65536), key=lambda c: abs(c - med))
    spread = max(est) - min(est)
    if abs(t - med) > 32768:
        return (None, None, 'estimate too loose to resolve the 18.2h wrap')
    return (int(t), max(60, int(spread)),
            f'derived: median over {len(est)} part files of '
            f'(file mtime - max ts_off in that file), snapped to the unique '
            f'second with int(t) & 0xFFFF == run_id (spread {int(spread)}s)')


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
    at_start = bool(getattr(args, 'at_start', False))
    if args.started:
        started, unc = float(args.started), 0
        src_note = 'operator-supplied via --started'
    elif at_start:
        started, unc, src_note = time.time(), 0, 'read from the clock at run start'
    else:
        started, unc, src_note = time.time(), 0, 'clock at write time'
    # run_id IS int(start) & 0xFFFF. A start that fails that congruence is not
    # this run's start, and saying so beats recording it as though it were.
    if int(started) & 0xFFFF != int(args.run_id) & 0xFFFF:
        print(f'WARNING: run_started {int(started)} & 0xFFFF = '
              f'{int(started) & 0xFFFF}, but run_id is {args.run_id} -- this '
              f'is not the second the run began', file=sys.stderr)
        src_note += ' (fails the run_id congruence)'
    doc = {
        'schema': SCHEMA,
        'provenance_class': MEASURED_AT_START if at_start else MEASURED_LATE,
        'gen': args.gen,
        'run_id': args.run_id,
        'run_started_epoch': int(started),
        'run_started': time.strftime('%Y-%m-%d %H:%M:%S',
                                     time.localtime(started)),
        'run_started_source': src_note,
        'run_started_uncertainty_s': unc,
        'engine_hashed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
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


def cmd_attest(args):
    """Record a human's statement about runs nothing measured.

    This exists because the alternative is worse in both directions. Leaving
    the runs unknown throws away the best evidence there is about 4.5M rows;
    hashing today's engine and writing the digest in would produce a record
    byte-identical to a true one whether or not it is right.

    So the statement goes in as a statement. The `model` block -- the one a
    query reaches for -- stays all-null, because nothing measured it. The claim
    lives in `attested_model`, which any reader must name explicitly and
    thereby acknowledge as a claim.
    """
    import subprocess
    root = runs_dir(args.repo, args.gen)
    os.makedirs(root, exist_ok=True)
    who = args.by
    if not who:
        try:
            n = subprocess.run(['git', 'config', 'user.name'], cwd=args.repo,
                               capture_output=True, text=True).stdout.strip()
            e = subprocess.run(['git', 'config', 'user.email'], cwd=args.repo,
                               capture_output=True, text=True).stdout.strip()
            who = f'{n} <{e}>' if n else ''
        except OSError:
            who = ''
    if not who:
        raise SystemExit('no attester: pass --by "Name <email>"')

    key, run, proj = (None, args.run, args.project)
    if args.run:
        try:
            with open(os.path.join(args.repo, 'data',
                                   'best_models.json')) as fh:
                reg = json.load(fh)
            for pname, d in (reg.get('projects') or {}).items():
                b = d.get('best') if isinstance(d, dict) else None
                if isinstance(b, dict) and b.get('run') == args.run:
                    key, proj = b.get('key'), pname
        except (OSError, ValueError, AttributeError):
            pass

    corr = {}
    if not args.no_corroborate:
        corr = cross_run_agreement(_detect_root(args.repo), args.gen)

    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    att_id = args.id or f'att-{time.strftime("%Y%m%d")}-{(args.run or "x")}'
    covered = sorted(args.run_ids)
    written = []
    for rid in covered:
        started, unc, how = derive_start(_detect_root(args.repo), args.gen, rid)
        c = corr.get(rid)
        doc = {
            'schema': SCHEMA,
            'provenance_class': ATTESTED,
            'gen': args.gen,
            'run_id': rid,
            'run_started_epoch': started,
            'run_started': (time.strftime('%Y-%m-%d %H:%M:%S',
                                          time.localtime(started))
                            if started else None),
            'run_started_source': how,
            'run_started_uncertainty_s': unc,
            'written_at': stamp,
            # RESERVED FOR MEASUREMENT. All-null, and present rather than
            # absent so a reader looking for a digest finds nothing instead of
            # something plausible.
            'model': {'engine': None, 'sha256': None, 'sha8': None,
                      'comet_key': None, 'comet_run': None,
                      'comet_project': None},
            'attested_model': {
                'comet_run': run, 'comet_project': proj, 'comet_key': key,
                'comet_key_source': ('data/best_models.json, matched on the '
                                     'ATTESTED RUN NAME; no engine digest was '
                                     'available to match on'),
                # the person knows the model; nobody knows the binary
                'engine_sha256': None, 'engine_sha8': None,
                'weights_sha256': None,
            },
            'attestation': {
                'id': att_id, 'statement': args.statement, 'by': who,
                'at': stamp, 'measured': False,
                'method': 'operator statement; no artifact of this run was '
                          'measured',
                # one statement, one id -- ten ids would make a single claim
                # look like ten independent corroborations
                'covers_run_ids': covered,
                'covers_fields': ['attested_model.comet_run',
                                  'attested_model.comet_project'],
            },
            # measured, and the only thing here that is
            'corroboration': (
                {'method': 'detections compared between runs that both '
                           'processed the same image_id, box for box',
                 'identical': c['identical'], 'differing': c['differing'],
                 'shares_with_run_ids': sorted(c['with'])} if c else
                {'method': 'detections compared between runs that both '
                           'processed the same image_id, box for box',
                 'identical': 0, 'differing': 0, 'shares_with_run_ids': [],
                 'note': 'no image in this run was processed by any other '
                         'run, so nothing corroborates it'}),
            # unknowable for a historical run. NOT today's config values:
            # the override file is untracked, and the env has since moved.
            'inference': {'conf': None, 'iou': None, 'max_det': None,
                          'imgsz': None},
            'versions': {'ultralytics': None, 'torch': None, 'tensorrt': None},
        }
        name = (f'run_{rid}_'
                + (time.strftime('%Y%m%dT%H%M%S', time.localtime(started))
                   if started else 'unknown') + '.attested.json')
        out = os.path.join(root, name)
        if os.path.exists(out) and not args.force:
            raise SystemExit(f'{out} exists; pass --force')
        tmp = out + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(doc, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
        written.append((rid, out, c))
    print(f'attested {len(written)} run(s) as {run!r}, by {who}')
    print(f'  statement: {args.statement}')
    for rid, out, c in written:
        n = (f'{c["identical"]:,} identical / {c["differing"]} differing'
             if c else 'NO corroboration')
        print(f'  run {rid:>6}  {n:<40} {os.path.basename(out)}')
    print('\nNothing about these runs was measured. The model block in each '
          'file is null;\nthe claim is in attested_model and is labelled as '
          'a claim.')
    return 0


def _detect_root(repo):
    sys.path.insert(0, os.path.join(repo, 'tools', 'detect'))
    import store as _store
    return _store.get_detect_root()


def cross_run_agreement(root, gen):
    """{run_id: {identical, differing, with}} -- measured, not asserted.

    Two runs that both processed an image_id and produced byte-identical boxes
    ran the same weights through the same graph: a different export precision
    or build moves the floats. This does not prove WHICH model, but it does
    tie the runs to each other, so a digest measured for one reaches the rest.
    """
    try:
        import duckdb
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import store as _store
        img = _store._sql_src(_store._store_globs(root, 'img'))
        det = _store._sql_src(_store._store_globs(root, 'det'))
        con = duckdb.connect()
        con.execute(f'CREATE TEMP TABLE sh AS SELECT image_id FROM {img} '
                    f'GROUP BY 1 HAVING count(DISTINCT run_id) > 1')
        con.execute(f'CREATE TEMP TABLE d AS SELECT d.image_id, d.run_id, '
                    f'd.det_idx, round(d.conf,4) c, round(d.x1,2) x1, '
                    f'round(d.y1,2) y1, round(d.x2,2) x2, round(d.y2,2) y2 '
                    f'FROM {det} d JOIN sh USING (image_id)')
        rows = con.execute(
            'SELECT a.run_id, b.run_id, '
            'count(*) FILTER (WHERE a.c=b.c AND a.x1=b.x1 AND a.y1=b.y1 '
            '                 AND a.x2=b.x2 AND a.y2=b.y2), '
            'count(*) FILTER (WHERE NOT (a.c=b.c AND a.x1=b.x1 AND a.y1=b.y1 '
            '                 AND a.x2=b.x2 AND a.y2=b.y2)) '
            'FROM d a JOIN d b ON a.image_id=b.image_id '
            'AND a.det_idx=b.det_idx AND a.run_id < b.run_id '
            'GROUP BY 1, 2').fetchall()
        con.close()
    except Exception as e:
        print(f'WARNING: could not measure cross-run agreement ({e})',
              file=sys.stderr)
        return {}
    out = {}
    for ra, rb, same, diff in rows:
        for r, o in ((int(ra), int(rb)), (int(rb), int(ra))):
            e = out.setdefault(r, {'identical': 0, 'differing': 0, 'with': []})
            e['identical'] += int(same)
            e['differing'] += int(diff)
            if o not in e['with']:
                e['with'].append(o)
    return out


def _describe(d, sha8):
    """One line for one manifest. Never renders a claim as a reading."""
    cls = d.get('provenance_class')
    m = d.get('model') or {}
    basis = BASIS.get(cls, 'unknown')
    if cls == ATTESTED:
        a = d.get('attestation') or {}
        am = d.get('attested_model') or {}
        c = d.get('corroboration') or {}
        # The attester and "NOT measured" are part of the sentence, not a
        # suffix that a narrower terminal could drop.
        who = f'{am.get("comet_run") or "?"} -- asserted by ' \
              f'{(a.get("by") or "?").split(" <")[0]} {(a.get("at") or "")[:10]}, NOT measured'
        if c.get('identical'):
            who += (f'; {c["identical"]:,} identical detections with runs '
                    f'{c.get("shares_with_run_ids")}, '
                    f'{c.get("differing", 0)} conflicting')
        else:
            who += '; NO corroboration (shares no detection with any run)'
        return basis, who
    if cls is None:
        return 'unknown', ('schema 1 manifest -- it does not say whether its '
                           'digest was measured at run start or later')
    who = f'{m.get("comet_run") or "?"}'
    if m.get('comet_key'):
        who += f'  [{m["comet_key"][:8]}]'
    if m.get('sha8'):
        who += f'  engine {m["sha8"]}'
    if cls == MEASURED_LATE:
        who += (f' -- hashed at {d.get("engine_hashed_at") or "?"}, '
                f'AFTER the run had already written rows')
    return basis, who


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
    rows = con.execute(
        f'SELECT gen, run_id, {sha} AS sha8, count(*) n '
        f'FROM {src} GROUP BY 1, 2, 3 ORDER BY n DESC').fetchall()
    if not has:
        print('note: no file carries model_sha8 yet -- these rows predate the '
              'column and are attributable only through a manifest\n')

    man = manifests_for(root)
    print(f"{'gen':>4}{'run_id':>8}{'rows':>12}  {'basis':<12} model")
    unattributed = 0
    for gen, rid, sha8, n in rows:
        docs = man.get((int(gen), int(rid)), [])
        if sha8:
            hit = [d for d in docs if (d.get('model') or {}).get('sha8') == sha8]
            if not hit and docs:
                # NOT a fallback to docs. A row whose digest matches no
                # manifest is the single most important thing on this screen;
                # rendering it as the manifest's model is how a wrong answer
                # gets a confident label.
                basis, who = ('CONFLICT',
                              f'rows carry engine {sha8}, but no manifest for '
                              f'this run names that engine')
                print(f'{int(gen):>4}{int(rid):>8}{n:>12,}  {basis:<12}{who}')
                continue
            docs = hit or docs
        if len(docs) > 1:
            print(f'{int(gen):>4}{int(rid):>8}{n:>12,}  {"AMBIGUOUS":<12}'
                  f'{len(docs)} runs share run_id={int(rid)}: '
                  + ', '.join(f'{(d.get("model") or {}).get("sha8") or "?"}'
                              f'@{d.get("run_started") or "?"}' for d in docs))
            continue
        if not docs:
            unattributed += n
            print(f'{int(gen):>4}{int(rid):>8}{n:>12,}  {"none":<12}'
                  f'no manifest -- provenance unknown')
            continue
        basis, who = _describe(docs[0], sha8)
        print(f'{int(gen):>4}{int(rid):>8}{n:>12,}  {basis:<12}{who}')
    print(f'\n  basis  measured     the sweep hashed the engine before its '
          f'first box\n'
          f'         hashed-late  a digest was taken, but after the run had '
          f'written rows\n'
          f'         ATTESTED     a person states the model; nothing about '
          f'the run was measured\n'
          f'         unknown      a schema 1 manifest, which does not say '
          f'which of those it is')
    if unattributed:
        print(f'\n{unattributed:,} rows have no manifest at all.')
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
    w.add_argument('--at-start', action='store_true',
                   help='the run is starting NOW and has written no box yet, '
                        'so hashing the engine measures what it will load. '
                        'Without this the record is classed '
                        'measured_after_the_fact, because hashing an engine '
                        'later measures the file today, not the process that '
                        'ran.')
    w.set_defaults(func=cmd_write)

    a = sub.add_parser('attest',
                       help='record a human statement about runs nothing '
                            'measured')
    a.add_argument('--gen', type=int, default=1)
    a.add_argument('--run-ids', type=int, nargs='+', required=True)
    a.add_argument('--run', required=True,
                   help='the model run name being asserted, e.g. train-30')
    a.add_argument('--project', default=None)
    a.add_argument('--statement', required=True,
                   help='what the person actually said, in their words')
    a.add_argument('--by', default=None,
                   help='defaults to the repo git identity')
    a.add_argument('--id', default=None)
    a.add_argument('--no-corroborate', action='store_true')
    a.add_argument('--force', action='store_true')
    a.set_defaults(func=cmd_attest)

    s = sub.add_parser('show', help='what produced the rows in the store')
    s.set_defaults(func=cmd_show)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
