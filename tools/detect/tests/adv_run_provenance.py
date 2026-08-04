#!/usr/bin/env python3
"""Adversarial checks on model provenance for the predictions store.

The store answers "which model drew this box" in two places, and both have a
failure mode where the broken state looks exactly like the working one:

  t1  A writer told a digest stamps it on EVERY row it commits, image and
      detection alike. Miss one table and half the store is unattributable
      while the other half looks fine.

  t2  A writer NOT told a digest writes NULL, never a guess. A default of
      "whatever model is in the config right now" would silently relabel
      rows produced by a different engine.

  t3  Old files lack the column entirely. Reads must union_by_name, or duckdb
      takes the FIRST file's schema and drops the column with no error --
      making every row look untagged.

  t4  run_id is time.time() & 0xFFFF and wraps every 18.2 hours, so two runs
      CAN share one. Writing a manifest must never overwrite another run's,
      and show() must report the ambiguity rather than picking one.

  t5  A run's ts_off is seconds since that run's own start. The manifest must
      record that start, or no timestamp in the store means anything.

Writes only to a temp directory.
"""

import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(DETECT))
sys.path.insert(0, DETECT)

FAILED = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}{("  -- " + detail) if detail else ""}')
    if not ok:
        FAILED.append(name)


def write_shard(writer, cell):
    sw = writer.open_shard(1, 'RX', cell, 'lynx', 0, 0, 1)
    sw.add_image({'image_id': 100000001, 'drive': 0, 'status': 0, 'n_det': 1,
                  'max_conf': 0.9, 'orig_w': 64, 'orig_h': 64, 'reduce': 1,
                  'guards': 0, 'ts_off': 3, 'run_id': 4242})
    sw.add_detections([{'image_id': 100000001, 'det_idx': 0, 'conf': 0.9,
                        'x1': 1, 'y1': 1, 'x2': 9, 'y2': 9, 'run_id': 4242}])
    sw.commit(1)
    return sw.dirpath


def main():
    import duckdb
    import store

    tmp = tempfile.mkdtemp(prefix='provenance-')
    try:
        con = duckdb.connect()

        # -- t1 / t2 --------------------------------------------------------
        told = write_shard(store.Writer(tmp, model_sha8='deadbeef'), 'CT')
        untold = write_shard(store.Writer(tmp), 'CU')

        def col(d, kind):
            import glob
            f = glob.glob(os.path.join(d, f'*{kind}*.parquet'))
            if not f:
                return '<no file>'
            return con.execute(
                f"SELECT model_sha8 FROM read_parquet('{f[0]}')").fetchone()[0]

        for kind in ('img', 'det'):
            check(f't1 {kind} rows carry the digest',
                  col(told, kind) == 'deadbeef', f'got {col(told, kind)!r}')
            check(f't2 {kind} rows are NULL when untold',
                  col(untold, kind) is None, f'got {col(untold, kind)!r}')

        # -- t3 a file WITHOUT the column must not hide it in the others ----
        # Same hive layout, so the only difference duckdb sees is the schema.
        import glob as _g
        legacy_dir = os.path.join(tmp, 'shards', 'gen=0001', 'region=RX',
                                  'cell=CL', 'drive=lynx')
        os.makedirs(legacy_dir, exist_ok=True)
        old = os.path.join(legacy_dir, 's00000.p000000_000001.img.parquet')
        con.execute(
            f"COPY (SELECT 1::BIGINT AS image_id, 4242::INT AS run_id) "
            f"TO '{old}' (FORMAT PARQUET)")
        # legacy FIRST: without union_by_name duckdb takes its schema and the
        # column vanishes from the result with no error at all
        files = [old] + [f for f in sorted(
            _g.glob(os.path.join(tmp, '**', '*img*.parquet'), recursive=True))
            if f != old]
        src = store._sql_src(files)
        cols = {r[0] for r in con.execute(f'DESCRIBE SELECT * FROM {src}')
                .fetchall()}
        check('t3 read keeps model_sha8 when the first file lacks it',
              'model_sha8' in cols, f'columns: {sorted(cols)}')
        if 'model_sha8' in cols:
            got = con.execute(
                f"SELECT coalesce(model_sha8, '(null)'), count(*) FROM {src} "
                f"GROUP BY 1 ORDER BY 1").fetchall()
            check('t3 filtering by digest still selects the tagged rows',
                  dict(got).get('deadbeef') == 1, f'{got}')

        # -- t4 / t5 --------------------------------------------------------
        import run_manifest

        fake_repo = os.path.join(tmp, 'repo')
        os.makedirs(os.path.join(fake_repo, 'data'))
        droot = os.path.join(tmp, 'droot')
        with open(os.path.join(fake_repo, 'data', 'detect_root.txt'), 'w') as fh:
            fh.write(droot + '\n')
        shutil.copy2(os.path.join(REPO, 'data', 'best_models.json'),
                     os.path.join(fake_repo, 'data', 'best_models.json'))
        eng = os.path.join(tmp, 'fake.engine')
        with open(eng, 'wb') as fh:
            fh.write(b'not a real engine')

        # store.get_detect_root reads the repo it is handed; run_manifest asks
        # for the real one, so point it at the fake repo for the duration
        real = store.get_detect_root

        rm_at_start = __import__('run_manifest').MEASURED_AT_START
        store.get_detect_root = lambda repo_root=None: droot
        try:
            cfg = {'engine': eng, 'conf': 0.05, 'iou': 0.9, 'max_det': 256,
                   'imgsz': None}
            t0 = 1754179772  # a fixed start, and one 18.2h later that wraps
            run_manifest.write_for_run(fake_repo, 1, 4242, cfg, started=t0)
            run_manifest.write_for_run(fake_repo, 1, 4242, cfg,
                                       started=t0 + 65536)
            man = run_manifest.manifests_for(droot)
            docs = man.get((1, 4242), [])
            check('t4 a wrapped run_id keeps BOTH manifests', len(docs) == 2,
                  f'{len(docs)} manifest(s) for run_id=4242')
            starts = {d.get('run_started_epoch') for d in docs}
            check('t5 each manifest records its own run start',
                  starts == {t0, t0 + 65536}, f'{sorted(starts)}')
            files = os.listdir(os.path.join(droot, 'runs', 'gen=0001'))
            check('t4 the two manifests are separate files',
                  len(files) == 2, f'{sorted(files)}')
            # the CLASS of every document these paths produce, asserted --
            # both directions could be inverted with the suite green
            check('t11 write_for_run classes its record measured_at_start',
                  all(d.get('provenance_class') == rm_at_start
                      for d in docs),
                  f'{[d.get("provenance_class") for d in docs]}')
            check('t11 and stamps the schema version',
                  all(d.get('schema') == run_manifest.SCHEMA for d in docs),
                  f'{[d.get("schema") for d in docs]}')
        finally:
            store.get_detect_root = real
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── t6-t9: how a provenance record is CLASSED ───────────────────────
    import run_manifest as rm

    # t6 an attested record must be impossible to render as a measurement
    att = {'schema': 2, 'provenance_class': rm.ATTESTED, 'gen': 1,
           'run_id': 1, 'model': {'sha8': None, 'comet_run': None,
                                  'comet_key': None},
           'attested_model': {'comet_run': 'train-30'},
           'attestation': {'by': 'Someone <a@b>', 'at': '2026-08-03'},
           'corroboration': {'identical': 0, 'differing': 0,
                             'shares_with_run_ids': []}}
    basis, line = rm._describe(att, None)
    # Against the literal, not against rm.BASIS[rm.ATTESTED] -- comparing
    # _describe's output to the dict it reads from can never fail.
    check('t6 an attested row is labelled ATTESTED', basis == 'ATTESTED',
          f'got {basis!r}')
    check('t6 an attested row always says NOT measured', 'NOT measured' in line,
          line[:90])
    check('t6 an attested row names its attester', 'Someone' in line, line[:90])
    check('t6 an uncorroborated attestation says so',
          'NO corroboration' in line, line[:90])

    # t7 a schema 1 record must not be read as measured
    basis, line = rm._describe({'gen': 1, 'run_id': 1,
                                'model': {'sha8': 'abcd1234'}}, None)
    check('t7 a class-less manifest is "unknown", not measured',
          basis == 'unknown', f'got {basis!r}')

    # t8 a measured-late record must say the digest came after the rows
    basis, line = rm._describe(
        {'schema': 2, 'provenance_class': rm.MEASURED_LATE, 'gen': 1,
         'run_id': 1, 'engine_hashed_at': '2026-08-03 22:57:49',
         'model': {'sha8': 'ac98daee', 'comet_run': 'train-30',
                   'comet_key': 'ef4a85c3c1ee4bfc8d6805fb413c43f3'}}, None)
    check('t8 a late digest says it was taken after the rows',
          'AFTER the run had already written rows' in line, line[:110])

    # t12 a class this version does not understand is not a measurement
    basis, line = rm._describe({'schema': 99, 'provenance_class': 'attested_v3',
                                'gen': 1, 'run_id': 1,
                                'model': {'sha8': 'abcd1234',
                                          'comet_run': 'train-99'}}, None)
    check('t12 an unrecognised provenance_class is not rendered as measured',
          basis == 'UNRECOGNISED' and 'train-99' not in line, f'{basis} {line[:70]}')

    # t13 a corroboration that never ran must not read as a measured zero
    _, line = rm._describe(dict(att, corroboration={'status': 'failed',
                                                    'identical': None}), None)
    check('t13 a skipped/failed corroboration says so',
          'NOT measured either way' in line, line[-70:])

    # t9 generation 0 must not widen the glob to every generation
    with tempfile.TemporaryDirectory() as tmp:
        for g, rid in ((0, 999), (1, 4242)):
            d = os.path.join(tmp, 'runs', f'gen={g:04d}')
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f'run_{rid}_x.json'), 'w') as fh:
                json.dump({'gen': g, 'run_id': rid}, fh)
        got = sorted(rm.manifests_for(tmp, 0))
        check('t9 gen=0 selects generation 0, not all of them',
              got == [(0, 999)], f'{got}')

    # t10 a manifest failure must never kill a sweep. run_manifest raises
    # SystemExit, which is NOT an Exception -- the guard has to be wider.
    # Structural, not a text window: a string search silently measures
    # whatever happens to be within N characters of the call.
    import ast as _ast
    tree = _ast.parse(open(os.path.join(DETECT, 'sweep.py')).read())
    wide = None
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Try):
            continue
        calls = [n for n in _ast.walk(node)
                 if isinstance(n, _ast.Attribute)
                 and n.attr == 'write_for_run']
        if not calls:
            continue
        names = []
        for h in node.handlers:
            t = h.type
            names += ([e.id for e in t.elts if isinstance(e, _ast.Name)]
                      if isinstance(t, _ast.Tuple)
                      else [t.id] if isinstance(t, _ast.Name)
                      else ['<bare>'] if t is None else ['?'])
        wide = names
    check('t10 the sweep guard catches SystemExit too',
          wide is not None and ('BaseException' in wide or '<bare>' in wide),
          f'handlers around write_for_run: {wide}')

    print()
    if FAILED:
        print(f'{len(FAILED)} FAILED: {", ".join(FAILED)}')
        return 1
    print('all provenance checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
