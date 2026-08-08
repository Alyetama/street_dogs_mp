#!/usr/bin/env python3
"""The gate panel must show a run that is running.

Progress was read off the written shards and nothing else. A shard is 20,000
frames, so for the first several minutes of a twelve-hour run every figure on
the panel was 0 or an em-dash while sixteen decoder processes were flat out --
pressing Run looked like it had done nothing at all. The runner now publishes
a heartbeat between shards.

Two things have to hold, and they pull against each other:

  * the panel must move while a shard is in flight, and
  * a heartbeat must never outlive the process that wrote it, or a killed run
    keeps "running" forever (a SIGTERM leaves the file behind by definition --
    the process is gone before it can clean up).

So the reader ages the file out, and everything in it is treated as a claim
about a process rather than a record of work. The shards stay the record.
"""

import ast
import json
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

PLAN = {'rows': 4688510, 'images': 3292062, 'model': 'dogbin_008',
        'created': '2026-08-07 22:28:52'}


def _beat_doc(**kw):
    doc = {'pid': 1, 'started': time.time() - 60, 'updated': time.time(),
           'shard': 0, 'shards_total': 165, 'images': 8123,
           'images_total': 3292062, 'rows_flight': 11500, 'boxes': 11500,
           'bad': 3, 'img_s': 135.4, 'box_s': 196.2, 'dog_share': 0.213}
    doc.update(kw)
    return doc


def reader_checks(bad):
    import dashboard as d
    tmp = tempfile.mkdtemp()
    d.GATE_DIR = tmp
    beat_p = os.path.join(tmp, 'progress.json')
    with open(os.path.join(tmp, 'plan.json'), 'w') as fh:
        json.dump(PLAN, fh)

    def prog(beat=None):
        if beat is None:
            try:
                os.remove(beat_p)
            except OSError:
                pass
        elif isinstance(beat, str):
            with open(beat_p, 'w') as fh:      # deliberately not JSON
                fh.write(beat)
        else:
            with open(beat_p, 'w') as fh:
                json.dump(beat, fh)
        d._GATE.update(at=0, doc=None)         # the shard scan is cached
        return d.gate_progress()

    # ── a live run with no shard written yet ────────────────────────────────
    g = prog(_beat_doc())
    if not g['running']:
        bad.append('a live heartbeat does not make the panel say running')
    if g['rows'] != 11500:
        bad.append(f"judged is {g['rows']}, expected the 11,500 in flight")
    if abs(g['pct'] - 11500 / PLAN['rows']) > 1e-9:
        bad.append(f"percentage {g['pct']} is not the in-flight rows over the "
                   f"plan total")
    if g['dog_share'] != 0.213:
        bad.append(f"share called dog is {g['dog_share']} with no shard to "
                   f"read it from -- the run measured 0.213")
    if (g['rate'], g['sustained']) != (196.2, 196.2):
        bad.append(f"rate {g['rate']}/{g['sustained']} is not the measured "
                   f"196.2 boxes/s")
    if g['images_done'] != 8123 or g['img_s'] != 135.4:
        bad.append(f"frames opened {g['images_done']} @ {g['img_s']}/s lost")
    if not g['eta_s'] or not 23000 < g['eta_s'] < 24500:
        bad.append(f"eta {g['eta_s']} is not the remaining rows over the "
                   f"measured rate")

    # ── the same file once the process is gone ──────────────────────────────
    g = prog(_beat_doc(updated=time.time() - 400))
    if g['running']:
        bad.append('a stale heartbeat still reports a running gate -- a '
                   'killed run cannot delete its own file')
    for k, want in (('rows', 0), ('images_done', None), ('img_s', None)):
        if g[k] != want:
            bad.append(f'stale heartbeat still counted: {k}={g[k]}')

    # ── hostile files must not take the panel down ──────────────────────────
    for junk in ('{not json', 'null', '[]', '{"updated":"soon"}',
                 '{"updated":null}', '{"updated":' + str(time.time()) +
                 ',"rows_flight":"lots"}'):
        try:
            g = prog(junk)
        except Exception as e:                 # noqa: BLE001 - that is the test
            bad.append(f'{junk!r} threw {type(e).__name__}: {e}')
            continue
        if g.get('rows'):
            bad.append(f'{junk!r} was believed: rows={g["rows"]}')
    g = prog(None)
    if g['running'] or g['rows'] or not g['ever']:
        bad.append(f'no heartbeat at all should read as an idle plan: {g}')

    # ── the beat must not be counted twice once a shard lands ───────────────
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        print('SKIP: no pyarrow — double-count check not run')
        return
    pq.write_table(
        pa.table({'label': pa.array(['dog'] * 300 + ['not_dog'] * 700)}),
        os.path.join(tmp, 'gate-00000.parquet'))
    g = prog(_beat_doc(rows_flight=250, dog_share=0.9, images=21000))
    if g['rows'] != 1250:
        bad.append(f"judged is {g['rows']}, expected 1,000 written + 250 in "
                   f"flight")
    if abs((g['dog_share'] or 0) - 0.30) > 1e-9:
        bad.append(f"share called dog is {g['dog_share']}, expected 0.30 from "
                   f"the written shard -- the record outranks the heartbeat")

    # ── the scan has to be incremental, or a 2 s window is unaffordable ─────
    # A shard is immutable once os.replace()d into place, so reading one twice
    # is pure waste -- and at 165 of them, every two seconds, it is the kind
    # of waste that shows up as a dashboard that stalls the box it runs on.
    reads = []
    real = pq.read_table

    def counted(path, **kw):
        reads.append(path)
        return real(path, **kw)

    pq.read_table = counted
    try:
        d._GATE.update(at=0, doc=None)
        d.gate_progress()                      # everything already known
        if reads:
            bad.append(f'{len(reads)} shard(s) re-read on a scan that had '
                       f'nothing new: {[os.path.basename(r) for r in reads]}')
        pq.write_table(pa.table({'label': pa.array(['dog'] * 40)}),
                       os.path.join(tmp, 'gate-00001.parquet'))
        d._GATE.update(at=0, doc=None)
        g = d.gate_progress()
        if len(reads) != 1:
            bad.append(f'a new shard cost {len(reads)} reads, expected 1')
        if g['rows'] != 1290 or g['shards'] != 2:
            bad.append(f"new shard not picked up: rows={g['rows']} "
                       f"shards={g['shards']}")
        if abs((g['dog_share'] or 0) - 340 / 1040) > 1e-9:
            bad.append(f"share called dog is {g['dog_share']}, expected the "
                       f"whole record's 340/1040")
        os.remove(os.path.join(tmp, 'gate-00001.parquet'))
        d._GATE.update(at=0, doc=None)
        g = d.gate_progress()
        if g['shards'] != 1 or g['rows'] != 1250:
            bad.append(f'a deleted shard is still counted: {g["shards"]} '
                       f'shards, {g["rows"]} rows')
    finally:
        pq.read_table = real


def runner_checks(bad):
    """The runner has to publish, and publish EARLY.

    Read out of the source, not out of a run: the runner needs ultralytics, a
    GPU and six mounted drives, none of which a test has. What can be checked
    is that the calls are where they have to be -- one before the pool opens
    (so the panel moves the moment the model is loaded) and one inside the
    per-image loop (so it keeps moving between shards).
    """
    src = os.path.join(REPO, 'tools', 'detect', 'gate_store.py')
    tree = ast.parse(open(src).read())
    fn = next((f for f in ast.walk(tree)
               if isinstance(f, ast.FunctionDef) and f.name == 'run'), None)
    if fn is None:
        bad.append('gate_store.run() is gone')
        return

    def calls(node):
        return {n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    withs = [n for n in fn.body if isinstance(n, ast.With)]
    pre = set()
    for n in fn.body:
        if isinstance(n, ast.With):
            break                                # everything after is in-pool
        pre |= calls(n)
    if 'beat' not in pre:
        bad.append('run() opens the worker pool without publishing anything '
                   'first -- the panel stays blank until the first shard')
    inner = [n for w in withs for n in ast.walk(w)
             if isinstance(n, ast.For) and 'imap_unordered' in ast.dump(n)]
    if not inner:
        bad.append('cannot find the per-image loop in run()')
    elif not any('beat' in calls(n) for n in inner):
        bad.append('the per-image loop never publishes -- progress would jump '
                   'only when a shard closes')

    import gate_store as gs
    tmp = tempfile.mkdtemp()
    gs.BEAT_FILE = os.path.join(tmp, 'progress.json')
    gs._beat(updated=1.0, rows_flight=7)
    with open(gs.BEAT_FILE) as fh:
        if json.load(fh) != {'updated': 1.0, 'rows_flight': 7}:
            bad.append('_beat did not write what it was given')
    gs._beat_clear()
    if os.path.exists(gs.BEAT_FILE):
        bad.append('_beat_clear left the file behind -- a finished run would '
                   'keep claiming to be running')
    gs._beat_clear()                            # idempotent: no file, no throw
    gs.BEAT_FILE = os.path.join(tmp, 'no', 'such', 'dir', 'p.json')
    gs._beat(updated=1.0)                       # unwritable: must not raise


def main():
    bad = []
    runner_checks(bad)
    try:
        reader_checks(bad)
    except ImportError as e:
        print(f'SKIP: cannot import the dashboard ({e})')
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('the gate panel moves while a shard is in flight, and stops '
          'claiming a run the moment the heartbeat goes stale')
    return 0


if __name__ == '__main__':
    sys.exit(main())
