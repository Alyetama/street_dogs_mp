#!/usr/bin/env python3
"""The gate panel shows a gate pass over a side store, read-only.

A candidate detector's boxes are judged under their own stages root
(data/gate_<store>/, gate_store's $GATE_ROOT), so data/gate keeps the live
store's verdicts. The panel read data/ only, so that pass was invisible.
Worse, the process check matched on script, subcommand and stage alone: a
side pass read as the LIVE gate running, with a Stop button that would have
killed it.

  p1 side_gate_root: a side root with a plan, newer than the live record, wins
  p2 an older side root does not; a side root with no plan never does
  p3 gate_progress(stage, root) reads the side root's shards and plan
  p4 gate_pids tells a live run from a side run by its $GATE_ROOT
  p5 a running side pass wins even when the live record is newer on disk
  p6 the /api/gate handler marks the side view and disables the controls

    python tools/detect/tests/adv_side_gate_panel.py      (mp14 env)
"""

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))

FAILS = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def shard(d, i, labels, model='m'):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({
        'image_id': [str(k) for k in range(len(labels))],
        'det_idx': list(range(len(labels))),
        'label': labels, 'p_dog': [0.9] * len(labels),
        'model': [model] * len(labels)}),
        os.path.join(d, f'gate-{i:05d}.parquet'))


def plan(d, rows, images):
    os.makedirs(d, exist_ok=True)
    json.dump({'rows': rows, 'images': images, 'shard_rows': 2,
               'model': 'dog-bin gate', 'created': 'x'},
              open(os.path.join(d, 'plan.json'), 'w'))


def main():
    import dashboard as dash
    repo = tempfile.mkdtemp()
    dash.REPO = repo
    data = os.path.join(repo, 'data')
    live = os.path.join(data, 'gate')
    side_root = os.path.join(data, 'gate_detect_x')
    old_root = os.path.join(data, 'gate_detect_old')
    noplan_root = os.path.join(data, 'gate_detect_noplan')
    plan(live, 4, 4)
    shard(live, 0, ['dog', 'not_dog'])
    t = time.time()
    os.utime(os.path.join(live, 'plan.json'), (t - 500, t - 500))
    os.utime(os.path.join(live, 'gate-00000.parquet'), (t - 400, t - 400))
    plan(os.path.join(old_root, 'gate'), 4, 4)
    os.utime(os.path.join(old_root, 'gate', 'plan.json'), (t - 900, t - 900))
    os.makedirs(os.path.join(noplan_root, 'gate'))
    shard(os.path.join(noplan_root, 'gate'), 0, ['dog'])
    plan(os.path.join(side_root, 'gate'), 6, 6)
    shard(os.path.join(side_root, 'gate'), 0, ['dog', 'dog'], model='new')

    got = dash.side_gate_root('gate')
    check('p1 newer side root with a plan wins', got == side_root, got)
    os.utime(os.path.join(side_root, 'gate', 'plan.json'), (t - 800, t - 800))
    os.utime(os.path.join(side_root, 'gate', 'gate-00000.parquet'),
             (t - 800, t - 800))
    got = dash.side_gate_root('gate')
    check('p2 older side roots and plan-less roots never win', got is None, got)

    dash._GATE.clear()
    dash._GATE_FILES.clear()
    doc = dash.gate_progress('gate', side_root)
    check('p3 side progress reads the side record',
          doc.get('total') == 6 and doc.get('rows') == 2
          and doc.get('dogs') == 2, doc)
    live_doc = dash.gate_progress('gate')
    check('p3b the live record is still the live record',
          live_doc.get('total') == 4 and live_doc.get('dogs') == 1, live_doc)

    # p4/p5: a fake runner -- argv shaped like the real one, under a side root
    fake = os.path.join(tempfile.mkdtemp(), 'gate_store.py')
    open(fake, 'w').write('import time\ntime.sleep(30)\n')
    env = dict(os.environ, GATE_ROOT=side_root)
    pr = subprocess.Popen([sys.executable, fake, 'run', '--stage', 'gate'],
                          env=env)
    env2 = {k: v for k, v in os.environ.items() if k != 'GATE_ROOT'}
    pl = subprocess.Popen([sys.executable, fake, 'run', '--stage', 'gate'],
                          env=env2)
    try:
        time.sleep(0.5)
        live_pids = dash.gate_pids('gate')
        side_pids = dash.gate_pids('gate', side_root)
        check('p4 the side run is not the live gate',
              pr.pid not in live_pids, live_pids)
        check('p4b the side run is found under its root',
              pr.pid in side_pids, side_pids)
        check('p4c a run with no GATE_ROOT is the live gate',
              pl.pid in live_pids and pl.pid not in side_pids)
        got = dash.side_gate_root('gate')
        check('p5 a running side pass wins over a newer live record',
              got == side_root, got)
    finally:
        pr.kill()
        pl.kill()

    src = open(os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')).read()
    i = src.index("if self.path.split('?', 1)[0] == '/api/gate':")
    body = src[i:i + 3000]
    check('p6 the handler marks the side view',
          "doc['side_store'] = bool(side)" in body)
    check('p6b the controls are off for a side view',
          "doc['can_run'] = (not side)" in body)
    check('p6c the panel says whose numbers these are',
          "'separate store '+(j.store||'')" in src)
    if FAILS:
        print(f'{len(FAILS)} failed')
        return 1
    print('the gate panel shows a side-store pass, read-only, and never '
          'mistakes it for the live gate')
    return 0


if __name__ == '__main__':
    sys.exit(main())
