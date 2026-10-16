#!/usr/bin/env python3
"""A stage of the model sweeps can be judged again with the promoted model,
and the verdicts it replaces are set aside, never deleted.

    python tools/detect/tests/adv_sweep_rerun.py

  g1  an archive moves the verdict shards and the heartbeat, byte for byte,
      keeps the work list, and its note names the model that wrote them, the
      one replacing them and the command that puts them back
  g2  a re-plan moves the work list and the pinned shard size as well
  g3  nothing to move is no archive at all
  g4  an archive is never written into one that exists, and then nothing moves
  g5  a restore puts the files back byte for byte, sets aside what was there
      first, and marks the archive restored
  g6  a restore refuses a name that is not one of the stage's archives
  g7  the dashboard refuses a re-run while the stage runs, while a work list
      is being built, with no model promoted, and before the upstream stage
      has finished -- and moves nothing when it refuses
  g8  a re-run sets aside the verdicts with the models read off the shards,
      starts the stage, says where the archive is and how to undo it, and
      the panel stops counting what was moved
  g9  the leash list is rebuilt when the gate has written since it was made,
      and kept when it has not
  g10 the panel counts verdicts per model; a shard without a stamp is unknown
  g11 the panel names the newest archive and whether it was restored
  g12 the page offers the re-run beside Run, only to an admin, posts it as
      `rerun`, and the confirm says the old verdicts are archived and how they
      come back

The detector sweep, the same way over its whole store:

  s1  an archive moves the shards, errors, run manifests, status and SQL
      snapshot, byte for byte, and keeps the worklist and the bootstrap
  s2  it refuses while a sweep holds the run lock, and moves nothing
  s3  nothing to move is no archive
  s4  a restore puts it back, sets aside what was there first, marks the
      archive restored, and refuses a name that is not an archive
  s5  the store's model is its newest run's that recorded an engine
  s6  the dashboard refuses a re-run while the sweep runs, while a stage
      reads the store, with no detector promoted or with no engine for it --
      and moves nothing; the same model again is allowed (s7b)
  s7  a re-run sets the store aside and starts the sweep on the promoted
      engine, saying where the archive is and how to undo it
  s8  Resume loads the engine the store's newest run loaded, and refuses
      when that engine is gone, rather than finishing on another model
  s9  the gate's list is out of date once the detector has run since it was
      built, and the gate is not re-run while the sweep runs
  s10 sweep.py --engine replaces the configured engine and refuses one that
      is not on disk
  s11 the page offers the detector re-run beside Resume, only to an admin,
      never folds the panel, posts `rerun`, and is always clickable; its
      confirm says when the model is the same one

What a re-run would replace is on screen before it is pressed:

  v1  the store's model carries when its run started, and the dashboard
      passes both on with the promoted model beside them
  v2  each tab opens with a line naming the model that wrote its results
      and the promoted one, saying whether they are the same; the names are
      escaped, and the header no longer hardcodes a model name
  v3  the gate panel is told when its newest shard landed

EVERY PATH IS REDIRECTED to a temp directory; the live stage directories are
fingerprinted before and compared after.
"""
import argparse
import fcntl
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print('SKIP: pyarrow is not importable here -- nothing checked the re-run')
    sys.exit(0)

import dashboard as d  # noqa: E402
import gate_store as gs  # noqa: E402
import sweep_archive as sa  # noqa: E402

_fails = []


def check(label, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + label + ('' if ok else '  ' + detail))
    if not ok:
        _fails.append(label.split()[0])


def fingerprint(path):
    out = {}
    for base, _, files in os.walk(path):
        for f in files:
            p = os.path.join(base, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out[os.path.relpath(p, path)] = (st.st_size, st.st_mtime_ns)
    return out


def raises(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    return None


def shard(path, labels, model='dogbin_008'):
    cols = {'image_id': ['1'] * len(labels), 'label': labels}
    if model is not None:
        cols['model'] = [model] * len(labels)
    pq.write_table(pa.table(cols), path)


def read(path):
    with open(path, 'rb') as fh:
        return fh.read()


LIVE = {s: os.path.join(REPO, 'data', s) for s in ('gate', 'leash')}
try:
    import store as _store
    LIVE_DETECT = _store.get_detect_root()
except Exception:  # noqa: BLE001 -- no store here is nothing to protect
    LIVE_DETECT = None


def detect_print():
    """The live detect store's top level and its run manifests: what a
    re-run would move. The shards themselves are thousands of files."""
    if not LIVE_DETECT or not os.path.isdir(LIVE_DETECT):
        return None
    top = sorted(os.listdir(LIVE_DETECT))
    return top, fingerprint(os.path.join(LIVE_DETECT, 'runs'))


live_before = {s: fingerprint(p) for s, p in LIVE.items()}
detect_before = detect_print()
tmp = tempfile.mkdtemp(prefix='sweep-rerun-')
real = {
    k: getattr(d, k)
    for k in ('REPO', 'gate_pids', 'gate_planning', 'best_models', '_gate_act',
              'sweep_pids', '_sweep_spawn', '_detect_root')
}
try:
    root = os.path.join(tmp, 'repo')
    gdir = os.path.join(root, 'data', 'gate')
    ldir = os.path.join(root, 'data', 'leash')
    os.makedirs(gdir)
    os.makedirs(ldir)

    def stage_files(dr, pre):
        shard(os.path.join(dr, f'{pre}-00000.parquet'), ['dog', 'not_dog'])
        shard(os.path.join(dr, f'{pre}-00001.parquet'), ['dog'])
        for f, body in (('progress.json', b'{"updated": 0}'),
                        ('plan.json', b'{"rows": 3, "images": 2}'),
                        ('work.parquet', b'work'), ('shard_size', b'20000')):
            with open(os.path.join(dr, f), 'wb') as fh:
                fh.write(body)

    # ── g1 ──────────────────────────────────────────────────────────────────
    stage_files(gdir, 'gate')
    before = {f: read(os.path.join(gdir, f)) for f in os.listdir(gdir)}
    note = gs.archive_run('gate', {'dogbin_008': 3},
                          'dogbin_009',
                          reason='probe',
                          repo=root,
                          now=1e9)
    left = sorted(os.listdir(gdir))
    arch = note and note['path']
    moved = sorted(f for f in os.listdir(arch) if f != gs.ARCHIVE_NOTE) \
        if arch else []
    same = bool(arch) and all(
        read(os.path.join(arch, f)) == before[f] for f in moved)
    check(
        'g1 an archive moves the shards and the heartbeat, byte for byte, and '
        'keeps the work list',
        moved == ['gate-00000.parquet', 'gate-00001.parquet', 'progress.json']
        and same
        and left == ['archive', 'plan.json', 'shard_size', 'work.parquet'],
        f'moved {moved}, identical {same}, left {left}')
    with open(os.path.join(arch, gs.ARCHIVE_NOTE)) as fh:
        disk = json.load(fh)
    check(
        'g1b its note names the old model, the new one and the way back',
        disk.get('models') == {'dogbin_008': 3}
        and disk.get('replaced_by') == 'dogbin_009'
        and disk.get('stage') == 'gate' and not disk.get('replanned')
        and disk.get('restore') == 'python tools/detect/gate_store.py restore '
        f'--stage gate --archive {note["name"]}', f'{disk}')

    # ── g2 ──────────────────────────────────────────────────────────────────
    stage_files(ldir, 'leash')
    n2 = gs.archive_run('leash', {},
                        'leash_v3',
                        replan=True,
                        repo=root,
                        now=1e9)
    check(
        'g2 a re-plan moves the work list and the shard size too', n2
        and os.listdir(ldir) == ['archive'] and sorted(n2['files']) == [
            'leash-00000.parquet', 'leash-00001.parquet', 'plan.json',
            'progress.json', 'shard_size', 'work.parquet'
        ], f'{n2 and n2["files"]}, left {os.listdir(ldir)}')

    # ── g3 ──────────────────────────────────────────────────────────────────
    n3 = gs.archive_run('leash', {}, 'leash_v3', repo=root, now=2e9)
    check(
        'g3 nothing to move is no archive', n3 is None
        and os.listdir(os.path.join(ldir, 'archive')) == [n2['name']],
        f'{n3}, {os.listdir(os.path.join(ldir, "archive"))}')

    # ── g4 ──────────────────────────────────────────────────────────────────
    shard(os.path.join(gdir, 'gate-00000.parquet'), ['dog'], 'dogbin_009')
    err = raises(lambda: gs.archive_run('gate', {}, 'x', repo=root, now=1e9))
    check(
        'g4 an archive is never written into one that exists, and nothing '
        'moves', err == 'FileExistsError'
        and os.path.exists(os.path.join(gdir, 'gate-00000.parquet'))
        and sorted(os.listdir(arch)) == sorted(moved + [gs.ARCHIVE_NOTE]),
        f'raised {err}, archive {sorted(os.listdir(arch))}')

    # ── g5 ──────────────────────────────────────────────────────────────────
    newer = read(os.path.join(gdir, 'gate-00000.parquet'))
    got = gs.restore_run('gate', note['name'], repo=root, now=3e9)
    back = all(
        read(os.path.join(gdir, f)) == before[f] for f in moved
        if os.path.exists(os.path.join(gdir, f)))
    aside = got.get('set_aside')
    kept = (os.path.join(gdir, 'archive', aside, 'gate-00000.parquet')
            if aside else '')
    with open(os.path.join(arch, gs.ARCHIVE_NOTE)) as fh:
        after = json.load(fh)
    check(
        'g5 a restore puts the files back byte for byte',
        sorted(got.get('restored') or []) == moved and back
        and all(os.path.exists(os.path.join(gdir, f)) for f in moved),
        f'{got}, identical {back}')
    check(
        'g5b what was there is set aside first, and the archive says it was '
        'restored',
        bool(aside) and os.path.exists(kept) and read(kept) == newer
        and bool(after.get('restored_at')), f'set aside {aside}, note {after}')

    # ── g6 ──────────────────────────────────────────────────────────────────
    # a note planted one level up, so '..' would resolve to a "note" if the
    # name itself were not refused
    planted = os.path.join(gdir, gs.ARCHIVE_NOTE)
    with open(planted, 'w') as fh:
        json.dump({'files': []}, fh)
    bad = [
        raises(lambda n=n: gs.restore_run('gate', n, repo=root))
        for n in ('', 'nope', '../leash', '.', '..', 'archive')
    ]
    os.remove(planted)
    check('g6 a restore refuses a name that is not an archive of the stage',
          all(x == 'ValueError' for x in bad), f'{bad}')

    # ── dashboard ───────────────────────────────────────────────────────────
    shutil.rmtree(os.path.join(root, 'data'))
    os.makedirs(gdir)
    os.makedirs(ldir)
    stage_files(gdir, 'gate')
    d.REPO = root
    d._GATE.clear()
    d._GATE_FILES.clear()
    best = {
        'projects': {
            'dog-bin': {
                'best': {
                    'run': 'dogbin_009',
                    'weights': 'w.pt'
                }
            },
            'leash-models': {
                'best': {
                    'run': 'leash_v3',
                    'weights': 'l.pt'
                }
            }
        }
    }
    d.best_models = lambda: best
    d.sweep_pids = lambda: []
    d._detect_root = lambda: os.path.join(root, 'nodetect')
    started = []
    d._gate_act = lambda action, stage, sp, pids: (started.append(
        (action, stage, sorted(os.listdir(d.gate_dir(stage))))) or {
            'ok': True,
            'running': True,
            'msg': 'started'
        })

    # ── g10 ─────────────────────────────────────────────────────────────────
    shard(os.path.join(gdir, 'gate-00002.parquet'), ['dog'], None)
    s10 = d._gate_shards('gate') or {}
    check('g10 the panel counts verdicts per model, unstamped as unknown',
          s10.get('models') == {
              'dogbin_008': 3,
              'unknown': 1
          }, f'{s10.get("models")}')

    # ── g7 ──────────────────────────────────────────────────────────────────
    snap = sorted(os.listdir(gdir))
    refusals = {}
    d.gate_pids = lambda stage='gate': [123]
    d.gate_planning = lambda: False
    refusals['running'] = d.gate_control('rerun', 'gate')
    d.gate_pids = lambda stage='gate': []
    d.gate_planning = lambda: True
    refusals['planning'] = d.gate_control('rerun', 'gate')
    d.gate_planning = lambda: False
    d.best_models = lambda: {'projects': {}}
    refusals['no best'] = d.gate_control('rerun', 'gate')
    d.best_models = lambda: best
    # the leash before the gate has finished: a plan of a million frames at
    # 20,000 a shard wants fifty shards, and three are on disk
    with open(os.path.join(gdir, 'plan.json'), 'w') as fh:
        json.dump({'rows': 10**6, 'images': 10**6}, fh)
    d._GATE.clear()
    stage_files(ldir, 'leash')
    lsnap = sorted(os.listdir(ldir))
    refusals['upstream'] = d.gate_control('rerun', 'leash')
    with open(os.path.join(gdir, 'plan.json'), 'w') as fh:
        json.dump({'rows': 4, 'images': 3}, fh)
    d._GATE.clear()
    check(
        'g7 a re-run is refused while running, while planning, with no '
        'promoted model and before the upstream is done -- and nothing moves',
        all(not r.get('ok') for r in refusals.values()) and not started
        and sorted(os.listdir(gdir)) == snap
        and sorted(os.listdir(ldir)) == lsnap,
        f'{ {k: (v.get("ok"), v.get("msg")) for k, v in refusals.items()} } '
        f'started {started}')

    # ── g9 (first half: the gate has NOT written since the leash list) ──────
    old = 1_000_000_000
    for f in os.listdir(gdir):
        if f.startswith('gate-'):
            os.utime(os.path.join(gdir, f), (old, old))
    os.utime(os.path.join(ldir, 'work.parquet'), (old + 60, old + 60))
    d._GATE.clear()
    d._GATE_FILES.clear()
    kept_list = d._gate_stale_plan('leash')
    os.utime(os.path.join(ldir, 'work.parquet'), (old - 60, old - 60))
    d._GATE.clear()
    d._GATE_FILES.clear()
    stale_list = d._gate_stale_plan('leash')
    check(
        'g9 the leash list is rebuilt when the gate has written since it was '
        'made, and kept when it has not', kept_list is False
        and stale_list is True and d._gate_stale_plan('gate') is False,
        f'newer list -> {kept_list}, older list -> {stale_list}')

    # ── g8 ──────────────────────────────────────────────────────────────────
    d._GATE.clear()
    d._GATE_FILES.clear()
    d._gate_shards('gate')  # the panel has counted them
    out = d.gate_control('rerun', 'gate')
    name = out.get('archived') or ''
    gnote = {}
    try:
        with open(os.path.join(gdir, 'archive', name, 'ARCHIVED.json')) as fh:
            gnote = json.load(fh)
    except (OSError, ValueError):
        pass
    after_rows = (d._gate_shards('gate') or {}).get('rows')
    check(
        'g8 a re-run sets the verdicts aside with the models off the shards, '
        'then starts the stage',
        bool(name) and gnote.get('models') == {
            'dogbin_008': 3,
            'unknown': 1
        } and gnote.get('replaced_by') == 'dogbin_009'
        and started == [('start', 'gate', [
            'archive', 'plan.json', 'shard_size', 'work.parquet'
        ])], f'{out}, note {gnote}, started {started}')
    check(
        'g8b it says where the archive is and how to undo it, and the panel '
        'stops counting what moved',
        out.get('ok') and f'data/gate/archive/{name}' in (out.get('msg') or '')
        and f'restore --stage gate --archive {name}' in (out.get('msg') or '')
        and after_rows == 0, f'{out.get("msg")}, rows now {after_rows}')

    # leash after a gate re-run: its list is the old gate's dogs
    started.clear()
    shard(os.path.join(gdir, 'gate-00000.parquet'), ['dog'] * 4, 'dogbin_009')
    with open(os.path.join(gdir, 'plan.json'), 'w') as fh:
        json.dump({'rows': 4, 'images': 1}, fh)
    os.utime(os.path.join(ldir, 'work.parquet'), (old - 60, old - 60))
    d._GATE.clear()
    d._GATE_FILES.clear()
    lout = d.gate_control('rerun', 'leash')
    check('g9b so a leash re-run after a gate re-run rebuilds its list',
          lout.get('ok') and started and started[0][2] == ['archive'],
          f'{lout}, started {started}')

    # ── g11 ─────────────────────────────────────────────────────────────────
    la = d.gate_last_archive('gate')
    gs.restore_run('gate', name, repo=root, now=4e9)
    lb = d.gate_last_archive('gate')
    check(
        'g11 the panel names the newest archive and whether it was restored',
        la and la['name'] == name and la['restored'] is False
        and la['models'].get('dogbin_008') == 3 and lb and lb['name'] != name,
        f'{la} then {lb}')
    names = sorted(os.listdir(os.path.join(gdir, 'archive')))
    with open(os.path.join(gdir, 'archive', name, 'ARCHIVED.json')) as fh:
        rnote = json.load(fh)
    check('g11b and a restored archive says so',
          bool(rnote.get('restored_at')) and len(names) == 2, f'{names}')

    # ── g12 ─────────────────────────────────────────────────────────────────
    page = d.TEMPLATE
    adm = page[page.index('id="gateCtl"'):]
    adm = adm[:adm.index('<!--/ADM-->')]
    js = page[page.index("var rr=document.getElementById('gateRerun');\n"
                         "  if(rr)rr.addEventListener"):]
    js = js[:js.index('});\n})();')]
    paint = page[page.index("var rr=document.getElementById('gateRerun');\n"
                            "    if(rr&&rr.dataset.busy"):]
    paint = paint[:paint.index('var ga=')]
    check('g12 the re-run sits beside Run, inside the admin-only controls',
          'id="gateRerun"' in adm and 'id="gateBtn"' in adm,
          'gateRerun is not in the admin span')
    check(
        'g12b it posts `rerun`, and the confirm says archived and restorable',
        "action:'rerun'" in js and 'confirm(' in js and 'archive' in js
        and 'restore puts them back' in js, 'click handler text changed')
    check(
        'g12c it is hidden while the stage runs or plans, or has nothing to '
        'replace', 'rr.hidden=!best||!j.rows||!!j.running||!!j.planning'
        in paint, 'visibility rule changed')

    # ── the detector sweep ──────────────────────────────────────────────────
    droot = os.path.join(tmp, 'detect')

    def store_files(r):
        for sub, f, body in (('shards/gen=0001/region=R/cell=C/drive=x',
                              'p0.img.parquet', b'img'),
                             ('errors/gen=0001', 'e0.parquet',
                              b'err'), ('runs/gen=0001', 'run_1_a.json',
                                        json.dumps({
                                            'run_started_epoch': 100,
                                            'model': {
                                                'engine':
                                                'data/engines/old.engine',
                                                'sha256': 'aa' * 32,
                                                'comet_run': 'train-30'
                                            }
                                        }).encode()),
                             ('runs/gen=0001', 'run_0_z.json',
                              json.dumps({
                                  'run_started_epoch': 50,
                                  'model': {
                                      'engine': 'data/engines/older.engine',
                                      'sha256': 'cc' * 32,
                                      'comet_run': 'train-29'
                                  }
                              }).encode()),
                             ('runs/gen=0001', 'run_2_b.attested.json',
                              json.dumps({
                                  'run_started_epoch': 200,
                                  'model': {
                                      'engine': None,
                                      'sha256': None
                                  }
                              }).encode()), ('', 'status.json', b'{}'),
                             ('', 'sweep.duckdb',
                              b'db'), ('worklist/gen=0001', 'ids.npy',
                                       b'ids'), ('_bootstrap/gen=_bootstrap',
                                                 'img.parquet', b'boot')):
            os.makedirs(os.path.join(r, sub), exist_ok=True)
            with open(os.path.join(r, sub, f), 'wb') as fh:
                fh.write(body)
        open(os.path.join(r, 'runs', 'gen=0001', '.lock'), 'w').close()

    def tree(r):
        out = {}
        for base, _, files in os.walk(r):
            for f in files:
                p = os.path.join(base, f)
                out[os.path.relpath(p, r)] = read(p)
        return out

    store_files(droot)
    whole = tree(droot)

    # ── s5 ──────────────────────────────────────────────────────────────────
    cm = sa.current_model(droot) or {}
    check('s5 the store\'s model is its newest run\'s that recorded an engine',
          cm.get('comet_run') == 'train-30', f'{cm}')

    # ── s2 ──────────────────────────────────────────────────────────────────
    holder = open(os.path.join(droot, 'runs', 'gen=0001', '.lock'), 'a')
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    err = raises(lambda: sa.archive_store(droot, {}, 'x', now=1e9))
    fcntl.flock(holder, fcntl.LOCK_UN)
    holder.close()
    check(
        's2 an archive refuses while a sweep holds the store, and moves '
        'nothing', err == 'RuntimeError' and tree(droot) == whole,
        f'raised {err}')

    # ── s1 ──────────────────────────────────────────────────────────────────
    note = sa.archive_store(droot, {'train-30': 'all'}, 'train-31', now=1e9)
    arch = note and note['path']
    kept_top = sorted(os.listdir(droot))
    moved_ok = bool(arch) and all(
        read(os.path.join(arch, k)) == v
        for k, v in whole.items() if k.split(os.sep)[0] in sa.MOVES)
    check(
        's1 an archive moves what the sweep wrote, byte for byte, and keeps '
        'the worklist and the bootstrap', moved_ok
        and kept_top == ['_bootstrap', 'archive', 'worklist']
        and sorted(note['files']) == sorted(sa.MOVES),
        f'left {kept_top}, moved {note and note["files"]}')

    # ── s3 ──────────────────────────────────────────────────────────────────
    check(
        's3 nothing to move is no archive',
        sa.archive_store(droot, {}, 'x', now=2e9) is None
        and os.listdir(os.path.join(droot, 'archive')) == [note['name']])
    with open(os.path.join(droot, 'status.json'), 'wb') as fh:
        fh.write(b'later')
    twice = raises(lambda: sa.archive_store(droot, {}, 'x', now=1e9))
    check(
        's3b an archive is never written into one that exists, and then '
        'nothing moves', twice == 'FileExistsError'
        and read(os.path.join(droot, 'status.json')) == b'later'
        and read(os.path.join(arch, 'status.json')) == b'{}', f'{twice}')
    os.remove(os.path.join(droot, 'status.json'))

    # ── s4 ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.join(droot, 'shards'))
    with open(os.path.join(droot, 'shards', 'new.img.parquet'), 'wb') as fh:
        fh.write(b'new')
    try:
        got = sa.restore_store(droot, note['name'], now=3e9)
    except Exception as e:  # noqa: BLE001 -- a raise is a failed restore
        got = {'raised': repr(e)}
    back = {
        k: v
        for k, v in tree(droot).items() if not k.startswith('archive')
    }
    aside = got.get('set_aside')
    planted = os.path.join(droot, sa.NOTE)
    with open(planted, 'w') as fh:
        json.dump({'files': []}, fh)
    bad = [
        raises(lambda n=n: sa.restore_store(droot, n))
        for n in ('', 'nope', '..', '.', '../x')
    ]
    os.remove(planted)
    restored_note = sa.archives(droot)
    check(
        's4 a restore puts the sweep back, sets aside what was there, marks '
        'the archive and refuses a name that is not one', back == whole
        and bool(aside) and os.path.exists(
            os.path.join(droot, 'archive', aside, 'shards', 'new.img.parquet'))
        and any(n['name'] == note['name'] and n.get('restored_at')
                for n in restored_note) and all(x == 'ValueError'
                                                for x in bad),
        f'{got.get("raised") or ""} same {back == whole}, aside {aside}, '
        f'refusals {bad}')

    # ── dashboard: s6 / s7 / s8 ─────────────────────────────────────────────
    shutil.rmtree(droot)
    store_files(droot)
    whole = tree(droot)
    drepo = os.path.join(tmp, 'drepo')
    os.makedirs(os.path.join(drepo, 'data', 'engines'))
    for e in ('old.engine', 'new.engine'):
        open(os.path.join(drepo, 'data', 'engines', e), 'wb').close()
    d.REPO = drepo
    d._detect_root = lambda: droot
    spawned = []
    d._sweep_spawn = lambda engine=None: spawned.append(engine)
    d.gate_pids = lambda stage='gate': []
    d.gate_planning = lambda: False
    dbest = {
        'projects': {
            'dogdetection': {
                'best': {
                    'run': 'train-31',
                    'engine': 'data/engines/new.engine',
                    'sha256_engine': 'bb' * 32
                }
            }
        }
    }
    refused = {}
    d.best_models = lambda: dbest
    d.sweep_pids = lambda: [7]
    refused['running'] = d.sweep_control('rerun')
    d.sweep_pids = lambda: []
    d.gate_pids = lambda stage='gate': [9] if stage == 'leash' else []
    refused['stage reading'] = d.sweep_control('rerun')
    d.gate_pids = lambda stage='gate': []
    d.best_models = lambda: {'projects': {}}
    refused['no best'] = d.sweep_control('rerun')
    d.best_models = lambda: {
        'projects': {
            'dogdetection': {
                'best': {
                    'run': 'train-31',
                    'engine': 'data/engines/missing.engine'
                }
            }
        }
    }
    refused['no engine'] = d.sweep_control('rerun')
    check(
        's6 a detector re-run is refused while it runs, while a stage reads '
        'the store, with nothing promoted or with no engine -- and nothing '
        'moves',
        all(not r.get('ok') for r in refused.values()) and not spawned
        and tree(droot) == whole,
        f'{ {k: (v.get("ok"), v.get("msg")) for k, v in refused.items()} }, '
        f'spawned {spawned}')

    # ── s8 ──────────────────────────────────────────────────────────────────
    res = d.sweep_control('resume')
    os.remove(os.path.join(drepo, 'data', 'engines', 'old.engine'))
    gone = d.sweep_control('resume')
    check(
        's8 Resume loads the store\'s own engine, and refuses when it is gone',
        res.get('ok') and spawned[:1] == ['data/engines/old.engine']
        and not gone.get('ok') and len(spawned) == 1,
        f'resume {res}, spawned {spawned}, gone {gone}')
    open(os.path.join(drepo, 'data', 'engines', 'old.engine'), 'wb').close()
    spawned.clear()

    # ── s7 ──────────────────────────────────────────────────────────────────
    d.best_models = lambda: dbest
    out = d.sweep_control('rerun')
    name = out.get('archived') or ''
    check(
        's7 a re-run sets the store aside and starts on the promoted engine, '
        'saying where the archive is and how to undo it',
        out.get('ok')
        and spawned == [os.path.join(drepo, 'data', 'engines', 'new.engine')]
        and os.path.isdir(os.path.join(droot, 'archive', name, 'shards'))
        and not os.path.exists(os.path.join(droot, 'shards'))
        and f'restore --archive {name}' in (out.get('msg') or ''),
        f'{out}, spawned {spawned}')

    # ── s7b ─────────────────────────────────────────────────────────────────
    # the same model again: allowed, and the answer says so
    shutil.rmtree(droot)
    store_files(droot)
    spawned.clear()
    d.best_models = lambda: {
        'projects': {
            'dogdetection': {
                'best': {
                    'run': 'train-30',
                    'engine': 'data/engines/old.engine',
                    'sha256_engine': 'aa' * 32
                }
            }
        }
    }
    same = d.sweep_control('rerun')
    check(
        's7b a re-run with the model that wrote the store is allowed, sets it '
        'aside like any other, and says it is the same model',
        same.get('ok')
        and spawned == [os.path.join(drepo, 'data', 'engines', 'old.engine')]
        and bool(same.get('archived'))
        and 'the model that wrote it' in (same.get('msg') or ''),
        f'{same}, spawned {spawned}')
    d.best_models = lambda: dbest

    # ── s9 ──────────────────────────────────────────────────────────────────
    gw = os.path.join(drepo, 'data', 'gate')
    os.makedirs(gw)
    with open(os.path.join(gw, 'work.parquet'), 'wb') as fh:
        fh.write(b'w')
    os.utime(os.path.join(gw, 'work.parquet'), (1e9, 1e9))
    os.makedirs(os.path.join(droot, 'runs', 'gen=0001'), exist_ok=True)
    mf = os.path.join(droot, 'runs', 'gen=0001', 'run_3_c.json')
    with open(mf, 'w') as fh:
        json.dump({'run_started_epoch': 300, 'model': {'engine': 'x'}}, fh)
    os.utime(mf, (1e9 - 60, 1e9 - 60))
    before_list = d._gate_stale_plan('gate')
    os.utime(mf, (1e9 + 60, 1e9 + 60))
    after_list = d._gate_stale_plan('gate')
    d.sweep_pids = lambda: [7]
    d._GATE.clear()
    gref = d.gate_control('rerun', 'gate')
    d.sweep_pids = lambda: []
    check(
        's9 the gate list is out of date once the detector has run since it '
        'was built, and the gate waits for a running sweep',
        before_list is False and after_list is True and not gref.get('ok')
        and 'sweep' in (gref.get('msg') or ''),
        f'before {before_list}, after {after_list}, gate {gref}')

    # ── s10 ─────────────────────────────────────────────────────────────────
    import sweep
    ns = argparse.Namespace(engine=os.path.join(tmp, 'nope.engine'))
    import contextlib
    import io
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            rc = sweep.cmd_run(ns)
    except BaseException as e:  # noqa: BLE001 -- it went past the refusal
        rc = repr(e)[:120]
    src = open(sweep.__file__).read()
    check(
        's10 sweep.py --engine replaces the configured engine and refuses one '
        'that is not on disk', rc == 1 and "r.add_argument('--engine'" in src
        and "cfg['engine'] = (args.engine" in src, f'rc {rc}')

    # ── s11 ─────────────────────────────────────────────────────────────────
    page = d.TEMPLATE
    adm = page[page.index('id="sweepCtl"'):]
    adm = adm[:adm.index('<!--/ADM-->')]
    js = page[page.index("if(rr)rr.addEventListener('click',function(e){\n"
                         "    /* inside <summary>, like"):]
    js = js[:js.index("b.addEventListener('click'")]
    paint_ = page[page.index('function paintRerun(j){'):]
    paint_ = paint_[:paint_.index('var la=')]
    check(
        's11 the detector re-run sits beside Resume, admin only, never folds '
        'the panel, posts `rerun`, and is never painted disabled -- a '
        'disabled button there looked enabled and did nothing',
        'id="sweepRerun"' in adm and "action:'rerun'" in js
        and 'e.preventDefault()' in js and 'sweep_archive.py restore' in js
        and 'rr.disabled=false;' in paint_ and 'rr.disabled=!' not in paint_
        and 'rr.hidden=!best||!!j.running;' in paint_, 'page text changed')
    check(
        's11b its confirm says when the model is the same one, and a '
        'disabled sweep control looks disabled',
        'expect the same boxes back' in js and 'rr.dataset.same' in js
        and '.rbtn.sw:disabled{opacity:.45' in page,
        'confirm or style changed')

    # ── v1 ──────────────────────────────────────────────────────────────────
    shutil.rmtree(droot)
    store_files(droot)
    cm = sa.current_model(droot) or {}
    d.best_models = lambda: dbest
    st = d.sweep_model_state()
    check(
        'v1 the store\'s model carries its run start, passed on beside the '
        'promoted model',
        cm.get('run_started') is None and 'run_started' in cm
        and st.get('store_model') == 'train-30'
        and st.get('best_model') == 'train-31' and st.get('same') is False
        and 'store_started' in st, f'{cm}, {st}')
    with open(os.path.join(droot, 'runs', 'gen=0001', 'run_1_a.json')) as fh:
        mdoc = json.load(fh)
    mdoc['run_started'] = '2026-08-02 20:09:33'
    with open(os.path.join(droot, 'runs', 'gen=0001', 'run_1_a.json'),
              'w') as fh:
        json.dump(mdoc, fh)
    check('v1b and the start is the newest run\'s own',
          d.sweep_model_state().get('store_started') == '2026-08-02 20:09:33',
          f'{d.sweep_model_state()}')

    # ── v2 ──────────────────────────────────────────────────────────────────
    page = d.TEMPLATE
    det_on = page[page.index('<div id="detOn">'):]
    det_on = det_on[:det_on.index('<div class="kpis"')]
    gate_on = page[page.index('<div id="gateOn" hidden>'):]
    gate_on = gate_on[:gate_on.index('<div class="kpis"')]
    ml = page[page.index('function modelLine(what,had,when,best){'):]
    ml = ml[:ml.index('\n}\n') + 2]
    hint = page[page.index('<span id="stgHint">'):]
    hint = hint[:hint.index('</span>')]
    switch = page[page.index("var h=document.getElementById('stgHint');"):]
    switch = switch[:switch.index('var dl=')]
    check(
        'v2 each tab opens with the model line, painted from the store\'s '
        'model and the promoted one', 'id="detModel"' in det_on
        and 'id="gModel"' in gate_on
        and "modelLine('Last sweep',j.store_model," in page
        and "modelLine('Verdicts by'," in page, 'model lines missing')
    got = None
    node = shutil.which('node')
    if node:
        import subprocess
        js_ = (ml + ';console.log(JSON.stringify(['
               "modelLine('Last sweep','train-30','started x','train-30'),"
               "modelLine('Last sweep','train-30','','train-31'),"
               "modelLine('Last sweep','<i>a</i>','','<b>')]))")
        try:
            got = json.loads(
                subprocess.run([node, '-e', js_],
                               capture_output=True,
                               text=True,
                               timeout=30).stdout)
        except Exception as e:  # noqa: BLE001
            got = repr(e)
    check(
        'v2b it says "same model" or "different", and escapes the names',
        isinstance(got, list) and 'same model' in got[0]
        and 'train-30' in got[0] and 'different' in got[1]
        and 'train-31' in got[1] and '<i>' not in got[2]
        and '&lt;i&gt;' in got[2] and '&lt;b&gt;' in got[2],
        f'{got}' if node else 'node is not installed -- the line was never '
        'rendered')
    check(
        'v2c the header names no model: the model line does',
        all(n not in hint + switch
            for n in ('yolo26x', 'dogbin_0', 'leash_v2', 'train-30')),
        f'{hint} / {switch}'[:200])

    # ── v3 ──────────────────────────────────────────────────────────────────
    src_d = open(d.__file__).read()
    check(
        'v3 the gate panel is told when its newest shard landed',
        "doc['last_shard_at'] = (_gate_shards(stage) or {}).get('last')"
        in src_d and "j.last_shard_at" in page, 'last_shard_at not wired')
finally:
    for k, v in real.items():
        setattr(d, k, v)
    d._GATE.clear()
    d._GATE_FILES.clear()
    shutil.rmtree(tmp, ignore_errors=True)

check('g0 the live stage directories were not touched', {
    s: fingerprint(p)
    for s, p in LIVE.items()
} == live_before, 'a live file changed')

check('g0b the live detect store was not touched',
      detect_print() == detect_before, 'its top level or its runs changed')

if _fails:
    print(f'\n{len(_fails)} re-run check(s) FAILED: ' + '; '.join(_fails))
    sys.exit(1)
print('\nall clear: a stage is judged again with the promoted model, and what '
      'it replaces is set aside, never deleted')
