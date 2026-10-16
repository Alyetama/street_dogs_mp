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

EVERY PATH IS REDIRECTED to a temp directory; the live stage directories are
fingerprinted before and compared after.
"""
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
live_before = {s: fingerprint(p) for s, p in LIVE.items()}
tmp = tempfile.mkdtemp(prefix='sweep-rerun-')
real = {
    k: getattr(d, k)
    for k in ('REPO', 'gate_pids', 'gate_planning', 'best_models', '_gate_act')
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

if _fails:
    print(f'\n{len(_fails)} re-run check(s) FAILED: ' + '; '.join(_fails))
    sys.exit(1)
print('\nall clear: a stage is judged again with the promoted model, and what '
      'it replaces is set aside, never deleted')
