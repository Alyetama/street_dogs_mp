#!/usr/bin/env python3
"""A tool that plans or sweeps over a store uses THAT store's generation.

`gate_store plan` and `sweep.py run` both defaulted --gen to 1. The gen-2
store holds only gen=0002, so after the live store switched:

  * `gate_store plan` (the dashboard spawns it with no --gen) matched no row,
    wrote plan.json with rows 0 and an empty work.parquet, and the runner
    "finished" it at once -- an empty verdict store, every count 0 dogs;
  * `sweep.py run` (Resume) locked runs/gen=0001 inside the gen-2 store and
    died on a worklist/gen=0001 that is not there.

What has to hold:

  s1  status.store_gen(root) is the gen the store's status.json carries;
      without a usable one, the newest worklist/gen=NNNN DIRECTORY; else None
  s2  `gate_store plan` with no --gen plans the store's gen; an explicit
      --gen still wins
  s3  a plan whose query returns 0 rows is refused (SystemExit) and writes
      nothing -- no plan.json, no work.parquet
  s4  `sweep.py run` with no --gen takes the store's gen, and a gen with no
      worklist is refused before anything is created in the store; an
      explicit --gen still wins
  s5  a store that names no generation is refused by both, not defaulted
  s6  a leash plan is refused (nothing written) unless the gate verdicts it
      joins on are the same gen of the same store: det_idx names a box only
      within one store's generation, and a gen-1 gate joined to gen-2 rows is
      NOT empty, so s3 would not catch it

Fixture stores in a temp directory, DETECT_ROOT and GATE_ROOT pointed at them.
Nothing here reads or writes data/. The MUTANTS put the old behaviour back by
editing the module source in memory and require a check to FAIL.

Run: python tools/detect/tests/adv_store_gen.py   (mp14 env)
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT = os.path.dirname(HERE)
sys.path.insert(0, DETECT)

TMP = tempfile.mkdtemp(prefix='adv_store_gen_')
# set before anything reads them; restored never (a test process)
os.environ['DETECT_ROOT'] = os.path.join(TMP, 'unset')
os.environ['GATE_ROOT'] = os.path.join(TMP, 'unset_gates')

import gate_store  # noqa: E402
import status  # noqa: E402
import store  # noqa: E402
import sweep  # noqa: E402

M = {'status': status, 'gate_store': gate_store, 'sweep': sweep}


# ── fixtures ───────────────────────────────────────────────────────────────
def _rows(iid, n):
    import pyarrow as pa
    img = pa.table({
        'image_id': pa.array([iid], pa.uint64()),
        'drive': pa.array([1], pa.uint8()),
        'status': pa.array([0], pa.uint8()),
        'n_det': pa.array([n], pa.uint16()),
        'max_conf': pa.array([0.5], pa.float32()),
        'orig_w': pa.array([100], pa.uint16()),
        'orig_h': pa.array([100], pa.uint16()),
        'reduce': pa.array([1], pa.uint8()),
        'guards': pa.array([0], pa.uint16()),
        'ts_off': pa.array([0], pa.uint32()),
        'run_id': pa.array([1], pa.uint16()),
        'shard_idx': pa.array([0], pa.uint32()),
        'model_sha8': pa.array(['ab'])}, schema=store.IMG_SCHEMA)
    det = pa.table({
        'image_id': pa.array([iid] * n, pa.uint64()),
        'det_idx': pa.array(list(range(n)), pa.uint8()),
        'conf': pa.array([0.9 - 0.1 * i for i in range(n)], pa.float32()),
        'x1': pa.array([10.0] * n, pa.float32()),
        'y1': pa.array([10.0] * n, pa.float32()),
        'x2': pa.array([90.0] * n, pa.float32()),
        'y2': pa.array([90.0] * n, pa.float32()),
        'run_id': pa.array([1] * n, pa.uint16()),
        'shard_idx': pa.array([0] * n, pa.uint32()),
        'leash_class': pa.array([None] * n, pa.uint8()),
        'leash_conf': pa.array([None] * n, pa.float32()),
        'model_sha8': pa.array(['ab'] * n)}, schema=store.DET_SCHEMA)
    return img, det


def fake_store(name, shards=None, status_gen=None, worklists=(),
               status_doc=None):
    """A store under TMP. shards: {gen: [(image_id, n_boxes), ...]}."""
    import pyarrow.parquet as pq
    root = os.path.join(TMP, name)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    store.ensure_bootstrap(root)
    for gen, frames in (shards or {}).items():
        d = os.path.join(root, 'shards', f'gen={gen:04d}', 'region=r',
                         'cell=c', 'drive=d1')
        os.makedirs(d)
        for k, (iid, n) in enumerate(frames):
            img, det = _rows(iid, n)
            base = f's00000.p{k:06d}_{k + 1:06d}'
            pq.write_table(img, os.path.join(d, base + '.img.parquet'))
            pq.write_table(det, os.path.join(d, base + '.det.parquet'))
    for gen, with_dirs in worklists:
        wd = os.path.join(root, 'worklist', f'gen={gen:04d}')
        os.makedirs(wd)
        if with_dirs:
            with open(os.path.join(wd, '_dirs.json'), 'w') as fh:
                json.dump([], fh)
    doc = status_doc if status_doc is not None else (
        None if status_gen is None else {'gen': status_gen,
                                         'state': 'stopped'})
    if doc is not None:
        with open(os.path.join(root, 'status.json'), 'w') as fh:
            fh.write(doc if isinstance(doc, str) else json.dumps(doc))
    return root


def plan_in(root, argv, stage='gate', upstream=None):
    """gate_store main(['plan', ...]) over `root`, into a fresh GATE_ROOT.
    upstream: (gate plan.json doc, [(image_id, det_idx, label), ...]) seeded
    into GATE_ROOT/gate first, for a leash plan.
    Returns (rc or 'exit: msg', plan doc or None, files in the stage dir)."""
    gates = os.path.join(TMP, 'gates_' + os.path.basename(root))
    shutil.rmtree(gates, ignore_errors=True)
    os.makedirs(gates)
    if upstream is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        updoc, verdicts = upstream
        ud = os.path.join(gates, 'gate')
        os.makedirs(ud)
        with open(os.path.join(ud, 'plan.json'), 'w') as fh:
            json.dump(updoc, fh)
        pq.write_table(pa.table({
            'image_id': pa.array([str(v[0]) for v in verdicts], pa.string()),
            'det_idx': pa.array([v[1] for v in verdicts], pa.int32()),
            'label': pa.array([v[2] for v in verdicts], pa.string())}),
            os.path.join(ud, 'gate-00000.parquet'))
    os.environ['DETECT_ROOT'] = root
    os.environ['GATE_ROOT'] = gates
    gs = M['gate_store']
    real = gs._resolve_roots
    gs._resolve_roots = lambda: {'d1': 'nowhere'}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = gs.main(['plan', '--stage', stage] + argv)
    except SystemExit as e:
        rc = 'exit: %s' % e
    finally:
        gs._resolve_roots = real
    d = os.path.join(gates, stage)
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    try:
        with open(os.path.join(d, 'plan.json')) as fh:
            doc = json.load(fh)
    except OSError:
        doc = None
    return rc, doc, files


def sweep_in(root, argv):
    """sweep.py main() with `run` + argv over `root`.
    Returns (exit code, stderr text, runs/ exists)."""
    os.environ['DETECT_ROOT'] = root
    sw = M['sweep']
    err = io.StringIO()
    real_argv = sys.argv
    sys.argv = ['sweep.py', 'run'] + argv
    # cmd_run imports torch before it reads the store; mp14 has none and
    # nothing here reaches a GPU
    fake_torch = sys.modules.setdefault('torch', types.ModuleType('torch'))
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(err):
            sw.main()
        rc = 0
    except SystemExit as e:
        rc = e.code
    except Exception as e:  # noqa: BLE001 -- it went past the refusal
        rc = 'threw %s: %s' % (type(e).__name__, e)
    finally:
        sys.argv = real_argv
        if sys.modules.get('torch') is fake_torch and not hasattr(
                fake_torch, '__version__'):
            del sys.modules['torch']
    return rc, err.getvalue(), os.path.exists(os.path.join(root, 'runs'))


# ── checks: each returns a list of failures ────────────────────────────────
def check_store_gen():
    sg = M['status'].store_gen
    bad = []
    cases = [
        ('status.json gen', dict(status_gen=2), 2),
        ('status.json first, over a newer worklist',
         dict(status_gen=2, worklists=[(5, True)]), 2),
        ('status.json gen 1 kept (gen 1 is a real gen)',
         dict(status_gen=1, worklists=[(2, True)]), 1),
        ('no status: newest worklist dir',
         dict(worklists=[(1, True), (3, False), (2, True)]), 3),
        ('a bool gen is not a gen',
         dict(status_doc={'gen': True}, worklists=[(4, True)]), 4),
        ('a string gen is not a gen',
         dict(status_doc={'gen': '7'}, worklists=[(4, True)]), 4),
        ('a payload that is not an object',
         dict(status_doc='[1, 2]', worklists=[(4, True)]), 4),
        ('unparsable status.json', dict(status_doc='{', worklists=[(4, True)]),
         4),
        ('nothing names a gen', dict(), None),
    ]
    for label, kw, want in cases:
        root = fake_store('sg', **kw)
        got = sg(root)
        if got != want or (got is not None and type(got) is not int):
            bad.append(f'store_gen: {label}: got {got!r}, want {want!r}')
    # a FILE named gen=0009 under worklist/ is not a generation; nor is gen=x
    root = fake_store('sg', worklists=[(2, True)])
    open(os.path.join(root, 'worklist', 'gen=0009'), 'w').close()
    os.makedirs(os.path.join(root, 'worklist', 'gen=x'))
    if sg(root) != 2:
        bad.append(f'store_gen: a stray file/dir read as a gen: {sg(root)!r}')
    # default root is the live one ($DETECT_ROOT here)
    root = fake_store('sg', status_gen=6)
    os.environ['DETECT_ROOT'] = root
    if sg() != 6:
        bad.append(f'store_gen(): not the live store ({sg()!r})')
    return bad


def _two_gen_store():
    # gen 1: one frame, one box. gen 2: two frames, three boxes. The store
    # says it is gen 2.
    return fake_store('two', shards={1: [(11, 1)], 2: [(21, 2), (22, 1)]},
                      status_gen=2, worklists=[(2, True)])


def check_plan_default_gen():
    bad = []
    root = _two_gen_store()
    rc, doc, files = plan_in(root, [])
    if rc != 0 or not doc or doc.get('gen') != 2 or doc.get('rows') != 3:
        bad.append(f'plan with no --gen: rc {rc}, plan {doc}')
    rc, doc, files = plan_in(root, ['--gen', '1'])
    if rc != 0 or not doc or doc.get('gen') != 1 or doc.get('rows') != 1:
        bad.append(f'plan --gen 1 not honoured: rc {rc}, plan {doc}')
    # no status.json: the newest worklist names the gen
    root = fake_store('wl', shards={1: [(11, 1)], 2: [(21, 2)]},
                      worklists=[(1, True), (2, True)])
    rc, doc, files = plan_in(root, [])
    if rc != 0 or not doc or doc.get('gen') != 2 or doc.get('rows') != 2:
        bad.append(f'plan, worklist fallback: rc {rc}, plan {doc}')
    return bad


def check_plan_refuses_empty():
    bad = []
    # the switch itself: a gen-2-only store, planned at the old gen 1
    root = fake_store('g2only', shards={2: [(21, 2)]}, status_gen=2,
                      worklists=[(2, True)])
    rc, doc, files = plan_in(root, ['--gen', '1'])
    if not (isinstance(rc, str) and 'no boxes' in rc):
        bad.append(f'a 0-row plan was not refused: rc {rc}, plan {doc}')
    if files:
        bad.append(f'a refused plan left files behind: {files}')
    # the store's own gen with no rows at all is refused the same way
    root = fake_store('empty', status_gen=3, worklists=[(3, True)])
    rc, doc, files = plan_in(root, [])
    if not (isinstance(rc, str) and 'no boxes' in rc and 'gen 3' in rc):
        bad.append(f'an empty store was planned: rc {rc}, plan {doc}')
    if files:
        bad.append(f'a refused plan left files behind: {files}')
    return bad


def check_sweep_default_gen():
    bad = []
    # status says gen 2, and gen 2 has no worklist: the refusal names gen 2
    # (the store's), and nothing is created in the store
    root = fake_store('sw', status_gen=2, worklists=[(1, True)])
    rc, err, runs = sweep_in(root, [])
    if rc != 1 or 'no worklist for gen 2' not in err:
        bad.append(f'sweep run with no --gen did not take gen 2: rc {rc}, '
                   f'stderr {err.strip()[-200:]!r}')
    if runs:
        bad.append('sweep run created runs/ before refusing')
    # no status.json: the newest worklist (here without _dirs.json)
    root = fake_store('sw', worklists=[(1, True), (4, False)])
    rc, err, runs = sweep_in(root, [])
    if rc != 1 or 'no worklist for gen 4' not in err or runs:
        bad.append(f'sweep run, worklist fallback: rc {rc}, runs {runs}, '
                   f'stderr {err.strip()[-200:]!r}')
    # explicit --gen still wins
    root = fake_store('sw', status_gen=2, worklists=[(2, True)])
    rc, err, runs = sweep_in(root, ['--gen', '7'])
    if rc != 1 or 'no worklist for gen 7' not in err or runs:
        bad.append(f'sweep run --gen 7 not honoured: rc {rc}, runs {runs}, '
                   f'stderr {err.strip()[-200:]!r}')
    return bad


def check_no_gen_refused():
    bad = []
    root = fake_store('nogen', shards={1: [(11, 1)]})
    rc, doc, files = plan_in(root, [])
    if not (isinstance(rc, str) and 'names no generation' in rc) or files:
        bad.append(f'plan over a store naming no gen: rc {rc}, files {files}')
    rc, err, runs = sweep_in(root, [])
    if rc != 1 or 'names no generation' not in err or runs:
        bad.append(f'sweep over a store naming no gen: rc {rc}, runs {runs}, '
                   f'stderr {err.strip()[-200:]!r}')
    return bad


def check_leash_upstream_gen():
    bad = []
    root = _two_gen_store()           # status gen 2; gen 2 = (21: 2), (22: 1)
    other = fake_store('other', shards={2: [(21, 2)]}, status_gen=2)
    one = [(21, 0, 'dog')]            # also a valid gen-1 key shape
    cases = [
        # the switch half-done: detect_root at gen 2, data/gate still gen 1
        # (the real gen-1 plan.json has no detect_root)
        ('gen-1 gate, gen-2 store', {'gen': 1, 'rows': 1}, one, []),
        ('gen-2 gate of ANOTHER store',
         {'gen': 2, 'rows': 1, 'detect_root': other}, one, []),
        ('a gate plan with no gen', {'rows': 1}, one, []),
        ('gen-2 gate, --gen 1', {'gen': 2, 'rows': 1, 'detect_root': root},
         one, ['--gen', '1']),
    ]
    for label, updoc, verdicts, argv in cases:
        rc, doc, files = plan_in(root, argv, 'leash', (updoc, verdicts))
        if not (isinstance(rc, str) and 'det_idx would name' in rc):
            bad.append(f'leash plan over {label} not refused: rc {rc}, '
                       f'plan {doc}')
        if files:
            bad.append(f'leash plan over {label} left files: {files}')
    # the matching gen of the same store plans (spelt through a symlink)
    link = os.path.join(TMP, 'two_link')
    if not os.path.islink(link):
        os.symlink(root, link)
    rc, doc, files = plan_in(root, [], 'leash',
                             ({'gen': 2, 'rows': 1, 'detect_root': link},
                              one))
    if rc != 0 or not doc or doc.get('gen') != 2 or doc.get('rows') != 1:
        bad.append(f'leash plan over its own gen-2 gate: rc {rc}, plan {doc}')
    # and a gen-1 store with the legacy gen-1 gate (no detect_root) still
    # plans: that is the live record until the switch
    g1 = fake_store('g1', shards={1: [(11, 2)]}, status_gen=1,
                    worklists=[(1, True)])
    rc, doc, files = plan_in(g1, [], 'leash',
                             ({'gen': 1, 'rows': 1}, [(11, 1, 'dog')]))
    if rc != 0 or not doc or doc.get('gen') != 1 or doc.get('rows') != 1:
        bad.append(f'leash plan over the legacy gen-1 gate: rc {rc}, '
                   f'plan {doc}')
    return bad


# ── mutants ────────────────────────────────────────────────────────────────
class AnchorGone(Exception):
    """A mutant's text is not in the module exactly once: it would test
    nothing, so it is a FAIL, never a kill."""


def _src_mutant(modname, old, new, check, more=()):
    """Run check with `old` -> `new` (and each pair in `more`) applied to the
    module's source."""
    def run():
        real = M[modname]
        with open(real.__file__) as fh:
            src = fh.read()
        for o, n in ((old, new),) + tuple(more):
            if src.count(o) != 1:
                raise AnchorGone(f'{modname}: {o!r}')
            src = src.replace(o, n)
        mod = types.ModuleType(modname)
        mod.__file__ = real.__file__
        saved = sys.modules.get(modname)
        exec(compile(src, real.__file__, 'exec'), mod.__dict__)
        M[modname] = mod
        # gate_store imports store_gen at call time, sweep at import time
        sys.modules[modname] = mod
        try:
            return check()
        finally:
            M[modname] = real
            if saved is not None:
                sys.modules[modname] = saved
    return run


MUTANTS = [
    ('store_gen ignores status.json',
     _src_mutant('status',
                 "        if isinstance(g, int) and not isinstance(g, bool) "
                 "and g >= 0:\n            return g",
                 '        pass', check_store_gen)),
    ('store_gen has no worklist fallback',
     _src_mutant('status', '    return max(gens) if gens else None',
                 '    return None', check_store_gen)),
    ('store_gen takes the oldest worklist',
     _src_mutant('status', '    return max(gens) if gens else None',
                 '    return min(gens) if gens else None', check_store_gen)),
    ('store_gen takes a bool for a gen',
     _src_mutant('status', 'and not isinstance(g, bool) ', '',
                 check_store_gen)),
    ('store_gen counts a file named gen=NNNN',
     _src_mutant('status', "num.isdigit() and os.path.isdir(\n"
                 "                os.path.join(root, 'worklist', name))",
                 'num.isdigit()', check_store_gen)),
    ('gate_store plan --gen defaults to 1 again',
     _src_mutant('gate_store',
                 "    p.add_argument('--gen', type=int, default=None,",
                 "    p.add_argument('--gen', type=int, default=1,",
                 check_plan_default_gen)),
    ('gate_store plan ignores an explicit --gen',
     _src_mutant('gate_store', '    if args.gen is None:\n'
                 '        # the store',
                 '    if True:\n        # the store',
                 check_plan_default_gen)),
    ('gate_store writes a 0-row plan',
     _src_mutant('gate_store', '    if not rows:\n', '    if False:\n',
                 check_plan_refuses_empty)),
    ('gate_store refuses an empty plan only after writing it',
     _src_mutant('gate_store',
                 "    with open(PLAN_FILE + '.tmp', 'w') as fh:",
                 "    if not rows:\n        os.makedirs(OUT_DIR, "
                 "exist_ok=True)\n        open(PLAN_FILE, 'w').write('{}')\n"
                 "        raise SystemExit('no boxes')\n"
                 "    with open(PLAN_FILE + '.tmp', 'w') as fh:",
                 check_plan_refuses_empty,
                 # and the early refusal taken out, so the late one is it
                 more=[('    if not rows:\n        # Refused',
                        '    if False:\n        # Refused')])),
    ('sweep run --gen defaults to 1 again',
     _src_mutant('sweep', "    r.add_argument('--gen', type=int, "
                 "default=None,", "    r.add_argument('--gen', type=int, "
                 "default=1,", check_sweep_default_gen)),
    ('sweep run has no worklist guard',
     _src_mutant('sweep', '    if not os.path.isfile(wl):',
                 '    if False:', check_sweep_default_gen)),
    ('sweep run ignores an explicit --gen',
     _src_mutant('sweep', '    if args.gen is None:\n        # the store',
                 '    if True:\n        # the store',
                 check_sweep_default_gen)),
    ('leash plan has no upstream gen/store guard',
     _src_mutant('gate_store', '        if up_gen != args.gen or (up_root',
                 '        if False and (up_root', check_leash_upstream_gen)),
    ('leash plan checks the gen but not the store',
     _src_mutant('gate_store', '        if up_gen != args.gen or (up_root',
                 '        if up_gen != args.gen or False and (up_root',
                 check_leash_upstream_gen)),
    ('leash plan compares the store paths unresolved',
     _src_mutant('gate_store', "os.path.realpath(up_root)\n"
                 "                                  != os.path.realpath(root)",
                 "up_root != root", check_leash_upstream_gen)),
    ('leash plan refuses a gate plan with no detect_root',
     _src_mutant('gate_store', '        if up_gen != args.gen or (up_root',
                 '        if up_gen != args.gen or not up_root or (up_root',
                 check_leash_upstream_gen)),
    ('a store naming no gen falls back to gen 1',
     _src_mutant('status', '    return max(gens) if gens else None',
                 '    return max(gens) if gens else 1',
                 check_no_gen_refused)),
]


def main():
    checks = [check_store_gen, check_plan_default_gen,
              check_plan_refuses_empty, check_sweep_default_gen,
              check_no_gen_refused, check_leash_upstream_gen]
    bad = []
    try:
        for fn in checks:
            try:
                got = fn()
            except (Exception, SystemExit) as e:  # noqa: BLE001
                got = ['threw %s: %s' % (type(e).__name__, e)]
            print(('ok   ' if not got else 'FAIL ') + fn.__name__)
            for g in got:
                print('       ' + g)
            bad += got
        for name, fn in MUTANTS:
            try:
                got = fn()
            except AnchorGone as e:
                got, killed = None, False
                print(f'FAIL mutant {name}: anchor gone {e}')
                bad.append(f'MUTANT {name}: anchor gone')
                continue
            except (Exception, SystemExit) as e:  # noqa: BLE001
                got = ['threw %s: %s' % (type(e).__name__, e)]
            killed = bool(got)
            print(('ok   mutant ' if killed else 'FAIL mutant ') + name)
            if not killed:
                bad.append(f'MUTANT {name}: the check still passes')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print()
    if bad:
        print(f'{len(bad)} FAILED')
        return 1
    print(f'all passed ({len(checks)} checks, {len(MUTANTS)} mutants)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
