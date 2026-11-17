#!/usr/bin/env python3
"""An audit pool and the frame picker never mix detector generations.

det_idx is numbered per detector generation: gen 1's box 3 of a frame is not
gen 2's box 3. The audit ledgers (verdicts, redraws, crops) are keyed
'<image_id>#<det_idx>' against a gen-1 pool, so the moment data/gate holds
gen-2 verdicts a "rebuild the pool to catch up" would re-key every human
verdict onto other boxes, silently. Same for pick_label_frames.py, which
joins data/gate to the gen-1 detect store on det_idx.

  a1 a pool.json with no gen key is gen 1; over a gen-1 plan it is no
     mismatch, and staleness reads as before
  a2 over a gen-2 plan, pool_info says 'different generation, start a new
     audit dir' -- stale False, behind 0 -- not 'rebuild'
  a3 build refuses that pool before touching it, override or not; a legacy
     pool (pool.parquet, no pool.json) is gen 1 too
  a4 build refuses a pool with verdicts against it, unless --overwrite-judged;
     an empty verdicts file is no verdicts
  a5 a fresh build writes gen and detect_root into pool.json (gen 1: the
     pinned ledger root; gen 2: the plan's own detect_root)
  a6 the audit page's banner says the generation message and never
     'Rebuild' for a gen mismatch (AUDIT_PY=<copy> drives an edited copy)
  p1 pick_label_frames refuses a gate plan of gen != 1 in both readers, and
     lets gen 1 (or no plan) through
  m  mutants: each guard removed is caught

    AUDIT_PY=<copy> checks an edited audit.py instead of the live file (an
    edit to audit.py restarts the running dashboard). Until the banner is
    installed (the live audit.py has no 'gen_mismatch' at all), a6 and the
    banner mutant print SKIP against the live file; with AUDIT_PY set, or
    once installed, they are strict.

    python tools/detect/tests/adv_audit_generation.py      (mp14 env)
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DETECT = os.path.join(REPO, 'tools', 'detect')
FA_PY = os.path.join(DETECT, 'fn_audit.py')
PLF_PY = os.path.join(DETECT, 'pick_label_frames.py')
AUDIT_PY = os.environ.get('AUDIT_PY') or os.path.join(
    REPO, 'tools', 'dashboard', 'audit.py')
# The banner edit waits for a sanctioned dashboard restart; until then the
# live audit.py is not the file under test for a6 (see docstring).
BANNER_PENDING = (not os.environ.get('AUDIT_PY')
                  and 'gen_mismatch' not in open(AUDIT_PY).read())
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
sys.path.insert(0, DETECT)
sys.path.insert(0, HERE)

FAILS = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def load(path, name, src=None):
    """A module from `path`, or from `src` standing in for it."""
    if src is not None:
        path2 = os.path.join(tempfile.mkdtemp(), os.path.basename(path))
        with open(path2, 'w') as fh:
            fh.write(src)
        path = path2
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # a copy elsewhere would read its repo off its own path
    mod.REPO = REPO
    return mod


def _repo(plan_gen=1, detect_root=None, pool=None, parquet=False,
          verdicts=None, shards=1, ledger_root='/ledger/gen1'):
    """A fake repo: data/gate with a plan, and data/fn_audit as asked."""
    repo = tempfile.mkdtemp(prefix='adv_audit_gen_')
    store = os.path.join(repo, 'data', 'gate')
    out = os.path.join(repo, 'data', 'fn_audit')
    os.makedirs(store)
    os.makedirs(out)
    if plan_gen is not None:
        plan = {'gen': plan_gen, 'rows': 1}
        if detect_root:
            plan['detect_root'] = detect_root
        json.dump(plan, open(os.path.join(store, 'plan.json'), 'w'))
    for i in range(shards):
        open(os.path.join(store, f'gate-{i:05d}.parquet'), 'w').close()
    if pool is not None:
        json.dump(pool, open(os.path.join(out, 'pool.json'), 'w'))
    if parquet:
        with open(os.path.join(out, 'pool.parquet'), 'w') as fh:
            fh.write('POOL')
    if verdicts is not None:
        with open(os.path.join(out, 'verdicts.jsonl'), 'w') as fh:
            fh.write(verdicts)
    with open(os.path.join(repo, 'data', 'ledger_detect_root.txt'), 'w') as fh:
        fh.write(ledger_root + '\n')
    return repo


def _build(fa, repo, overwrite=False):
    """build() on a fake repo: (refusal message or '', pool.parquet text)."""
    fa.REPO = repo
    args = types.SimpleNamespace(stage='gate', memory='1GB',
                                 tmp=tempfile.mkdtemp(),
                                 overwrite_judged=overwrite)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fa.build(args)
        why = ''
    except SystemExit as e:
        why = str(e)
    except Exception as e:                    # noqa: BLE001 - past the guard
        why = f'PAST-GUARD {type(e).__name__}'
    p = os.path.join(repo, 'data', 'fn_audit', 'pool.parquet')
    return why, (open(p, 'rb').read().decode('latin-1')
                 if os.path.exists(p) else None)


def fa_checks(fa, pre=''):
    """True when every check passed."""
    n0 = len(FAILS)
    made = []
    try:
        # a1
        r = _repo(plan_gen=1, pool={'shards': 1, 'protocol': 'legacy'})
        made.append(r)
        fa.REPO = r
        pi = fa.pool_info('gate')
        check(pre + 'a1 a gen-less pool over a gen-1 plan is gen 1, no '
              'mismatch', pi.get('pool_gen') == 1 and pi.get('plan_gen') == 1
              and pi.get('gen_mismatch') is False and not pi.get('stale'),
              str({k: pi.get(k) for k in ('pool_gen', 'plan_gen',
                                          'gen_mismatch', 'stale')}))
        r = _repo(plan_gen=1, pool={'shards': 1, 'protocol': 'legacy'},
                  shards=3)
        made.append(r)
        fa.REPO = r
        pi = fa.pool_info('gate')
        check(pre + 'a1 same generation, more shards: still stale as before',
              pi.get('stale') is True and pi.get('behind') == 2
              and not pi.get('gen_mismatch'), str(pi))
        # a2
        r = _repo(plan_gen=2, detect_root='/x/gen2',
                  pool={'shards': 1, 'protocol': 'legacy'}, shards=4)
        made.append(r)
        fa.REPO = r
        pi = fa.pool_info('gate')
        msg = pi.get('message') or ''
        check(pre + 'a2 a gen-2 plan under a gen-1 pool is a generation '
              'mismatch, not stale', pi.get('gen_mismatch') is True
              and pi.get('stale') is False and pi.get('behind') == 0
              and pi.get('pool_gen') == 1 and pi.get('plan_gen') == 2,
              str(pi))
        check(pre + 'a2 the message names both generations and says start a '
              'new audit dir, do not rebuild', 'gen-1' in msg
              and 'gen-2' in msg and 'new audit dir' in msg
              and 'do not rebuild' in msg, msg)
        # a pool that recorded gen 2 over a gen-2 plan is fine
        r = _repo(plan_gen=2, pool={'shards': 4, 'gen': 2,
                                    'protocol': 'legacy'}, shards=4)
        made.append(r)
        fa.REPO = r
        pi = fa.pool_info('gate')
        check(pre + 'a2 a gen-2 pool over a gen-2 plan is no mismatch',
              pi.get('gen_mismatch') is False and pi.get('pool_gen') == 2,
              str(pi))
        # no pool at all: nothing to mismatch
        r = _repo(plan_gen=2, shards=0)
        made.append(r)
        fa.REPO = r
        check(pre + 'a2 no pool at all is no mismatch',
              fa.pool_info('gate').get('gen_mismatch') is False)
        # a3
        for ov in (False, True):
            r = _repo(plan_gen=2, pool={'shards': 1}, parquet=True)
            made.append(r)
            why, body = _build(fa, r, overwrite=ov)
            check(pre + f'a3 build refuses a gen-1 pool over a gen-2 plan '
                  f'(overwrite={ov}), pool untouched',
                  'gen-1' in why and 'gen-2' in why and body == 'POOL',
                  f'{why!r} {body!r}')
        r = _repo(plan_gen=2, parquet=True)
        made.append(r)
        why, body = _build(fa, r)
        check(pre + 'a3 a legacy pool (no pool.json) is gen 1 and refused',
              'gen-1' in why and body == 'POOL', f'{why!r} {body!r}')
        fa.REPO = r
        check(pre + 'a3 ...and pool_info calls it a mismatch',
              fa.pool_info('gate').get('gen_mismatch') is True)
        r = _repo(plan_gen=2, shards=0)
        made.append(r)
        why, _ = _build(fa, r)
        check(pre + 'a3 no pool yet: a gen-2 build is not refused for '
              'generation', 'gen-' not in why and 'shards' in why, why)
        # a4
        r = _repo(plan_gen=1, pool={'shards': 1}, parquet=True,
                  verdicts='{"key": "1#0", "verdict": "dog"}\n')
        made.append(r)
        why, body = _build(fa, r)
        check(pre + 'a4 build refuses a pool with verdicts against it',
              'verdicts' in why and '--overwrite-judged' in why
              and body == 'POOL', f'{why!r} {body!r}')
        r = _repo(plan_gen=1, pool={'shards': 0}, parquet=True,
                  verdicts='{"key": "1#0", "verdict": "dog"}\n', shards=0)
        made.append(r)
        why, _ = _build(fa, r, overwrite=True)
        check(pre + 'a4 --overwrite-judged gets past the verdict guard',
              'verdicts' not in why and 'shards' in why, why)
        r = _repo(plan_gen=1, pool={'shards': 0}, parquet=True, verdicts='',
                  shards=0)
        made.append(r)
        why, _ = _build(fa, r)
        check(pre + 'a4 an empty verdicts file is no verdicts',
              'verdicts' not in why and 'shards' in why, why)
        # a5
        a5(fa, pre, made)
    finally:
        for r in made:
            shutil.rmtree(r, ignore_errors=True)
    return len(FAILS) == n0


def a5(fa, pre, made):
    try:
        import duckdb
    except ImportError:
        print('SKIP a5: duckdb missing')
        return
    for gen, root, want in ((1, None, '/ledger/gen1'),
                            (2, '/x/gen2', '/x/gen2')):
        r = _repo(plan_gen=gen, detect_root=root, shards=0)
        made.append(r)
        store = os.path.join(r, 'data', 'gate')
        man = os.path.join(r, 'grid', 'cell')
        os.makedirs(man)
        con = duckdb.connect()
        con.execute(f"""copy (select * from (values ('11', 0, 0.3::float))
            t(image_id, det_idx, p_dog)) to
            '{os.path.join(store, 'gate-00000.parquet')}' (format parquet)""")
        con.execute(f"""copy (select * from (values
            ('11', 0, 1.0, 2.0, 3.0, 4.0, 0.5, 'c', 'd'))
            t(image_id, det_idx, x1, y1, x2, y2, conf, cell, drive)) to
            '{os.path.join(store, 'work.parquet')}' (format parquet)""")
        con.execute(f"""copy (select * from (values (11, 'sq'))
            t(image_id, "sequence")) to
            '{os.path.join(man, 'all_data_0.parquet')}' (format parquet)""")
        con.close()
        real = fa._roots
        fa._roots = lambda: {'x': os.path.join(r, 'grid')}
        try:
            why, body = _build(fa, r)
        finally:
            fa._roots = real
        try:
            doc = json.load(open(os.path.join(r, 'data', 'fn_audit',
                                              'pool.json')))
        except (OSError, ValueError):
            doc = {}
        check(pre + f'a5 a gen-{gen} build records gen and detect_root',
              why == '' and doc.get('gen') == gen
              and doc.get('detect_root') == want and doc.get('rows') == 1,
              f'{why!r} {doc}')


def plf_checks(plf, pre=''):
    n0 = len(FAILS)
    tmp = tempfile.mkdtemp(prefix='adv_plf_gen_')
    real = (plf.GATE, plf.CATALOG)
    try:
        os.makedirs(os.path.join(tmp, 'gate'))
        plf.GATE = os.path.join(tmp, 'gate', 'gate-*.parquet')
        plf.CATALOG = os.path.join(tmp, 'no_catalog.duckdb')
        plan = os.path.join(tmp, 'gate', 'plan.json')

        def run():
            out = {}
            for name, fn in (
                    ('_load', lambda: plf._load(None, 0.9)),
                    ('seed_boxes', lambda: plf.seed_boxes(
                        [{'image_id': '1', 'seed_id': '2'}], 0.9))):
                try:
                    fn()
                    out[name] = ''
                except SystemExit as e:
                    out[name] = str(e)
                except Exception as e:        # noqa: BLE001 - past the guard
                    out[name] = 'PAST-GUARD ' + type(e).__name__
            return out
        json.dump({'gen': 2}, open(plan, 'w'))
        got = run()
        check(pre + 'p1 a gen-2 gate plan is refused by _load and seed_boxes',
              all('gen 2' in v and 'refusing' in v for v in got.values()),
              str(got))
        json.dump({'gen': 1}, open(plan, 'w'))
        got = run()
        check(pre + 'p1 a gen-1 gate plan gets past the guard',
              all('refusing' not in v for v in got.values()), str(got))
        os.remove(plan)
        got = run()
        check(pre + 'p1 no plan.json gets past the guard',
              all('refusing' not in v for v in got.values()), str(got))
    finally:
        plf.GATE, plf.CATALOG = real
        shutil.rmtree(tmp, ignore_errors=True)
    return len(FAILS) == n0


TAIL = r"""
function chk(c, m){ if(!c) console.log('FAIL ' + m) }
STATS.pool_info = {unknown:false, stale:false, gen_mismatch:true, pool_gen:1,
                   plan_gen:2, shards:165, shards_now:258, behind:0,
                   message:'this pool is gen-1 and the store is now gen-2; '+
                           'start a new audit dir, do not rebuild'};
paintStats(STATS);
var t = String(els.poolwarn.textContent);
chk(els.poolwarn.hidden === false, 'a generation mismatch shows no banner');
chk(/gen-1/.test(t) && /gen-2/.test(t) && /new audit dir/.test(t)
    && /do not rebuild/.test(t), 'mismatch banner reads: ' + t);
chk(!/Rebuild/.test(t), 'mismatch banner prompts a rebuild: ' + t);
// even a payload that also says stale must not prompt the rebuild
STATS.pool_info.stale = true; STATS.pool_info.unknown = true;
paintStats(STATS);
t = String(els.poolwarn.textContent);
chk(/new audit dir/.test(t) && !/Rebuild/.test(t),
    'stale/unknown outrank the generation mismatch: ' + t);
STATS.pool_info = {unknown:false, stale:true, gen_mismatch:false, shards:2,
                   shards_now:9};
paintStats(STATS);
chk(/2 shards/.test(String(els.poolwarn.textContent)),
    'a same-generation stale pool lost its banner');
"""


def page_checks(audit_mod, pre=''):
    n0 = len(FAILS)
    if not shutil.which('node'):
        print('SKIP a6: node not on PATH')
        return True
    import adv_fn_audit as harness
    probe = os.path.join(HERE, 'audit_page_stub.js')
    harness.TAIL = TAIL
    bad = []
    harness._page_once(bad, audit_mod, probe, 'gate')
    check(pre + 'a6 the audit banner says the generation message, never '
          'Rebuild', not bad, '; '.join(bad))
    return len(FAILS) == n0


def quiet(fn, *a):
    saved = list(FAILS)
    with contextlib.redirect_stdout(io.StringIO()):
        ok = fn(*a, 'mut: ')
    FAILS[:] = saved
    return ok


def main():
    fa_checks(load(FA_PY, 'fn_audit_t'))
    plf_checks(load(PLF_PY, 'plf_t'))
    try:
        audit_mod = load(AUDIT_PY, 'audit')
    except Exception as e:                    # noqa: BLE001
        audit_mod = None
        check('a6 audit.py loads', False, f'{type(e).__name__}: {e}')
    if BANNER_PENDING:
        print('SKIP a6: the live audit.py has no gen_mismatch banner yet '
              '(install it, or run with AUDIT_PY=<edited copy>)')
    elif audit_mod is not None:
        page_checks(audit_mod)

    fsrc, psrc = open(FA_PY).read(), open(PLF_PY).read()
    asrc = open(AUDIT_PY).read()
    mutants = [
        ('fa', 'gen refusal removed',
         "    if exists and plan_gen is not None and plan_gen != "
         "_pool_gen(doc):\n",
         "    if False:\n"),
        ('fa', 'verdict refusal removed',
         "    if judged and not overwrite_judged:\n", "    if False:\n"),
        ('fa', 'override ignored',
         "_build_refusal(P, getattr(args, 'overwrite_judged', False))",
         "_build_refusal(P, False)"),
        ('fa', 'refusal not raised',
         "    if why:\n        raise SystemExit(why)\n", ""),
        ('fa', 'missing gen not read as 1',
         "return int((doc or {}).get('gen') or 1)",
         "return int((doc or {}).get('gen') or 0)"),
        ('fa', 'legacy pool.parquet not a pool',
         "        return {}, os.path.exists(P['pool'])",
         "        return {}, False"),
        ('fa', 'mismatch still stale',
         "        doc['stale'] = False\n        doc['behind'] = 0\n", ""),
        ('fa', 'mismatch never flagged',
         "doc['gen_mismatch'] = bool(exists and",
         "doc['gen_mismatch'] = bool(False and"),
        ('fa', 'gen not written', "'gen': _plan_gen(plan) or 1,", ""),
        ('fa', 'detect_root not written',
         "'detect_root': _plan_detect_root(plan),", "'detect_root': None,"),
        ('fa', 'no ledger-root fallback for gen 1',
         "if root or (_plan_gen(plan) or 1) != 1:", "if True:"),
        ('plf', 'guard gone from _load',
         "def _load(con, p_dog):\n    _gate_is_gen1()\n",
         "def _load(con, p_dog):\n"),
        ('plf', 'guard gone from seed_boxes',
         "        return {}\n    _gate_is_gen1()\n", "        return {}\n"),
        ('plf', 'only an unreadable plan refused',
         "    if gen != 1:\n", "    if gen is None:\n"),
        ('audit', 'banner ignores the mismatch',
         "    warn.textContent=pi.gen_mismatch\n",
         "    warn.textContent=false\n"),
    ]
    for kind, name, old, new in mutants:
        if kind == 'audit' and BANNER_PENDING:
            print(f'SKIP m mutant "{name}": banner not installed yet')
            continue
        src = {'fa': fsrc, 'plf': psrc, 'audit': asrc}[kind]
        if src.count(old) != 1:
            check(f'm mutant "{name}" applies', False, 'target text gone')
            continue
        mut = src.replace(old, new)
        if kind == 'fa':
            ok = quiet(fa_checks, load(FA_PY, 'fn_audit_m', mut))
        elif kind == 'plf':
            ok = quiet(plf_checks, load(PLF_PY, 'plf_m', mut))
        else:
            ok = quiet(page_checks, load(AUDIT_PY, 'audit', mut))
        check(f'm mutant "{name}" is caught', not ok)
    if FAILS:
        print(f'{len(FAILS)} failed')
        return 1
    print('an audit pool is never re-cut across detector generations, and '
          'the frame picker never joins gen-2 verdicts to gen-1 boxes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
