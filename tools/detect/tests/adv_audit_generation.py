#!/usr/bin/env python3
"""An audit pool and the frame picker never mix detector generations.

det_idx is numbered per detector generation: gen 1's box 3 of a frame is not
gen 2's box 3. The audit ledgers (verdicts, redraws, crops) are keyed
'<image_id>#<det_idx>' against a gen-1 pool, so the moment data/gate holds
gen-2 verdicts a "rebuild the pool to catch up" would re-key every human
verdict onto other boxes, silently. Same for pick_label_frames.py, which
joins data/gate to the gen-1 detect store on det_idx.

  a1 a pool.json with no gen key is gen 1; over a live gen-1 plan it is no
     mismatch, and staleness reads as before. A plan naming no detect_root
     leaves no generation open (the ledger pin is never a stand-in)
  a2 over a gen-2 plan the gen-1 audit is FROZEN: pool_info says closed for
     new pages, answers can still be changed -- stale False, behind 0, no
     mismatch to act on, never 'rebuild'; a gen-2 pool that does not exist
     is not ready, never gen 1
  a3 build refuses --gen 1 outright (gen 1 untouched), a gen-2 dir that
     holds a pool, a plan of another gen, a detect store of another gen, and
     a plan naming no detect_root
  a4 build refuses a gen-2 dir holding verdicts; a stopped build's *.tmp is
     no reason to refuse
  a5 a fresh gen-2 build lands in data/fn_audit_gen2 with gen, pool_id,
     model and the plan's own detect_root, leaves gen 1 alone, is live and
     drawable; a pool box 1 px off its detect store's is never swapped in;
     a new model swept over the same generation closes it (model_moved)
  a6 the audit page's banner says the frozen-generation and model-moved
     messages and never 'Rebuild' (AUDIT_PY=<copy> drives an edited copy)
  p1 pick_label_frames refuses a gate plan of gen != 1 in both readers, and
     lets gen 1 (or no plan) through
  m  mutants: each guard removed is caught

    AUDIT_PY=<copy> checks an edited audit.py instead of the live file (an
    edit to audit.py restarts the running dashboard). a6 and the banner
    mutant are always strict: they used to SKIP while the live audit.py had
    no banner, and a gate keyed on a word in the source would turn them
    into a silent SKIP the day that word was renamed.

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


def _store(repo, name, gen):
    """A fake detect store that says it is generation `gen`."""
    root = os.path.join(repo, name)
    os.makedirs(os.path.join(root, 'shards', 'gen=%04d' % gen), exist_ok=True)
    json.dump({'gen': gen}, open(os.path.join(root, 'status.json'), 'w'))
    return root


def _repo(plan_gen=1, detect_root=True, pool=None, parquet=False,
          verdicts=None, shards=1, out2=None, store_gen=None):
    """A fake repo: data/gate with a plan, data/fn_audit (gen 1) as asked,
    data/fn_audit_gen2 holding the files named in `out2`, a gen-1 detect
    store pinned as the ledger root, and the plan's detect store (generation
    `store_gen`, default plan_gen) when `detect_root` is True."""
    repo = tempfile.mkdtemp(prefix='adv_audit_gen_')
    store = os.path.join(repo, 'data', 'gate')
    out = os.path.join(repo, 'data', 'fn_audit')
    os.makedirs(store)
    os.makedirs(out)
    if plan_gen is not None:
        plan = {'gen': plan_gen, 'rows': 1}
        if detect_root:
            plan['detect_root'] = _store(
                repo, 'det_plan', plan_gen if store_gen is None
                else store_gen)
        json.dump(plan, open(os.path.join(store, 'plan.json'), 'w'))
    if (plan_gen or 1) >= 2:
        # the gen-2 store's protocol (a gen-1 store predates protocol.json)
        json.dump({'crop': 'full', 'imgsz': 640},
                  open(os.path.join(store, 'protocol.json'), 'w'))
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
    if out2 is not None:
        o2 = os.path.join(repo, 'data', 'fn_audit_gen2')
        os.makedirs(o2)
        for name in out2:
            with open(os.path.join(o2, name), 'w') as fh:
                fh.write('KEEP')
    with open(os.path.join(repo, 'data', 'ledger_detect_root.txt'), 'w') as fh:
        fh.write(_store(repo, 'det_pin', 1) + '\n')
    return repo


def _build(fa, repo, gen=2):
    """build() on a fake repo: (refusal message or '', {path: text} of every
    file under data/fn_audit and data/fn_audit_gen2 afterwards)."""
    fa.REPO = repo
    args = types.SimpleNamespace(stage='gate', gen=gen, memory='1GB',
                                 tmp=tempfile.mkdtemp())
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fa.build(args)
        why = ''
    except SystemExit as e:
        why = str(e)
    except Exception as e:                    # noqa: BLE001 - past the guard
        why = f'PAST-GUARD {type(e).__name__}'
    files = {}
    for d in ('fn_audit', 'fn_audit_gen2'):
        top = os.path.join(repo, 'data', d)
        for name in sorted(os.listdir(top)) if os.path.isdir(top) else ():
            p = os.path.join(top, name)
            if os.path.isfile(p):
                files[f'{d}/{name}'] = open(p, 'rb').read().decode('latin-1')
    return why, files


def fa_checks(fa, pre=''):
    """True when every check passed."""
    n0 = len(FAILS)
    made = []
    try:
        # a1 -- the gen-1 world: a gen-1 plan that names its gen-1 store
        r = _repo(plan_gen=1, pool={'shards': 1, 'protocol': 'legacy'})
        made.append(r)
        fa.REPO = r
        pi = fa.pool_info('gate', gen=1)
        check(pre + 'a1 a gen-less pool over a live gen-1 plan is gen 1, no '
              'mismatch, not frozen', pi.get('pool_gen') == 1
              and pi.get('plan_gen') == 1 and pi.get('frozen') is False
              and pi.get('gen_mismatch') is False and not pi.get('stale'),
              str({k: pi.get(k) for k in ('pool_gen', 'plan_gen', 'frozen',
                                          'gen_mismatch', 'stale')}))
        r = _repo(plan_gen=1, pool={'shards': 1, 'protocol': 'legacy'},
                  shards=3)
        made.append(r)
        fa.REPO = r
        pi = fa.pool_info('gate', gen=1)
        check(pre + 'a1 same generation, more shards: still stale as before',
              pi.get('stale') is True and pi.get('behind') == 2
              and not pi.get('gen_mismatch'), str(pi))
        # a plan that names no detect_root: no live generation, so none is
        # open (the ledger pin is never a stand-in)
        r = _repo(plan_gen=1, detect_root=False,
                  pool={'shards': 1, 'protocol': 'legacy'})
        made.append(r)
        fa.REPO = r
        pi = fa.pool_info('gate', gen=1)
        check(pre + 'a1 a plan naming no detect_root leaves no generation '
              'open', pi.get('frozen') is True and pi.get('live_gen') is None
              and fa.is_frozen('gate', 1), str(pi))
        try:
            fa.pool_info('gate')
            nodefault = False
        except TypeError:
            nodefault = True
        check(pre + 'a1 pool_info has no default generation', nodefault)
        # a2 -- the switch: the live store is gen 2
        r = _repo(plan_gen=2, pool={'shards': 1, 'protocol': 'legacy'},
                  shards=4)
        made.append(r)
        fa.REPO = r
        pi = fa.pool_info('gate', gen=1)
        msg = pi.get('message') or ''
        check(pre + 'a2 the gen-1 audit under a gen-2 store is frozen -- not '
              'stale, not behind, not a mismatch to act on',
              pi.get('frozen') is True and pi.get('stale') is False
              and pi.get('behind') == 0 and pi.get('gen_mismatch') is False
              and pi.get('unknown') is False and pi.get('live_gen') == 2
              and 'shards_now' not in pi, str(pi))
        check(pre + 'a2 the message says gen-1, closed for new pages, and '
              'answers can still be changed -- never rebuild',
              'gen-1' in msg and 'closed for new pages' in msg
              and 'can still be changed' in msg
              and 'rebuild' not in msg.lower(), msg)
        try:
            fa.pool_info('gate', gen=2)
            raised = False
        except fa.PoolIdentityError:
            raised = True
        check(pre + 'a2 no gen-2 pool at all: pool_info says not ready, '
              'it does not fall back to gen 1', raised)
        check(pre + 'a2 audit_gens lists gen 1 only until a gen-2 pool is '
              'ready', fa.audit_gens('gate') == [1], fa.audit_gens('gate'))
        # a3 -- gen 1 is never built; gen 2 only into an empty dir
        for kw in ({'pool': {'shards': 1}, 'parquet': True},
                   {'parquet': True}, {}):
            r = _repo(plan_gen=2, **kw)
            made.append(r)
            why, files = _build(fa, r, gen=1)
            check(pre + f'a3 build --gen 1 is refused ({sorted(kw)}), gen 1 '
                  f'untouched', 'gen-1' in why and 'frozen' in why
                  and files.get('fn_audit/pool.parquet') ==
                  ('POOL' if kw.get('parquet') else None)
                  and not any(k.startswith('fn_audit_gen2') for k in files),
                  f'{why!r} {files}')
        r = _repo(plan_gen=2, out2=['pool.parquet'])
        made.append(r)
        why, files = _build(fa, r)
        check(pre + 'a3 a gen-2 dir that holds a pool is never re-cut',
              'already holds' in why
              and files.get('fn_audit_gen2/pool.parquet') == 'KEEP',
              f'{why!r} {files}')
        r = _repo(plan_gen=3)
        made.append(r)
        why, _ = _build(fa, r)
        check(pre + 'a3 build --gen 2 over a gen-3 plan is refused',
              'gen-2' in why and 'gen 3' in why, why)
        r = _repo(plan_gen=2, store_gen=3)
        made.append(r)
        why, _ = _build(fa, r)
        check(pre + 'a3 a plan whose detect store is another generation is '
              'refused', 'says it is gen 3' in why, why)
        r = _repo(plan_gen=2, detect_root=False)
        made.append(r)
        why, _ = _build(fa, r)
        check(pre + 'a3 a plan naming no detect_root is refused (no pin '
              'fallback)', 'names no detect_root' in why, why)
        r = _repo(plan_gen=2, shards=0)
        made.append(r)
        why, _ = _build(fa, r)
        check(pre + 'a3 no gen-2 dir yet: not refused for generation',
              'gen-' not in why and 'shards' in why, why)
        # a4 -- answers keyed against a gen-2 pool are never re-keyed
        r = _repo(plan_gen=2, out2=['verdicts.jsonl'])
        made.append(r)
        why, files = _build(fa, r)
        check(pre + 'a4 a gen-2 dir that holds verdicts is refused',
              'already holds' in why and 'verdicts.jsonl' in why
              and files == {'fn_audit_gen2/verdicts.jsonl': 'KEEP'},
              f'{why!r} {files}')
        r = _repo(plan_gen=2, out2=['pool.parquet.tmp'], shards=0)
        made.append(r)
        why, _ = _build(fa, r)
        check(pre + 'a4 a stopped build\'s *.tmp is no reason to refuse',
              'already holds' not in why and 'shards' in why, why)
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
    for moved in (0.0, 1.0):
        r = _repo(plan_gen=2, shards=0,
                  pool={'stage': 'gate', 'rows': 7, 'shards': 1})
        made.append(r)
        g1_before = open(os.path.join(r, 'data', 'fn_audit',
                                      'pool.json')).read()
        store = os.path.join(r, 'data', 'gate')
        plan = json.load(open(os.path.join(store, 'plan.json')))
        det = os.path.join(plan['detect_root'], 'shards', 'gen=0002', 'x')
        os.makedirs(det)
        man = os.path.join(r, 'grid', 'cell')
        os.makedirs(man)
        con = duckdb.connect()
        con.execute(f"""copy (select * from (values
            ('11', 0, 0.3::float, 'gate_m'))
            t(image_id, det_idx, p_dog, model)) to
            '{os.path.join(store, 'gate-00000.parquet')}' (format parquet)""")
        con.execute(f"""copy (select * from (values
            ('11', 0, 1.0, 2.0, 3.0, 4.0, 0.5, 'c', 'd'))
            t(image_id, det_idx, x1, y1, x2, y2, conf, cell, drive)) to
            '{os.path.join(store, 'work.parquet')}' (format parquet)""")
        con.execute(f"""copy (select * from (values
            (11::UBIGINT, 0::UTINYINT, (1.0 + {moved})::FLOAT, 2.0::FLOAT,
             3.0::FLOAT, 4.0::FLOAT, 0.5::FLOAT))
            t(image_id, det_idx, x1, y1, x2, y2, conf)) to
            '{os.path.join(det, 'a.det.parquet')}' (format parquet)""")
        con.execute(f"""copy (select * from (values (11, 'sq'))
            t(image_id, "sequence")) to
            '{os.path.join(man, 'all_data_0.parquet')}' (format parquet)""")
        con.close()
        real = fa._roots
        fa._roots = lambda: {'x': os.path.join(r, 'grid')}
        try:
            why, files = _build(fa, r)
        finally:
            fa._roots = real
        g1_after = open(os.path.join(r, 'data', 'fn_audit',
                                     'pool.json')).read()
        if moved:
            check(pre + 'a5 a pool box 1 px off its detect store\'s is '
                  'never swapped in', 'disagree' in why
                  and 'fn_audit_gen2/pool.json' not in files
                  and 'fn_audit_gen2/pool.parquet' not in files,
                  f'{why!r} {sorted(files)}')
            continue
        try:
            doc = json.load(open(os.path.join(r, 'data', 'fn_audit_gen2',
                                              'pool.json')))
        except (OSError, ValueError):
            doc = {}
        check(pre + 'a5 a gen-2 build lands in data/fn_audit_gen2 and '
              'records gen, pool, model and the plan\'s detect_root',
              why == '' and doc.get('gen') == 2 and doc.get('rows') == 1
              and doc.get('detect_root') == os.path.realpath(
                  plan['detect_root'])
              and doc.get('model') == 'gate_m' and doc.get('pool_id')
              and doc.get('state') == 'ready'
              and fa.verify_pool('gate', 2) is None, f'{why!r} {doc}')
        check(pre + 'a5 ...and leaves the gen-1 pool as it was',
              g1_after == g1_before)
        fa.REPO = r
        c = fa.ctx('gate', 2)
        pi = fa.pool_info('gate', gen=2)
        check(pre + 'a5 the gen-2 pool is the live one: not frozen, no '
              'mismatch, its model still the store\'s, drawable',
              not c.frozen and pi.get('frozen') is False
              and pi.get('gen_mismatch') is False
              and pi.get('model_moved') is False
              and fa.draw_refusal(c) is None
              and fa.audit_gens('gate') == [1, 2],
              str({k: pi.get(k) for k in ('frozen', 'gen_mismatch',
                                          'model_moved', 'models_now')}))
        # a later model swept over the same generation: no new pages here
        con = duckdb.connect()
        con.execute(f"""copy (select * from (values
            ('11', 0, 0.3::float, 'gate_m2'))
            t(image_id, det_idx, p_dog, model)) to
            '{os.path.join(store, 'gate-00000.parquet')}' (format parquet)""")
        con.close()
        pi = fa.pool_info('gate', gen=2)
        why_draw = fa.draw_refusal(fa.ctx('gate', 2)) or ''
        check(pre + 'a5 a new model over the same generation: model_moved, '
              'and no new pages', pi.get('model_moved') is True
              and 'gate_m2' in why_draw and 'new audit dir' in why_draw,
              f'{pi.get("model_moved")} {why_draw!r}')


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
// a FROZEN generation: not behind anything, so never "rebuild"
STATS.pool_info = {frozen:true, stale:false, behind:0, unknown:false,
                   gen_mismatch:false, model_moved:false, gen:1, live_gen:2,
                   pool_gen:1, plan_gen:2, shards:165, shards_now:258,
                   message:'gen-1 audit \u2014 closed for new pages; answers '+
                           'can still be changed'};
paintStats(STATS);
var t = String(els.poolwarn.textContent);
chk(els.poolwarn.hidden === false, 'a frozen generation shows no banner');
chk(/gen-1/.test(t) && /closed for new pages/.test(t)
    && /answers can still be changed/.test(t), 'frozen banner reads: ' + t);
chk(!/Rebuild/.test(t), 'frozen banner prompts a rebuild: ' + t);
// even a payload that also says stale must not prompt the rebuild
STATS.pool_info.stale = true; STATS.pool_info.unknown = true;
paintStats(STATS);
t = String(els.poolwarn.textContent);
chk(/closed for new pages/.test(t) && !/Rebuild/.test(t),
    'stale/unknown outrank the frozen generation: ' + t);
// a live pool whose store has been judged by another model since
STATS.pool_info = {frozen:false, model_moved:true, stale:true, unknown:false,
                   gen_mismatch:false, shards:2, shards_now:9,
                   message:'the dog-bin gate store is now judged by m2, this '+
                           'pool by m1: no new pages here; a new model\u2019s '+
                           'audit needs a new audit dir'};
paintStats(STATS);
t = String(els.poolwarn.textContent);
chk(els.poolwarn.hidden === false && /new audit dir/.test(t)
    && !/Rebuild/.test(t), 'model-moved banner reads: ' + t);
// the old mismatch payload still never says rebuild
STATS.pool_info = {unknown:false, stale:true, gen_mismatch:true, pool_gen:1,
                   plan_gen:2, shards:165, shards_now:258, behind:0};
paintStats(STATS);
t = String(els.poolwarn.textContent);
chk(/new audit dir/.test(t) && !/Rebuild/.test(t),
    'a generation mismatch prompts a rebuild: ' + t);
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
    check(pre + 'a6 the audit banner says frozen / model moved, never '
          'Rebuild', not bad, '; '.join(bad))
    return len(FAILS) == n0


def quiet(fn, *a):
    saved = list(FAILS)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ok = fn(*a, 'mut: ')
    # a guard removed so that a later step now raises is caught too
    except (Exception, SystemExit):           # noqa: BLE001
        ok = False
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
    if audit_mod is not None:
        page_checks(audit_mod)

    fsrc, psrc = open(FA_PY).read(), open(PLF_PY).read()
    asrc = open(AUDIT_PY).read()
    mutants = [
        ('fa', 'gen-1 build not refused',
         "    if gen == 1:\n        return (f\"refusing to build a gen-1",
         "    if False:\n        return (f\"refusing to build a gen-1"),
        ('fa', 'non-empty gen-2 dir re-cut',
         "    if other:\n        return (f\"refusing to build into",
         "    if False:\n        return (f\"refusing to build into"),
        ('fa', 'refusal not raised',
         "    if why:\n        raise SystemExit(why)\n"
         "    P = paths(stage, gen=gen)\n",
         "    P = paths(stage, gen=gen)\n"),
        ('fa', 'plan gen not compared',
         "    if plan.get('gen') != gen:\n",
         "    if False:\n"),
        ('fa', 'detect store gen not compared',
         "    got = _store_gen_of(root)\n    if got != gen:\n"
         "        return (f\"refusing to build: {root}",
         "    got = _store_gen_of(root)\n    if False:\n"
         "        return (f\"refusing to build: {root}"),
        ('fa', 'live_gen falls back to the ledger pin',
         "    g, root = plan.get('gen'), plan.get('detect_root')\n",
         "    g, root = plan.get('gen'), plan.get('detect_root') or "
         "_ledger_pin()\n"),
        ('fa', 'a frozen generation still reads stale',
         "        doc.update({'frozen': True, 'stale': False, 'behind': 0,",
         "        doc.update({'frozen': True, 'stale': True, 'behind': 0,"),
        ('fa', 'frozen never reported',
         "    if live != g:\n        doc.update(",
         "    if False:\n        doc.update("),
        ('fa', 'gen not written',
         "    doc = {'stage': stage, 'gen': gen, 'pool_id'",
         "    doc = {'stage': stage, 'pool_id'"),
        ('fa', 'detect_root not the plan\'s',
         "           'detect_root': detect_root,\n",
         "           'detect_root': _ledger_pin(),\n"),
        ('fa', 'geometry never checked',
         "    _geometry_check(con, tmp, detect_root, gen)\n", ""),
        ('fa', 'model change not noticed',
         "        doc['model_moved'] = models is None or models != "
         "[doc['model']]\n",
         "        doc['model_moved'] = False\n"),
        ('fa', 'draw_refusal ignores a model change',
         "    if models != [c.meta['model']]:\n",
         "    if False:\n"),
        ('plf', 'guard gone from _load',
         "def _load(con, p_dog):\n    _gate_is_gen1()\n",
         "def _load(con, p_dog):\n"),
        ('plf', 'guard gone from seed_boxes',
         "        return {}\n    _gate_is_gen1()\n", "        return {}\n"),
        ('plf', 'only an unreadable plan refused',
         "    if gen != 1:\n", "    if gen is None:\n"),
        ('audit', 'banner ignores a frozen generation or a moved model',
         "    warn.textContent=pi.frozen||pi.model_moved\n",
         "    warn.textContent=false\n"),
    ]
    for kind, name, old, new in mutants:
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
