#!/usr/bin/env python3
"""DETECT_ROOT env override: one process can write a separate store.

A sweep with a candidate detector runs with DETECT_ROOT pointing at its own
store, so the live store (named by data/detect_root.txt) and everything that
reads it stay untouched. Every reader of the pointer -- store, status,
worklist -- must agree, or a sweep would write its shards to one root and its
status or worklist to another.

The human ledgers are the exception: their det_idx and confidences belong to
the gen-1 detector, so crop_sources.default_detect_root() follows neither the
env nor data/detect_root.txt (the live store, gen 2 after the switch-over)
but the pinned data/ledger_detect_root.txt, and refuses a missing pin, a
pinned store whose shards are not only gen=0001, and an env that disagrees
with the pin (p1-p8). harvest_flagged refuses an explicit --detect-root that
is not gen-1-only unless --any-generation (h1-h3), and so does crop_sources
collect, which build_dataset runs with the root it resolved (c1-c3). Every
MUTANT puts a wrong
behaviour back and must make its check fail.

    python tools/detect/tests/adv_detect_root_env.py
"""

import argparse
import contextlib
import inspect
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import crop_sources  # noqa: E402
import harvest_flagged  # noqa: E402
import status  # noqa: E402
import store  # noqa: E402
import worklist  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def no_env():
    os.environ.pop('DETECT_ROOT', None)


# -- the pinned ledger root ------------------------------------------------

def _store(td, name, *gens):
    root = os.path.join(td, name)
    os.makedirs(os.path.join(root, 'shards'), exist_ok=True)
    for g in gens:
        os.makedirs(os.path.join(root, 'shards', g), exist_ok=True)
    return root


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(text)


def _call(fn):
    """('ok', value) or ('refused', message); the env is always restored."""
    try:
        return 'ok', fn()
    except SystemExit as e:
        return 'refused', str(e.code)


def _mutated(fn, old, new):
    """fn re-compiled from its own source with `old` -> `new`, in a copy of
    its module's namespace (so the real function is left alone)."""
    src = inspect.getsource(fn)
    if src.count(old) != 1:
        raise AssertionError('mutant anchor %r not found once in %s'
                             % (old, fn.__name__))
    ns = dict(fn.__globals__)
    exec(compile(src.replace(old, new), fn.__code__.co_filename, 'exec'), ns)
    return ns[fn.__name__]


def check_pin(ddr=None, refusal=None):
    """p1-p8 against a fake repo. `ddr`/`refusal` stand in for
    crop_sources.default_detect_root / ledger_store_refusal (mutants)."""
    ddr = ddr or crop_sources.default_detect_root
    real_refusal = crop_sources.ledger_store_refusal
    bad = []
    saved_repo, saved_cwd = crop_sources.REPO, os.getcwd()
    saved_env = os.environ.get('DETECT_ROOT')
    with tempfile.TemporaryDirectory() as td:
        g1 = _store(td, 'g1', 'gen=0001')
        g2 = _store(td, 'g2', 'gen=0002')
        mixed = _store(td, 'mixed', 'gen=0001', 'gen=0002')
        empty = _store(td, 'empty')
        repo = os.path.join(td, 'repo')
        pin = os.path.join(repo, 'data', 'ledger_detect_root.txt')
        live = os.path.join(repo, 'data', 'detect_root.txt')
        crop_sources.REPO = repo
        if refusal:
            crop_sources.ledger_store_refusal = refusal
            ddr.__globals__['ledger_store_refusal'] = refusal
        os.environ.pop('DETECT_ROOT', None)
        try:
            # p1: no pin -> refused, even with a gen-1 live pointer to fall
            # back to
            _write(live, g1 + '\n')
            got = _call(ddr)
            if got[0] != 'refused' or 'ledger_detect_root.txt' not in got[1]:
                bad.append('p1 a missing pin is not refused: %r' % (got,))
            # p2: the pin decides, not the live pointer
            _write(live, g2 + '\n')
            _write(pin, g1 + '/\n')
            got = _call(ddr)
            if got != ('ok', g1):
                bad.append('p2 the pinned gen-1 store is not returned while '
                           'detect_root.txt names gen 2: %r' % (got,))
            # p3-p5: the pinned store must hold gen=0001 shards and only them
            for tag, root in (('p3 gen-2', g2), ('p4 gen-1+gen-2', mixed),
                              ('p5 no shards', empty),
                              ('p5 missing', os.path.join(td, 'nope'))):
                _write(pin, root + '\n')
                got = _call(ddr)
                if got[0] != 'refused' or ('gen=0001' not in got[1] and
                                           'cannot be listed' not in got[1]):
                    bad.append('%s pinned store is not refused: %r'
                               % (tag, got))
            # p6: a relative or empty pin is refused, never resolved against
            # the cwd
            os.chdir(td)
            for text in ('g1', ''):
                _write(pin, text + '\n')
                got = _call(ddr)
                if got[0] != 'refused':
                    bad.append('p6 pin %r is not refused: %r' % (text, got))
            os.chdir(saved_cwd)
            # p7: $DETECT_ROOT is compared with the PIN, not detect_root.txt
            _write(pin, g1 + '\n')
            os.environ['DETECT_ROOT'] = g2 + '/'
            got = _call(ddr)
            if got[0] != 'refused' or '$DETECT_ROOT' not in got[1]:
                bad.append('p7 $DETECT_ROOT equal to detect_root.txt but not '
                           'the pin is not refused: %r' % (got,))
            os.environ['DETECT_ROOT'] = g1 + '/'
            got = _call(ddr)
            if got != ('ok', g1):
                bad.append('p7 $DETECT_ROOT equal to the pin is refused: %r'
                           % (got,))
            # p8: the env cannot smuggle a gen-2 store past the pin either
            _write(pin, g2 + '\n')
            os.environ['DETECT_ROOT'] = g2
            got = _call(ddr)
            if got[0] != 'refused':
                bad.append('p8 a gen-2 pin agreeing with $DETECT_ROOT is not '
                           'refused: %r' % (got,))
        finally:
            os.chdir(saved_cwd)
            crop_sources.REPO = saved_repo
            crop_sources.ledger_store_refusal = real_refusal
            ddr.__globals__['ledger_store_refusal'] = real_refusal
            if saved_env is None:
                os.environ.pop('DETECT_ROOT', None)
            else:
                os.environ['DETECT_ROOT'] = saved_env
    return bad


def _harvest(main, argv):
    saved = sys.argv
    sys.argv = ['harvest_flagged.py'] + argv
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            return _call(main)
    finally:
        sys.argv = saved


def check_harvest(main=None):
    """h1-h3: harvest_flagged's explicit --detect-root. The ledger is an
    empty file, so a root that passes the guard ends at 'no flags' (0)."""
    main = main or harvest_flagged.main
    bad = []
    saved_repo = crop_sources.REPO
    saved_env = os.environ.get('DETECT_ROOT')
    with tempfile.TemporaryDirectory() as td:
        g1 = _store(td, 'g1', 'gen=0001')
        g2 = _store(td, 'g2', 'gen=0002')
        mixed = _store(td, 'mixed', 'gen=0001', 'gen=0002')
        flags = os.path.join(td, 'labels.jsonl')
        _write(flags, '')
        # no pin in this repo: an explicit root must not need one
        crop_sources.REPO = os.path.join(td, 'repo')
        os.environ.pop('DETECT_ROOT', None)
        base = ['--flags', flags, '--corrections', '']
        try:
            for tag, root in (('h1 gen-2', g2), ('h1 gen-1+gen-2', mixed)):
                got = _harvest(main, base + ['--detect-root', root])
                if got[0] != 'refused' or 'gen=0001' not in got[1]:
                    bad.append('%s --detect-root is not refused: %r'
                               % (tag, got))
            got = _harvest(main, base + ['--detect-root', g2,
                                         '--any-generation'])
            if got != ('ok', 0):
                bad.append('h2 --any-generation does not let a gen-2 '
                           '--detect-root through: %r' % (got,))
            got = _harvest(main, base + ['--detect-root', g1])
            if got != ('ok', 0):
                bad.append('h3 an explicit gen-1 --detect-root is refused: '
                           '%r' % (got,))
            got = _harvest(main, base)
            if got[0] != 'refused' or 'ledger_detect_root' not in got[1]:
                bad.append('h3 no --detect-root and no pin is not refused: '
                           '%r' % (got,))
        finally:
            crop_sources.REPO = saved_repo
            if saved_env is not None:
                os.environ['DETECT_ROOT'] = saved_env
    return bad


def check_collect(cs_main=None):
    """c1-c3: crop_sources collect's explicit --detect-root. collect() and
    low_priority() are stubbed: a root that passes the guard is returned."""
    cs_main = cs_main or crop_sources.main
    g = cs_main.__globals__
    saved = {k: g[k] for k in ('collect', 'low_priority')}
    saved_repo = crop_sources.REPO
    saved_env = os.environ.get('DETECT_ROOT')
    seen = []
    g['collect'] = lambda *a, **kw: seen.append(kw['detect_root'])
    g['low_priority'] = lambda: None
    bad = []

    def run(argv):
        del seen[:]
        got = _call(lambda: cs_main(['collect', '--family', 'leash',
                                  '--stage', 'x'] + argv))
        return ('ok', seen[0]) if got == ('ok', 0) and seen else got
    with tempfile.TemporaryDirectory() as td:
        g1 = _store(td, 'g1', 'gen=0001')
        g2 = _store(td, 'g2', 'gen=0002')
        mixed = _store(td, 'mixed', 'gen=0001', 'gen=0002')
        crop_sources.REPO = os.path.join(td, 'repo')  # no pin
        os.environ.pop('DETECT_ROOT', None)
        try:
            for tag, root in (('c1 gen-2', g2), ('c1 gen-1+gen-2', mixed)):
                got = run(['--detect-root', root])
                if got[0] != 'refused' or 'gen=0001' not in got[1]:
                    bad.append('%s --detect-root is not refused: %r'
                               % (tag, got))
            got = run(['--detect-root', g2, '--any-generation'])
            if got != ('ok', g2):
                bad.append('c2 --any-generation does not let a gen-2 '
                           '--detect-root through: %r' % (got,))
            got = run(['--detect-root', g1])
            if got != ('ok', g1):
                bad.append('c3 an explicit gen-1 --detect-root is refused: '
                           '%r' % (got,))
            got = run([])
            if got[0] != 'refused' or 'ledger_detect_root' not in got[1]:
                bad.append('c3 no --detect-root and no pin is not refused: '
                           '%r' % (got,))
        finally:
            g.update(saved)
            crop_sources.REPO = saved_repo
            if saved_env is not None:
                os.environ['DETECT_ROOT'] = saved_env
    return bad


MUTANTS = [
    ('p1 fall back to detect_root.txt',
     lambda: check_pin(_mutated(crop_sources.default_detect_root,
                                'pin = D(LEDGER_ROOT_FILE)',
                                "pin = D('detect_root.txt')"))),
    ('p3 no generation guard',
     lambda: check_pin(_mutated(crop_sources.default_detect_root,
                                'why = ledger_store_refusal(root)',
                                'why = None'))),
    ('p4 gen=0001 among others accepted',
     lambda: check_pin(refusal=_mutated(
         crop_sources.ledger_store_refusal, 'if gens != LEDGER_SHARDS:',
         "if 'gen=0001' not in gens:"))),
    ('p5 an unlistable store accepted',
     lambda: check_pin(refusal=_mutated(
         crop_sources.ledger_store_refusal,
         "return '%s cannot be listed (%s)' % (shards, e.strerror or e)",
         'return None'))),
    ('p6 a relative pin accepted',
     lambda: check_pin(_mutated(crop_sources.default_detect_root,
                                'if not os.path.isabs(root):',
                                'if False:'))),
    ('p7 $DETECT_ROOT compared with detect_root.txt',
     lambda: check_pin(_mutated(
         crop_sources.default_detect_root,
         'os.path.realpath(env) != os.path.realpath(root)',
         "os.path.realpath(env) != os.path.realpath("
         "open(D('detect_root.txt')).read().strip())"))),
    ('p7 $DETECT_ROOT not checked',
     lambda: check_pin(_mutated(crop_sources.default_detect_root,
                                'if env and os.path.realpath',
                                'if False and os.path.realpath'))),
    ('h1 explicit --detect-root unguarded',
     lambda: check_harvest(_mutated(
         harvest_flagged.main, 'if why and not args.any_generation:',
         'if False:'))),
    ('h2 --any-generation ignored',
     lambda: check_harvest(_mutated(
         harvest_flagged.main, 'if why and not args.any_generation:',
         'if why:'))),
    ('h3 explicit --detect-root routed through the pin',
     lambda: check_harvest(_mutated(
         harvest_flagged.main, '        detect_root = args.detect_root\n',
         '        detect_root = crop_sources.default_detect_root()\n'))),
    ('c1 collect --detect-root unguarded',
     lambda: check_collect(_mutated(
         crop_sources.main, 'if why and not a.any_generation:',
         'if False:'))),
    ('c2 collect --any-generation ignored',
     lambda: check_collect(_mutated(
         crop_sources.main, 'if why and not a.any_generation:',
         'if why:'))),
    ('c3 collect --detect-root routed through the pin',
     lambda: check_collect(_mutated(
         crop_sources.main, '        root = a.detect_root\n',
         '        root = default_detect_root()\n'))),
]


def pinned_root_checks():
    for fn in (check_pin, check_harvest, check_collect):
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001 - report, not die
            got = ['threw %s: %s' % (type(e).__name__, e)]
        check(fn.__name__, not got, '; '.join(got))
    for name, fn in MUTANTS:
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001
            got = ['threw %s: %s' % (type(e).__name__, e)]
        check('mutant ' + name + ' is caught', bool(got),
              '' if got else 'the check still passes with it -- it proves '
              'nothing')

    # the live pin on this machine, when there is one
    no_env()
    if os.path.exists(crop_sources.D(crop_sources.LEDGER_ROOT_FILE)):
        got = _call(crop_sources.default_detect_root)
        check('p9 the live pin names a gen-1-only store', got[0] == 'ok',
              got[1])
    else:
        print('skip p9: no data/ledger_detect_root.txt on this machine')


def main():
    no_env()
    try:
        file_root = store.get_detect_root()
    except store.StoreError:
        file_root = None
    status_file_root = status.detect_root()
    wl_file_root = (worklist.resolve_detect_root(
        argparse.Namespace(detect_root=None)) if file_root else None)

    # e1: no env -> every reader answers what it answered before
    if file_root:
        check(
            'e1 store without env reads the pointer file', file_root == open(
                store._REPO_ROOT / 'data' / 'detect_root.txt').read().strip())
        check('e1 status agrees with store without env',
              status_file_root == file_root.rstrip('/'), status_file_root)
        check('e1 worklist agrees with store without env',
              wl_file_root.rstrip('/') == file_root.rstrip('/'), wl_file_root)
    else:
        print('skip e1: no data/detect_root.txt on this machine')

    with tempfile.TemporaryDirectory() as td:
        alt = os.path.join(td, 'alt_store')
        os.environ['DETECT_ROOT'] = alt + '/'
        try:
            # e2: env wins, the same answer from all three, trailing / dropped
            check('e2 store honours DETECT_ROOT',
                  store.get_detect_root() == alt, store.get_detect_root())
            check('e2 status honours DETECT_ROOT',
                  status.detect_root() == alt, status.detect_root())
            check(
                'e2 status.json lands under DETECT_ROOT',
                status.default_status_path() == os.path.join(
                    alt, 'status.json'), status.default_status_path())
            check(
                'e2 worklist honours DETECT_ROOT',
                worklist.resolve_detect_root(
                    argparse.Namespace(detect_root=None)) == alt)
            check(
                'e2 --detect-root still beats the env',
                worklist.resolve_detect_root(
                    argparse.Namespace(detect_root='/x/y')) == '/x/y')
            check(
                'e2 store paths follow the env',
                store.pair_dir(store.get_detect_root(), 2, 'r', 'c',
                               'd').startswith(alt + os.sep))

            # e3: a relative env is refused, never resolved against the cwd
            os.environ['DETECT_ROOT'] = 'data/detect_alt'
            try:
                store.get_detect_root()
                check('e3 store refuses a relative DETECT_ROOT', False)
            except store.StoreError:
                check('e3 store refuses a relative DETECT_ROOT', True)

            # e4: an empty env means unset
            os.environ['DETECT_ROOT'] = '   '
            if file_root:
                check('e4 blank DETECT_ROOT falls back to the file',
                      store.get_detect_root() == file_root)
        finally:
            no_env()

    if file_root:
        check('e5 unsetting restores the file root',
              store.get_detect_root() == file_root)

    pinned_root_checks()
    print(f'{len(FAILS)} failed' if FAILS else 'all DETECT_ROOT checks passed')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
