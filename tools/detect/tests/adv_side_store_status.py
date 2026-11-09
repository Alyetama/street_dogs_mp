#!/usr/bin/env python3
"""The sweep panel shows a side-store sweep while it runs, and only then.

A candidate detector sweeps into its own store (DETECT_ROOT at a sibling
``detect_*`` directory). The dashboard reads the live store's status.json,
so that sweep was invisible and the panel kept showing the finished run.

  s1 side_status_paths finds sibling detect_* stores that have a status.json,
     never the live root, never a non-detect_ dir, never a dir without one
  s2 a running side sweep newer than the live document wins
  s3 a stale side document (paused or crashed sweep) still wins while it is
     newer than the live one, as not running, with its totals
  s4 so does a stopped/done/failed one; s4b an OLDER side store never does
  s4c/d a bad side file or a failing listing blanks nothing
  s5 two side sweeps: the most recently published wins
  s6 detect_payload carries side_store/store so the panel can label it
  s7 no side store at all: the live document, side_store False

    python tools/detect/tests/adv_side_store_status.py
"""

import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))

import status  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def put(root, name, state='running', age=0.0, run_id=1):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'status.json'), 'w') as f:
        json.dump(
            {
                'state': state,
                'ts': time.time() - age,
                'run_id': run_id,
                'imgs_done': 10,
                'imgs_total': 100
            }, f)
    return os.path.join(d, 'status.json')


def main():
    import dashboard as dash

    with tempfile.TemporaryDirectory() as td:
        live = os.path.join(td, 'detect')
        live_p = put(td, 'detect', state='done', age=86400 * 30, run_id=100)
        side_a = put(td, 'detect_cand_a', run_id=200)
        os.makedirs(os.path.join(td, 'detect_empty'))
        put(td, 'other_store', run_id=300)
        open(os.path.join(td, 'detect_file'), 'w').close()

        # s1
        got = status.side_status_paths(live)
        check('s1 only sibling detect_* stores with a status.json',
              got == [side_a], got)
        check('s1 trailing slash on the live root changes nothing',
              status.side_status_paths(live + '/') == [side_a])

        def run_with(paths):
            dash._side_status_paths = lambda: paths
            dash._live_status_path = lambda: live_p
            dash._detect_memo.update(t=0.0, body=None)
            return dash._detect_status_path(), dash.detect_payload()

        # s2 + s6
        (p, side), body = run_with([side_a])
        check('s2 a running side sweep wins', p == side_a and side, p)
        check(
            's6 payload names the side store',
            body.get('side_store') is True
            and body.get('store') == 'detect_cand_a', body.get('store'))
        check('s6 payload is the side document', body.get('run_id') == 200)
        check('s6 side payload counts as running', body.get('running'))

        # s3: a paused/crashed side sweep still newer than the live run
        stale = put(td, 'detect_stale', age=3600, run_id=400)
        (p, side), body = run_with([stale])
        check('s3 a stale side document newer than live still wins', p == stale
              and side, p)
        check('s3 payload is not running, keeps its totals',
              body.get('running') is False and body.get('imgs_done') == 10)

        # s4: stopped/done/failed side runs newer than live stay on show
        for st in ('stopped', 'done', 'failed'):
            term = put(td, 'detect_term', state=st, run_id=500)
            (p, side), _ = run_with([term])
            check(f's4 a {st} side document newer than live wins', p == term
                  and side, p)

        # s4b: a side store OLDER than the live one never stands in for it
        # (after promotion the old store is a sibling of the new live root)
        old = put(td, 'detect_old_gen', state='done', age=86400 * 60)
        (p, side), body = run_with([old])
        check('s4b an older side document loses to live', p == live_p
              and not side, p)

        # s4c: one bad file beside the live store blanks nothing
        bad = os.path.join(td, 'detect_bad')
        os.makedirs(bad)
        with open(os.path.join(bad, 'status.json'), 'w') as f:
            f.write('{"state": "running", "ts": NaN}')
        junk = os.path.join(td, 'detect_junk')
        os.makedirs(junk)
        with open(os.path.join(junk, 'status.json'), 'w') as f:
            f.write('{trunc')
        bad_p = os.path.join(bad, 'status.json')
        junk_p = os.path.join(junk, 'status.json')
        (p, side), body = run_with([bad_p, junk_p, side_a])
        check('s4c bad side files are skipped, the good one wins', p == side_a
              and body.get('ever') is True, p)
        (p, side), body = run_with([bad_p, junk_p])
        check('s4c only bad side files: the live document', p == live_p
              and body.get('run_id') == 100, p)

        def boom():
            raise OSError('listing failed')

        dash._side_status_paths = boom
        dash._detect_memo.update(t=0.0, body=None)
        check('s4d a failing side listing falls back to live',
              dash._detect_status_path() == (live_p, False))

        # s5
        older = put(td, 'detect_old', age=60, run_id=600)
        newer = put(td, 'detect_new', age=1, run_id=700)
        (p, _), body = run_with([older, newer])
        check('s5 newest running side sweep wins', p == newer, p)
        (p, _), _ = run_with([newer, older])
        check('s5 order of discovery does not matter', p == newer, p)

        # s7
        (p, side), body = run_with([])
        check('s7 no side store: the live document', p == live_p and not side
              and body.get('store') == 'detect')

    print(f'{len(FAILS)} failed' if FAILS else 'all side-store checks passed')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
