#!/usr/bin/env python3
"""Sweep pacing overrides: per-drive caps/readers and the GPU ceiling.

  p1 drive=value pairs parse; malformed, non-numeric, zero or negative fail
  p2 finish_time without a GPU cap is the slowest lane at its read cap
  p3 a GPU cap below the drives' combined caps sets the finish time
  p4 a GPU cap above them changes nothing
  p5 no priority work: plan_rates is the plain waterfill
  p6 GPU-bound priority phase: priority lanes share the GPU, others pause
  p7 read-bound priority phase: others get only the GPU left over
  p8 no GPU cap: others pause; a lane whose priority work is done joins them
  p9 a paused TokenBucket blocks, resumes on a rate, and returns on stop

    python tools/detect/tests/adv_sweep_pacing.py
"""

import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import sweep  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def refused(pairs, cast=int):
    try:
        sweep.parse_drive_map(pairs, cast, '--x')
    except SystemExit:
        return True
    return False


def main():
    got = sweep.parse_drive_map(['jackal=8', 'bobcat=16'], int, '--readers')
    check('p1 pairs parse', got == {'jackal': 8, 'bobcat': 16}, got)
    check('p1 none is empty', sweep.parse_drive_map(None, int, '--x') == {})
    for bad in (['jackal'], ['=3'], ['jackal=x'], ['jackal=0'], ['jackal=-2']):
        check(f'p1 refuses {bad}', refused(bad))
    check(
        'p1 float caps parse',
        sweep.parse_drive_map(['jackal=90.5'], float,
                              '--cap') == {'jackal': 90.5})

    rem = {'a': 1000.0, 'b': 3000.0}
    cap = {'a': 10.0, 'b': 10.0}
    check('p2 slowest lane at its cap', sweep.finish_time(rem, cap) == 300.0)
    check('p3 GPU cap binds', sweep.finish_time(rem, cap, 5.0) == 800.0)
    check('p4 loose GPU cap is a no-op',
          sweep.finish_time(rem, cap, 1000.0) == 300.0)

    def close(a, b):
        return set(a) == set(b) and all(abs(a[k] - b[k]) < 1e-6 for k in a)

    rem = {'capy': 1000.0, 'jack': 1000.0, 'bob': 500.0}
    cap = {'capy': 50.0, 'jack': 100.0, 'bob': 50.0}
    t = sweep.finish_time(rem, cap, 100.0)
    check('p5 no priority left is the waterfill',
          close(sweep.plan_rates(rem, {'capy': 0, 'jack': 0, 'bob': 0}, cap,
                                 100.0), {d: rem[d] / t for d in rem}))
    got = sweep.plan_rates(rem, {'capy': 600, 'jack': 600, 'bob': 0}, cap,
                           100.0)
    check('p6 GPU-bound: priority lanes split the GPU, the other pauses',
          close(got, {'capy': 50.0, 'jack': 50.0, 'bob': 0.0}), got)
    got = sweep.plan_rates(rem, {'capy': 0, 'jack': 500, 'bob': 0}, cap,
                           160.0)
    check('p7 read-bound: priority lane at its cap',
          abs(got['jack'] - 100.0) < 1e-6, got)
    check('p7 the rest share the 60 img/s left over',
          abs(got['capy'] + got['bob'] - 60.0) < 1e-6
          and got['capy'] > 0 and got['bob'] > 0, got)
    got = sweep.plan_rates(rem, {'capy': 0, 'jack': 500, 'bob': 0}, cap)
    check('p8 no GPU cap: everything but the priority lane waits',
          got['capy'] == 0 and got['bob'] == 0 and got['jack'] > 0, got)

    stop = threading.Event()
    tb = sweep.engine.TokenBucket(rate=0.0, stop=stop)
    done = threading.Event()
    th = threading.Thread(target=lambda: (tb.take(), done.set()), daemon=True)
    th.start()
    time.sleep(1.2)
    check('p9 a paused bucket blocks', not done.is_set())
    tb.set_rate(1000.0)
    check('p9 a rate releases it', done.wait(2.0))
    tb.set_rate(0.0)
    done.clear()
    th = threading.Thread(target=lambda: (tb.take(), done.set()), daemon=True)
    th.start()
    time.sleep(0.7)
    stop.set()
    check('p9 stop releases a paused bucket', done.wait(2.0))
    check('p9 None is still unlimited',
          sweep.engine.TokenBucket().take() is None)
    print(f'{len(FAILS)} failed' if FAILS else 'all pacing checks passed')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
