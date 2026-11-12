#!/usr/bin/env python3
"""Sweep pacing overrides: per-drive caps/readers and the GPU ceiling.

  p1 drive=value pairs parse; malformed, non-numeric, zero or negative fail
  p2 finish_time without a GPU cap is the slowest lane at its read cap
  p3 a GPU cap below the drives' combined caps sets the finish time
  p4 a GPU cap above them changes nothing
  p5 no priority work: every lane keeps its finish-together pace, and
     the GPU the slowest drive cannot use goes to the others on top
  p6 GPU-bound priority phase: priority lanes share the GPU, others pause
  p7 read-bound priority phase: others get only the GPU left over
  p8 no GPU cap: others pause; a lane whose priority work is done joins them
  p11 adapt_caps: a lane that fell short drops toward what it delivered, one
      that kept up grows back (never past its configured cap), a paused
      lane is untouched
  p9 a paused TokenBucket blocks, resumes on a rate, and returns on stop
  p12 gpu_ceiling: two or more lanes short at once is the GPU (plan against what
      was delivered, drive caps untouched); one lane short is its disk; the
      ceiling grows back, never past --gpu-cap
  p13 closed loop, GPU shared with another job: the slowest drive keeps
      its full read rate, and no GPU is planned for a finished lane

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
    check('p5 GPU-bound, no priority left: plain finish-together',
          close(sweep.plan_rates(rem, {'capy': 0, 'jack': 0, 'bob': 0}, cap,
                                 100.0), {d: rem[d] / t for d in rem}))
    check('p5 no GPU cap: plain finish-together',
          close(sweep.plan_rates(rem, {'capy': 0, 'jack': 0, 'bob': 0}, cap),
                {d: rem[d] / sweep.finish_time(rem, cap) for d in rem}))
    # read-bound tail (after North America): capybara is the slowest, and
    # the others must not be held to its pace while the GPU has room
    trem = {'capy': 4.4e6, 'jack': 2.3e6, 'bob': 1.2e6, 'lynx': 1.0}
    tcap = {'capy': 23.0, 'jack': 90.0, 'bob': 75.0, 'lynx': 71.0}
    got = sweep.plan_rates(trem, {d: 0 for d in trem}, tcap, 126.0)
    check('p5 read-bound tail: the slowest drive stays at its cap',
          abs(got['capy'] - 23.0) < 1e-6, got)
    check('p5 read-bound tail: the whole GPU is planned',
          abs(sum(got.values()) - 126.0) < 1e-6, got)
    check('p5 read-bound tail: others run past the finish-together pace',
          got['jack'] > 2.3e6 / (4.4e6 / 23.0) * 2
          and got['bob'] > 1.2e6 / (4.4e6 / 23.0) * 2, got)
    check('p5 read-bound tail: no lane over its cap',
          all(got[d] <= tcap[d] + 1e-9 for d in got), got)
    check('p5 everything done: every lane rests',
          sweep.plan_rates({'a': 0.0}, {'a': 0}, {'a': 9.0}, 50.0) == {
              'a': 0.0})
    got = sweep.plan_rates(rem, {'capy': 600, 'jack': 600, 'bob': 0}, cap,
                           100.0)
    check('p6 GPU-bound: priority lanes split the GPU, the other pauses',
          close(got, {'capy': 50.0, 'jack': 50.0, 'bob': 0.0}), got)
    got = sweep.plan_rates(rem, {'capy': 0, 'jack': 500, 'bob': 0}, cap,
                           160.0)
    check('p7 read-bound: priority lane at its cap',
          abs(got['jack'] - 100.0) < 1e-6, got)
    got = sweep.plan_rates(rem, {'capy': 0, 'jack': 1, 'bob': 0}, cap,
                           160.0)
    check('p7b a lane with a sliver of priority work still gets spare GPU',
          got['jack'] > 50.0, got)
    got = sweep.plan_rates(rem, {'capy': 0, 'jack': 500, 'bob': 0}, cap,
                           160.0)
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
    rem = {'capy': 14.6e6, 'jack': 8.0e6, 'bob': 3.75e6, 'lynx': 3.38e6}
    cap = {'lynx': 71.0, 'capy': 45.0, 'jack': 32.0, 'bob': 60.0}
    got = sweep.plan_rates(rem, {'capy': 7.8e6, 'jack': 5.45e6, 'bob': 0,
                                 'lynx': 60}, cap, 126.0)
    check('p10 read-bound priority: the whole GPU is planned',
          abs(sum(got.values()) - 126.0) < 1e-6, got)
    check('p10 no lane over its cap',
          all(got[d] <= cap[d] + 1e-9 for d in got), got)
    check('p10 priority lanes keep their pace',
          abs(got['capy'] - 45.0) < 1e-6 and got['jack'] >= 31.4, got)
    cap = {'a': 90.0, 'b': 90.0, 'c': 60.0}
    eff = dict(cap)
    sweep.adapt_caps(eff, cap, {'a': 74.0, 'b': 50.0, 'c': 0.0},
                     {'a': 26.0, 'b': 49.0, 'c': 0.0})
    check('p11 a disk-bound lane drops toward what it delivered',
          abs(eff['a'] - (45.0 + 14.3)) < 1e-6, eff)
    check('p11 a lane that kept up stays at its cap', eff['b'] == 90.0, eff)
    check('p11 a paused lane is untouched', eff['c'] == 60.0, eff)
    for _ in range(20):
        sweep.adapt_caps(eff, cap, {'a': eff['a']}, {'a': 26.0})
    check('p11 converges near the delivered rate',
          26.0 < eff['a'] < 30.0, eff)
    sweep.adapt_caps(eff, cap, {'a': eff['a']}, {'a': eff['a']})
    check('p11 grows back when it keeps up', eff['a'] > 30.0, eff)
    for _ in range(20):
        sweep.adapt_caps(eff, cap, {'a': eff['a']}, {'a': eff['a']})
    check('p11 never past the configured cap', eff['a'] == 90.0, eff)
    # p12
    g, bound = sweep.gpu_ceiling(126.0, 126.0, {'a': 23.0, 'b': 68.0,
                                                'c': 35.0},
                                 {'a': 15.0, 'b': 20.0, 'c': 15.0})
    check('p12 every lane short: GPU-bound, ceiling at what was delivered',
          bound and abs(g - 50.0) < 1e-6, (g, bound))
    g, bound = sweep.gpu_ceiling(55.0, 126.0, {'a': 23.0, 'b': 68.0},
                                 {'a': 10.0, 'b': 68.0})
    check('p12 one lane short is its disk, and the ceiling grows back',
          not bound and abs(g - 57.75) < 1e-6, (g, bound))
    g, _ = sweep.gpu_ceiling(124.0, 126.0, {'a': 1.0}, {'a': 1.0})
    check('p12 never past --gpu-cap', g == 126.0, g)
    check('p12 no --gpu-cap: nothing to plan against',
          sweep.gpu_ceiling(None, None, {'a': 1.0}, {'a': 0.0}) == (None,
                                                                    False))
    got = sweep.plan_rates({'capy': 4.4e6, 'jack': 2.3e6},
                           {'capy': 0, 'jack': 0},
                           {'capy': 23.0, 'jack': 5.0, 'lynx': 71.0}, 126.0)
    check('p13 a lane with no work gets no rate at all', 'lynx' not in got,
          got)
    got = sweep.plan_rates({'capy': 4.4e6, 'lynx': 0.0}, {'capy': 0,
                                                          'lynx': 0},
                           {'capy': 23.0, 'lynx': 71.0}, 126.0)
    check('p13 no GPU is planned for a finished lane',
          got.get('lynx', 0.0) == 0.0 and abs(got['capy'] - 23.0) < 1e-6,
          got)

    # p13: the live tail, simulated. Disks read at most `disk`; the GPU
    # serves G img/s, split round-robin over the lanes that want more than
    # their share (the decoder takes one item per non-empty queue in turn).
    def serve(want, G):
        got, left, pool = {}, G, {d: w for d, w in want.items() if w > 0}
        while pool:
            share = left / len(pool)
            small = {d: w for d, w in pool.items() if w <= share}
            if not small:
                got.update({d: share for d in pool})
                break
            for d, w in small.items():
                got[d] = w
                left -= w
                del pool[d]
        return {d: got.get(d, 0.0) for d in want}

    disk = {'capy': 23.0, 'jack': 90.0, 'bob': 75.0}
    for G in (50.0, 60.0, 100.0, 140.0):
        rem = {'capy': 4.4e6, 'jack': 2.3e6, 'bob': 1.2e6}
        cap = {'capy': 90.0, 'jack': 90.0, 'bob': 75.0}
        eff, gpu, capy = dict(cap), None, []
        planned = sweep.plan_rates(rem, {d: 0 for d in rem}, eff, 126.0)
        for step in range(120):
            want = {d: min(planned[d], disk[d]) for d in rem}
            obs = serve(want, G)
            gpu, bound = sweep.gpu_ceiling(gpu, 126.0, planned, obs)
            if not bound:
                sweep.adapt_caps(eff, cap, planned, obs)
            planned = sweep.plan_rates(rem, {d: 0 for d in rem}, eff,
                                       gpu or 126.0)
            if step >= 60:
                capy.append(obs['capy'])
        mean = sum(capy) / len(capy)
        # capybara needs 23 of the GPU; G/3 round-robin is the floor
        ideal = min(23.0, G)
        check(f'p13 GPU {G:.0f} img/s shared: capybara holds ~its disk rate',
              mean >= 0.9 * ideal, f'{mean:.1f} vs {ideal:.1f}')
        check(f'p13 GPU {G:.0f}: capybara eff cap not cut below its disk',
              eff['capy'] >= 0.9 * 23.0, eff)

    print(f'{len(FAILS)} failed' if FAILS else 'all pacing checks passed')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
