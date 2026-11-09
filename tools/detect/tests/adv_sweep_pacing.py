#!/usr/bin/env python3
"""Sweep pacing overrides: per-drive caps/readers and the GPU ceiling.

  p1 drive=value pairs parse; malformed, non-numeric, zero or negative fail
  p2 finish_time without a GPU cap is the slowest lane at its read cap
  p3 a GPU cap below the drives' combined caps sets the finish time
  p4 a GPU cap above them changes nothing

    python tools/detect/tests/adv_sweep_pacing.py
"""

import os
import sys

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
    print(f'{len(FAILS)} failed' if FAILS else 'all pacing checks passed')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
