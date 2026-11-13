#!/usr/bin/env python3
"""Reading a drive's work in on-disk order (sweep --disk-order).

  d1 every item comes out exactly once, whatever the window
  d2 within a window, items come out in block order; windows stay in turn
  d3 offsets are looked up in inode order
  d4 a file without an offset sorts first and is still yielded
  d5 no window (0/None/1) passes the work through untouched
  d6 phys_offset reads a real file's extent and is None for a missing one
  d7 DirInodes takes inode numbers from the directory listing
  d8 --disk-order parses DRIVE=WINDOW and refuses a bad window
  d9 an error in the worklist reaches the reader; a set stop cuts a lookup
  d10 no lookup runs while a window is being read (the head stays put)

    python tools/detect/tests/adv_disk_order.py
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import sweep  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def main():
    n = 103
    work = [({'pos': i}, f'/x/{i}.jpg') for i in range(n)]
    off = {f'/x/{i}.jpg': (i * 7919) % 1000 for i in range(n)}
    ino = {f'/x/{i}.jpg': (i * 31) % 211 for i in range(n)}
    looked = []

    def phys(p):
        looked.append(p)
        return off[p]

    for w in (1, 2, 10, 50, 103, 500):
        got = list(sweep.disk_ordered(iter(work), w, phys, ino.get))
        check(f'd1 window {w}: every item exactly once',
              sorted(m['pos'] for m, _ in got) == list(range(n)))
    looked.clear()
    got = list(sweep.disk_ordered(iter(work), 10, phys, ino.get))
    ok = True
    for k in range(0, n, 10):
        chunk = got[k:k + 10]
        ok &= {m['pos'] for m, _ in chunk} == set(range(k, min(n, k + 10)))
        ok &= [off[p] for _, p in chunk] == sorted(off[p] for _, p in chunk)
    check('d2 block order within each window, windows in turn', ok)
    first = looked[:10]
    check('d3 offsets are looked up in inode order',
          [ino[p] for p in first] == sorted(ino[p] for p in first), first)
    off['/x/5.jpg'] = None
    got = list(sweep.disk_ordered(iter(work), 10, lambda p: off[p], ino.get))
    check('d4 an unmapped file sorts first in its window, still yielded',
          got[0][1] == '/x/5.jpg' and len(got) == n)
    for w in (0, None, 1):
        got = list(sweep.disk_ordered(iter(work), w, phys, ino.get))
        check(f'd5 window {w!r} passes through', got == work)

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'a.jpg')
        with open(p, 'wb') as f:
            f.write(os.urandom(1 << 16))
            f.flush()
            os.fsync(f.fileno())
        o = sweep.phys_offset(p)
        check('d6 a real file has an offset (or FIEMAP is unsupported here)',
              o is None or (isinstance(o, int) and o >= 0), o)
        check('d6 a missing file has none',
              sweep.phys_offset(os.path.join(td, 'nope.jpg')) is None)
        di = sweep.DirInodes()
        check('d7 inode from the directory listing',
              di(p) == os.stat(p).st_ino and di(os.path.join(td, 'z')) == 0)
    check('d8 DRIVE=WINDOW parses',
          sweep.parse_drive_map(['capybara=40000'], int, '--disk-order') == {
              'capybara': 40000})
    for bad in (['capybara=0'], ['capybara=-5'], ['capybara']):
        try:
            sweep.parse_drive_map(bad, int, '--disk-order')
            check(f'd8 {bad} refused', False)
        except SystemExit:
            check(f'd8 {bad} refused', True)

    def boom():
        yield work[0]
        raise RuntimeError('worklist broke')

    try:
        list(sweep.disk_ordered(boom(), 10, phys, ino.get))
        check('d9 a worklist error reaches the reader', False)
    except RuntimeError as e:
        check('d9 a worklist error reaches the reader',
              'worklist broke' in str(e))
    import threading
    ev = threading.Event()
    ev.set()
    got = list(sweep.disk_ordered(iter(work), 10, phys, ino.get, stop=ev))
    check('d9 a set stop skips the lookup, still yields every item',
          len(got) == n and got[:10] == work[:10], len(got))

    # d10: the next window is NOT looked up while this one is read
    seen = []

    def counted(p):
        seen.append(p)
        return off.get(p) or 0

    off['/x/5.jpg'] = 5
    g = sweep.disk_ordered(iter(work), 50, counted, ino.get)
    for _ in range(50):
        next(g)
    check('d10 no lookup runs while a window is being read', len(seen) == 50,
          len(seen))
    rest = list(g)
    check('d10 and nothing is lost', len(rest) == n - 50)

    print(f'{len(FAILS)} failed' if FAILS else 'all disk-order checks passed')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
