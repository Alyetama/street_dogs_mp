"""
Self-test for tools/detect/status.py (spec §7.2/§7.3, Addendum A.5).

Plain python, no pytest. Exits non-zero on the first failure.

Covers:
  1. payload shape + derived fields (rates, ETA, band flag, regions, drives)
  2. atomic write under concurrent readers — a reader must NEVER observe a
     missing or half-written file while the writer republishes in a loop
  3. staleness detection in read_status (fresh / stale / terminal / absent)
  4. §7.3 guards: null ETA at zero rate, allow_nan (no bare NaN/Infinity),
     serialization failure counted, not written
"""

import json
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import status as st  # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    tag = 'ok  ' if cond else 'FAIL'
    print(f'[{tag}] {name}' + (f' — {detail}' if detail and not cond else ''))
    if not cond:
        FAILED.append(name)


class Clock:
    """Deterministic time source so window rates are exactly computable."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_writer(tmp, clock):
    return st.StatusWriter(
        run_id='r-test', gen=3, imgs_total=1000,
        status_path=os.path.join(tmp, 'status.json'),
        interval=5.0,
        drive_totals={'lynx': 600, 'bobcat': 400},
        region_totals={'South_Asia': 700, 'Europe': 300},
        gpu_fn=lambda: {'util': 97, 'mem_used_mb': 8000,
                        'mem_total_mb': 24000, 'temp': 71},
        time_fn=clock)


def test_payload(tmp):
    clock = Clock()
    w = make_writer(tmp, clock)
    # first frame: nothing done yet -> ETA must be null, not a crash (§7.3)
    check('first publish succeeds', w.publish_now())
    d = json.load(open(w.path))
    check('null ETA at zero rate', d['eta_s'] is None, repr(d['eta_s']))
    check('null positive_rate at zero done', d['positive_rate'] is None)
    check('run identity', d['run_id'] == 'r-test' and d['gen'] == 3)
    check('pid present', d['pid'] == os.getpid())

    # 60 s of progress at exactly 10 img/s
    for i in range(1, 13):
        clock.t += 5
        w.update(imgs_done=i * 50, boxes_total=i * 5, positives=i * 4,
                 drive_done={'lynx': i * 30, 'bobcat': i * 20},
                 drive_queue={'lynx': 7, 'bobcat': 3},
                 region_done={'South_Asia': i * 35, 'Europe': i * 15},
                 crops_classified=i * 10,
                 class_counts={'leashed': i * 5, 'unleashed': i * 4,
                               'not_a_dog': i * 1},
                 errors={'decode': i})
        w.publish_now()
    d = json.load(open(w.path))
    check('w60 rate = 10 img/s', d['img_per_sec']['w60'] == 10.0,
          repr(d['img_per_sec']))
    check('w900 rate = 10 img/s', d['img_per_sec']['w900'] == 10.0)
    check('eta = remaining/rate', d['eta_s'] == int((1000 - 600) / 10.0),
          repr(d['eta_s']))
    check('drive fields', d['drives']['lynx'] ==
          {'done': 360, 'total': 600, 'rate': 6.0, 'queue_depth': 7,
           'stalled': False}, repr(d['drives']))
    check('region percent', d['regions'] ==
          {'South_Asia': 60.0, 'Europe': 60.0}, repr(d['regions']))
    check('positive_rate pct', d['positive_rate'] == 8.0,
          repr(d['positive_rate']))
    check('boxes_per_img trailing', d['boxes_per_img'] == 0.1,
          repr(d['boxes_per_img']))
    check('errors surfaced', d['errors']['decode'] == 12
          and d['errors']['mount_lost'] == 0)
    check('gpu passthrough', d['gpu']['util'] == 97 and d['gpu']['temp'] == 71)

    # not_a_dog band (A.5): 1/10 classified = 10% -> inside 7–16
    check('not_a_dog in band', d['not_a_dog_rate'] == 10.0
          and d['not_a_dog_band'] == {'lo': 7.0, 'hi': 16.0, 'in_band': True})
    w.update(class_counts={'not_a_dog': 60})     # 60/120 = 50% -> out of band
    w.publish_now()
    d = json.load(open(w.path))
    check('not_a_dog out of band', d['not_a_dog_rate'] == 50.0
          and d['not_a_dog_band']['in_band'] is False)

    # stall: 35 s with no lynx progress while bobcat keeps moving
    bdone = d['drives']['bobcat']['done']
    for k in range(7):
        clock.t += 5
        w.update(drive_done={'bobcat': bdone + k + 1})
        w.publish_now()
    d = json.load(open(w.path))
    check('stalled drive flagged', d['drives']['lynx']['stalled'] is True
          and d['drives']['bobcat']['stalled'] is False, repr(d['drives']))

    # no bare NaN/Infinity anywhere in the blob (§7.3)
    blob = open(w.path).read()
    check('no NaN/Infinity in blob',
          'NaN' not in blob and 'Infinity' not in blob)
    check('blob compact', len(blob) < 12000, f'{len(blob)} bytes')

    # last_error round-trips
    w.update(last_error='decode: /gone.jpg', state='paused')
    w.publish_now()
    d = json.load(open(w.path))
    check('last_error + state', d['last_error'] == 'decode: /gone.jpg'
          and d['state'] == 'paused')

    # unserializable counter: publish fails, counted, file NOT clobbered
    before = open(w.path).read()
    w.update(last_error=object())
    check('bad payload refused', w.publish_now() is False)
    check('publish_errors counted', w.publish_errors == 1)
    check('previous file intact', open(w.path).read() == before)
    w.update(last_error=None)
    w.publish_now()
    d = json.load(open(w.path))
    check('publish_errors surfaced', d['publish_errors'] == 1)


def test_atomic_concurrent(tmp):
    """Republishing in a tight loop must never expose a bad file."""
    path = os.path.join(tmp, 'status.json')
    w = st.StatusWriter('r-atomic', 1, 10**6, status_path=path,
                        interval=0.001, gpu_fn=lambda: None)
    w.publish_now()                              # file exists before readers
    stop = threading.Event()
    bad = []

    def reader():
        while not stop.is_set():
            try:
                with open(path) as f:
                    json.load(f)
            except Exception as e:               # missing OR truncated
                bad.append(repr(e))

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    n = 0
    t0 = time.time()
    while time.time() - t0 < 2.0:
        w.update(imgs_done=n)
        w.publish_now()
        n += 1
    stop.set()
    for t in threads:
        t.join()
    check('writes during read storm', n > 50, f'only {n} writes')
    check('no torn/missing reads', not bad,
          f'{len(bad)} bad reads, first: {bad[:1]}')

    # background thread mode: starts, publishes, stop() writes terminal frame
    w2 = st.StatusWriter('r-bg', 1, 10, status_path=path, interval=0.05,
                         gpu_fn=lambda: None)
    w2.start()
    w2.update(imgs_done=10)
    time.sleep(0.3)
    w2.stop(state='done')
    d = json.load(open(path))
    check('stop publishes terminal state', d['state'] == 'done'
          and d['imgs_done'] == 10)


def test_reader(tmp):
    path = os.path.join(tmp, 's.json')
    clock = Clock()
    w = st.StatusWriter('r-read', 1, 100, status_path=path,
                        gpu_fn=lambda: None, time_fn=clock)
    w.publish_now()
    # fresh -> running
    d = st.read_status(path, stale_after=120, now_fn=lambda: clock.t + 10)
    check('fresh reads running', d['running'] is True and d['age_s'] == 10)
    # stale (>120 s) -> not running, reason visible
    d = st.read_status(path, stale_after=120, now_fn=lambda: clock.t + 121)
    check('stale reads not running', d == {
        'running': False, 'stale': True, 'age_s': 121, 'state': 'running',
        'run_id': 'r-read', 'pid': os.getpid()}, repr(d))
    # fresh but terminal state -> not running
    w.update(state='stopped')
    w.publish_now()
    d = st.read_status(path, stale_after=120, now_fn=lambda: clock.t + 5)
    check('terminal state not running', d['running'] is False
          and d['state'] == 'stopped')
    # absent / garbage -> {'running': False}
    check('absent file', st.read_status(os.path.join(tmp, 'nope.json'))
          == {'running': False})
    with open(path, 'w') as f:
        f.write('{"truncat')
    check('garbage file', st.read_status(path) == {'running': False})
    with open(path, 'w') as f:
        f.write('[1,2]')
    check('non-object payload', st.read_status(path) == {'running': False})


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_payload(os.path.join(tmp, 'a'))
        test_atomic_concurrent(os.path.join(tmp, 'b'))
        test_reader(os.path.join(tmp, 'c'))
    if FAILED:
        print(f'\n{len(FAILED)} FAILED: {FAILED}')
        return 1
    print('\nall status.py self-tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
