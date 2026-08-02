"""
Adversarial tests for tools/detect/status.py — hostile inputs and environment.

Written by the verification pass (not the module author). Plain python, no
pytest; exits non-zero on failure.

Hunts:
  A. _payload() exceptions (raising gpu_fn, malformed engine push) must be
     contained by publish_now(): counted, previous file intact, no raise —
     including through stop() (engine shutdown path).
  B. NaN / Infinity pushed as counters can never reach the published file.
  C. Denominator mismatch (done > total): ETA clamps to 0, regions cap 100.
  D. Unknown drive appearing mid-run (done without a total) is tolerated.
  E. Unwritable / vanished target directory (unmounted drive analogue):
     publish fails soft, counted, and recovers when the dir returns.
  F. Stale .tmp leftover from a killed writer is harmlessly overwritten.
  G. Full-corpus-scale payload (32,542,334 imgs, 4 drives, 16 regions,
     53-char run_id) stays under the 12 kB blob budget (§7.3).
  H. update() hammered from 4 threads while the daemon publishes at 1 ms:
     no exception, no torn file, zero publish errors.
  I. read_status against hostile files: directory-at-path, empty file,
     string ts (falls back to mtime), future ts (age clamps to 0).
"""

import json
import os
import shutil
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


def test_payload_exception_contained(tmp):
    path = os.path.join(tmp, 's.json')
    w = st.StatusWriter('r', 1, 100, status_path=path, gpu_fn=lambda: None)
    check('A: baseline publish', w.publish_now() is True)
    before = open(path).read()

    boom = st.StatusWriter('r', 1, 100, status_path=path,
                           gpu_fn=lambda: 1 / 0)
    try:
        ok = boom.publish_now()
        raised = False
    except Exception:
        ok, raised = None, True
    check('A: raising gpu_fn does not escape publish_now', not raised)
    check('A: raising gpu_fn returns False + counted',
          ok is False and boom.publish_errors == 1,
          f'ok={ok} errors={boom.publish_errors}')
    check('A: previous file intact after gpu_fn failure',
          open(path).read() == before)

    # malformed engine push: drive_done replaced by a non-dict
    bad = st.StatusWriter('r', 1, 100, status_path=path, gpu_fn=lambda: None)
    bad.update(drive_done=None)
    try:
        ok = bad.publish_now()
        raised = False
    except Exception:
        ok, raised = None, True
    check('A: malformed push contained', not raised and ok is False
          and bad.publish_errors == 1)
    # stop() is the engine's shutdown path — it must never raise either
    try:
        bad.stop(state='failed')
        raised = False
    except Exception:
        raised = True
    check('A: stop() never raises on a broken payload', not raised)


def test_nan_inf_never_published(tmp):
    path = os.path.join(tmp, 's.json')
    w = st.StatusWriter('r', 1, 100, status_path=path, gpu_fn=lambda: None)
    w.update(imgs_done=10)
    w.publish_now()
    before = open(path).read()
    for evil in (float('nan'), float('inf'), -float('inf')):
        w.update(imgs_done=evil)
        w.publish_now()
        w.update(positives=evil)
        w.publish_now()
        w.update(imgs_done=10, positives=0)
        w.publish_now()                          # recover with sane counters
    blob = open(path).read()
    check('B: no NaN/Infinity ever reaches the file',
          'NaN' not in blob and 'Infinity' not in blob, blob[:200])
    check('B: file always parseable', isinstance(json.loads(blob), dict))
    check('B: recovered after hostile counters',
          json.loads(blob)['imgs_done'] == 10)
    del before


def test_denominator_mismatch(tmp):
    path = os.path.join(tmp, 's.json')
    w = st.StatusWriter('r', 1, 100, status_path=path, gpu_fn=lambda: None,
                        drive_totals={'lynx': 50},
                        region_totals={'Europe': 50})
    # raw per-drive counts exceed the deduped total (§7.3's exact failure)
    w.update(imgs_done=140, drive_done={'lynx': 70},
             region_done={'Europe': 70})
    w.publish_now()
    d = json.load(open(path))
    check('C: overshoot ETA clamps to 0, never negative', d['eta_s'] == 0,
          repr(d['eta_s']))
    check('C: region percent capped at 100', d['regions']['Europe'] == 100.0)
    check('C: overshot drive not flagged stalled',
          d['drives']['lynx']['stalled'] is False)


def test_unknown_drive(tmp):
    path = os.path.join(tmp, 's.json')
    w = st.StatusWriter('r', 1, 100, status_path=path, gpu_fn=lambda: None,
                        drive_totals={'lynx': 100})
    w.update(drive_done={'ghost': 5}, region_done={'Atlantis': 5})
    ok = w.publish_now()
    d = json.load(open(path))
    check('D: unknown drive tolerated', ok and d['drives']['ghost'] ==
          {'done': 5, 'total': 0, 'rate': 0.0, 'queue_depth': 0,
           'stalled': False}, repr(d['drives']))
    check('D: unknown region tolerated (0 total -> 0%)',
          d['regions']['Atlantis'] == 0.0)


def test_unwritable_dir(tmp):
    # unmounted-drive analogue: the parent dir exists but cannot be written
    root = os.path.join(tmp, 'ro')
    os.makedirs(root)
    path = os.path.join(root, 'status.json')
    w = st.StatusWriter('r', 1, 100, status_path=path, gpu_fn=lambda: None)
    os.chmod(root, 0o500)
    try:
        ok = w.publish_now()
        check('E: unwritable dir fails soft', ok is False
              and w.publish_errors == 1)
    finally:
        os.chmod(root, 0o700)
    check('E: recovers when dir returns', w.publish_now() is True
          and json.load(open(path))['publish_errors'] == 1)

    # dir vanishes entirely -> makedirs recreates it (mount restored case)
    shutil.rmtree(root)
    check('E: recreates missing dir', w.publish_now() is True)


def test_stale_tmp(tmp):
    path = os.path.join(tmp, 's.json')
    with open(path + '.tmp', 'w') as f:
        f.write('{"half":')                      # killed writer's leftover
    w = st.StatusWriter('r', 1, 100, status_path=path, gpu_fn=lambda: None)
    check('F: publishes over stale .tmp', w.publish_now() is True)
    check('F: published file valid', json.load(open(path))['run_id'] == 'r')


def test_full_scale_blob(tmp):
    path = os.path.join(tmp, 's.json')
    regions = ['Africa', 'Australia', 'Central_America', 'East_Asia',
               'Eastern_Europe', 'Greenland', 'Middle_East', 'North_America',
               'Northern_Europe', 'Oceania', 'Russia', 'South_America',
               'South_Asia', 'Southeast_Asia', 'Southern_Europe',
               'Western_Europe']
    w = st.StatusWriter(
        'detect-20260801-023114-yolo26x-1280-fp16-trt-gen0007', 7,
        32_542_334, status_path=path, interval=5,
        drive_totals={d: 8_135_583 for d in
                      ('lynx', 'bobcat', 'capybara', 'jackal')},
        region_totals={r: 2_033_895 for r in regions},
        gpu_fn=lambda: {'util': 97, 'mem_used_mb': 23888,
                        'mem_total_mb': 24564, 'temp': 83})
    w.update(imgs_done=31_999_999, boxes_total=2_961_351,
             positives=2_612_398, crops_classified=2_961_351,
             class_counts={'leashed': 411_223, 'unleashed': 2_101_774,
                           'not_a_dog': 448_354},
             drive_done={d: 7_999_999 for d in
                         ('lynx', 'bobcat', 'capybara', 'jackal')},
             drive_queue={d: 4096 for d in
                          ('lynx', 'bobcat', 'capybara', 'jackal')},
             region_done={r: 1_999_999 for r in regions},
             errors={'read': 12345, 'decode': 999, 'missing': 40_017,
                     'infer': 3, 'mount_lost': 1},
             last_error='decode: cell 12345_67890 image 99887766554433.jpg '
                        'premature end of data segment')
    check('G: full-scale publish', w.publish_now() is True)
    blob = open(path).read()
    check('G: blob under 12 kB at full scale (§7.3)', len(blob) < 12000,
          f'{len(blob)} bytes')
    d = json.loads(blob)
    check('G: 53-char run_id + huge ints survive',
          d['run_id'].startswith('detect-2026') and
          d['imgs_done'] == 31_999_999)


def test_concurrent_update_hammer(tmp):
    path = os.path.join(tmp, 's.json')
    w = st.StatusWriter('r', 1, 10**7, status_path=path, interval=0.001,
                        gpu_fn=lambda: None,
                        drive_totals={'a': 1, 'b': 1, 'c': 1})
    w.start()
    stop = threading.Event()
    errs = []

    def pusher(i):
        n = 0
        while not stop.is_set():
            n += 1
            try:
                w.update(imgs_done=n, boxes_total=n,
                         drive_done={'abc'[i % 3]: n},
                         class_counts={'not_a_dog': n},
                         errors={'decode': n}, crops_classified=n * 2)
            except Exception as e:
                errs.append(repr(e))

    threads = [threading.Thread(target=pusher, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    bad = 0
    t0 = time.time()
    while time.time() - t0 < 2.0:
        try:
            json.load(open(path))
        except FileNotFoundError:
            pass                                 # before the first publish
        except Exception:
            bad += 1
    stop.set()
    for t in threads:
        t.join()
    w.stop(state='done')
    check('H: no update() exceptions under 4-thread hammer', not errs,
          errs[:1])
    check('H: no torn reads while hammered', bad == 0, f'{bad} bad')
    check('H: zero publish errors under contention', w.publish_errors == 0,
          f'{w.publish_errors}')
    check('H: terminal frame wins', json.load(open(path))['state'] == 'done')


def test_reader_hostile(tmp):
    # directory where the file should be
    dirpath = os.path.join(tmp, 'iamadir.json')
    os.makedirs(dirpath)
    check('I: directory at path -> not running',
          st.read_status(dirpath) == {'running': False})
    # empty file
    empty = os.path.join(tmp, 'empty.json')
    open(empty, 'w').close()
    check('I: empty file -> not running',
          st.read_status(empty) == {'running': False})
    # string ts -> mtime fallback keeps a fresh foreign payload readable
    p = os.path.join(tmp, 'strts.json')
    with open(p, 'w') as f:
        json.dump({'ts': 'yesterday', 'state': 'running'}, f)
    d = st.read_status(p, stale_after=120)
    check('I: string ts falls back to mtime (fresh -> running)',
          d.get('running') is True, repr(d))
    # ...and an old mtime is stale
    os.utime(p, (time.time() - 500, time.time() - 500))
    d = st.read_status(p, stale_after=120)
    check('I: string ts + old mtime -> stale', d['running'] is False
          and d.get('stale') is True, repr(d))
    # writer clock ahead of reader clock: age clamps to 0, still running
    p2 = os.path.join(tmp, 'future.json')
    with open(p2, 'w') as f:
        json.dump({'ts': time.time() + 3600, 'state': 'running'}, f)
    d = st.read_status(p2, stale_after=120)
    check('I: future ts clamps age to 0', d['running'] is True
          and d['age_s'] == 0, repr(d))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for i, fn in enumerate([
                test_payload_exception_contained, test_nan_inf_never_published,
                test_denominator_mismatch, test_unknown_drive,
                test_unwritable_dir, test_stale_tmp, test_full_scale_blob,
                test_concurrent_update_hammer, test_reader_hostile]):
            sub = os.path.join(tmp, str(i))
            os.makedirs(sub)
            fn(sub)
    if FAILED:
        print(f'\n{len(FAILED)} FAILED: {FAILED}')
        return 1
    print('\nall adversarial status tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
