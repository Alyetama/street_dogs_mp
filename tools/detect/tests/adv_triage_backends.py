#!/usr/bin/env python3
"""
Two guessers, ONE status file. Prove a run is never credited to the other one.

    python tools/detect/tests/adv_triage_backends.py

tools/detect/triage_crops.py publishes progress to a single
data/dashboard/triage_status.json, whatever backend it is running. The review
page's strip asks about one backend at a time. Nothing in the file layout stops
the strip answering for the wrong one, and before this was guarded it did:
starting RF-DETR and moving the dropdown to SigLIP showed SigLIP as running,
with RF-DETR's progress bar under it and a Pause button that would have stopped
a run the reader was not looking at.

This drives the real dashboard functions against a fabricated status file. It
needs duckdb (dashboard.py imports it) and a 3.12+ interpreter -- the same one
the dashboard itself runs under.

Also checks the two numbers the dropdown puts on screen are a like-for-like
comparison, because one of them was not: SigLIP's was carried over from a
different measurement and overstated the gap by twenty points.
"""

import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))

fails = []


def check(name, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + name + ('' if ok else '  ' + detail))
    if not ok:
        fails.append(name)


try:
    import dashboard as d
except Exception as e:                                   # pragma: no cover
    print(f'SKIP: cannot import dashboard.py ({type(e).__name__}: {e})')
    print('  needs duckdb and a 3.12+ interpreter — the env the dashboard runs on')
    raise SystemExit(0)


def status_doc(model, **kw):
    doc = {'running': True, 'model': model, 'done': 12, 'total': 100,
           'watch': 300, 'updated': time.time(), 'pid': os.getpid(),
           'schema': 1}
    doc.update(kw)
    return doc


tmp = tempfile.mkdtemp()
STATUS = os.path.join(tmp, 'triage_status.json')
d.TRIAGE_STATUS = STATUS
# A live run, from the process table's point of view. The real function scans
# for a python running triage_crops.py; what matters here is only that
# something IS alive, so the backend attribution is the sole variable.
d.triage_pids = lambda: [os.getpid()]

with open(STATUS, 'w') as fh:
    json.dump(status_doc('rfdetr'), fh)

rf = d.triage_status('rfdetr')
sg = d.triage_status('siglip')
check('a live RF-DETR run is reported as running under RF-DETR',
      rf['running'] is True, f'running={rf["running"]}')
check('the same run is NOT reported as running under SigLIP',
      sg['running'] is False, f'running={sg["running"]}')
check('the idle backend is told who has the card',
      sg.get('busy_with') == 'RF-DETR', f'busy_with={sg.get("busy_with")!r}')
check('the busy backend is not told it is waiting for itself',
      rf.get('busy_with') is None, f'busy_with={rf.get("busy_with")!r}')

# ...and the mirror image, so this is not passing by hardcoding one name
with open(STATUS, 'w') as fh:
    json.dump(status_doc('google/siglip2-so400m-patch14-384'), fh)
rf = d.triage_status('rfdetr')
sg = d.triage_status('siglip')
check('a live SigLIP run is reported as running under SigLIP',
      sg['running'] is True, f'running={sg["running"]}')
check('the same run is NOT reported as running under RF-DETR',
      rf['running'] is False, f'running={rf["running"]}')
check('the idle backend is told who has the card (mirror)',
      rf.get('busy_with') == 'SigLIP 2', f'busy_with={rf.get("busy_with")!r}')

# A stall belongs to the run that stalled. Same trap one field over: an
# RF-DETR run that goes quiet must not raise "Run stopped" under SigLIP.
with open(STATUS, 'w') as fh:
    json.dump(status_doc('rfdetr', watch=0,
                         updated=time.time() - (d.TRIAGE_STALE_S + 60)), fh)
check('a stalled run does not alarm the other backend',
      d.triage_status('siglip').get('stalled') is False,
      f'stalled={d.triage_status("siglip").get("stalled")!r}')
check('a stalled run does alarm its own backend',
      d.triage_status('rfdetr').get('stalled') is True,
      f'stalled={d.triage_status("rfdetr").get("stalled")!r}')

# Starting the other guesser while one holds the card can only be refused, and
# the refusal has to name what is in the way -- 'already running' under a
# dropdown set to the other one reads as a broken button.
with open(STATUS, 'w') as fh:
    json.dump(status_doc('rfdetr'), fh)
body = d._triage_control('start', 'siglip')
check('starting SigLIP while RF-DETR runs is refused', body.get('ok') is False,
      f'{body}')
check('the refusal names the guesser in the way',
      'RF-DETR' in str(body.get('msg', '')), f'msg={body.get("msg")!r}')

# ── the two numbers on screen have to be the same measurement ──────────────
info = d.BACKEND_INFO
recalls = {k: v.get('recall') for k, v in info.items()}
check('every offered backend carries a recall',
      all(isinstance(v, (int, float)) for v in recalls.values()),
      f'{recalls}')
# Both directions, for all three, measured on one set: the dog-bin validation
# split, 342 dogs and 300 not-dogs. A window per number, because one moving
# without the others being re-measured is exactly how a figure from an
# unrelated measurement once ended up on screen beside two that were not.
MEASURED = {'siglip': (0.977, 0.943), 'dogbin': (0.936, 0.943),
            'rfdetr': (0.678, 0.957)}
for _b, (_r, _c) in MEASURED.items():
    got = info.get(_b) or {}
    check(f'{_b} finds-dogs matches its measured value',
          abs((got.get('recall') or 0) - _r) < 0.02,
          f'{got.get("recall")} vs {_r} — re-measure ALL THREE on one set '
          f'before changing any')
    check(f'{_b} clears-not-dogs matches its measured value',
          abs((got.get('clears') or 0) - _c) < 0.02,
          f'{got.get("clears")} vs {_c}')
# One number is not a claim: a guesser that called everything a dog would read
# 100% on the first and 0% on the second, so the pair has to travel together.
check('every guesser carries both directions',
      all('recall' in v and 'clears' in v for v in info.values()),
      f'{ {k: sorted(v) for k, v in info.items()} }')
# ...and the page must be told that two of the three were tuned against this
# split, which flatters exactly the two that lead.
check('the page is told which guessers were tuned on the test split',
      'tuned' in d.RECALL_CAVEAT.lower()
      and 'RF-DETR' in d.RECALL_CAVEAT,
      'no caveat naming the exposure')
# Deliberately NOT checking that a comment names the measurement set: doing so
# means spelling a verdict store's name here, and t1c in adv_triage_isolation
# rightly fails any allowlisted module that does. The weaker check loses.

# ── Pause must stop MY guesser, never the other one ────────────────────────
# The stop branch SIGTERMed every triage_crops.py alive, whatever backend it
# was running. The button reads 'Pause' for a moment after the dropdown moves,
# before the next poll corrects it, so a fast click could end a run the reader
# was not looking at.
killed = []
d.os_kill_orig = os.kill
try:
    os.kill = lambda pid, sig: killed.append(pid)
    with open(STATUS, 'w') as fh:
        json.dump(status_doc('rfdetr'), fh)
    body = d._triage_control('stop', 'siglip')
    check('pausing SigLIP does not kill the RF-DETR run',
          body.get('ok') is False and not killed, f'{body} killed={killed}')
    check('the refusal says which run it is',
          'RF-DETR' in str(body.get('msg', '')), f'msg={body.get("msg")!r}')
    body = d._triage_control('stop', 'rfdetr')
    check('pausing RF-DETR does stop the RF-DETR run',
          body.get('ok') is True and killed, f'{body} killed={killed}')
finally:
    os.kill = d.os_kill_orig

# ── a backend is validated one way, by every caller ────────────────────────
# The strip validated against the names that exist, the queue against the
# names that can run, so a saved preference for an unrunnable one had the two
# describing different guessers with nothing on screen admitting it.
check('an unknown backend resolves to something offered',
      d.pick_backend('nonsense') in d.backends_offered(),
      f'got {d.pick_backend("nonsense")!r}')
check('the strip and the queue resolve a backend identically',
      d.triage_status('nonsense')['backend'] == d.pick_backend('nonsense'),
      f'strip={d.triage_status("nonsense")["backend"]!r} '
      f'rule={d.pick_backend("nonsense")!r}')
check('a backend with guesses on file is offered even if it cannot be run',
      all(b in d.backends_offered()
          for b in d.TRIAGE_BACKENDS if d.triage_seen(b)),
      f'offered={d.backends_offered()}')

# ── coverage counts what the guesser has DEALT WITH ────────────────────────
# RF-DETR is allowed to find nothing and writes that down. Counting only the
# crops carrying a bucket left its coverage permanently short of the pool, so
# the strip could never stop warning however long it ran.
for b in d.TRIAGE_BACKENDS:
    idx, seen = d.triage_index(b), d.triage_seen(b)
    check(f'{b}: every guess is also counted as looked at',
          set(idx) <= set(seen),
          f'{len(set(idx) - set(seen))} guesses missing from the seen set')
check('a no-guess record still counts as dealt with',
      len(d.triage_seen('rfdetr')) >= len(d.triage_index('rfdetr')),
      'seen must be a superset of the index')

# ...and the number the STRIP publishes has to be the one built that way.
# The invariant above holds whichever set triage_status() reaches for, so it
# does not guard the fix on its own -- this pins the published figure. The
# pool is frozen for the comparison because the real one rotates under it.
_no_guess = sorted(d.triage_seen('rfdetr') - set(d.triage_index('rfdetr')))
if not _no_guess:
    print('SKIP no-guess coverage: this checkout has no such records yet')
else:
    _guessed = sorted(d.triage_index('rfdetr'))[:5]
    _pool = [(n, '/pool') for n in _no_guess[:5] + _guessed]
    _real_pool = d.review_pool_names
    try:
        d.review_pool_names = lambda: _pool
        with open(STATUS, 'w') as fh:
            json.dump(status_doc('rfdetr', running=False), fh)
        st = d.triage_status('rfdetr')
        check('the published coverage counts crops looked at, not bucketed',
              st['guessed'] == len(_pool),
              f'published {st["guessed"]} of {len(_pool)}; the '
              f'{len(_no_guess[:5])} it looked at and declined were dropped')
    finally:
        d.review_pool_names = _real_pool

# ── a guesser is only offered filters it can actually answer ───────────────
# The dog-bin gate is binary: dog, or not a dog, with no opinion on what a
# not-dog is. Offering it 'Other animal' offers a filter that can only ever
# return nothing, and folding its 'not_dog' into 'object' would file every cow
# under "not an animal".
for _b in d.TRIAGE_BACKENDS:
    declared = set((d.BACKEND_INFO.get(_b) or {}).get('buckets') or ())
    if not declared:
        continue
    check(f'{_b}: the filter offers only what it can say',
          declared <= set(d.TRIAGE_BUCKETS),
          f'buckets outside the known set: {declared - set(d.TRIAGE_BUCKETS)}')
    seen_b = {v['b'] for v in d.triage_index(_b).values()}
    check(f'{_b}: nothing it has written falls outside those buckets',
          seen_b <= declared or not seen_b,
          f'wrote {sorted(seen_b - declared)}, declared {sorted(declared)}')
    check(f'{_b}: every offered bucket has a label',
          all(k in d.BUCKET_LABELS for k in declared),
          f'unlabelled: {sorted(declared - set(d.BUCKET_LABELS))}')
check('the binary gate does not claim the three-bucket vocabulary',
      set((d.BACKEND_INFO.get('dogbin') or {}).get('buckets') or ())
      == {'dog', 'not_dog'},
      f'{(d.BACKEND_INFO.get("dogbin") or {}).get("buckets")}')

# ── a filter the page cannot show must not be applied ──────────────────────
# The review page hides the guess filter whenever the dog-bin gate's own axis
# covers it. Hiding a control does not unset it: the value stayed in the
# request and the server kept honouring it, so choosing the gate could empty
# the queue with no chip, no cross to clear it, no control on screen and no
# "narrowed from" — every surface that exists to make an empty queue
# explainable, silent at once. The server decides, and echoes what it decided.
_offered = d.backends_offered()
if 'dogbin' not in _offered or not d.triage_index('dogbin'):
    print('SKIP guess-filter drop: no dog-bin verdicts in this checkout')
else:
    for _sg in ('animal', 'object', 'dog'):
        j = d.review_payload(page=0, size=1, backend='dogbin', suggest=_sg)
        check(f'the gate ignores a guess filter it does not offer ({_sg})',
              j['suggest'] == '' and j['total_unflagged'] == j['pool_unfiltered'],
              f'applied {j["suggest"]!r}, showing {j["total_unflagged"]} of '
              f'{j["pool_unfiltered"]}')
    # ...and it is still honoured where the page DOES offer it
    j = d.review_payload(page=0, size=1, backend='siglip', suggest='dog')
    check('the guess filter still works where it is offered',
          j['suggest'] == 'dog', f'applied {j["suggest"]!r}')

# ── the audit baseline must not move with the filter it measures ───────────
# annotated_payload's `label` decides which ledger files are read at all, so a
# baseline taken from what was read collapsed onto the total exactly when the
# one filter that view has was on — and the caption could never report the
# narrowing it caused.
# Judged WITHIN one payload. Comparing a baseline taken now against totals
# fetched a moment later reads a verdict recorded in between as a defect --
# the ledgers are live, and this dashboard is in use while its tests run.
# The bug was that the baseline collapsed onto the total whenever the verdict
# filter was on, and that is visible in a single response.
for _lab in ('false_positive', 'true_positive'):
    j = d.annotated_payload(page=0, size=1, label=_lab)
    check(f'the audit baseline does not collapse onto the total '
          f'({_lab})',
          j['pool_unfiltered'] > j['total'],
          f'baseline {j["pool_unfiltered"]} == total {j["total"]}: the filter '
          f'is narrowing upstream of the number meant to measure it')
_all = d.annotated_payload(page=0, size=1, label='all')
check('with no verdict filter the baseline IS the total',
      _all['pool_unfiltered'] == _all['total'],
      f'{_all["pool_unfiltered"]} vs {_all["total"]}')

print()
if fails:
    raise SystemExit(f'{len(fails)} backend check(s) FAILED: '
                     + ', '.join(fails))
print('one status file, two guessers, and neither is credited with the '
      "other's run")
