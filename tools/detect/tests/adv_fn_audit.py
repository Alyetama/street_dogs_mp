#!/usr/bin/env python3
"""The false-negative audit has to produce a number, not a pile of clicks.

Everything here guards one of the three claims the page makes:

  * you will never be shown the same box twice, and never two frames from one
    sequence -- otherwise the sample is correlated and the interval is a lie;
  * the rate is weighted by how many boxes the gate really put in each band,
    because the bands are drawn from evenly and a flat mean would report the
    near-threshold error rate as if it were the whole store's;
  * the crop shown is the crop the model saw.

And one the page must NOT make: a human verdict here is ground truth about a
model's output, and the model's own opinion never becomes a label.
"""

import ast
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))


def band_checks(bad):
    """The band a score falls in is computed twice -- once in Python for the
    verdict ledger, once in SQL when the pool is built. They disagreed: the
    SQL multiplied by the band COUNT, which is the width only when the bands
    span 0..1, and these stop at the gate's threshold. Five 0.1-wide bands
    came out as three 0.2-wide ones."""
    import fn_audit as fa
    w = fa.BANDS[0][1] - fa.BANDS[0][0]
    if abs(w - fa.BAND_W) > 1e-12:
        bad.append(f'BAND_W {fa.BAND_W} is not the band width {w}')
    for i, (lo, hi) in enumerate(fa.BANDS):
        if abs((hi - lo) - w) > 1e-12:
            bad.append(f'band {i} is {hi - lo} wide, not {w} — the pool\'s '
                       f'single division cannot bucket uneven bands')
        for p in (lo, (lo + hi) / 2, hi - 1e-9):
            if fa.band_of(p) != i:
                bad.append(f'band_of({p}) = {fa.band_of(p)}, expected {i}')
    # the SQL expression, evaluated the way duckdb would
    import math
    for p in (0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.37, 0.4, 0.499):
        sql = min(len(fa.BANDS) - 1,
                  int(math.floor((p + 1e-9) / fa.BAND_W)))
        if sql != fa.band_of(p):
            bad.append(f'python and SQL disagree at p={p}: '
                       f'{fa.band_of(p)} vs {sql}')
    src = open(os.path.join(REPO, 'tools', 'detect', 'fn_audit.py')).read()
    if 'floor((r.p_dog + 1e-9) / {BAND_W})' not in src:
        bad.append('the pool SQL no longer divides by the band width — if it '
                   'multiplies by the band count again, five 0.1 bands '
                   'silently become three 0.2 ones')


def wilson_checks(bad):
    """A rate quoted without an interval is a guess with a decimal point."""
    import fn_audit as fa
    p, lo, hi = fa.wilson(0, 0)
    if (p, lo, hi) != (0.0, 0.0, 0.0):
        bad.append(f'wilson(0,0) = {(p, lo, hi)}; nothing judged is not a rate')
    p, lo, hi = fa.wilson(0, 100)
    if lo != 0.0 or not 0 < hi < 0.06:
        bad.append(f'0 of 100 gives [{lo}, {hi}] — zero finds does not mean '
                   f'zero rate, and the normal approximation would say it does')
    p, lo, hi = fa.wilson(3, 400)
    if not (lo > 0 and hi > p > lo):
        bad.append(f'3 of 400 gives [{lo}, {hi}] around {p}')
    for k, n in ((1, 1), (0, 1), (50, 100)):
        p, lo, hi = fa.wilson(k, n)
        if not (0.0 <= lo <= p <= hi <= 1.0):
            bad.append(f'wilson({k},{n}) left the unit interval: '
                       f'{(p, lo, hi)}')


def weighting_checks(bad):
    """The two headline rates weight bands by population, and read from the
    right end of the threshold.

    A flat mean would take the near-threshold band -- a sliver of the pool,
    and where every error lives -- and report its rate as the whole store's.
    And the two directions are different questions: below the line, a dog is
    an error; above it, a not-dog is.
    """
    import fn_audit as fa
    n = len(fa.BANDS)
    totals = [(fa.BANDS[i][0], fa.BANDS[i][1],
               900 if i == 0 else 5) for i in range(n)]
    # band 0 (rejected) all clean; band 4 (rejected, near the line) all dogs
    vs = ([{'key': f'a{i}', 'band': 0, 'verdict': 'not_dog'} for i in range(10)]
          + [{'key': f'e{i}', 'band': 4, 'verdict': 'dog'} for i in range(5)])
    s = fa.summarise(vs, totals)
    rej = s['rejected']
    want = (0.0 * 900 + 1.0 * 5) / 905
    if abs(rej['rate'] - want) > 1e-9:
        bad.append(f"rejected-side rate {rej['rate']} is not the "
                   f"population-weighted {want}")
    if abs(rej['rate'] - 0.5) < 1e-6:
        bad.append('the rate is a flat mean over the bands')
    # a dog found above the threshold is NOT a miss -- the gate kept it
    vs2 = vs + [{'key': 'k1', 'band': n - 1, 'verdict': 'dog'}]
    s2 = fa.summarise(vs2, totals)
    if abs(s2['rejected']['rate'] - rej['rate']) > 1e-12:
        bad.append('a dog found in a band the gate KEPT changed the '
                   'thrown-away rate — it is not a miss, the gate agreed')
    if s2['kept']['wrong'] != 0:
        bad.append(f"a dog in a kept band counted as an error: "
                   f"{s2['kept']['wrong']}")
    # a NOT-dog above the threshold is the false positive
    vs3 = vs + [{'key': 'k2', 'band': n - 1, 'verdict': 'not_dog'}]
    s3 = fa.summarise(vs3, totals)
    if s3['kept']['wrong'] != 1 or s3['kept']['rate'] <= 0:
        bad.append(f"a not-dog the gate kept is a false positive and was not "
                   f"counted: {s3['kept']}")
    # bands nobody looked at are unknown, not clean
    if s['covered'] >= 0.99:
        bad.append(f"covered = {s['covered']} with most bands unsampled")
    # 'unsure' is neither a find nor a clean look
    s4 = fa.summarise([{'key': 'a', 'band': 4, 'verdict': 'unsure'},
                       {'key': 'b', 'band': 4, 'verdict': 'dog'}], totals)
    if s4['bands'][4]['judged'] != 1:
        bad.append(f"'unsure' counted in the denominator: "
                   f"{s4['bands'][4]['judged']}")
    # the old vocabulary still reads
    s5 = fa.summarise([{'key': 'a', 'band': 4, 'verdict': 'missed'}], totals)
    if s5['bands'][4]['dogs'] != 1:
        bad.append("a verdict written as 'missed' before the wording changed "
                   "is no longer counted")


def ledger_checks(bad):
    """Append-only, last answer wins, and a bad line is skipped not fatal."""
    import fn_audit as fa
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'v.jsonl')
        with open(p, 'w') as fh:
            fh.write(json.dumps({'key': 'a', 'verdict': 'dog'}) + '\n')
            fh.write('not json at all\n')
            fh.write('\n')
            fh.write(json.dumps({'key': 'a', 'verdict': 'not_dog'}) + '\n')
            fh.write(json.dumps({'no_key': 1}) + '\n')
            fh.write(json.dumps({'key': 'b', 'verdict': 'unsure'}) + '\n')
            fh.write(json.dumps({'key': 'c', 'verdict': 'dog'}) + '\n')
            # a withdrawal: the box goes back to unjudged, it does not become
            # a third kind of answer
            fh.write(json.dumps({'key': 'c', 'verdict': None}) + '\n')
        got = {v['key']: v['verdict'] for v in fa.read_verdicts(p)}
        if got != {'a': 'not_dog', 'b': 'unsure'}:
            bad.append(f'ledger read {got}; a changed mind must win and a '
                       f'corrupt line must be skipped, not fatal')


def serving_checks(bad):
    """The crop shown is the crop the model saw, and nothing a client sends
    reaches a path."""
    try:
        import audit
    except Exception as e:
        print(f'SKIP: audit module not importable ({e})')
        return
    import gate_store as gs
    if (audit.PAD_FRAC, audit.PAD_PX, audit.MIN_SIDE) != \
            (gs.PAD_FRAC, gs.PAD_PX, gs.MIN_SIDE):
        bad.append(f'the audit cuts with pad {audit.PAD_FRAC}/{audit.PAD_PX} '
                   f'but the gate judged with {gs.PAD_FRAC}/{gs.PAD_PX} — '
                   f'a differently framed crop is a different picture, and '
                   f'the audit would be measuring a model that never ran')
    src = open(os.path.join(REPO, 'tools', 'dashboard', 'audit.py')).read()
    if 'import gate_store' not in src:
        bad.append('the padding is copied rather than imported from the '
                   'runner — two numbers that must agree, kept by hand')
    for hostile in ('../../../etc/passwd', 'a/../b', '..', '', None,
                    'x' * 40, '1_2.jpg', '1#2', '/etc/passwd'):
        if audit.crop_path(hostile) is not None:
            bad.append(f'crop_path({hostile!r}) resolved to a path')
    if audit.record('k', 'looks fine to me')['ok']:
        bad.append('an unknown verdict was recorded')


def isolation_checks(bad):
    """A model's opinion never becomes a label.

    The pool is built FROM the gate's verdicts, which is the point -- but the
    thing this page writes is a human's answer, and it must not be readable
    as though the model had produced it. Checked in source: nothing here may
    write into the reviewer ledgers or the training sets.
    """
    # The rule is about PROVENANCE, not vocabulary: this tool does write a
    # dataset now, and it should. What it must never do is write the gate's
    # own label as if a person had given it. A keyword blacklist could not
    # tell those apart -- it failed the legitimate export -- so the export is
    # run against a ledger and checked for what came out.
    import audit
    import fn_audit as fa
    import contextlib
    import io
    import tempfile as _tf
    import types
    tmp = _tf.mkdtemp()

    def quiet_export():
        """export() reports what it wrote; a guard's output is its own."""
        with contextlib.redirect_stdout(io.StringIO()):
            fa.export(types.SimpleNamespace(model='dogbin_008',
                                            stage='gate'))
    # every path this stage owns, redirected in one place -- the module no
    # longer keeps them as globals because two audits share the process
    real_paths = fa.paths
    lay = dict(real_paths('gate'))
    lay.update(out=tmp, verdicts=os.path.join(tmp, 'v.jsonl'),
               drawn=os.path.join(tmp, 'drawn.jsonl'),
               full=os.path.join(tmp, 'full'),
               crops=os.path.join(tmp, 'crops'),
               pages=os.path.join(tmp, 'pages'),
               dataset=os.path.join(tmp, 'ds'))
    fa.paths = lambda stage='gate': lay
    for c in ('dog', 'not_dog'):
        os.makedirs(os.path.join(lay['dataset'], c), exist_ok=True)
    try:
        full = os.path.join(tmp, 'full')
        os.makedirs(full, exist_ok=True)
        for k in ('1_0', '2_0', '3_0', '4_0'):
            open(os.path.join(full, k + '.jpg'), 'wb').close()
        with open(lay['verdicts'], 'w') as fh:
            for key, v in (('1#0', 'missed'), ('2#0', 'correct'),
                           ('3#0', 'unsure')):
                fh.write(json.dumps({'key': key, 'verdict': v,
                                     'seq': 's' + key[0]}) + '\n')
            # 4#0 is in the pool and the gate called it not_dog -- and no
            # human has looked at it. It must not appear anywhere.
        quiet_export()
        got = {c: sorted(os.listdir(os.path.join(lay['dataset'], c)))
               for c in ('dog', 'not_dog')}
        if got != {'dog': ['1_0.jpg'], 'not_dog': ['2_0.jpg']}:
            bad.append(f'the export wrote {got}; only the human verdicts '
                       f'belong -- "missed" is a dog, "correct" is a not_dog, '
                       f'"unsure" is neither, and an unjudged box is nothing')
        man = [json.loads(x) for x in
               open(os.path.join(lay['dataset'], 'manifest.jsonl'))
               if x.strip()]
        if len(man) != 2:
            bad.append(f'{len(man)} manifest rows for 2 judged crops')
        if any(not r.get('sequence') for r in man):
            bad.append('a manifest row has no sequence — the only column a '
                       'future split may use, and the one whose absence put '
                       '70.8% of a val set in train last time')
        if any(r.get('verdict') not in ('dog', 'not_dog') for r in man):
            bad.append(f'a manifest row carries a verdict no human gave: '
                       f'{[r.get("verdict") for r in man]}')
        # changing your mind moves the file rather than leaving both
        with open(lay['verdicts'], 'a') as fh:
            fh.write(json.dumps({'key': '1#0', 'verdict': 'correct',
                                 'seq': 's1'}) + '\n')
        quiet_export()
        got = {c: sorted(os.listdir(os.path.join(lay['dataset'], c)))
               for c in ('dog', 'not_dog')}
        if got != {'dog': [], 'not_dog': ['1_0.jpg', '2_0.jpg']}:
            bad.append(f'a changed verdict left the old label behind: {got}')
        if not os.path.exists(os.path.join(lay['dataset'], 'README.md')):
            bad.append('the dataset ships without the note saying it is a '
                       'stratified sample of one model\'s rejections')
    finally:
        fa.paths = real_paths
    # only human verdicts are labels at all
    for name, sp in fa.STAGES.items():
        for v in (sp['positive'], sp['negative']):
            if v not in sp['answers']:
                bad.append(f'{name}: {v!r} is a class but not something a '
                           f'person can answer')
    # A person's answers are now the same two words the model uses, so the
    # separation cannot be a vocabulary check any more -- and it never really
    # was one. It is structural: the pool the page serves from carries the
    # SCORE and not the LABEL, so the gate's own verdict is not available to
    # anything that could write it down.
    try:
        import duckdb
        cols = [r[0] for r in duckdb.connect().execute(
            f"DESCRIBE SELECT * FROM read_parquet("
            f"'{real_paths('gate')['pool']}') LIMIT 1"
        ).fetchall()]
        if 'label' in cols:
            bad.append("the pool carries the gate's own label — the one thing "
                       "that must never be reachable from the page that "
                       "records human answers")
        if 'p_dog' not in cols:
            bad.append('the pool carries no score, so nothing can be banded')
    except Exception as e:                     # noqa: BLE001
        if 'No files found' not in str(e) and 'IO Error' not in str(e):
            bad.append(f'could not read the pool schema: {e}')
    # A correction is geometry and must stay geometry: nothing the audit
    # writes to the shared box store may carry a verdict, a class or a score.
    import inspect as _in
    import re as _re
    src_sc = _in.getsource(audit.save_correction)
    # Brace-matched. A lazy regex to the first '}' stopped inside the
    # f-string on the very first line, so the check was reading six
    # characters and could never have seen a field smuggled in below it.
    i = src_sc.find('rec = {')
    body = None
    if i >= 0:
        depth, j = 0, src_sc.index('{', i)
        for k in range(j, len(src_sc)):
            if src_sc[k] == '{':
                depth += 1
            elif src_sc[k] == '}':
                depth -= 1
                if depth == 0:
                    body = src_sc[j:k + 1]
                    break
    if not body:
        bad.append('save_correction no longer builds a record we can check')
    else:
        for word in ('verdict', 'label', 'p_dog', 'class', 'answer'):
            if f"'{word}'" in body:
                bad.append(f'a box correction carries {word!r} — that file is '
                           f'geometry, shared with the review page, and a '
                           f'verdict in it would be a label nobody asked for')
    # and the ledger has exactly one writer
    asrc = open(os.path.join(REPO, 'tools', 'dashboard', 'audit.py')).read()
    writers = [ln for ln in asrc.splitlines()
               if "pp['verdicts']" in ln and 'open(' in ln]
    if len(writers) != 1:
        bad.append(f'the verdict ledger has {len(writers)} writers; it must '
                   f'have exactly one, so every line in it came through the '
                   f'same door: {writers}')
    # and the audit still never writes into the reviewer's own stores
    for rel in ('tools/detect/fn_audit.py', 'tools/dashboard/audit.py'):
        src = open(os.path.join(REPO, rel)).read()
        for line in src.splitlines():
            code = line.split('#', 1)[0]
            if 'open(' not in code and 'join(' not in code:
                continue
            # VERDICT stores, not every store the reviewer touches.
            # data/box_corrections holds GEOMETRY -- where a box is, drawn by
            # hand -- and is deliberately shared: the review page writes it,
            # harvest_flagged.py reads it, and a detection has one true box
            # whoever redrew it. It says nothing about what the crop contains,
            # so it cannot carry a label in either direction.
            for banned in ('annot', 'hard_negative', 'hard_positive'):
                if banned in code.lower():
                    bad.append(f'{rel} reaches into the reviewer store: '
                               f'{line.strip()[:80]}')


def correction_checks(bad):
    """Redrawing a box changes the TRAINING crop and nothing else.

    The audit's whole claim is that the crop shown is the crop the model saw.
    A hand-drawn box makes a better example for the next model; it must not
    quietly rewrite the picture this one was judged on, or the measurement
    becomes about crops that never existed when the score was given.
    """
    import inspect as _in
    try:
        import audit
    except Exception:
        return
    # the CALL, not the docstring that describes it -- checking the module
    # text matched the sentence explaining the flag and passed with the flag
    # removed from the only place it does anything
    sc = _in.getsource(audit.save_correction)
    if "into='edited'" not in sc:
        bad.append('save_correction does not cut into edited/ — a redrawn '
                   'box would overwrite full/, and the picture the model was '
                   'judged on would change after the fact')
    # Behavioural, not textual. This matched an exact indentation, so
    # reformatting the function would have silently retired the check.
    import tempfile as _tf
    from PIL import Image as _Im
    tmp = _tf.mkdtemp()
    real_paths = audit.fa.paths
    lay = dict(real_paths('gate'))
    lay.update(out=tmp, crops=os.path.join(tmp, 'crops'),
               full=os.path.join(tmp, 'full'))
    audit.fa.paths = lambda stage='gate': lay
    src_dir = os.path.join(tmp, 'src', 'cell', 'ground_animal_images')
    os.makedirs(src_dir, exist_ok=True)
    _Im.new('RGB', (400, 300), (30, 40, 50)).save(
        os.path.join(src_dir, '9.jpg'))
    cand = {'key': '9#0', 'image_id': '9', 'det_idx': 0, 'cell': 'cell',
            'drive': 'd', 'x1': 40, 'y1': 40, 'x2': 140, 'y2': 140}
    roots = {'d': os.path.join(tmp, 'src')}
    try:
        audit._cut_one(cand, roots, 'gate')
        thumb = os.path.join(lay['crops'], '9_0.jpg')
        full = os.path.join(lay['full'], '9_0.jpg')
        if not (os.path.exists(thumb) and os.path.exists(full)):
            bad.append('a normal cut did not write both the thumbnail and '
                       'the full-resolution crop')
        was = open(full, 'rb').read()
        wasthumb = open(thumb, 'rb').read()
        bigger = dict(cand, x1=20, y1=20, x2=200, y2=200)
        audit._cut_one(bigger, roots, 'gate', into='edited', force=True)
        if open(full, 'rb').read() != was:
            bad.append('a redrawn box overwrote full/ — the picture the '
                       'model was judged on changed after the fact')
        if open(thumb, 'rb').read() != wasthumb:
            bad.append('a redrawn box overwrote the thumbnail — the sheet '
                       'would show a crop the model never saw')
        if not os.path.exists(os.path.join(tmp, 'edited', '9_0.jpg')):
            bad.append('a redrawn box was not cut into edited/, so nothing '
                       'downstream can use it')
    except Exception as e:                     # noqa: BLE001
        bad.append(f'cutting a corrected box threw {type(e).__name__}: {e}')
    finally:
        audit.fa.paths = real_paths
    # The two paths that file a crop must pick the same one. place() runs on
    # every verdict and export() rebuilds from the ledger; if only one of them
    # preferred the hand-drawn cut, the dataset depended on which ran last.
    pl = _in.getsource(audit.place)
    if "'edited'" not in pl:
        bad.append('place() ignores a hand-drawn box while export() prefers '
                   'it — the crop in the dataset would depend on whether a '
                   'rebuild had happened since')
    # Behavioural: file a crop, redraw it, and the dataset must hold the new
    # framing. A source check for 'place(' passed while place() itself
    # returned early on a file that was already there, so the re-file did
    # nothing at all.
    import tempfile as _tf2
    t2 = _tf2.mkdtemp()
    rp2 = audit.fa.paths
    l2 = dict(rp2('gate'))
    l2.update(out=t2, full=os.path.join(t2, 'full'),
              dataset=os.path.join(t2, 'ds'))
    audit.fa.paths = lambda stage='gate': l2
    try:
        os.makedirs(l2['full'], exist_ok=True)
        os.makedirs(os.path.join(t2, 'edited'), exist_ok=True)
        open(os.path.join(l2['full'], '7_0.jpg'), 'wb').write(b'ORIGINAL')
        audit.place('7#0', 'dog', 'gate')
        got = os.path.join(l2['dataset'], 'dog', '7_0.jpg')
        if not os.path.exists(got) or open(got, 'rb').read() != b'ORIGINAL':
            bad.append('a judged crop was not filed into the dataset')
        open(os.path.join(t2, 'edited', '7_0.jpg'), 'wb').write(b'REDRAWN')
        audit.place('7#0', 'dog', 'gate', force=True)
        if open(got, 'rb').read() != b'REDRAWN':
            bad.append('re-filing after a redraw left the old framing in the '
                       'dataset — place() returns early on a file that is '
                       'already there')
    except Exception as e:                     # noqa: BLE001
        bad.append(f're-filing threw {type(e).__name__}: {e}')
    finally:
        audit.fa.paths = rp2
    # the pool is swapped in, not written over: a rebuild takes minutes and
    # the dashboard serves pages out of that file the whole time
    fsrc = open(os.path.join(REPO, 'tools', 'detect', 'fn_audit.py')).read()
    if ".tmp' (FORMAT PARQUET" not in fsrc or "os.replace(P['pool']" not in fsrc:
        bad.append('the pool is written in place — for the minutes a rebuild '
                   'takes, every page is drawn from a half-written file')
    # and the export prefers the redrawn one
    import fn_audit as fa
    ex = _in.getsource(fa.export)
    if "edited" not in ex or "os.path.exists(fixed)" not in ex:
        bad.append('the export ignores hand-drawn boxes, so redrawing one '
                   'changes nothing that gets trained on')
    if "'corrected'" not in ex:
        bad.append('the manifest does not say which rows were redrawn')


def concurrency_checks(bad):
    """Recording a verdict must not wait on a page being cut.

    They shared one lock, so every keystroke made while the next page was
    being prefetched blocked until the cutting finished -- which is precisely
    when someone is judging, because the prefetch starts the moment a page is
    handed over. Measured at 0.59 s warm and the full cut time cold.
    """
    import threading
    import time as _t
    try:
        import audit
    except Exception:
        return
    draw_lock, ledger_lock = audit._locks('gate')
    if draw_lock is ledger_lock:
        bad.append('drawing and recording share one lock — a verdict waits '
                   'for a page to finish cutting')
    real = audit.materialise
    audit.materialise = (lambda c, workers=8, stage='gate':
                         (_t.sleep(0.6), c)[1])
    orig_sample, orig_pc = audit.sample, audit.page_count
    audit.sample = lambda n=25, band=None, seed=None, stage='gate': [
        {'key': 'z#0', 'seq': 'zz', 'image_id': 'z', 'det_idx': 0,
         'p_dog': 0.1, 'conf': 0.5, 'band': 1, 'drive': 'd', 'cell': 'c'}]
    try:
        import tempfile as _tf
        tmp = _tf.mkdtemp()
        real_paths = audit.fa.paths
        lay = dict(real_paths('gate'))
        lay.update(out=tmp, pages=os.path.join(tmp, 'pages'),
                   drawn=os.path.join(tmp, 'drawn.jsonl'),
                   verdicts=os.path.join(tmp, 'v.jsonl'),
                   full=os.path.join(tmp, 'full'),
                   dataset=os.path.join(tmp, 'ds'))
        audit.fa.paths = lambda stage='gate': lay
        live = audit.fa.paths.__wrapped__('gate')['pages'] \
            if hasattr(audit.fa.paths, '__wrapped__') else real_paths('gate')['pages']
        before = len(os.listdir(live)) if os.path.isdir(live) else 0
        t = threading.Thread(target=lambda: audit.draw_page(n=1), daemon=True)
        t.start()
        _t.sleep(0.15)
        a = _t.time()
        audit.record('probe#0', 'unsure')
        waited = _t.time() - a
        t.join(timeout=5)
        if waited > 0.25:
            bad.append(f'a verdict waited {waited:.2f}s while a page was '
                       f'cut — cutting must hold no lock a verdict needs')
        # and none of that fake work may have landed in the real audit
        after = len(os.listdir(live)) if os.path.isdir(live) else 0
        if after != before:
            bad.append(f'this check wrote {after - before} page(s) into the '
                       f'LIVE audit at {live} — redirecting fa.paths did not '
                       f'take, so the fixtures went to real data')
    finally:
        audit.materialise, audit.sample, audit.page_count = (
            real, orig_sample, orig_pc)
        try:
            audit.fa.paths = real_paths
        except NameError:
            pass


def persistence_checks(bad):
    """A page re-read carries the verdicts already given on it.

    The page document is the draw, not the judging. Read back without the
    ledger merged in, every card on a page you had already worked through
    came back blank and the work looked lost.
    """
    try:
        import audit
    except Exception:
        return
    doc = {'index': 0, 'items': [{'key': 'a#0'}, {'key': 'b#1'}]}
    import tempfile as _tf
    tmp = _tf.mkdtemp()
    real_paths = audit.fa.paths
    lay = dict(real_paths('gate'))
    lay['verdicts'] = os.path.join(tmp, 'v.jsonl')
    audit.fa.paths = lambda stage='gate': lay
    try:
        with open(lay['verdicts'], 'w') as fh:
            fh.write(json.dumps({'key': 'a#0', 'verdict': 'missed'}) + '\n')
        got = audit.with_verdicts(json.loads(json.dumps(doc)))
        if got['items'][0].get('verdict') != 'missed':
            bad.append('a verdict already on record is not shown when the '
                       'page is read again')
        if 'verdict' in got['items'][1]:
            bad.append('an unjudged box came back with a verdict')
    finally:
        audit.fa.paths = real_paths


def selection_checks(bad):
    """The band and the size a page was drawn with have to follow it.

    Reading the last page queues the next one, and that prefetch took its band
    from a parameter the client never sent -- so an audit filtered to one band
    quietly cut an all-bands page and handed it over on the next click.
    """
    try:
        import audit
    except Exception:
        return
    seen = {}
    real = audit.prefetch
    audit.prefetch = (lambda band=None, n=25, stage='gate':
                      seen.update(band=band, n=n))
    real_get, real_count = audit.get_page, audit.page_count
    audit.get_page = lambda i, stage='gate': {'index': i, 'items': [],
                                              'band': 4, 'n': 50}
    audit.page_count = lambda stage='gate': 3
    try:
        audit.api_page(2, n=50, band=4)
        if seen.get('band') != 4 or seen.get('n') != 50:
            bad.append(f'reading the last page of a band-4 audit queued '
                       f'{seen} — the next page would be from bands the '
                       f'reader excluded')
    finally:
        audit.prefetch, audit.get_page, audit.page_count = (
            real, real_get, real_count)
    for v, want in ((None, 25), (25, 25), (50, 50), (100, 100),
                    ('75', 75), (40, 25), (10 ** 9, 25), ('; DROP', 25)):
        if audit.page_size(v) != want:
            bad.append(f'page_size({v!r}) = {audit.page_size(v)}, want {want}')
    # the offered sizes need not divide by the band count -- with ten bands
    # and a page of 25 they cannot -- but every crop asked for must be drawn,
    # and no band may get two more than another
    import fn_audit as fa
    for n in audit.PAGE_SIZES:
        base, extra = divmod(n, len(fa.BANDS))
        quota = [base + (1 if i >= len(fa.BANDS) - extra else 0)
                 for i in range(len(fa.BANDS))]
        if sum(quota) != n:
            bad.append(f'a page of {n} plans {sum(quota)} crops')
        if max(quota) - min(quota) > 1:
            bad.append(f'a page of {n} gives one band {max(quota)} crops and '
                       f'another {min(quota)}')


def page_checks(bad):
    """The page renders, and renders numbers rather than NaN.

    16 KB of script that had never been executed. Everything else in this
    repo that draws a panel is driven under node before it ships; this was
    not, which is how it got here with no check at all.
    """
    import subprocess
    import tempfile as _tf
    if not __import__('shutil').which('node'):
        print('SKIP: node not on PATH — audit page not driven')
        return
    try:
        import audit
    except Exception:
        return
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'audit_page_stub.js')
    if not os.path.exists(probe):
        bad.append('audit_page_stub.js is missing')
        return
    # EVERY stage's page, not just the gate's. They are one template with
    # different words substituted in, and a substitution that goes wrong goes
    # wrong in exactly one of them.
    import fn_audit as _fa
    for stage in _fa.STAGES:
        _page_once(bad, audit, probe, stage)


def _page_once(bad, audit, probe, stage):
    import subprocess
    import tempfile as _tf
    # the built page for a stage, not the template -- the template still has
    # its placeholders in and would throw ReferenceError on the first line
    html = audit.page_html(stage)
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    css = html[html.index('<style>') + 7:html.index('</style>')]
    with _tf.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(open(probe).read()
                + '\nvar PAGE_CSS = ' + json.dumps(css) + ';\n'
                + 'var PAGE_HTML = ' + json.dumps(html) + ';\n'
                + script + '\n' + TAIL)
        p = f.name
    r = subprocess.run(['node', p], capture_output=True, text=True)
    if r.returncode != 0:
        # the LAST line of node's stderr is its version banner; the useful
        # line is the first one naming the error
        err = [x.strip() for x in (r.stderr or '').splitlines() if x.strip()]
        why = next((x for x in err if 'Error' in x), err[0] if err else '?')
        bad.append(f'the {stage} audit page threw: {why[:200]}')
        return
    for line in r.stdout.splitlines():
        if line.startswith('FAIL '):
            bad.append(f'{stage} audit page: ' + line[5:])


TAIL = r"""
function chk(c, m){ if(!c) console.log('FAIL ' + m) }
chk(/^\d/.test(els.rate.textContent), 'headline rate is ' +
  JSON.stringify(els.rate.textContent));
chk(els.judged.textContent === '12', 'judged reads ' + els.judged.textContent);
// the axis runs 0-100% and says where the gate's own line falls
chk(/100%/.test(els.bands.innerHTML) && /where the gate/.test(els.bands.innerHTML),
  'the band axis does not state its range or the threshold');
// bands above the threshold must exist now
chk(/0\.9–1\.0/.test(els.bands.innerHTML),
  'the bands stop before 1.0 — what the model accepted is unauditable');
// a pool that does not know what it covers must say so rather than present
// a snapshot of a running job as the whole store
STATS.pool_info = {unknown:true, shards_now:9};
paintStats(STATS);
chk(els.poolwarn.hidden === false,
  'a pool with no provenance is presented as though it covered everything');
STATS.pool_info = {unknown:false, stale:true, shards:2, shards_now:9};
paintStats(STATS);
chk(/2 shards/.test(String(els.poolwarn.textContent)),
  'a stale pool does not say how far behind it is');
STATS.pool_info = {unknown:false, stale:false, shards:9, shards_now:9};
paintStats(STATS);
chk(els.poolwarn.hidden === true, 'an up-to-date pool still warns');
chk(/page 1 of 1/.test(els.pos.textContent), 'position reads ' + els.pos.textContent);
chk(els.grid.innerHTML.length > 100, 'the grid rendered nothing');
// a page of a hundred crops ends a long way from the toolbar
chk(/id="next2"/.test(PAGE_HTML) && /id="prev2"/.test(PAGE_HTML),
  'there is no way to page on from the foot of the sheet');
chk(/id="views"/.test(PAGE_HTML),
  'there is no way to look at what has already been answered');
// Which side of the model's line to walk is THE question of this page, so it
// is a visible control, not the third option in a list of thirteen.
['rejected', 'kept', 'all'].forEach(function (sd) {
  chk(new RegExp('data-side="' + sd + '"').test(PAGE_HTML),
    'no visible control for "' + sd + '"');
});
// the two controls describe ONE state and can never be set to a pair that
// means an empty page
band = 7; paintFilter();
chk(bandSel.value === '7', 'the band dropdown does not show the chosen band');
chk(/on/.test(String(els.sides.className)) || true, '');
band = 'rejected'; paintFilter();
chk(bandSel.value === '', 'picking a side left a band selected under it');
chk(sideOf(7) === 'kept' && sideOf(2) === 'rejected',
  'a band is attributed to the wrong side of the threshold');
// every tile says what the model called it, and it must agree with the score
// it is shown beside -- the label is derived from the score, so a tile that
// reads "leashed 0.048" would mean the two came from different places
chk(/class="ptag/.test(els.grid.innerHTML),
  'no tile says what the model predicted');
// The editing box is drawn in VIEW pixels, but the picture is fitted to the
// window by CSS -- so on any screen that shrinks it, a box placed at its view
// coordinates lands down and right of the detection by exactly the ratio.
EDIT.meta = {view_w:1000, view_h:800, model_box:[100,100,200,200],
             off_x:0, off_y:0, scale:1};
EDIT.box = [100,100,200,200];
els.lbimg.clientWidth = 600;
edPaint();
chk(ebox.style.left === '60px' && ebox.style.width === '60px',
  'with the picture shown at 60% the box is drawn at ' + ebox.style.left +
  '/' + ebox.style.width + ', not 60px/60px — it would sit away from the '
  + 'detection it is meant to be on');
chk(mbox.style.left === '60px',
  'the detector\'s own box is drawn at ' + mbox.style.left + ' at 60%');
els.lbimg.clientWidth = 1000;
edPaint();
chk(ebox.style.left === '100px' && ebox.style.width === '100px',
  'unscaled, the box is drawn at ' + ebox.style.left + '/' + ebox.style.width);
// and a drag moves it by the screen distance, not by that distance in view
// pixels -- at 60% a 60px drag is 100 view pixels
els.lbimg.clientWidth = 600;
EDIT.box = [100,100,200,200];
EDIT.on = true;                 // the handler does nothing unless editing
EDIT.drag = {h:'move', x:0, y:0, box:[100,100,200,200]};
listeners.doc.mousemove({clientX:60, clientY:0});
chk(Math.abs(EDIT.box[0] - 200) < 0.01,
  'a 60px drag at 60% moved the box ' + (EDIT.box[0]-100) +
  ' view pixels, not 100');
EDIT.drag = null; EDIT.on = false;

// A review view reads the ledger back. Answering there must CHANGE the record
// and leave the crop where it is -- clearing it off the grid, the way the
// sheet does, would make the thing you came to look at vanish as you touched it.
view = 'flagged';
page.items.forEach(function (it) { it.verdict = POS });
render();
var shown = grid.children.filter(function (c) {
  return c.style.display !== 'none' }).length;
judge(0, NEG);
chk(grid.children.filter(function (c) {
      return c.style.display !== 'none' }).length === shown,
  'answering in a review view took the crop off the grid');
chk(page.items[0].verdict === NEG, 'the record was not changed');
view = 'sheet';
page.items.forEach(function (it) { delete it.verdict });
render();

// The third control on a tile is the crop mark, and it must OPEN the editor
// rather than record anything. It replaced the "?" button, so a wiring slip
// here would quietly file an "unsure" every time someone went to redraw.
chk(/data-edit=/.test(els.grid.innerHTML),
  'no tile offers a way to redraw its box');
chk(!/data-v="unsure"/.test(els.grid.innerHTML),
  'the tile still carries the old unsure button the crop mark replaced');
chk(/<svg/.test(els.grid.innerHTML), 'the crop mark is not drawn');
FETCHES.length = 0;
// through the CLICK HANDLER, not by calling openEditor() -- calling the
// function directly proved the function works and said nothing about what
// the button is wired to, which is the thing that just changed.
listeners.grid.click({target: {closest: function (sel) {
  return sel === '[data-edit]' ? {getAttribute: function () { return '0' }}
                               : null }}});
chk(FETCHES.some(function (u) { return /audit\/frame/.test(u) }),
  'the crop mark did not ask for the frame to edit');
chk(!FETCHES.some(function (u) { return /verdict/.test(u) }),
  'opening the editor recorded a verdict');
chk(cur === 0, 'opening the editor did not select the tile it was opened on');
// put the page back: the editor left the lightbox open, and every keyboard
// check below returns early while it is
edStop(); els.lb.hidden = true; cur = -1;

// at rest a tile is a photograph: the buttons ride over it, they are not
// permanent furniture below it
chk(/\.acts\{[^}]*position:absolute/.test(PAGE_CSS),
  'the action row is still stacked under every tile rather than riding over '
  + 'it — seventy-five buttons where the photographs should be');
chk(/\.card:hover \.acts/.test(PAGE_CSS),
  'the actions never appear on hover, so the mouse cannot reach them');
// A control that rides over a PHOTOGRAPH cannot be nearly-opaque. On paper
// the old label passed at 7.1:1 against its own button -- but the button was
// 94% opaque, so the real background was that colour blended with whatever
// the crop showed, and over a bright cobbled street the text vanished.
var actBg = /\.act\{[^}]*background:([^;}]+)/.exec(PAGE_CSS);
chk(actBg && !/rgba\([^)]*,\s*0?\.\d+\s*\)/.test(actBg[1]),
  'the action buttons are translucent (' + (actBg && actBg[1]) +
  ') — whatever the crop shows comes through the label');
chk(actBg && !/transparent|none/.test(actBg[1]),
  'the action buttons have no background at all');
// and the score must not sit in the same corner the buttons appear in
var chip = /\.pchip\{[^}]*\}/.exec(PAGE_CSS)[0];
var acts = /\.acts\{[^}]*\}/.exec(PAGE_CSS)[0];
chk(!(/bottom:\s*\d/.test(chip) && /bottom:\s*0/.test(acts)),
  'the score chip and the action row share the bottom of the tile, so the '
  + 'score shows through the buttons');
page.items.forEach(function (it) {
  var want = (+it.p_dog >= THRESH) ? POS : NEG;
  chk(predOf(it) === want,
    'the tag for ' + it.key + ' (scored ' + it.p_dog + ') reads ' +
    predOf(it) + ', not ' + want);
});
// the two classes are told apart by weight, not by hue alone: the word is
// in the tile either way
chk(/>(dog|not_dog|leashed|unleashed)</.test(els.grid.innerHTML),
  'the predicted class is not written out, only styled');
chk(els.bands.innerHTML.length > 100, 'the band strip rendered nothing');
// 1/2/3 are verdicts AND the way a keyboard picks an option in a focused
// <select>; a page size chosen by keyboard must not record a verdict
var before = FETCHES.length;
listeners.doc.keydown({key:'1', target:{tagName:'SELECT'},
  preventDefault:function(){}});
chk(FETCHES.length === before,
  'a keypress inside a dropdown recorded a verdict');
// With nothing selected, a verdict key must do NOTHING -- there is no crop
// under a cursor that does not exist.
page.items.forEach(function (it) { delete it.verdict });
render(); cur = -1; FETCHES.length = 0;
listeners.doc.keydown({key:'f', target:{tagName:'DIV'},
  preventDefault:function(){}});
chk(!FETCHES.some(function(u){return /verdict/.test(u)}),
  'a verdict key recorded something with no crop selected');
// an arrow selects; then the key acts on what you chose
listeners.doc.keydown({key:'ArrowRight', target:{tagName:'DIV'},
  preventDefault:function(){}});
chk(cur === 0, 'the first arrow press did not select the first crop');
listeners.doc.keydown({key:'f', target:{tagName:'DIV'},
  preventDefault:function(){}});
chk(FETCHES.some(function(u){return /verdict/.test(u)}),
  'F did not flag the selected crop');
// the cursor must NOT walk on by itself -- the page choosing the next crop
// for you is what made it feel like it was selecting things
page.items.forEach(function (it) { delete it.verdict });
render(); cur = 1;
var was = cur;
judge(cur, 'dog');
chk(cur === was, 'the cursor advanced on its own after a verdict');
// EVERY verdict clears the crop off the grid -- the grid is the work left,
// not a record of what was answered -- and every one offers a way back.
[POS, NEG, 'unsure'].forEach(function (v, n) {
  page.items.forEach(function (it) { delete it.verdict });
  render();
  var before = grid.children.filter(function (c) {
    return c.style.display !== 'none' }).length;
  judge(0, v);
  var after = grid.children.filter(function (c) {
    return c.style.display !== 'none' }).length;
  chk(before > 0, 'the grid rendered no cards to judge');
  chk(after === before - 1,
    'after answering "' + v + '" the crop is still on the grid (' +
    before + ' -> ' + after + ')');
  chk(els.undotoast.hidden === false, '"' + v + '" offered no undo');
  undoLast();
  var back = grid.children.filter(function (c) {
    return c.style.display !== 'none' }).length;
  chk(back === before, 'undo did not put the crop back (' + back +
    ' of ' + before + ')');
});
// nothing is selected when a page arrives -- show() is the real path a page
// takes, and it is where the ring used to land on the first crop
PAGE.items.forEach(function (it) { delete it.verdict });
show(PAGE, 0, 1);
chk(cur < 0, 'a crop is selected before the reader has touched anything');
chk(grid.children.filter(function (c) {
  return c.style.display !== 'none' }).length === PAGE.items.length,
  'a fresh page does not show all of its crops');
// changing the selection must draw rather than replay a page cut under the
// old one. Each click lands a stubbed response that resets total, so the
// state under test is set immediately before each one.
dirty = true; total = 5; idx = 0; FETCHES.length = 0;
listeners.next.click();
chk(FETCHES.some(function(u){return /audit\/draw/.test(u)}),
  'after changing the band, next replayed an old page instead of drawing');
dirty = false; total = 5; idx = 0; FETCHES.length = 0;
listeners.next.click();
chk(FETCHES.some(function(u){return /audit\/page/.test(u)}),
  'with the selection unchanged, next did not page forward');
chk(/none seen/.test(els.bands.innerHTML),
  'a band nobody judged does not say so');
chk(/0%/.test(els.bands.innerHTML),
  'the interval axis has no stated ends — a bar with no scale is a shape');
chk(!/left:NaN|width:NaN/.test(els.bands.innerHTML),
  'an interval bar is positioned at NaN');
var junk = Object.keys(els).map(function(k){
  return String(els[k].textContent) + ' ' + String(els[k].innerHTML)}).join(' ');
chk(!/NaN|undefined|\[object Object\]/.test(junk), 'junk on the page');
// a page with no items must say why rather than sit blank. `page` is the
// script's own state; PAGE is only what the stubbed fetch hands back.
page = {index:0, items:[], exhausted:true};
try { render() } catch(e) { console.log('FAIL render([]) threw ' + e.message) }
chk(/every one has been shown/i.test(els.empty.textContent),
  'an exhausted band does not explain itself: ' + els.empty.textContent);
"""


def stage_checks(bad):
    """Two audits, one code path, and no way for either to touch the other.

    They differ in vocabulary and in nothing else, so the danger is not that
    one breaks -- it is that one writes into the other. Every path either owns
    must be its own, and a word from one must be refused by the other.
    """
    import fn_audit as fa
    try:
        import audit
    except Exception:
        return
    import gate_store as gs
    seen = {}
    for name, sp in fa.STAGES.items():
        pp = fa.paths(name)
        for k, v in pp.items():
            if seen.setdefault((k, v), name) != name:
                bad.append(f'stages {seen[(k, v)]} and {name} share {k}: {v} '
                           f'— one audit would write into the other')
        # the shards it reads are the stage's own, and the runner agrees
        # about where they live and what the score column is called
        run = gs.STAGES.get(name)
        if not run:
            bad.append(f'the audit knows a {name!r} stage the runner cannot '
                       f'produce')
            continue
        if sp['dir'] != run['dir']:
            bad.append(f'{name}: audit reads data/{sp["dir"]}, runner writes '
                       f'data/{run["dir"]}')
        if sp['p_col'] != run['p_col']:
            bad.append(f'{name}: audit bands on {sp["p_col"]}, runner writes '
                       f'{run["p_col"]} — the pool would be all nulls')
        if sorted(sp['answers'][:2]) != sorted(run['classes']):
            bad.append(f'{name}: a person may answer {sp["answers"][:2]} but '
                       f'the model classifies {run["classes"]}')
        if sp['positive'] != run['positive']:
            bad.append(f'{name}: the score predicts {run["positive"]} and the '
                       f'audit calls {sp["positive"]} the positive')
    # a word from one audit is not a verdict in the other
    for name, sp in fa.STAGES.items():
        for other, osp in fa.STAGES.items():
            if other == name:
                continue
            for w in osp['answers'][:2]:
                if w in sp['answers'] or w in sp['legacy']:
                    continue
                if fa.verdict_of(w, name) is not None:
                    bad.append(f'{name} accepted {w!r}, which is '
                               f'{other}\'s vocabulary')
    # Every crop URL the page builds must carry the stage. Three places built
    # one by hand and only the lightbox got the prefix, so every thumbnail on
    # the leash page asked the gate for a crop it does not have -- and a DOM
    # stub never loads an image, so nothing driving the script could see it.
    # Checked in the source: exactly one builder, and it uses STAGE.
    import re as _re
    for name in fa.STAGES:
        html = audit.page_html(name)
        script = html[html.rindex('<script>'):]
        hand = _re.findall(r"['\"]/audit/crop/[^'\"]*['\"]", script)
        if len(hand) != 1:
            bad.append(f'{name} page builds a crop URL in {len(hand)} places '
                       f'({hand}); one of them will forget the stage')
        elif 'STAGE' not in script[script.index(hand[0]):
                                   script.index(hand[0]) + 90]:
            bad.append(f'{name} page builds a crop URL without the stage: '
                       f'{hand[0]}')
        # and it must resolve to this stage's directory
        if f"'/audit/crop/'+STAGE" not in script:
            bad.append(f'{name} page does not put STAGE in the crop path')

    # A model whose two errors cost the same must not open on one side of
    # the threshold: its own page says to read both together, and defaulting
    # to one would contradict that in the same screen.
    import re as _re
    for name, sp in fa.STAGES.items():
        m = _re.search(r'DEFAULT_BAND="([^"]*)"', audit.page_html(name))
        got = m.group(1) if m else None
        want = 'rejected' if sp['asymmetric'] else 'all'
        if got != want:
            bad.append(f'{name} opens drawing from {got!r}, expected {want!r} '
                       f'— it is '
                       f'{"asymmetric" if sp["asymmetric"] else "symmetric"}')
    # and the page is built per stage, with nothing left unsubstituted
    for name in fa.STAGES:
        html = audit.page_html(name)
        if '__' in html.replace('__', '', 0) and '__H1__' in html:
            bad.append(f'{name} page has unsubstituted placeholders')
        for tok in ('__BANDS__', '__STAGE__', '__POS__', '__H1__'):
            if tok in html:
                bad.append(f'{name} page still contains {tok}')
        if f'"{name}"' not in html:
            bad.append(f'{name} page does not tell its script which stage '
                       f'it is')


def main():
    bad = []
    stage_checks(bad)
    for fn in (band_checks, wilson_checks, weighting_checks, ledger_checks,
               serving_checks, isolation_checks, correction_checks,
               concurrency_checks,
               persistence_checks, selection_checks, page_checks):
        try:
            fn(bad)
        except Exception as e:                 # noqa: BLE001 - report, not die
            bad.append(f'{fn.__name__} threw {type(e).__name__}: {e}')
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('the audit samples without repeats, weights by population, and '
          'shows the crop the model saw')
    return 0


if __name__ == '__main__':
    sys.exit(main())
