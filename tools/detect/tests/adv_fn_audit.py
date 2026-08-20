#!/usr/bin/env python3
"""The false-negative audit has to produce a number, not a pile of clicks.

Everything here guards one of the four claims the page makes:

  * you will never be shown the same box twice, and never two frames from one
    sequence -- otherwise the sample is correlated and the interval is a lie;
  * the rate is weighted by how many boxes the gate really put in each band,
    because the bands are drawn from evenly and a flat mean would report the
    near-threshold error rate as if it were the whole store's;
  * and it says what it is a share OF -- the sheets record every crop put in
    front of a person, a fraction of them come back, and until one is answered
    in full the rate off it is a ceiling rather than an estimate;
  * the crop shown is the crop the model saw;
  * a draw is what it says it is -- 'least/most confident' really walks the
    distance from the threshold, the annotated-date filter really narrows the
    ledger on the server, and the shared tab strip is the contract's markup.

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


def _sheet(pages, i, items, band=None):
    """One page document, written the way draw_page writes it: under an index
    that is the count of the pages before it, and never touched again.

    `band` is the page's own band value -- the compound 'rejected~most' when
    the sheet was drawn by a confidence walk, which is the only record of how
    its crops were chosen."""
    with open(os.path.join(pages, f'{i:05d}.json'), 'w') as fh:
        json.dump({'index': i, 'band': band, 'n': len(items),
                   'items': [{'key': k, 'band': b} for b, k in items]}, fh)


def _leaked(P, mark):
    """Where a fixture's own keys turn up in a store it must never reach.

    Not a byte comparison: the dashboard is serving these audits while this
    runs and somebody answering a crop mid-check is not a fault. A key only
    this file invents is, and it says exactly which redirect did not take.
    """
    hit = []
    for k in ('verdicts', 'drawn'):
        try:
            with open(P[k]) as fh:
                if mark in fh.read():
                    hit.append(os.path.basename(P[k]))
        except OSError:
            pass
    try:
        for f in sorted(os.listdir(P['pages'])):
            with open(os.path.join(P['pages'], f)) as fh:
                if mark in fh.read():
                    hit.append('pages/' + f)
    except OSError:
        pass
    return hit


def sheets_checks(bad):
    """What was SHOWN is the page documents, and it is the denominator.

    A band's answers are not a sample of the band. They are the crops somebody
    chose to click on a sheet where every other tile could be scrolled past,
    and the one worth stopping on is the one that already looks like the
    answer -- so a band's share is a ceiling until its sheets are answered in
    full, and the answer rate has to travel beside it for that to be readable
    at all.

    Driven against a store built for it. The live one is open in the dashboard
    while this runs, and a session that redirected one path and not another
    put seventeen invented verdicts into it, so the last thing here is that
    nothing this check invented reached either real store.
    """
    import fn_audit as fa
    import tempfile as _tf
    mark = 'zzguard_'
    real_paths = fa.paths
    tmp = _tf.mkdtemp()
    lay = dict(real_paths('gate'))
    lay.update(out=tmp, pages=os.path.join(tmp, 'pages'),
               verdicts=os.path.join(tmp, 'v.jsonl'),
               drawn=os.path.join(tmp, 'drawn.jsonl'),
               full=os.path.join(tmp, 'full'),
               crops=os.path.join(tmp, 'crops'),
               dataset=os.path.join(tmp, 'ds'),
               pool=os.path.join(tmp, 'pool.parquet'))
    fa.paths = lambda stage='gate': lay
    try:
        os.makedirs(lay['pages'])
        # Bands 2 and 5 were shown and nobody answered them. That is the case
        # the two side totals have to include: a band with no answer in it is
        # exactly what the rate beside it must be read against, and summing
        # over the answered bands only would report a rate off six crops as a
        # rate off eight.
        _sheet(lay['pages'], 0, [(0, mark + 'a0'), (0, mark + 'a1'),
                                 (0, mark + 'a2'), (0, mark + 'a3'),
                                 (9, mark + 'k0'), (9, mark + 'k1')])
        _sheet(lay['pages'], 1, [(0, mark + 'a4'), (0, mark + 'a5'),
                                 (2, mark + 'b0'), (2, mark + 'b1'),
                                 (5, mark + 'c0'), (5, mark + 'c1'),
                                 (9, mark + 'k2')])
        with open(lay['verdicts'], 'w') as fh:
            for key, band, v in (('a0', 0, 'dog'), ('a1', 0, 'not_dog'),
                                 ('k0', 9, 'not_dog'),
                                 # answered before pages/ was kept: a verdict
                                 # with no sheet to be a share of
                                 ('x0', 0, 'dog')):
                fh.write(json.dumps({'key': mark + key, 'band': band,
                                     'verdict': v}) + '\n')
        got = {b: sorted(k) for b, k in fa.sheets('gate').items()}
        want = {0: [mark + k for k in ('a0', 'a1', 'a2', 'a3', 'a4', 'a5')],
                2: [mark + 'b0', mark + 'b1'],
                5: [mark + 'c0', mark + 'c1'],
                9: [mark + 'k0', mark + 'k1', mark + 'k2']}
        if got != want:
            bad.append(f'the sheets read back as {got}; the page documents '
                       f'are the record of what was put in front of a person '
                       f'and nothing else is')
        totals = [(lo, hi, 1000 if i == 0 else 500 if i == 9 else 10)
                  for i, (lo, hi) in enumerate(fa.BANDS)]
        s = fa.summarise(totals=totals)
        b = s['bands']
        for i, sh, an in ((0, 6, 2), (2, 2, 0), (5, 2, 0), (9, 3, 1),
                          (1, 0, 0)):
            if (b[i]['shown'], b[i]['answered']) != (sh, an):
                bad.append(f"band {i} was shown {b[i]['shown']} crops and "
                           f"answered {b[i]['answered']}, not {sh} and {an}")
        if (s['rejected']['shown'], s['rejected']['answered']) != (8, 2):
            bad.append(f"below the line the sheets showed "
                       f"{s['rejected']['shown']} crops and "
                       f"{s['rejected']['answered']} came back, not 8 and 2 "
                       f"— a band that was shown and never answered is part "
                       f"of what the rate beside it is short of")
        if (s['kept']['shown'], s['kept']['answered']) != (5, 1):
            bad.append(f"above the line the sheets showed "
                       f"{s['kept']['shown']} crops and "
                       f"{s['kept']['answered']} came back, not 5 and 1")
        if s['sheets'] != {'shown': 13, 'answered': 3, 'unrecorded': 1}:
            bad.append(f"the overall answer rate reads {s['sheets']}, not "
                       f"3 of 13 with 1 judged before the sheets were kept — "
                       f"a verdict with no sheet is not a share of anything, "
                       f"and dropping it would lose a human answer")
        # a page still being written is not a sheet
        with open(os.path.join(lay['pages'], '00002.json.tmp'), 'w') as fh:
            json.dump({'items': [{'key': mark + 'half', 'band': 3}]}, fh)
        if 3 in fa.sheets('gate'):
            bad.append('a half-written page counted as shown — draw_page '
                       'writes the document beside its final name and moves '
                       'it, so a glob that matches the temp reads a page '
                       'nobody has been served')
        # and a sheet drawn since the last poll is counted
        _sheet(lay['pages'], 2, [(3, mark + 'd0')])
        s2 = fa.summarise(totals=totals)
        if s2['sheets']['shown'] != 14:
            bad.append(f"a sheet drawn after the last poll is not counted: "
                       f"{s2['sheets']} — the cache is keyed on something "
                       f"coarser than the document")
    finally:
        fa.paths = real_paths
    for name in fa.STAGES:
        hit = _leaked(fa.paths(name), mark)
        if hit:
            bad.append(f'this check wrote its own keys into the LIVE {name} '
                       f'audit ({hit}) — redirecting fa.paths did not take, '
                       f'and the fixtures went to human data')


def targeted_draw_checks(bad):
    """A confidence draw is a search, not a sample, and the estimate knows it.

    'least/most confident' hand back one END of a band on purpose. Every rate
    here multiplies a band's share by the band's WHOLE population, so a page
    drawn at the edge and answered enters the estimate as though it had been
    drawn evenly across the band: measured on the live gate store, one 'most
    confident' page below the line came back 25 crops all scoring exactly
    0.000 out of a band holding 3,530,147 boxes, and answering it moved the
    headline anywhere from 52.6% to 97.3% away from 95.5%. The page prints
    that headline as "at most", so a targeted page that drags it DOWN turns a
    stated upper bound into a claim that is false in the unsafe direction.

    The page document is the only record of how its crops were chosen, so
    that is what this reads. The finds are still finds -- they are counted,
    named and kept out of the shares.
    """
    import fn_audit as fa
    import tempfile as _tf
    for v, want in (('rejected~most', 'most'), ('4~least', 'least'),
                    ('rejected~bogus', None), ('rejected', None),
                    (9, None), (None, None)):
        if fa.draw_mode_of(v) != want:
            bad.append(f'draw_mode_of({v!r}) = {fa.draw_mode_of(v)!r}, want '
                       f'{want!r} — a page whose draw cannot be read back is '
                       f'a page the estimate cannot hold out')
    try:
        import audit
        if audit.DRAW_MODES is not fa.DRAW_MODES:
            bad.append('audit.py keeps its own list of draw modes — the '
                       'control and the estimator must spell them the same '
                       'way or one of them stops recognising a targeted page')
    except Exception:
        pass
    mark = 'zzaim_'
    real_paths = fa.paths
    tmp = _tf.mkdtemp()
    lay = dict(real_paths('gate'))
    lay.update(out=tmp, pages=os.path.join(tmp, 'pages'),
               verdicts=os.path.join(tmp, 'v.jsonl'),
               drawn=os.path.join(tmp, 'drawn.jsonl'),
               full=os.path.join(tmp, 'full'), crops=os.path.join(tmp, 'crops'),
               dataset=os.path.join(tmp, 'ds'),
               pool=os.path.join(tmp, 'pool.parquet'))
    fa.paths = lambda stage='gate': lay
    try:
        os.makedirs(lay['pages'])
        spread = [(0, mark + f's{i}') for i in range(4)]
        aimed = [(0, mark + f'm{i}') for i in range(4)]
        _sheet(lay['pages'], 0, spread, band='rejected')
        _sheet(lay['pages'], 1, aimed, band='rejected~most')
        with open(lay['verdicts'], 'w') as fh:
            for _, key in spread:
                fh.write(json.dumps(
                    {'key': key, 'band': 0,
                     'verdict': 'dog' if key.endswith('s0') else 'not_dog'})
                    + '\n')
            for _, key in aimed:          # a targeted page, every one a dog
                fh.write(json.dumps({'key': key, 'band': 0,
                                     'verdict': 'dog'}) + '\n')
        totals = [(lo, hi, 1000 if i == 0 else 10)
                  for i, (lo, hi) in enumerate(fa.BANDS)]
        fa._SHEETS.clear()
        s = fa.summarise(totals=totals)
        b0 = s['bands'][0]
        if (b0['judged'], b0['dogs']) != (4, 1) or abs(b0['rate'] - .25) > 1e-9:
            bad.append(f"band 0 reads {b0['dogs']}/{b0['judged']} = "
                       f"{b0['rate']} — the four answers off the targeted "
                       f"sheet are in the share, and that share is "
                       f"multiplied by all 1,000 boxes in the band")
        if (b0['aimed'], b0['aimed_dogs'], b0['aimed_wrong']) != (4, 4, 4):
            bad.append(f"the targeted answers are not counted at all: "
                       f"aimed={b0['aimed']} dogs={b0['aimed_dogs']} "
                       f"wrong={b0['aimed_wrong']} — held out of the rate is "
                       f"not the same as thrown away")
        if b0['shown'] != 4 or b0['answered'] != 4:
            bad.append(f"band 0 counts {b0['shown']} crops shown and "
                       f"{b0['answered']} answered — a targeted sheet is not "
                       f"a denominator for the share it is held out of")
        if s['aimed'] != 4 or s['rejected']['aimed'] != 4:
            bad.append(f"nothing says how many answers were held out: "
                       f"aimed={s['aimed']}, below the line "
                       f"{s['rejected'].get('aimed')} — a page that holds "
                       f"answers out has to say how many")
        if abs(s['rejected']['rate'] - .25) > 1e-6:
            bad.append(f"the headline is {s['rejected']['rate']}, not the "
                       f"0.25 the spread sheet measured — a targeted page "
                       f"moved the population-weighted estimate")
        # THE CONTROL: the same eight answers, with the second sheet drawn
        # the ordinary way, must ALL count. Without this the check passes
        # just as well against a summarise() that counts nothing.
        _sheet(lay['pages'], 1, aimed, band='rejected')
        fa._SHEETS.clear()
        s2 = fa.summarise(totals=totals)
        c0 = s2['bands'][0]
        if (c0['judged'], c0['dogs'], c0['aimed']) != (8, 5, 0):
            bad.append(f"a SPREAD sheet's answers are being held out too: "
                       f"{c0['dogs']}/{c0['judged']}, aimed {c0['aimed']} — "
                       f"the even draw is the measurement, not an exception "
                       f"to it")
        if abs(s2['rejected']['rate'] - .625) > 1e-9:
            bad.append(f"the headline off two spread sheets is "
                       f"{s2['rejected']['rate']}, not 0.625")
    finally:
        fa.paths = real_paths
        fa._SHEETS.clear()
    for name in fa.STAGES:
        hit = _leaked(fa.paths(name), mark)
        if hit:
            bad.append(f'this check wrote its own keys into the LIVE {name} '
                       f'audit ({hit}) — redirecting fa.paths did not take, '
                       f'and the fixtures went to human data')


def legacy_checks(bad):
    """Answers written in the older words, and they still count.

    'missed' and 'correct' only made sense while the pool was the gate's
    rejections; the wording changed when it grew to hold what the gate KEPT,
    and the rows already on disk did not change with it. So every reader goes
    through verdict_of, and a rename that forgets one of them drops human
    answers with nothing on screen to say so -- the tile paints from this
    stage's own two words, and a find written in the old ones matched none of
    them, so it read as an ordinary answered card.
    """
    import fn_audit as fa
    # The two words that are really on disk, named here rather than read back
    # out of the table that is supposed to map them -- a table that
    # reclassified them would agree with itself, and thirty-eight rows would
    # quietly become an answer nobody gave.
    for old, new in (('missed', 'dog'), ('correct', 'not_dog')):
        if fa.verdict_of(old, 'gate') != new:
            bad.append(f'a gate answer written {old!r} now reads as '
                       f'{fa.verdict_of(old, "gate")!r}, not {new!r} — every '
                       f'row on disk in that wording is dropped or recounted')
    for name, sp in fa.STAGES.items():
        for old, new in sp['legacy'].items():
            if new not in sp['answers']:
                bad.append(f'{name}: {old!r} maps to {new!r}, which is not '
                           f'something a person can answer here')
    # and they are counted, not merely readable
    totals = [(lo, hi, 100) for lo, hi in fa.BANDS]
    s = fa.summarise([{'key': 'a', 'band': 4, 'verdict': 'missed'},
                      {'key': 'b', 'band': 4, 'verdict': 'correct'}],
                     totals, shown={})
    if (s['bands'][4]['judged'], s['bands'][4]['dogs']) != (2, 1):
        bad.append(f"the older wording is not counted: band 4 reads "
                   f"{s['bands'][4]['judged']} judged and "
                   f"{s['bands'][4]['dogs']} dogs, want 2 and 1")
    try:
        import audit
    except Exception:
        return
    # what the server stamps on a tile is a word the tile can paint
    import tempfile as _tf
    tmp = _tf.mkdtemp()
    real_paths = fa.paths
    try:
        for name, sp in fa.STAGES.items():
            if not sp['legacy']:
                continue
            lay = dict(real_paths(name))
            lay['verdicts'] = os.path.join(tmp, name + '.jsonl')
            fa.paths = lambda stage=name, _l=lay: _l
            words = sorted(sp['legacy'])
            with open(lay['verdicts'], 'w') as fh:
                for i, w in enumerate(words):
                    fh.write(json.dumps({'key': f'{i}#0',
                                         'verdict': w}) + '\n')
            doc = audit.with_verdicts(
                {'items': [{'key': f'{i}#0'} for i in range(len(words))]},
                name)
            got = [it.get('verdict') for it in doc['items']]
            if [g for g in got if g not in sp['answers']]:
                bad.append(f'{name}: a page read back carries {got}; the '
                           f'tiles are painted by comparing against this '
                           f'stage\'s own two words, so a verdict in any '
                           f'other lights nothing')
    finally:
        fa.paths = real_paths
    # and every word anybody has ever written into a live ledger still reads
    for name in fa.STAGES:
        p = real_paths(name)['verdicts']
        if not os.path.exists(p):
            continue
        lost = {}
        for v in fa.read_verdicts(p, name):
            w = v.get('verdict')
            if w is not None and fa.verdict_of(w, name) is None:
                lost[w] = lost.get(w, 0) + 1
        for w, n in sorted(lost.items()):
            bad.append(f'{n} answers in the live {name} ledger are written '
                       f'{w!r} and nothing maps that any more — they are '
                       f'human verdicts and they have stopped counting')


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
    except ModuleNotFoundError:
        # the pool is duckdb's file and only one of this repo's interpreters
        # has it; reported rather than failed, because a check that cannot run
        # has not found anything
        print('SKIP: no duckdb in this interpreter — pool schema not read')
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
            bad.append('a redrawn box overwrote the model\'s thumbnail — '
                       'nothing is destroyed by a redraw, the new cut goes '
                       'beside the old one')
        if not os.path.exists(os.path.join(tmp, 'edited', '9_0.jpg')):
            bad.append('a redrawn box was not cut into edited/, so nothing '
                       'downstream can use it')
        # ...and the tile shows what you drew
        rt = os.path.join(tmp, 'edited_thumbs', '9_0.jpg')
        if not os.path.exists(rt):
            bad.append('a redraw wrote no thumbnail, so the tile would keep '
                       'showing the framing you just replaced')
        elif audit.crop_path('9_0', 'gate') != rt:
            bad.append(f'the served crop is {audit.crop_path("9_0", "gate")}, '
                       f'not the redrawn one')
        # A crop is cached for a day, so the tile must be able to ASK for the
        # new one. Without a version from the SERVER this worked until the
        # page was reloaded and then quietly reverted to the old cut.
        import time as _t3
        bf = audit.BOX_FILE
        audit.BOX_FILE = os.path.join(tmp, 'boxes.jsonl')
        try:
            with open(audit.BOX_FILE, 'w') as fh:
                fh.write(json.dumps({'image_id': '9', 'det_idx': 0,
                                     'x1': 1, 'y1': 1, 'x2': 9, 'y2': 9,
                                     'saved_at': 1234}) + '\n')
            doc = audit.with_verdicts(
                {'items': [{'key': '9#0'}]}, 'gate')
            it = doc['items'][0]
            if not it.get('corrected'):
                bad.append('a redrawn box is not marked on the tile')
            if it.get('v') != 1234:
                bad.append(f'the tile is given v={it.get("v")}, so after a '
                           f'reload it asks for the same URL the browser has '
                           f'already cached and the redraw disappears')
        finally:
            audit.BOX_FILE = bf
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
        # named, or the write is refused before it ever reaches the lock this
        # is timing and the measurement passes by measuring nothing
        audit.record('probe#0', 'unsure', by='probe')
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
    import fn_audit as fa
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
        # in this stage's own wording, not the ledger's: the row is one of the
        # thirty-eight written before the vocabulary changed, and the tile can
        # only paint the words it was built with
        if got['items'][0].get('verdict') != fa.STAGES['gate']['positive']:
            bad.append(f"a verdict already on record comes back as "
                       f"{got['items'][0].get('verdict')!r} when the page is "
                       f"read again")
        if 'verdict' in got['items'][1]:
            bad.append('an unjudged box came back with a verdict')
    finally:
        audit.fa.paths = real_paths


def passive_load_checks(bad):
    """A GET that merely LOOKS at the audit page spends no sample.

    Boxes are retired the moment they are drawn, and api_page used to queue a
    fresh draw whenever the last page was read -- so every passive load (a
    health checker, a crawler, a reviewer glancing) permanently retired 25
    stratified boxes: measured live, three page-loads moved the gate store
    from 55 to 58 pages while the verdict ledger sat untouched for a week.
    The contract now: an UNJUDGED last page queues nothing, a judged one
    queues the next -- that is a reviewer working, and the draw buys their
    Next click.
    """
    try:
        import audit
    except Exception:
        return
    st = {'queued': 0}
    real = (audit.prefetch, audit.get_page, audit.page_count,
            audit.with_verdicts)
    audit.prefetch = (lambda band=None, n=25, stage='gate':
                      st.update(queued=st['queued'] + 1))
    audit.page_count = lambda stage='gate': 3
    audit.with_verdicts = lambda doc, stage='gate': doc
    audit.get_page = lambda i, stage='gate': {
        'index': i, 'items': [{'key': f'p{i}#0'}]}
    try:
        # the load every /audit/<stage> page issues: i=-1, nothing judged
        for _ in range(3):
            audit.api_page(-1)
        if st['queued']:
            bad.append(f"{st['queued']} page draw(s) were queued by bare "
                       f"page-loads of an unjudged audit — a monitoring GET "
                       f"is spending stratified sample")
        # and the moment the page carries one verdict, the next page is cut
        audit.get_page = lambda i, stage='gate': {
            'index': i, 'items': [{'key': f'p{i}#0', 'verdict': 'dog'}]}
        audit.api_page(-1)
        if st['queued'] != 1:
            bad.append('a judged last page did not queue the next draw, so '
                       'every Next now pays the full cut')
    finally:
        (audit.prefetch, audit.get_page, audit.page_count,
         audit.with_verdicts) = real


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
    real_wv = audit.with_verdicts
    # one verdict on the page: the prefetch follows JUDGING now, so a page
    # that queues the next one has to be a page somebody has worked on --
    # see passive_load_checks for the other half of that contract
    audit.get_page = lambda i, stage='gate': {
        'index': i, 'band': 4, 'n': 50,
        'items': [{'key': 'a#0', 'verdict': 'dog'}]}
    audit.with_verdicts = lambda doc, stage='gate': doc
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
        audit.with_verdicts = real_wv
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


def draw_filter_checks(bad):
    """The two confidence draws pick the crops they claim to pick.

    'least' is the crops the model could barely decide -- scores nearest the
    threshold; 'most' is its surest calls -- the far ends of whichever side
    is asked for, and BOTH ends on 'all', where millions of boxes at 0.0 tie
    with a few hundred thousand at 1.0 and a tie broken at random is a page
    entirely of the crowded end. The walk is deterministic, so the no-repeat
    rule is the only thing standing between the reader and the same page for
    ever: a drawn sequence must move the walk on.
    """
    try:
        import duckdb
    except ModuleNotFoundError:
        print('SKIP: no duckdb in this interpreter — draw modes not exercised')
        return
    try:
        import audit
    except Exception:
        return
    import fn_audit as fa
    import tempfile as _tf
    for v, want in (('rejected~least', 'rejected~least'),
                    ('kept~most', 'kept~most'),
                    ('4~most', '4~most'),
                    ('rejected~bogus', 'rejected'),   # a typo is not a mode
                    ('bogus~least', None),            # nor a band
                    ('rejected', 'rejected'), (4, 4), (None, None)):
        if audit.band_arg(v) != want:
            bad.append(f'band_arg({v!r}) = {audit.band_arg(v)!r}, want '
                       f'{want!r} — the mode rides the band value and a '
                       f'wrong parse draws from somewhere else')
    mark = 'zzdraw_'
    tmp = _tf.mkdtemp()
    real_paths = fa.paths
    lay = dict(real_paths('gate'))
    lay.update(out=tmp, pool=os.path.join(tmp, 'pool.parquet'),
               drawn=os.path.join(tmp, 'drawn.jsonl'),
               verdicts=os.path.join(tmp, 'v.jsonl'),
               pages=os.path.join(tmp, 'pages'),
               crops=os.path.join(tmp, 'crops'),
               full=os.path.join(tmp, 'full'),
               dataset=os.path.join(tmp, 'ds'))
    fa.paths = lambda stage='gate': lay
    real_mat = audit.materialise
    audit.materialise = lambda c, workers=8, stage='gate': c
    try:
        # seq 'm1' holds a coin-flip AND a sure call: each walk must offer
        # the sequence through the box that belongs on that walk.
        #
        # LOPSIDED ON PURPOSE, because that is the shape of the pool and the
        # shape the interleave exists for: nine sequences below the line
        # against three above, and the four boxes farthest from the
        # threshold (0.001..0.004, distance 0.496+) all sit below it, ahead
        # of the surest call above (0.99, distance 0.49). An 'all~most' walk
        # that does NOT partition by side therefore returns 4/0 here. With
        # an even fixture it returned 2/2 either way and the assertion below
        # asserted nothing -- the interleave could be deleted and this file
        # stayed green.
        rows = [(0.49, 'm1'), (0.02, 'm1'), (0.45, 's2'), (0.48, 's3'),
                (0.05, 's4'), (0.30, 's8'),
                (0.001, 's9'), (0.002, 's10'), (0.003, 's11'),
                (0.004, 's12'),
                (0.55, 's5'), (0.95, 's6'), (0.99, 's7')]
        con = duckdb.connect()
        con.execute('CREATE TEMP TABLE t(band INT, image_id VARCHAR, '
                    'det_idx INT, p_dog DOUBLE, x1 DOUBLE, y1 DOUBLE, '
                    'x2 DOUBLE, y2 DOUBLE, cell VARCHAR, drive VARCHAR, '
                    'seq VARCHAR, conf DOUBLE)')
        con.executemany(
            'INSERT INTO t VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            [(fa.band_of(p), f'{mark}{i}', 0, p, 0, 0, 50, 50,
              'c', 'd', seq, 0.5) for i, (p, seq) in enumerate(rows)])
        con.execute(f"COPY t TO '{lay['pool']}' (FORMAT PARQUET)")
        con.close()

        def ps(cands):
            # three places, not two: the far-below rows that make the
            # 'all~most' fixture lopsided differ in the third, and rounding
            # them together would hide which of them a walk actually drew
            return sorted(round(c['p_dog'], 3) for c in cands)

        got = audit.sample(n=3, band='rejected~least', stage='gate')
        if ps(got) != [0.45, 0.48, 0.49]:
            bad.append(f'least-confident below the line drew {ps(got)}, not '
                       f'the three scores nearest the threshold')
        m1 = [c for c in got if c['seq'] == 'm1']
        if not m1 or round(m1[0]['p_dog'], 2) != 0.49:
            bad.append('a sequence holding a 0.49 and a 0.02 was offered '
                       'through the wrong box on a least-confident walk')
        got = audit.sample(n=2, band='rejected~most', stage='gate')
        if ps(got) != [0.001, 0.002]:
            bad.append(f'most-confident below the line drew {ps(got)}, not '
                       f'the scores nearest 0')
        got = audit.sample(n=2, band='kept~most', stage='gate')
        if ps(got) != [0.95, 0.99]:
            bad.append(f'most-confident above the line drew {ps(got)}, not '
                       f'the scores nearest 1')
        got = audit.sample(n=4, band='all~most', stage='gate')
        lo = sum(1 for c in got if c['p_dog'] < fa.THRESHOLD)
        if (lo, len(got) - lo) != (2, 2):
            bad.append(f"most-confident over 'all' split "
                       f'{lo}/{len(got) - lo} — the surest calls are BOTH '
                       f'ends, and one ordered walk over the pool hands back '
                       f'the crowded end alone (the fixture is lopsided the '
                       f'way the pool is, so 4/0 is what dropping the '
                       f'per-side partition looks like)')
        # and it is the surest TWO of each side, not any two: the walk is
        # ordered within a side as well as interleaved across the two
        if ps(got) != [0.001, 0.002, 0.95, 0.99]:
            bad.append(f"most-confident over 'all' drew {ps(got)}, not the "
                       f'two boxes farthest from the threshold on each side')
        # the walk is deterministic, so only the drawn record moves it on
        with open(lay['drawn'], 'w') as fh:
            fh.write(json.dumps({'key': f'{mark}0#0', 'seq': 'm1'}) + '\n')
        got = audit.sample(n=2, band='rejected~least', stage='gate')
        if ps(got) != [0.45, 0.48] or any(c['seq'] == 'm1' for c in got):
            bad.append(f'a drawn sequence came round again on a '
                       f'least-confident walk ({ps(got)}) — a deterministic '
                       f'order that ignores the record is the same page for '
                       f'ever')
        # A LEDGER ROW WITH A KEY AND NO SEQ. The pool was rebuilt once and
        # the log went with it, so this shape has existed here. The ANTI JOIN
        # only knows sequences, so such a row is dropped by the key filter in
        # Python afterwards -- and while the SQL fetched exactly n, that came
        # back as a short page reporting dropped=0, which reads as nothing at
        # all. The spread branch over-draws and absorbs it; the walk must too.
        with open(lay['drawn'], 'w') as fh:
            fh.write(json.dumps({'key': f'{mark}2#0'}) + '\n')
        got = audit.sample(n=3, band='rejected~least', stage='gate')
        if len(got) != 3:
            bad.append(f'a least-confident page came back {len(got)} of 3 '
                       f'after one key-only ledger row ({ps(got)}) — the '
                       f'walk must draw past a row the key filter removes, '
                       f'or a short page is served as a full one')
        if any(c['key'] == f'{mark}2#0' for c in got):
            bad.append('a key already in the ledger was drawn again')
        # and it replaces the dropped crop with the NEXT one on the walk,
        # rather than with whatever the order happened to reach
        if len(got) == 3 and ps(got) != [0.3, 0.48, 0.49]:
            bad.append(f'the walk filled the gap with {ps(got)}, not the '
                       f'next scores along from the threshold')
        os.remove(lay['drawn'])
        # the page document keeps the compound value, so a page read back
        # says how it was chosen, not only where from
        doc = audit.draw_page(n=2, band='rejected~least', stage='gate')
        if doc.get('band') != 'rejected~least':
            bad.append(f"a page drawn least-confident is stored as "
                       f"band={doc.get('band')!r} — the position line can "
                       f"no longer say what was drawn")
        doc2 = audit.draw_page(n=2, band='rejected~least', stage='gate')
        if {c['key'] for c in doc.get('items') or []} & \
                {c['key'] for c in doc2.get('items') or []}:
            bad.append('two least-confident pages shared a crop — drawn '
                       'boxes must never be shown twice, whatever the draw')
        # and the spread is untouched: no mode means the even draw
        got = audit.sample(n=4, band='rejected', stage='gate')
        if any(c['p_dog'] >= fa.THRESHOLD for c in got):
            bad.append('a plain rejected draw reached above the threshold')
    finally:
        audit.materialise = real_mat
        fa.paths = real_paths
    for name in fa.STAGES:
        hit = _leaked(fa.paths(name), mark)
        if hit:
            bad.append(f'this check wrote its own keys into the LIVE {name} '
                       f'audit ({hit}) — redirecting fa.paths did not take, '
                       f'and the fixtures went to human data')


def split_checks(bad):
    """Who answered what, split between the two classes.

    Two scopes on one sheet: what this reader put there, and what the ledger
    holds. Counted through read_verdicts() rather than by walking the file,
    which is what makes a crop answered twice count once, a withdrawal count
    as nothing, and a row written before this project had accounts belong to
    the admin who wrote it.

    The number that would go wrong quietly is `mine`. A split that credits
    one annotator with another's answers is not an error anybody will spot
    -- it just makes the two rows agree.
    """
    try:
        import audit
    except Exception:
        return
    import fn_audit as fa
    import tempfile as _tf
    mark = 'zzsplit_'
    tmp = _tf.mkdtemp()
    real_paths = fa.paths
    lay = dict(real_paths('gate'))
    lay.update(out=tmp, verdicts=os.path.join(tmp, 'v.jsonl'),
               drawn=os.path.join(tmp, 'drawn.jsonl'),
               pages=os.path.join(tmp, 'pages'),
               pool=os.path.join(tmp, 'pool.parquet'))
    fa.paths = lambda stage='gate': lay
    try:
        rows = [
            (f'{mark}a#0', 'dog', 'sam'), (f'{mark}b#0', 'dog', 'sam'),
            (f'{mark}c#0', 'not_dog', 'sam'),
            (f'{mark}d#0', 'dog', 'ana'),
            (f'{mark}e#0', 'not_dog', 'ana'),
            (f'{mark}f#0', 'unsure', 'ana'),
            (f'{mark}g#0', 'dog', None),          # legacy -> the admin
            # answered, then answered again: the LATEST answer, counted once
            (f'{mark}h#0', 'dog', 'sam'), (f'{mark}h#0', 'not_dog', 'sam'),
            # answered, then withdrawn: counted for nobody
            (f'{mark}i#0', 'dog', 'sam'), (f'{mark}i#0', None, 'sam'),
        ]
        with open(lay['verdicts'], 'w') as fh:
            for key, v, who in rows:
                rec = {'key': key, 'verdict': v, 'band': 1, 'p_dog': 0.1,
                       'seq': 's' + key[-4:], 'ts': 1_800_000_000}
                if who:
                    rec[fa.AUTHOR_FIELD] = who
                fh.write(json.dumps(rec) + '\n')
        got = audit.class_split('gate', who='sam')
        want_all = {'dog': 4, 'not_dog': 3, 'unsure': 1}
        want_mine = {'dog': 2, 'not_dog': 2, 'unsure': 0}
        if got['all'] != want_all:
            bad.append(f'the whole sheet splits {got["all"]}, want {want_all}')
        if got['mine'] != want_mine:
            bad.append(f'sam\'s share is {got["mine"]}, want {want_mine} — '
                       f'one annotator is credited with another\'s answers')
        if audit.class_split('gate', who='ana')['mine'] != \
                {'dog': 1, 'not_dog': 1, 'unsure': 1}:
            bad.append('the split does not follow the annotator asked for')
        if audit.class_split('gate', who=fa.LEGACY_AUTHOR)['mine']['dog'] != 1:
            bad.append('a row written before accounts existed is not the '
                       'admin\'s in the split')
        # NOBODY IS NOT EVERYBODY. Signed out, or a name nobody holds, gets a
        # share of nothing -- never the whole sheet relabelled as theirs.
        for nobody in (None, '', 'ghost'):
            share = audit.class_split('gate', who=nobody)
            if any(share['mine'].values()):
                bad.append(f'{nobody!r} is credited with {share["mine"]}')
            if share['all'] != want_all:
                bad.append('the whole-sheet count moves with who is asking')
        sp = fa.spec('gate')
        if got['positive'] != sp['positive'] or got['negative'] != \
                sp['negative']:
            bad.append('the split names the wrong ends for this stage')
        for k in (sp['positive'], sp['negative']):
            if not got['words'].get(k):
                bad.append(f'{k} has no word on the readout, so the bar is '
                           f'two colours and no names')
        # the words are the BUTTONS' words: a reader should not have to work
        # out that "dog" and "it's a dog" are the same answer
        if got['words'][sp['positive']] != sp['yes'] or \
                got['words'][sp['negative']] != sp['no']:
            bad.append('the split names the classes differently from the '
                       'buttons that record them')
    finally:
        fa.paths = real_paths
    for name in fa.STAGES:
        hit = _leaked(fa.paths(name), mark)
        if hit:
            bad.append(f'this check wrote its own keys into the LIVE {name} '
                       f'audit ({hit})')

    # THE NAME COMES OFF THE SESSION THE GATE RESOLVED. A split is a
    # measurement of a person; a route that took it from the query string
    # would hand any signed-in reader a scoreboard of their colleagues, and
    # that is a hole nothing on the page would look wrong about. Read off the
    # source because the route needs a built pool to answer at all, and a
    # check that quietly does nothing on a machine without one is worse than
    # no check.
    try:
        src = open(os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py'),
                   encoding='utf-8').read()
    except OSError as e:
        bad.append(f'could not read the route: {e}')
        return
    i = src.find("if path == '/api/audit/stats':")
    blk = src[i:src.find('return True', i)] if i >= 0 else ''
    if not blk:
        bad.append('the stats route is gone, or moved somewhere this check '
                   'cannot see it')
        return
    if 'self.session' not in blk:
        bad.append('the stats route does not take the annotator off the '
                   'session it was handed')
    for tell in ("q.get('who'", 'q.get("who"', "q.get('user'",
                 "q.get('username'"):
        if tell in blk:
            bad.append(f'the stats route reads the annotator out of the '
                       f'query string ({tell}) — every signed-in reader '
                       f'could ask about anybody')


def period_filter_checks(bad):
    """The annotated-date filter narrows the ledger read-back on the server.

    The rows carry ts, so "what did I answer this week" is a filter over the
    record -- and it has to happen in judged() before pagination, because a
    hide on the client leaves 'page 2 of 5' describing rows nobody can see.
    The wire value rides the verdict filter ('all~7d'), so the route's
    membership check must know the compounds or every period request quietly
    becomes 'all time'.
    """
    try:
        import audit
    except Exception:
        return
    import fn_audit as fa
    import tempfile as _tf
    import time as _t
    for name in fa.STAGES:
        for w in ('all~2026-08-12..2026-08-19', 'flagged~2026-08-12..',
                  'wrong~..2026-08-19'):
            if not audit.judged_which_ok(w, name):
                bad.append(f'{name}: {w!r} is refused by the route, so the '
                           f'window never reaches judged()')
        # A stale bookmark holding a preset is REFUSED, not quietly served as
        # the whole ledger: a list wider than the address asked for is the
        # failure that looks like an answer.
        for w in ('all~7d', 'flagged~today', 'bogus', 'bogus~2026-08-12..'):
            if audit.judged_which_ok(w, name):
                bad.append(f'{name}: {w!r} is accepted — a view or a window '
                           f'nothing can read is served as something else')
    now = _t.time()
    lt = _t.localtime(now)
    midnight = _t.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0,
                          0, 0, -1))
    today = _t.strftime('%Y-%m-%d', lt)
    lo, hi = audit.period_range(f'{today}..{today}')
    if lo != midnight:
        bad.append(f'a window over today opens at {lo}, not the server\'s '
                   f'own midnight {midnight}')
    if hi is None or abs(hi - (midnight + 86400)) > 1:
        bad.append('one day does not cover its whole day — picking today '
                   'twice returns whatever was judged at 00:00')
    if audit.period_range(None) != (None, None) or \
            audit.period_range('bogus') != (None, None) or \
            audit.period_norm('2026-02-31..'):
        bad.append('an unreadable date filters something — a typo must mean '
                   'any time, not a window over the wrong days')
    mark = 'zzperiod_'
    tmp = _tf.mkdtemp()
    real_paths = fa.paths
    lay = dict(real_paths('gate'))
    lay.update(out=tmp, verdicts=os.path.join(tmp, 'v.jsonl'),
               drawn=os.path.join(tmp, 'drawn.jsonl'),
               pages=os.path.join(tmp, 'pages'),
               pool=os.path.join(tmp, 'pool.parquet'))
    fa.paths = lambda stage='gate': lay
    try:
        rows = ((f'{mark}k1#0', 'dog', midnight + 60),       # today
                (f'{mark}k2#0', 'dog', midnight - 3600),     # yesterday
                (f'{mark}k3#0', 'not_dog', now - 10 * 86400),
                (f'{mark}k4#0', 'dog', now - 40 * 86400),
                (f'{mark}k5#0', 'dog', None))                # ts predates ts
        with open(lay['verdicts'], 'w') as fh:
            for key, v, ts in rows:
                rec = {'key': key, 'verdict': v, 'band': 1, 'p_dog': 0.1,
                       'seq': 's' + key[-4:]}
                if ts is not None:
                    rec['ts'] = ts
                fh.write(json.dumps(rec) + '\n')
        # the windows the four presets used to be, spelled as dates -- plus
        # the two they could never express: one day, and an open near end
        def day(back):
            return _t.strftime('%Y-%m-%d', _t.localtime(now - back * 86400))
        d30, d7, d0 = day(30) + '..', day(7) + '..', day(0) + '..'
        for which, want in (('all', 5), ('all~' + d30, 3), ('all~' + d7, 2),
                            ('all~' + d0, 1),
                            ('all~' + day(0) + '..' + day(0), 1),
                            ('all~..' + day(0), 4),
                            ('all~' + day(1) + '..' + day(1), 1),
                            ('dog~' + d30, 2),
                            ('dog', 4), ('not_dog~' + d0, 0)):
            got = audit.judged('gate', which, 0, 25)
            if got['total'] != want:
                bad.append(f'judged({which!r}) holds {got["total"]} rows, '
                           f'want {want} — the window is not filtering, or '
                           f'is filtering the wrong rows')
            if which.startswith('all') and got['counts'].get('all') != want:
                bad.append(f'judged({which!r}) counts {got["counts"]} beside '
                           f'{got["total"]} rows — the button would count '
                           f'rows the view no longer shows')
        # pagination is over the FILTERED rows, so the page count is truthful
        got = audit.judged('gate', 'all~' + d30, 0, 2)
        if (got['total'], got['pages']) != (3, 2):
            bad.append(f'a thirty-day window at 2 per page reads '
                       f'{got["total"]} rows on {got["pages"]} pages — the '
                       f'filter must run before pagination, not after it')
        if got.get('period') != d30:
            bad.append('the response does not say which window produced it')
        # ...and what comes back is what BIT: an unreadable date is echoed as
        # no window, never as itself, or the page names a narrowing the
        # server did not make
        junk = audit.judged('gate', 'all~7d', 0, 25)
        if junk['total'] != 5 or junk.get('period'):
            bad.append(f'a stale preset window ({junk.get("period")!r}) is '
                       f'served as something, or claimed after the fact')
        # a row with no ts cannot prove it is inside any window
        keys30 = {i['key']
                  for i in audit.judged('gate', 'all~' + d30, 0, 25)['items']}
        if f'{mark}k5#0' in keys30:
            bad.append('a row with no ts passed a date window — it cannot '
                       'prove when it was judged')
    finally:
        fa.paths = real_paths
    for name in fa.STAGES:
        hit = _leaked(fa.paths(name), mark)
        if hit:
            bad.append(f'this check wrote its own keys into the LIVE {name} '
                       f'audit ({hit}) — redirecting fa.paths did not take, '
                       f'and the fixtures went to human data')


def tab_checks(bad):
    """The shared tab strip, exactly as the contract spells it.

    Three judging surfaces carry one strip -- same markup, same classes,
    rendered by two different owners (audit.py here, dashboard.py on the
    review page) -- so any drift in it is two pages disagreeing about the
    same navigation. The current page is marked with 'jtab on' and
    aria-current, and the old gate/leash header cross-link folded INTO the
    strip: a second link to the other audit is the duplication it replaced.
    """
    try:
        import audit
    except Exception:
        return
    import fn_audit as fa
    import re as _re
    labels = {'review': 'Review queue', 'gate': 'Dog-bin audit',
              'leash': 'Leash audit'}
    order = ('review', 'gate', 'leash')
    for stage in fa.STAGES:
        html = audit.page_html(stage)
        nav = '<nav class="jtabs" aria-label="judging surfaces">'
        if html.count(nav) != 1:
            bad.append(f'{stage}: the page carries {html.count(nav)} shared '
                       f'tab strips, not one')
            continue
        if not _re.search(r'</header>\s*<nav class="jtabs"', html):
            bad.append(f'{stage}: the tab strip is not directly under the '
                       f'header — the contract puts it in the same place on '
                       f'every judging page')
        at = []
        for k in order:
            a = (f'<a href="/audit/{k}" class="jtab on" '
                 f'aria-current="page">{labels[k]}</a>' if k == stage
                 else f'<a href="/audit/{k}" class="jtab">{labels[k]}</a>')
            if a not in html:
                bad.append(f'{stage}: the strip is missing the exact tab '
                           f'{a!r} — three agents render this markup and '
                           f'they must render it identically')
                at.append(-1)
            else:
                at.append(html.index(a))
        if -1 not in at and at != sorted(at):
            bad.append(f'{stage}: the tabs are out of order — review queue, '
                       f'then the two audits')
        if html.count('aria-current') != 1:
            bad.append(f'{stage}: aria-current appears '
                       f'{html.count("aria-current")} times; exactly one tab '
                       f'IS the current page')
        if '<span class="tabs">' in html:
            bad.append(f'{stage}: the old header tab strip is still there '
                       f'beside the shared one')
        for k in order:
            n = html.count(f'href="/audit/{k}"')
            if n != 1:
                bad.append(f'{stage}: /audit/{k} is linked {n} times — the '
                           f'old cross-link folds into the strip, it does '
                           f'not ride beside it')


def mobile_checks(bad):
    """The page on a phone, measured rather than read off the sheet.

    The band table is five tracks, 470px of them fixed before the bar column
    gets a pixel, and it sits in a panel that is shut by default -- so the
    sheet looks fine and one tap on "the numbers" used to put the whole
    DOCUMENT into horizontal scroll at 390px, carrying the flagged and
    answered columns off the right edge of every other block on the page.
    A table wider than a phone has to scroll inside its own box.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print('SKIP: playwright is not in this interpreter — the audit page '
              'was never measured at phone width')
        return
    try:
        import audit
    except Exception:
        return
    import fn_audit as _fa

    def serve(stage, html):
        def handler(route, request):
            u = request.url
            if u.endswith('/audit/' + stage):
                route.fulfill(status=200,
                              content_type='text/html; charset=utf-8',
                              body=html)
            elif '/api/' in u:
                route.fulfill(status=200, content_type='application/json',
                              body=json.dumps({'page': {'index': 0,
                                                        'items': [],
                                                        'band': 'rejected'},
                                               'index': 0, 'total': 1,
                                               'bands': [], 'judged': 0,
                                               'counts': {'all': 0,
                                                          'ledger': 0}}))
            else:
                route.abort()
        return handler

    try:
        with sync_playwright() as p:
            br = p.chromium.launch()
            for stage in _fa.STAGES:
                html = audit.page_html(stage)
                ctx = br.new_context(viewport={'width': 390, 'height': 844},
                                     is_mobile=True, has_touch=True)
                pg = ctx.new_page()
                pg.route('**/*', serve(stage, html))
                pg.goto(f'http://audit.fixture/audit/{stage}',
                        wait_until='domcontentloaded', timeout=30000)
                pg.wait_for_timeout(400)
                shut = pg.evaluate(
                    '()=>[document.documentElement.scrollWidth,'
                    'document.documentElement.clientWidth]')
                pg.evaluate("()=>{document.getElementById('figures')"
                            ".open=true}")
                pg.wait_for_timeout(300)
                open_ = pg.evaluate(
                    '()=>[document.documentElement.scrollWidth,'
                    'document.documentElement.clientWidth,'
                    "getComputedStyle(document.getElementById('bands'))"
                    '.overflowX]')
                ctx.close()
                if shut[0] > shut[1]:
                    bad.append(f'/audit/{stage} scrolls sideways at 390px '
                               f'with the numbers shut: {shut[0]} > {shut[1]}')
                if open_[0] > open_[1]:
                    bad.append(f'/audit/{stage} puts the whole document into '
                               f'horizontal scroll at 390px as soon as "the '
                               f'numbers" is opened: scrollWidth {open_[0]} '
                               f'against clientWidth {open_[1]} — the band '
                               f'table has to scroll inside its own panel '
                               f'(overflow-x is {open_[2]})')
            br.close()
    except Exception as e:                 # noqa: BLE001
        print(f'SKIP: playwright would not run a browser here '
              f'({type(e).__name__}: {str(e).splitlines()[0][:100]}) — the '
              f'audit page was never measured at phone width')


def prefetch_checks(bad):
    """The page after the last one is the page already being cut.

    Handing a page over queues the next, so "next" always asks for an index
    the count does not have yet -- and clamping it back onto the page already
    on screen meant the client drew instead. That pays the twenty-second cut
    the prefetch exists to avoid AND retires the queued page unseen: its boxes
    and sequences are reserved the moment they are chosen and never come round
    again, which is how nineteen of the gate store's thirty pages ended up
    carrying no verdict at all.
    """
    import threading
    import time as _t
    try:
        import audit
    except Exception:
        return
    st = {'pages': 5, 'drew': 0, 'queued': 0}

    def drew(n=25, band=None, stage='gate'):
        st['drew'] += 1
        st['pages'] += 1
        return {'index': st['pages'] - 1, 'items': [{'key': 'new#0'}]}

    real = (audit.page_count, audit.get_page, audit.draw_page, audit.prefetch,
            audit.with_verdicts)
    audit.page_count = lambda stage='gate': st['pages']
    audit.get_page = lambda i, stage='gate': {'index': i,
                                              'items': [{'key': f'p{i}#0'}]}
    audit.draw_page = drew
    audit.prefetch = (lambda band=None, n=25, stage='gate':
                      st.update(queued=st['queued'] + 1))
    audit.with_verdicts = lambda doc, stage='gate': doc
    held = audit._PREFETCH.get('gate')
    try:
        # a prefetch mid-flight, which is exactly the state a reader is in one
        # click after a page was handed to them
        t = threading.Thread(target=lambda: (_t.sleep(0.4),
                                             st.update(pages=st['pages'] + 1)),
                             daemon=True)
        t.start()
        audit._PREFETCH['gate'] = t
        a = _t.time()
        r = audit.api_page(5)
        waited = _t.time() - a
        if r.get('index') != 5 or r.get('total') != 6:
            bad.append(f"asking for the page after the last one gave page "
                       f"{r.get('index')} of {r.get('total')} — the "
                       f"prefetched page cannot be reached, so every Next "
                       f"draws over it and its crops are never shown")
        if st['drew']:
            bad.append(f"{st['drew']} page(s) were cut beside the one already "
                       f"being prefetched")
        if waited < 0.3:
            bad.append(f'the request came back in {waited:.2f}s without '
                       f'waiting for the page being cut, so it cannot have '
                       f'served it')
        t.join(timeout=5)
        # nothing queued: cut it now, and line up the one after it
        audit._PREFETCH.pop('gate', None)
        st['queued'] = 0
        r = audit.api_page(6)
        if st['drew'] != 1 or r.get('index') != 6:
            bad.append(f"with nothing queued, the next page drew "
                       f"{st['drew']} page(s) and came back as "
                       f"{r.get('index')}")
        if st['queued'] != 1:
            bad.append('a freshly cut page did not queue the one after it, so '
                       'the next Next pays the cut again')
        # and a hand-typed index still cannot spend sample
        st['drew'] = 0
        r = audit.api_page(9999)
        if r.get('index') != st['pages'] - 1 or st['drew']:
            bad.append(f"i=9999 drew {st['drew']} page(s) and came back as "
                       f"{r.get('index')}; anything past the queued page is a "
                       f"typo, not a request to cut crops")
    finally:
        (audit.page_count, audit.get_page, audit.draw_page, audit.prefetch,
         audit.with_verdicts) = real
        if held is None:
            audit._PREFETCH.pop('gate', None)
        else:
            audit._PREFETCH['gate'] = held


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
// The axis runs 0-100% and says where THIS stage's line falls. Asserted
// against TITLE rather than the word "gate": the literal used to be the gate's
// and read "where the gate draws its line" on /audit/leash, which is the
// per-stage-vocabulary bug this file exists to catch -- so a guard spelling the
// gate's own word was holding the defect in place.
chk(/100%/.test(els.bands.innerHTML)
    && els.bands.innerHTML.indexOf('where the ' + TITLE + ' draws its line') >= 0,
  'the band axis does not state its range, or names a model other than ' +
  TITLE + ': ' + els.bands.innerHTML.slice(-160));
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
// The denominator every rate on this page was missing. The sheets record what
// was put in front of you and the ledger records what came back -- 120 of
// 1,575 on the gate's own store -- so a band's share is the share among crops
// somebody chose to click, and until a sheet is answered in full it is a
// bound rather than an estimate.
STATS.sheets = {shown:1575, answered:120, unrecorded:62};
STATS.rejected.shown = 1080; STATS.rejected.answered = 70;
STATS.rejected.judged = 40;
STATS.bands[0].shown = 216; STATS.bands[0].answered = 9;
paintStats(STATS);
chk(/7\.6%/.test(els.ansrate.textContent),
  'the answer rate reads ' + els.ansrate.textContent + ', not 7.6%');
chk(/1,575/.test(els.answ.textContent) && /62/.test(els.answ.textContent),
  'the answer rate does not say what it is a share of, nor how many answers '
  + 'predate the sheets: ' + els.answ.textContent);
chk(/at most/.test(els.rate.innerHTML),
  'a headline read off sheets nobody finished is presented as an estimate: '
  + els.rate.innerHTML);
chk(/ceiling/.test(els.ci.textContent) &&
    /70 of the 1,080/.test(els.ci.textContent),
  'nothing under the headline says why it is a bound: ' + els.ci.textContent);
chk(!/somewhere between/.test(els.ci.textContent),
  'a sheet nobody finished still extrapolates to a range over the whole '
  + 'store: ' + els.ci.textContent);
chk(/<b>9<\/b>\/216/.test(els.bands.innerHTML),
  'the band table has no answered column, and that rate ran 1.7% to 10.2% '
  + 'across these ten rows');
// The two errors are not one number. Below the line a wrong answer is
// something thrown away that nothing downstream will ever see again; above it
// a wrong answer is a click. A single "N the model got wrong" said which of
// the two the run was making only by accident of which side had been judged.
STATS.rejected.wrong = 136; STATS.kept.wrong = 25; STATS.judged = 270;
paintStats(STATS);
chk(/136<\/b> false negatives/.test(els.figline.innerHTML),
  'the summary does not report false negatives on their own: '
  + els.figline.innerHTML);
chk(/25<\/b> false positives/.test(els.figline.innerHTML),
  'the summary does not report false positives on their own: '
  + els.figline.innerHTML);
chk(/below/.test(els.figline.title) && /above/.test(els.figline.title),
  'nothing says which side of the threshold each error is: '
  + JSON.stringify(els.figline.title));
// singular reads as singular -- "1 false negatives" is how a count betrays
// that nobody looked at it with one find on the board
STATS.rejected.wrong = 1; STATS.kept.wrong = 1;
paintStats(STATS);
chk(/1<\/b> false negative,/.test(els.figline.innerHTML) &&
    /1<\/b> false positive /.test(els.figline.innerHTML),
  'one error is reported in the plural: ' + els.figline.innerHTML);
STATS.rejected.wrong = 1; STATS.kept.wrong = 0; STATS.judged = 12;
// answered in full, and the estimate is a measurement again
STATS.rejected.answered = 1080;
paintStats(STATS);
chk(!/at most/.test(els.rate.innerHTML),
  'a side whose sheets are answered in full still reads as a ceiling');
chk(/somewhere between/.test(els.ci.textContent),
  'a sheet answered in full does not get its interval back: ' +
  els.ci.textContent);
delete STATS.sheets; delete STATS.rejected.shown;
delete STATS.rejected.answered; STATS.rejected.judged = 12;
delete STATS.bands[0].shown; delete STATS.bands[0].answered;
paintStats(STATS);
chk(/page 1 of 1/.test(els.pos.textContent), 'position reads ' + els.pos.textContent);
chk(els.grid.innerHTML.length > 100, 'the grid rendered nothing');
// a page of a hundred crops ends a long way from the toolbar
chk(/id="next2"/.test(PAGE_HTML) && /id="prev2"/.test(PAGE_HTML),
  'there is no way to page on from the foot of the sheet');
chk(/id="views"/.test(PAGE_HTML),
  'there is no way to look at what has already been answered');
// one button for the annotations, with a filter beside it that speaks this
// stage's vocabulary
chk((PAGE_HTML.match(/class="viewbtn"|class="viewbtn on"/g) || []).length === 2,
  'the annotations should be one button, not a row of them');
chk(/id="anno"/.test(PAGE_HTML), 'no filter over the annotations');
[POS, NEG, 'unsure', 'all', 'wrong'].forEach(function (v) {
  chk(new RegExp('value="' + v + '"').test(PAGE_HTML),
    'the annotation filter cannot select ' + v);
});
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
// Dragging an edge past its opposite must SWAP them, the way every editor
// does. Forcing a minimum width instead dragged the far edge along with the
// near one, so the box crept across the picture instead of flipping.
EDIT.meta = {view_w:1000, view_h:800, model_box:[0,0,1,1],
             off_x:0, off_y:0, scale:1};
els.lbimg.clientWidth = 1000;
EDIT.on = true;
EDIT.box = [100,100,200,200];
EDIT.drag = {h:'se', x:0, y:0, box:[100,100,200,200]};
listeners.doc.mousemove({clientX:-150, clientY:-150});   // past the far corner
chk(EDIT.box[0] < EDIT.box[2] && EDIT.box[1] < EDIT.box[3],
  'dragging past the opposite corner left an inverted box: ' +
  JSON.stringify(EDIT.box));
chk(EDIT.box[2] <= 100.01,
  'dragging the corner past its opposite pushed the far edge along instead ' +
  'of flipping: ' + JSON.stringify(EDIT.box));
// moving keeps the box's size when it meets the frame edge
EDIT.box = [900,700,1000,800];
EDIT.drag = {h:'move', x:0, y:0, box:[900,700,1000,800]};
listeners.doc.mousemove({clientX:500, clientY:500});
chk(Math.abs((EDIT.box[2]-EDIT.box[0]) - 100) < 0.01 &&
    Math.abs((EDIT.box[3]-EDIT.box[1]) - 100) < 0.01,
  'moving the box into the frame edge squashed it: ' +
  JSON.stringify(EDIT.box));
EDIT.drag = null; EDIT.on = false;

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
// The editor is armed by a load event on the lightbox <img>, and edStop puts
// the CROP back into that same <img> -- which fires load again. With the
// handler still attached, cancel, close and Escape all re-opened the editor
// holding the FRAME's off_x/off_y and scale, and one press of save then wrote
// a box for a different picture under this crop's key.
chk(typeof lbimg.onload === 'function',
  'opening the editor armed no load handler, so nothing below proves anything');
// guarded, because an unarmed editor is a finding and not a crash: calling it
// anyway reported a TypeError from the page instead of the sentence above
if (typeof lbimg.onload === 'function') lbimg.onload();
chk(EDIT.on === true && !!EDIT.meta,
  'the frame arrived and the editor did not open');
edStop();
chk(!EDIT.meta && !EDIT.box,
  'edStop kept the frame\'s geometry: ' + JSON.stringify(EDIT.meta));
chk(!lbimg.onload,
  'edStop left its load handler on the picture it is about to swap');
if (lbimg.onload) lbimg.onload();      // the crop going back in fires load
chk(EDIT.on === false,
  'cancelling re-opened the editor over the crop, in the frame\'s scale');
chk(!EDIT.meta,
  'a cancelled editor still holds another picture\'s offsets: ' +
  JSON.stringify(EDIT.meta));
// put the page back: the editor left the lightbox open, and every keyboard
// check below returns early while it is
edStop(); els.lb.hidden = true; cur = -1;
// The other door into the same room. Cutting the window on a frame takes
// seconds off a cold drive, and a reader who closes the lightbox or moves to
// another crop in the meantime must not have the editor opened over it.
cur = 0; EDIT.meta = null; lbimg.onload = null;
edStart();
chk(!EDIT.meta && EDIT.on === false && !lbimg.onload,
  'a frame that landed after the lightbox was closed armed the editor anyway');
cur = -1; lbimg.onload = null;

// What the tile says about the answer on record, in THIS stage's words. The
// three comparisons were the gate's 'dog' and 'not_dog' written out, so on the
// leash page every answer painted the same grey card with no button lit, and
// the review view -- whose whole job is showing what you recorded -- could not
// say which answer that was.
function acts(i){ return grid.children[i].querySelectorAll('.act') }
function watchActs(i){
  acts(i).forEach(function (b) {
    b.lit = false;
    b.classList = {add:function(){}, remove:function(){},
      toggle:function(c, on){ if (c === 'on') b.lit = !!on }};
  });
}
function lits(i){ return acts(i).map(function (b) { return b.lit }).join(',') }
page.items.forEach(function (it) { delete it.verdict; delete it.corrected });
render(); watchActs(0);
page.items[0].verdict = POS;
paintCard(0);
chk(/ miss\b/.test(grid.children[0].className),
  'a "' + POS + '" find paints as "' + grid.children[0].className + '"');
chk(acts(0)[0].lit && !acts(0)[1].lit,
  'a "' + POS + '" verdict lit ' + lits(0) + ' — the answer on record is not '
  + 'shown on the button that gave it');
page.items[0].verdict = NEG;
paintCard(0);
chk(/ done\b/.test(grid.children[0].className) &&
    / ok\b/.test(grid.children[0].className),
  'a "' + NEG + '" answer paints as "' + grid.children[0].className + '"');
chk(acts(0)[1].lit && !acts(0)[0].lit,
  'a "' + NEG + '" verdict lit ' + lits(0));
// and the third button is the crop mark, so it says whether the box was
// redrawn -- it was still being lit for the 'unsure' verdict it replaced
page.items[0].verdict = 'unsure';
paintCard(0);
chk(!acts(0)[2].lit, 'the crop mark lights for an "unsure" verdict');
page.items[0].corrected = true;
paintCard(0);
chk(acts(0)[2].lit, 'a redrawn box is not marked on its tile');
page.items.forEach(function (it) { delete it.verdict; delete it.corrected });
render();

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
// AND IT IS ON WITHOUT THE CURSOR. A band IS a range of scores, so how sure
// the model was is the reason a tile is on this sheet at all. Revealed on
// hover it could be read one tile at a time, which on a fifty-tile sheet is
// fifty passes of the cursor to learn what one glance should carry.
chk(!/opacity:\s*0(?![.\d])/.test(chip),
  'the score chip starts invisible (opacity:0) — the score is back to one '
  + 'tile at a time, under the cursor');
chk(!/\.pchip[^{}]*\{[^}]*opacity:\s*1/.test(
      PAGE_CSS.replace(/\.pchip\{[^}]*\}/, '')),
  'a rule still switches the score chip on, so something switches it off');
// Two marks are now permanent furniture over a PHOTOGRAPH: the class tag and
// the score. Neither can be a wash — whatever the crop shows comes through
// the text on exactly the bright frames that most need reading, which is the
// defect the verdict buttons above were fixed for, two elements along.
[['pchip', chip], ['ptag', /\.ptag\{[^}]*\}/.exec(PAGE_CSS)[0]]
].forEach(function (pair) {
  var bg = /background:([^;}]+)/.exec(pair[1]);
  chk(bg && !/rgba\([^)]*,\s*0?\.\d+\s*\)/.test(bg[1]),
    'the ' + pair[0] + ' is translucent (' + (bg && bg[1]) + ') — the crop '
    + 'comes through a mark that is always on screen');
});
page.items.forEach(function (it) {
  var want = (+it.p_dog >= THRESH) ? POS : NEG;
  chk(predOf(it) === want,
    'the tag for ' + it.key + ' (scored ' + it.p_dog + ') reads ' +
    predOf(it) + ', not ' + want);
});
// the two classes are told apart by weight, not by hue alone: the word is
// in the tile either way. Built from the stage's OWN vocabulary -- an
// alternation spelling the existing stages' four words passed when the
// OTHER stage's words were on the grid and false-failed any honest third
// stage, which is the same two-hardcoded-words pathology the band-axis
// check above was cured of.
var reWord = function (w) {
  return String(w).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
};
chk(new RegExp('>(' + reWord(POS) + '|' + reWord(NEG) + ')<')
      .test(els.grid.innerHTML),
  'the predicted class (' + POS + '/' + NEG + ') is not written out, '
  + 'only styled');
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
// The LAST page is where the prefetched one lives: it was queued when this
// page was handed over, so the `total` on this side was counted before it
// existed. Deciding here from that count drew a fresh page over the queued
// one every time -- twenty seconds, and twenty-five boxes and sequences
// retired without anyone seeing them. Only the server knows whether a page
// is already being cut.
dirty = false; total = 3; idx = 2; FETCHES.length = 0;
listeners.next.click();
chk(FETCHES.some(function(u){return /audit\/page\?[^"]*[?&]i=3(&|$)/.test(u)}),
  'at the last page, next did not ask for the page after it: ' +
  JSON.stringify(FETCHES));
chk(!FETCHES.some(function(u){return /audit\/draw/.test(u)}),
  'at the last page, next cut a new page beside the one already queued');
// and N is the same door, not a second copy of the rule
dirty = false; total = 3; idx = 2; FETCHES.length = 0;
listeners.doc.keydown({key:'n', target:{tagName:'DIV'},
  preventDefault:function(){}});
chk(FETCHES.some(function(u){return /audit\/page/.test(u)}) &&
    !FETCHES.some(function(u){return /audit\/draw/.test(u)}),
  'the N key keeps its own copy of the paging rule and drew over the page '
  + 'already queued: ' + JSON.stringify(FETCHES));
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
// ...and WHICH view is empty decides what the sentence is. Drawing a page
// adds unjudged crops and cannot produce an annotation, so the empty
// "my annotations" view telling the reader to draw one was instructions
// that do not lead to the thing the view shows.
page = {index:0, items:[]};
view = 'judged';
try { render() } catch(e) { console.log('FAIL render(judged []) threw ' + e.message) }
chk(!/draw a page/i.test(String(els.empty.textContent)),
  'the empty "my annotations" view says "' + els.empty.textContent +
  '" — drawing a page cannot produce an annotation');
chk(/judged|annotat|verdict/i.test(String(els.empty.textContent)),
  'the empty "my annotations" view does not say nothing has been judged: '
  + els.empty.textContent);
view = 'sheet';
render();
chk(/draw a page/i.test(String(els.empty.textContent)),
  'the empty SHEET lost its own invitation: ' + els.empty.textContent);

// ── the confidence draws ──
// A sheet of the model's surest calls and a sheet of its coin-flips look
// identical, so the choice has to exist, say what it means in this stage's
// words, and travel with the page so the position line can name the draw.
chk(/data-draw="least"/.test(PAGE_HTML) && /data-draw="most"/.test(PAGE_HTML),
  'no visible control for the least/most confident draws');
chk(/data-draw="spread"/.test(PAGE_HTML),
  'the even spread is no longer a choice — there is no way back to the '
  + 'measurement\'s own draw');
chk(typeof DRAW_TITLE === 'object' &&
    new RegExp(reWord(TITLE)).test(String(DRAW_TITLE.least)) &&
    /barely decide/.test(String(DRAW_TITLE.least)),
  'the least-confident draw is not explained in this stage\'s vocabulary: '
  + JSON.stringify(DRAW_TITLE && DRAW_TITLE.least));
chk(new RegExp(reWord(BELOW)).test(String(DRAW_TITLE.most)) &&
    new RegExp(reWord(ABOVE)).test(String(DRAW_TITLE.most)),
  'the most-confident draw does not say which ends it draws from: '
  + JSON.stringify(DRAW_TITLE && DRAW_TITLE.most));
// the mode rides the band value on the wire, exactly the way the band does
band = 'rejected'; mode = 'most';
chk(wireBand() === 'rejected~most',
  'wireBand() is ' + wireBand() + ' with mode=most — the server never '
  + 'learns how to draw');
mode = 'spread';
chk(wireBand() === 'rejected',
  'the spread must put the plain band on the wire, not ' + wireBand());
// chosen like the band: the click paints, remembers, and redraws (debounced)
var _setT = setTimeout; setTimeout = function (f) { f(); return 1 };
var BODIES = []; var _oldFetch = fetch;
// the boot-time stub never learned the judged endpoint, so answer it here
// with an empty ledger page rather than the bare {} that crashes render()
fetch = function (u, o) {
  BODIES.push((o && o.body) || '');
  if (/audit\/judged/.test(u)) {
    FETCHES.push(u);
    return {then:function (f) {
      var r = f({ok:true, json:function () {
        return {items:[], total:0, page:0, pages:1, counts:{all:0}} }});
      return {then:function (g) { g && g(r);
          return {catch:function () { return {} }} },
        catch:function () { return {then:function () { return {} }} }};
    }};
  }
  return _oldFetch(u, o);
};
view = 'sheet'; page = null; busy = false; FETCHES.length = 0;
listeners.draws.click({target:{closest:function (sel) {
  return sel === '.sidebtn'
    ? {getAttribute:function () { return 'least' }} : null }}});
chk(mode === 'least', 'clicking a draw button did not set the mode: ' + mode);
chk(localStorage.getItem('sdAuditDraw:' + STAGE) === 'least',
  'the draw choice is not remembered the way the band choice is');
chk(FETCHES.some(function (u) { return /audit\/draw/.test(u) }),
  'picking a draw did not redraw the sheet the way the band filter does');
chk(BODIES.some(function (b) { return /rejected~least/.test(String(b)) }),
  'the draw request does not carry the mode: ' + JSON.stringify(BODIES));
// re-picking the draw already on screen cuts NOTHING -- each cut spends
// sample the pool never offers again
page = {index:0, band:'rejected~least', n:25, dropped:0, items:[]};
FETCHES.length = 0; dirty = true;
applyFilter();
chk(!FETCHES.some(function (u) { return /audit\/draw/.test(u) }),
  'reselecting the draw already on screen cut a fresh page');
// the position line says what was drawn, off the page document itself
show({index:0, band:'rejected~least', n:25, dropped:0, items:[]}, 0, 1);
chk(/least confident/.test(els.pos.textContent) &&
    new RegExp(reWord(BELOW)).test(els.pos.textContent),
  'the position line does not say what was drawn: ' + els.pos.textContent);
chk(/most confident/.test(bandName('kept~most')),
  'a stored most-confident page reads back as: ' + bandName('kept~most'));
chk(/0\.3–0\.4/.test(bandName('3~least')) &&
    /least confident/.test(bandName('3~least')),
  'a single-band mode page is not named: ' + bandName('3~least'));

// ── the class split ──
// Two facts the page could not answer before: what this reader contributed
// to each class, and what the sheet holds in each. Drawn as ONE scale per
// scope rather than four figures, because on a two-class question the number
// that matters is where the split falls.
chk(/id="split"/.test(PAGE_HTML), 'no class split on the page');
chk(/<div class="split" id="split" hidden>/.test(PAGE_HTML),
  'the split ships visible, so a stage nobody has touched grows an empty '
  + 'panel above the crops');
var SP = {positive:'dog', negative:'not_dog',
          words:{dog:'a dog', not_dog:'not a dog', unsure:'unsure'}};
paintSplit(null);
chk(els.split.hidden, 'a page with no split payload shows the panel anyway');
paintSplit(Object.assign({mine:{dog:0,not_dog:0,unsure:0},
                          all:{dog:0,not_dog:0,unsure:0}}, SP));
chk(els.split.hidden,
  'a sheet nobody has answered draws a panel of zeros over the crops');
// YOURS AND EVERYONE'S, two rows, when they differ
paintSplit(Object.assign({mine:{dog:318,not_dog:84,unsure:2},
                          all:{dog:1204,not_dog:1572,unsure:11}}, SP));
chk(!els.split.hidden, 'the split stayed hidden with answers to show');
chk((els.split.innerHTML.match(/class="sprow/g) || []).length === 2,
  'yours and everyone\'s are not two rows: ' + els.split.innerHTML);
['318', '84', '1,204', '1,572'].forEach(function (n) {
  chk(els.split.innerHTML.indexOf('>' + n + '<') >= 0,
    'the split does not print ' + n + ' — the counts are the thing that was '
    + 'asked for, the bar is how they read');
});
chk(/a dog/.test(els.split.innerHTML) &&
    /not a dog/.test(els.split.innerHTML),
  'the two ends of the scale are unnamed');
// ...and ONE row when every answer on the sheet is this reader's, because
// two identical bars is the same bar drawn twice
paintSplit(Object.assign({mine:{dog:242,not_dog:102,unsure:0},
                          all:{dog:242,not_dog:102,unsure:0}}, SP));
chk((els.split.innerHTML.match(/class="sprow/g) || []).length === 1,
  'the same numbers are drawn as two identical rows');
chk(/every answer here is yours/.test(els.split.innerHTML),
  'nothing says why there is only one row');
// the segments are a PROPORTION of that row, so they fill the track
var segs = els.split.innerHTML.match(/width:([\d.]+)%/g) || [];
chk(segs.length === 3, 'a row is not three segments: ' + segs.join(' '));
var tot = segs.reduce(function (a, s) {
  return a + parseFloat(s.replace(/[^\d.]/g, '')) }, 0);
chk(Math.abs(tot - 100) < 0.5,
  'the segments of a row add to ' + tot.toFixed(1) + '%, not 100 — the bar '
  + 'is drawn to a scale it does not have');

// ── the annotated-date window ──
// Two calendars, not four presets. 'any time / today / last 7 / last 30' is
// four windows out of every window there is, and the one somebody wants is
// the day the gate changed. A date input is the platform's own calendar --
// a hand-built one is locale order, week start and every keyboard path, to
// arrive somewhere worse.
['pfrom', 'pto'].forEach(function (id) {
  chk(new RegExp('id="' + id + '"[^>]*|type="date"[^>]*id="' + id + '"')
        .test(PAGE_HTML) && /type="date"/.test(PAGE_HTML),
    'the annotated-date window has no ' + id + ' calendar');
});
chk(!/value="7d"|value="30d"|>last 7 days</.test(PAGE_HTML),
  'the presets are still in the markup beside the calendars');
chk(/aria-label="judged on or after/.test(PAGE_HTML) &&
    /aria-label="judged on or before/.test(PAGE_HTML),
  'the two calendars are unnamed — a screen reader gets two date fields '
  + 'and no way to tell which end is which');
chk(/server(&#8217;|’|')s local day/.test(PAGE_HTML),
  'nothing says whose midnight a date is — the title attribute went missing');
// server-side, not a client hide: choosing dates must re-fetch with the
// compound which, or the page counts describe rows nobody can see
view = 'judged'; anno = 'all'; FETCHES.length = 0;
els.pfrom.value = '2026-08-12';
els.pto.value = '2026-08-19';
listeners.pfrom.change.call(els.pfrom);
chk(FETCHES.some(function (u) {
      return /judged\?[^"]*which=all~2026-08-12\.\.2026-08-19/.test(u) }),
  'the chosen dates did not reach the server: ' + JSON.stringify(FETCHES));
chk(els.pclr.hidden === false,
  'nothing offers to clear a window that is set');
// one open end is a window too
FETCHES.length = 0;
els.pto.value = '';
listeners.pto.change.call(els.pto);
chk(FETCHES.some(function (u) {
      return /judged\?[^"]*which=all~2026-08-12\.\.(&|$)/.test(u) }),
  'an open far end is not sent as one: ' + JSON.stringify(FETCHES));
// ...and the x is the way back to any time, fields and all
FETCHES.length = 0;
listeners.pclr.click.call(els.pclr);
chk(els.pfrom.value === '' && els.pto.value === '' &&
    els.pclr.hidden === true,
  'clearing left a date on screen over a list it no longer narrows');
chk(FETCHES.some(function (u) { return /judged\?[^"]*which=all(&|$)/.test(u) }),
  'clearing the window did not reach the server: ' + JSON.stringify(FETCHES));
setTimeout = _setT; fetch = _oldFetch;
view = 'sheet'; period = ''; mode = 'spread';

// ── an empty SLICE is not an empty ledger ──
// The sentence a reader gets when a filter matches nothing used to be
// "nothing judged at this stage yet -- verdicts land here as you record them
// on the sheet", which is instructions for a ledger that is empty, handed to
// someone with 344 verdicts who asked for today's. The review queue says the
// same thing in the same words next door.
view = 'judged'; anno = 'all'; period = 'today';
page = {index:0, items:[], dropped:0}; render();
chk(/period/.test(els.empty.textContent) &&
    !/land here as you record/.test(els.empty.textContent),
  'an empty PERIOD reads as an empty ledger: ' + els.empty.textContent);
anno = 'unsure';
render();
chk(/verdict/.test(els.empty.textContent) &&
    /period/.test(els.empty.textContent),
  'an empty verdict-and-period slice does not name both filters: ' +
  els.empty.textContent);
anno = 'all'; period = '';
render();
chk(/land here as you record/.test(els.empty.textContent),
  'a genuinely empty ledger lost its own sentence: ' + els.empty.textContent);

// ── the tab's count follows the view ──
// The period narrows what judged() counts, and the period control is hidden
// on the sheet: leaving today's sixteen on the tab there under-reports the
// ledger with nothing on screen to explain it.
view = 'judged';
counts({all:16, ledger:344});
chk(els.nAll.textContent === '16',
  'the annotations tab does not count the slice it is showing: ' +
  els.nAll.textContent);
view = 'sheet'; paintNAll();
chk(els.nAll.textContent === '344',
  'back on the sheet the tab still reads the period-narrowed count: ' +
  els.nAll.textContent);
counts({all:344});                      // an older server, no ledger field
chk(els.nAll.textContent === '344',
  'a payload without the ledger count zeroed the tab: ' +
  els.nAll.textContent);

// ── a targeted draw says what it costs, and where it went ──
// 'least/most confident' draw one END of a band; every rate on this page
// multiplies a band's share by the band's whole population, so those answers
// are held out of the rates -- and a page that holds answers out has to say
// so where the button is and where the number is.
mode = 'most'; paintFilter();
var _dn = document.getElementById('drawnote');
chk(!!_dn && _dn.hidden === false &&
    /held out|not measurement/.test(String(_dn.textContent)),
  'nothing beside the confidence buttons says the answers do not feed the ' +
  'measurement: ' + JSON.stringify(_dn && _dn.textContent));
mode = 'spread'; paintFilter();
chk(!!_dn && _dn.hidden === true,
  'the spread draw is warned about as though it were targeted');
STATS.aimed = 25; STATS.aimed_dogs = 24; STATS.aimed_wrong = 24;
STATS.rejected.aimed = 25; STATS.rejected.aimed_wrong = 24;
paintStats(STATS);
chk(/held out/.test(els.figline.innerHTML) && /25/.test(els.figline.innerHTML),
  'the headline does not say how many of its answers came off a targeted ' +
  'draw: ' + els.figline.innerHTML);
chk(els.judged.textContent === '37',
  'the work done does not count the targeted answers (12 + 25): ' +
  els.judged.textContent);
chk(/^25 /.test(els.found.textContent),
  'a find drawn by a confidence walk is not counted as a find: ' +
  els.found.textContent);
STATS.aimed = 0; STATS.aimed_dogs = 0; STATS.aimed_wrong = 0;
delete STATS.rejected.aimed; delete STATS.rejected.aimed_wrong;
paintStats(STATS);
chk(!/held out/.test(els.figline.innerHTML),
  'the headline claims answers were held out when none were: ' +
  els.figline.innerHTML);
view = 'sheet'; anno = 'all'; period = ''; mode = 'spread';
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
    for fn in (band_checks, wilson_checks, weighting_checks, sheets_checks,
               targeted_draw_checks, legacy_checks, ledger_checks,
               serving_checks, isolation_checks, correction_checks,
               concurrency_checks,
               persistence_checks, selection_checks, passive_load_checks,
               draw_filter_checks, period_filter_checks, split_checks,
               tab_checks,
               prefetch_checks, page_checks, mobile_checks):
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
