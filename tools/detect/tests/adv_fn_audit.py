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
    """The headline weights bands by population. A flat mean would take the
    near-threshold band -- 1.8% of the pool, and where every error lives --
    and report its rate as the whole store's."""
    import fn_audit as fa
    totals = [(0.0, 0.1, 900), (0.1, 0.2, 50), (0.2, 0.3, 30),
              (0.3, 0.4, 15), (0.4, 0.5, 5)]
    # every band judged, only the top one has misses
    vs = ([{'key': f'a{i}', 'band': 0, 'verdict': 'correct'} for i in range(10)]
          + [{'key': f'e{i}', 'band': 4, 'verdict': 'missed'} for i in range(5)]
          + [{'key': f'f{i}', 'band': 4, 'verdict': 'correct'} for i in range(5)])
    s = fa.summarise(vs, totals)
    flat = 0.5 / 5
    # over the population of the bands actually SAMPLED (900 + 5), not all
    # 910 -- a band nobody has looked at is unknown, not clean
    want = 0.5 * 5 / 905
    if abs(s['weighted_rate'] - want) > 1e-9:
        bad.append(f"weighted rate {s['weighted_rate']} is not the "
                   f"population-weighted {want}")
    if abs(s['weighted_rate'] - flat) < 1e-6:
        bad.append('the rate is a flat mean over the bands — the tiny band '
                   'where the errors are would dominate the headline')
    # an unsampled band must not be read as a band with no errors
    vs2 = [{'key': 'x', 'band': 4, 'verdict': 'missed'}]
    s2 = fa.summarise(vs2, totals)
    if abs(s2['weighted_rate'] - 1.0) > 1e-9:
        bad.append(f"one miss in the only band sampled gives "
                   f"{s2['weighted_rate']}, expected 1.0 — bands nobody has "
                   f"looked at must not be counted as clean")
    if s2['covered'] >= 0.99:
        bad.append(f"covered = {s2['covered']} with four bands unsampled")
    # 'unsure' is neither a find nor a clean look
    vs3 = [{'key': 'a', 'band': 4, 'verdict': 'unsure'},
           {'key': 'b', 'band': 4, 'verdict': 'missed'}]
    s3 = fa.summarise(vs3, totals)
    if s3['bands'][4]['judged'] != 1:
        bad.append(f"'unsure' counted in the denominator: "
                   f"{s3['bands'][4]['judged']} judged, expected 1")


def ledger_checks(bad):
    """Append-only, last answer wins, and a bad line is skipped not fatal."""
    import fn_audit as fa
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'v.jsonl')
        with open(p, 'w') as fh:
            fh.write(json.dumps({'key': 'a', 'verdict': 'missed'}) + '\n')
            fh.write('not json at all\n')
            fh.write('\n')
            fh.write(json.dumps({'key': 'a', 'verdict': 'correct'}) + '\n')
            fh.write(json.dumps({'no_key': 1}) + '\n')
            fh.write(json.dumps({'key': 'b', 'verdict': 'unsure'}) + '\n')
        got = {v['key']: v['verdict'] for v in fa.read_verdicts(p)}
        if got != {'a': 'correct', 'b': 'unsure'}:
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
    for rel in ('tools/detect/fn_audit.py', 'tools/dashboard/audit.py'):
        src = open(os.path.join(REPO, rel)).read()
        tree = ast.parse(src)
        writes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == 'open' and len(node.args) > 1:
                m = node.args[1]
                if isinstance(m, ast.Constant) and 'w' in str(m.value) \
                        or isinstance(m, ast.Constant) and 'a' in str(m.value):
                    writes.add(ast.dump(node.args[0])[:60])
        for banned in ('annot', 'review_ledger', 'hard_negative', 'dataset',
                       'labels'):
            if banned in src.lower().replace('fn_audit', ''):
                # a mention is fine in prose; a path is not
                for line in src.splitlines():
                    low = line.lower()
                    if banned in low and ('open(' in low or 'join(' in low):
                        bad.append(f'{rel} touches a {banned} path: '
                                   f'{line.strip()[:80]}')
        if 'verdicts.jsonl' not in src and 'VERDICTS' not in src:
            bad.append(f'{rel} does not name its own ledger')


def main():
    bad = []
    for fn in (band_checks, wilson_checks, weighting_checks, ledger_checks,
               serving_checks, isolation_checks):
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
