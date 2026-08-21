#!/usr/bin/env python3
"""The model's suggestions must never become labels. Prove it from the source.

triage_crops.py writes guesses about what each unreviewed crop contains, so
the review queue can be worked one kind of mistake at a time. The standing
constraint is that those guesses are for FILTERING ONLY: they may not be
written into the annotation ledgers, may not reach the crop dataset, and may
not be used as training labels. Only a human verdict is ground truth.

A comment saying so is worth very little. These checks fail the build if any
of the following becomes true:

  t1  any module under tools/ other than the writer and the dashboard
      mentions the triage file
  t2  the triage writer opens a ledger or a dataset directory for writing
  t3  the review page's flag path lets a suggestion reach the ledger record
  t4  a triage record is missing its unverified marks
  t5  any store a human decision lands in holds something the model wrote --
      the flag ledgers, the looked-at-and-kept ledger, the hand-drawn
      geometry, both audit ledgers and what they export, the leash verdicts
      and the label flags; or a store grows that is in none of those lists;
      or a flag ledger holds a row the review page could not have written
  t6  the dashboard's suggestion fields are not separate from the verdict one
  t7  the bucket mapping silently disagrees with the ImageNet class order

WHICH ASSERTION t5d MAKES ABOUT A BOX, AND WHY IT IS NOT THE OTHER ONE.
A correction whose four numbers equal the detector's box at the ledger's own
precision is nobody's geometry: it is Reset box putting the model's box back.
The tempting rule is "no such record may be written", and it is wrong. The
ledger is append-only and last write wins, so writing the detector's box IS
how a reviewer withdraws a box they drew earlier -- refusing the write would
leave the withdrawn box in force, and only the reviewer knows which they
meant. Four such records are on disk across three boxes, and every one of the
three has a history holding real corrections too. So the rule is the other
one: such a record must stay tellable from a real correction, and the only
thing that can tell them apart is the geometry itself. t5c keeps the fields
that make that comparison possible and t5d makes it, failing when a box is on
record in the corrections store by the detector's geometry ALONE -- no human
framing anywhere in its history. That is the one state a consumer cannot read
as anything but "a person framed this".

Run: python tools/detect/tests/adv_triage_isolation.py

t5d needs duckdb to read the predictions store, so run it under the env the
dashboard runs on; it says so and skips loudly elsewhere. Everything else
runs anywhere. Point ISOLATION_STORES at a repo root holding COPIES of the
stores to scan them somewhere else -- which is the only safe way to prove
these checks bite, since proving it means writing a poisoned verdict and a
probe session has already left fake ones in the live audit ledger.
"""

import json
import os
import re
import shutil
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TOOLS = os.path.join(REPO, 'tools')
TRIAGE = os.path.join(REPO, 'tools', 'detect', 'triage_crops.py')
DASH = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')

# WHO IS ALLOWED to know this file exists. An allowlist of builders was the
# first attempt and it rotted immediately: three of its seven names were not
# real files and three real builders (build_detector_negatives.py,
# dedup_crops.py, reserve_acceptance_set.py) were never scanned, so the check
# that was supposed to catch a leak was reading almost nothing.
#
# Inverted, it cannot rot. EVERY python file under tools/ is scanned, and only
# these two may mention the triage file: the tool that writes it and the
# dashboard that filters with it. A new script that reads it fails this test
# until someone adds it here deliberately -- which is the review the rule
# actually needs.
# Being on this list means t1 SKIPS the file, so an allowlisted module could
# reach for a ledger and t1 would never notice -- verified by appending one to
# crop_search.py and watching the suite still pass. t1c below is what actually
# holds these two, so the exemption buys silence about the triage file and
# nothing else.
ALLOWED = {'triage_crops.py', 'dashboard.py', 'adv_triage_isolation.py',
           'crop_search.py',
           # It drives the review page, and the page polls /api/triage for the
           # guesser strip, so the word is unavoidable there. What t1 is for --
           # a module quietly growing a path from a guess to a ledger -- is
           # covered for this file by NO_LEDGER below: a test may READ the
           # guess API, and may not name a verdict store.
           'adv_review_render.py',
           # It exists to prove one guesser's run is never credited to the
           # other, which cannot be written without naming the status file.
           'adv_triage_backends.py',
           # The guard on the LLM tier. It names this file, because its own
           # allowlist has to say which modules are permitted to know the LLM
           # store exists and this is one of them; and it names the model's
           # predictions store, because part of what it asserts is that an
           # LLM answer can reach neither that nor a human ledger. It cannot
           # go in NO_LEDGER for the same reason -- it names every ledger on
           # purpose, to prove each one is refused. t1d below is what pays
           # for the exemption instead.
           'adv_llm_tier.py',
           # The guard on who may operate the machine. /api/triage starts and
           # stops a guesser, so it is on the admin-only list that guard
           # checks, and the list cannot be checked without naming it. Same
           # deal as adv_review_render above: it may name the API, and
           # NO_LEDGER still forbids it naming a verdict store.
           'adv_train_page.py'}
# The allowlisted modules that have no business writing a verdict. dashboard.py
# is not here: it owns the ledgers. triage_crops.py is covered by t2.
NO_LEDGER = ('crop_search.py', 'adv_review_render.py',
             'adv_triage_backends.py')
LEDGER_WORDS = ('hard_negatives', 'hard_positives', 'labels.jsonl',
                'label_flags', 'leash.db', 'flag_crop',
                # the five that grew after this list was written, and that t5
                # went blind to for the same reason
                'box_corrections', 'boxes.jsonl', 'reviewed.jsonl',
                'verdicts.jsonl', 'audit_finds')

# The stores are read from here rather than from REPO so that a poisoned COPY
# can be scanned instead of the real thing. Proving any of t5 bites means
# writing a record that must never exist, and a probe session has already put
# fake verdicts into the live audit ledger by doing that in place; there is no
# command to take one back out.
STORES = os.environ.get('ISOLATION_STORES') or REPO
DATA = os.path.join(STORES, 'data')

# EVERY file a human verdict or a hand-drawn box reaches, and the model's own
# file it must stay out of. t5 used to name two of these -- the two
# labels.jsonl -- while ten more grew up beside them, so the check that was
# supposed to catch a model's opinion reaching the training data was reading
# two of the twelve places one can land, and the geometry store was not among
# them. t5g walks data/ and fails on any store that is in none of the lists,
# so the next one cannot go quiet the same way: classifying it is the review
# the rule actually needs.
HUMAN_STORES = ('hard_negatives/labels.jsonl',      # flag verdicts
                'hard_positives/labels.jsonl',
                'hard_negatives/reviewed.jsonl',    # looked at and kept
                'box_corrections/boxes.jsonl',      # hand-drawn geometry
                'fn_audit/verdicts.jsonl',          # audit answers
                'leash_audit/verdicts.jsonl',
                # not answers, but the denominator of every rate the audit
                # reports -- a model writing here inflates what it was asked
                'fn_audit/drawn.jsonl',
                'leash_audit/drawn.jsonl',
                'audit_finds/manifest.jsonl',       # the exported dataset
                'audit_finds_leash/manifest.jsonl')
LEASH_DB = 'leash_labels/leash.db'
FLAGS_DB = 'label_flags/label_flags.db'
MODEL_OWN = ('dashboard/triage.jsonl',)
# A THIRD KIND OF STORE, and it is neither of the other two. An LLM was asked
# whether a crop holds a dog and its answers are kept in llm_annotations/ --
# below a human verdict, and beside rather than under a model's own score,
# because nobody here trained it and nobody here has measured it. It is named
# so t5g stops failing on it and goes back to failing on the NEXT store nobody
# has classified, which is the whole point of that check; the list is separate
# from the two above so that "classified" never reads as "human". What holds
# this one apart from the human stores is tools/detect/tests/adv_llm_tier.py.
LLM_OWN = ('llm_annotations/llm_guesses.jsonl',)
# A FOURTH KIND, and the only thing under data/ that holds nothing about a
# crop at all: who may open the dashboard. accounts.db carries password
# hashes, invite tokens and failed-login counters and is written by
# tools/dashboard/accounts.py; session.key is the secret every session cookie
# is signed with. Named here rather than swept into MODEL_WORKING (which is a
# model's leftovers) or waved through by suffix, because t5g's whole job is
# that a new store gets read by somebody before it goes quiet -- and because
# "classified" must never read as "human": nothing in this pipeline may open
# either of them, and what guards that is tools/detect/tests/adv_accounts.py.
ACCOUNT_STORES = ('dashboard/accounts.db', 'dashboard/session.key')
# The two ledgers the review page's flag button writes, which is the pair a
# promotion would aim at.
FLAG_LEDGERS = (('hard_positives/labels.jsonl', 'true_positive'),
                ('hard_negatives/labels.jsonl', 'false_positive'))
# The crop name the review page mints, and every field it writes with it. Both
# come from flag_crop() in dashboard.py.
CROP_NAME = re.compile(r'^(\d+)_(\d+)_(\d{2,3})\.jpg$')
FLAG_FIELDS = ('image_id', 'conf', 'ts', 'crop', 'label', 'copied',
               'flagged_at')
# What a record from the LLM store looks like wherever it turns up. Its own
# words are all prefixed, so a value check catches a record whose fields were
# renamed on the way in.
LLM_KEYS = ('llm_says', 'prompt_version', 'tier', 'reply')
LLM_WORDS = ('llm_yes', 'llm_no', 'llm_unparsed', 'llm_error',
             'llm_experimental', 'human_says_dog', 'human_says_no_dog')
BOXES = 'box_corrections/boxes.jsonl'
# The surfaces a person can answer from. leash_store stamps every row with the
# one it was written through, which is the whole point of the column: a batch
# import of the leash model's own parquet predictions arrives naming itself.
HUMAN_SOURCES = ('review_page',)
TRIAGE_FILE = os.path.join(DATA, 'dashboard', 'triage.jsonl')

fails = []
skipped = []


def check(name, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + name + (('  ' + detail) if detail
                                                 and not ok else ''))
    if not ok:
        fails.append(name)


def read(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return fh.read()
    except OSError:
        return None


def records(rel):
    """[(line number, record)] from one jsonl store. Missing reads as empty.

    A fresh clone has none of these and a store nobody has written to yet is
    an empty file, so absence is not a failure anywhere below -- what would be
    a failure is a store that exists and is never opened, which is what t5g
    is for.
    """
    out = []
    try:
        with open(os.path.join(DATA, rel), encoding='utf-8') as fh:
            for i, ln in enumerate(fh, 1):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    out.append((i, r))
    except OSError:
        pass
    return out


def rows(rel, sql):
    """Query one sqlite store read-only, or None if it is not there."""
    p = os.path.join(DATA, rel)
    if not os.path.exists(p):
        return None
    con = sqlite3.connect('file:%s?mode=ro' % p, uri=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def write_opens(text):
    """[(first argument, mode)] for every open() in a write mode.

    Read off the parse tree rather than out of a regex, and that is not
    fastidiousness -- two simpler versions were tried and both were blind. A
    pattern ending at the first comma reads straight past a path built by
    joining: the comma inside the join ends the first group, the mode is not
    where the pattern then looks, and the check reports nothing wrong.
    Matching parentheses over the raw text fixed that and found the same call
    written out in a docstring, which is prose and not a write at all.
    """
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, 'attr', '')
        # os.fdopen takes a descriptor, and the question about a descriptor is
        # what opened it, which is asked where that happens.
        if name != 'open' or len(node.args) < 2:
            continue
        mode = node.args[1]
        if not (isinstance(mode, ast.Constant)
                and isinstance(mode.value, str)
                and mode.value[:1] in ('w', 'a', 'x')):
            continue
        out.append((ast.unparse(node.args[0]), mode.value))
    return out


def wrote_it(r):
    """True if the model wrote this record, whatever file it turned up in.

    The triage writer's own stamps, so a record that travelled from
    triage.jsonl into a ledger is recognisable there by the marks t4 makes it
    carry -- which is the point of making it carry them.
    """
    return (r.get('source') == 'model_suggestion'
            or r.get('unverified') is True
            or 'bucket' in r or 'sg' in r)


# ── t1 nothing but the writer and the filter knows this file exists ────────
scanned, guilty = 0, []
for base, _, files in os.walk(TOOLS):
    for f in files:
        if not f.endswith('.py') or f in ALLOWED:
            continue
        scanned += 1
        if re.search(r'triage', read(os.path.join(base, f)) or '', re.I):
            guilty.append(os.path.relpath(os.path.join(base, f), REPO))
check(f't1 none of the {scanned} other tools/ modules mention the triage file',
      not guilty, 'mentions triage: ' + ', '.join(guilty))
# An allowlisted module is invisible to t1, so the thing t1 would have caught
# has to be caught somewhere. A search index reads embeddings and writes
# embeddings; the moment one of these names a ledger, it is doing something
# else and this fails.
_ledger_guilty = []
for _name in NO_LEDGER:
    for _base, _, _fs in os.walk(TOOLS):
        if _name in _fs:
            _txt = read(os.path.join(_base, _name))
            for _w in LEDGER_WORDS:
                if _w in _txt:
                    _ledger_guilty.append(f'{_name} mentions {_w}')
check('t1c allowlisted search modules touch no ledger', not _ledger_guilty,
      '; '.join(_ledger_guilty))
check('t1b the allowlist names files that exist', all(
    any(f in fs for _, _, fs in os.walk(TOOLS)) for f in ALLOWED),
    'an allowed name matches no file -- the list has rotted')
# The LLM guard is allowlisted above and names every ledger there is, so the
# usual counter-check cannot hold it. This one can: it asserts against stores
# it has REDIRECTED into a temp directory, so no write in it may land under
# data/. A guard whose paths pointed at the live gate audit once left
# seventeen invented verdicts in it, which is the accident this refuses to let
# happen again.
#
# BY RUNNING IT, not by reading it. This check used to be a regex over the
# source text of each write target -- `lay[|self.dir|\btmp\b` -- and asked
# whether the write was SPELLED like a redirect, which is a different question
# from whether the redirect took. The accident it exists to prevent is
# literally spelled `lay['verdicts']`: `lay = dict(fa.paths('gate'))` starts as
# the LIVE layout and .update() overwrites only some of its eleven keys, so
# every key the update forgets is a live path wearing an approved name. The
# old check called that ok. It also short-circuited to the literal True if
# `def judged_names` ... (see t2b) and saw only open() with a literal mode,
# missing os.replace, shutil.copy*, Path.write_text and sqlite3.connect
# entirely.
#
# So: run adv_llm_tier.py in a subprocess with every write API refused inside
# data/, and fail on any attempt. Refused, not observed -- a check that
# noticed the write after it happened would be the seventeen verdicts again.
_WRITE_GUARD = r'''
import builtins, io, json, os, pathlib, runpy, shutil, sqlite3, sys
GUARDED, REPORT, TARGET = os.path.realpath(sys.argv[1]), sys.argv[2], sys.argv[3]
hits = []
def _under(p):
    try:
        rp = os.path.realpath(os.fspath(p))
    except (TypeError, ValueError):
        return False
    return rp == GUARDED or rp.startswith(GUARDED + os.sep)
def _trip(api, p):
    hits.append('%s -> %s' % (api, os.fspath(p)))
    raise PermissionError('write-guard: %s(%s) inside %s' % (api, p, GUARDED))
def _wrap_path(mod, name, argi=0):
    fn = getattr(mod, name, None)
    if fn is None:
        return
    label = getattr(mod, '__name__', '?') + '.' + name
    def w(*a, **kw):
        if len(a) > argi and _under(a[argi]):
            _trip(label, a[argi])
        return fn(*a, **kw)
    setattr(mod, name, w)
def _wrap_open(mod, name):
    fn = getattr(mod, name)
    label = getattr(mod, '__name__', 'builtins') + '.' + name
    def w(file, mode='r', *a, **kw):
        m = kw.get('mode', mode)
        if isinstance(m, str) and (m[:1] in ('w', 'a', 'x') or '+' in m):
            if _under(file):
                _trip(label, file)
        return fn(file, mode, *a, **kw)
    setattr(mod, name, w)
_wrap_open(builtins, 'open')
_wrap_open(io, 'open')
_osopen = os.open
def _os_open(path, flags, *a, **kw):
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC):
        if _under(path):
            _trip('os.open', path)
    return _osopen(path, flags, *a, **kw)
os.open = _os_open
for _n in ('remove', 'unlink', 'rmdir', 'mkdir', 'makedirs', 'truncate'):
    _wrap_path(os, _n)
for _n in ('replace', 'rename', 'renames', 'link', 'symlink'):
    _wrap_path(os, _n, 1)
    _wrap_path(os, _n, 0)
for _n in ('copy', 'copy2', 'copyfile', 'copytree', 'move'):
    _wrap_path(shutil, _n, 1)
_wrap_path(shutil, 'rmtree', 0)
def _mk(f, n):
    def w(self, *a, **kw):
        if _under(self):
            _trip('Path.' + n, self)
        return f(self, *a, **kw)
    return w
for _n in ('write_text', 'write_bytes', 'touch', 'mkdir', 'unlink', 'rmdir'):
    _f = getattr(pathlib.Path, _n, None)
    if _f is not None:
        setattr(pathlib.Path, _n, _mk(_f, _n))
_pathopen = pathlib.Path.open
def _p_open(self, mode='r', *a, **kw):
    if isinstance(mode, str) and (mode[:1] in ('w', 'a', 'x') or '+' in mode):
        if _under(self):
            _trip('Path.open', self)
    return _pathopen(self, mode, *a, **kw)
pathlib.Path.open = _p_open
_conn = sqlite3.connect
def _sq(database, *a, **kw):
    s = str(database)
    if not (s.startswith('file:') and 'mode=ro' in s):
        if _under(s.split('?')[0].replace('file:', '')):
            _trip('sqlite3.connect', s)
    return _conn(database, *a, **kw)
sqlite3.connect = _sq
rc, err = 0, ''
try:
    sys.argv = [TARGET]
    runpy.run_path(TARGET, run_name='__main__')
except SystemExit as e:
    rc = e.code if isinstance(e.code, int) else 1
except BaseException as e:
    rc, err = 99, '%s: %s' % (type(e).__name__, e)
with open(REPORT, 'w') as fh:
    json.dump({'done': True, 'hits': hits, 'rc': rc, 'err': err}, fh)
'''

_tier_path = None
for _base, _, _fs in os.walk(TOOLS):
    if 'adv_llm_tier.py' in _fs:
        _tier_path = os.path.join(_base, 'adv_llm_tier.py')
_tier_bad, _tier_skip = [], ''
if _tier_path is None:
    _tier_bad.append('adv_llm_tier.py is missing -- nothing was run')
else:
    import subprocess                                           # noqa: E402
    import tempfile                                             # noqa: E402
    _wd = tempfile.mkdtemp(prefix='wguard_')
    _wg, _rep = os.path.join(_wd, 'wg.py'), os.path.join(_wd, 'report.json')
    with open(_wg, 'w', encoding='utf-8') as _fh:
        _fh.write(_WRITE_GUARD)
    subprocess.run([sys.executable, _wg, os.path.join(REPO, 'data'), _rep,
                    _tier_path], capture_output=True, text=True)
    try:
        with open(_rep, encoding='utf-8') as _fh:
            _doc = json.load(_fh)
    except (OSError, ValueError):
        _doc = None
    if not _doc or not _doc.get('done'):
        # The trap this suite keeps meeting: an abort prints nothing and looks
        # like a pass. It is not one.
        _tier_bad.append('the write-guarded run did not complete -- nothing '
                         'was proved')
    elif _doc['hits']:
        _tier_bad += _doc['hits']
    elif _doc.get('rc') == 99:
        _tier_bad.append(f'adv_llm_tier.py died before it finished '
                         f'({_doc["err"]}), so most of it never ran')
    shutil.rmtree(_wd, ignore_errors=True)
check('t1d the LLM guard attempts no write under data/, run and watched',
      not _tier_bad, '; '.join(_tier_bad[:5]))

# ── t2 the writer never opens a ledger or dataset for writing ───────────────
src = read(TRIAGE)
if src is None:
    check('t2 triage_crops.py exists', False, TRIAGE)
else:
    # every open() with a write mode in that file, read with the parenthesis
    # matcher rather than a regex -- see write_opens() for the call shape the
    # regex version read straight past
    writes = write_opens(src)
    # Two files, both its own: the predictions, and the progress the dashboard
    # reads. Everything else is a leak. This list is the deliberate review --
    # adding the status file made this check fail until it was named here,
    # which is the check doing its job.
    OWN = ('args.out', 'OUT_FILE', 'args.status', 'STATUS_FILE',
           'tmp_status')      # named, not a bare 'tmp' anything could hold
    bad = [t for t, _ in writes if not any(o in t for o in OWN)]
    check('t2 the writer only ever writes its own files', not bad,
          'other write targets: ' + '; '.join(bad))
    # t2b THE WHOLE FILE, not one function's body.
    #
    # This used to read `'labels.jsonl' not in src.split('def judged_names')[1]
    # .split('def ')[0].replace("'labels.jsonl'", '')` -- a window one function
    # wide, reported under a name ("the writer names no ledger path") and a
    # failure detail ("referenced outside the read-only helper") that both
    # claim the whole file. A module-level `LEDGER = 'data/hard_negatives/
    # labels.jsonl'` two hundred lines away was invisible to it, and t2 could
    # not cover the gap either: t2 reads write_opens(), which only sees
    # open(path, <literal mode>). Worse, the whole expression was `... if 'def
    # judged_names' in src else True`, so renaming the helper made the check
    # the literal True and it passed having read nothing.
    #
    # Now: every string constant in the file that names a store, from the parse
    # tree so a path built by joining is seen piece by piece, and only two
    # places may hold one -- the read-only helper's body, and the module
    # docstring, which states the rule in prose. A missing helper is a
    # failure, not a pass.
    import ast                                                   # noqa: E402
    _t2b_bad, _helper = [], None
    try:
        _tree = ast.parse(src)
    except SyntaxError as _exc:
        _tree = None
        _t2b_bad.append(f'triage_crops.py does not parse: {_exc}')
    if _tree is not None:
        # the NODE, not get_docstring()'s cleaned text -- those differ by
        # indentation and comparing them would exempt nothing
        _doc = None
        if (_tree.body and isinstance(_tree.body[0], ast.Expr)
                and isinstance(_tree.body[0].value, ast.Constant)
                and isinstance(_tree.body[0].value.value, str)):
            _doc = _tree.body[0].value
        for _n in ast.walk(_tree):
            if isinstance(_n, ast.FunctionDef) and _n.name == 'judged_names':
                _helper = _n
        _ok_lines = (range(_helper.lineno, (_helper.end_lineno or
                                            _helper.lineno) + 1)
                     if _helper else ())
        if _helper is None:
            _t2b_bad.append('judged_names() is gone -- the one place a ledger '
                            'name is allowed no longer exists, so nothing '
                            'here was scanned')
        for _n in ast.walk(_tree):
            if not (isinstance(_n, ast.Constant)
                    and isinstance(_n.value, str)):
                continue
            _w = [w for w in LEDGER_WORDS if w in _n.value]
            if not _w:
                continue
            if _n is _doc:
                continue        # the module docstring says what the rule is
            if _n.lineno in _ok_lines:
                continue
            _t2b_bad.append(f'line {_n.lineno} names {_w[0]}: '
                            f'{_n.value[:48]!r}')
    check('t2b the writer names a store only inside its read-only helper',
          not _t2b_bad, '; '.join(_t2b_bad[:5]))

# ── t3 a suggestion cannot ride into a ledger record ────────────────────────
dash = read(DASH)
if dash is None:
    check('t3 dashboard.py readable', False)
else:
    fn = ''
    m = re.search(r'\ndef flag_crop\(.*?\n(?=\ndef )', dash, re.S)
    if m:
        fn = m.group(0)
    check('t3 the flag writer never touches a suggestion',
          bool(fn) and not re.search(r"\bsg\b|triage|suggest", fn),
          'flag_crop mentions the suggestion')

    # ── t6 the payload keeps guesses under their own keys ───────────────────
    check("t6 suggestion fields are separate from 'label'",
          "'sg': sg.get('b', '')" in dash and "'label'" not in
          dash.split("'sg': sg.get('b', '')")[1].split('})')[0],
          'a suggestion is written into the verdict field')

# ── t4 every record marks itself unverified ─────────────────────────────────
recs = []
if os.path.exists(TRIAGE_FILE):
    with open(TRIAGE_FILE) as fh:
        for ln in fh:
            try:
                recs.append(json.loads(ln))
            except ValueError:
                pass
if recs:
    bad = [r for r in recs if r.get('unverified') is not True
           or r.get('source') != 'model_suggestion']
    check(f't4 all {len(recs):,} records marked unverified', not bad,
          f'{len(bad)} records missing the marks')
else:
    print('ok   t4 no predictions written yet (nothing to verify)')

# ── t5 nothing the model wrote is in a store a human decision lands in ─────
led_bad, n_recs = [], 0
for _rel in HUMAN_STORES:
    for _i, _r in records(_rel):
        n_recs += 1
        if wrote_it(_r):
            led_bad.append(f'{_rel}:{_i}')
check(f't5 none of the {n_recs:,} records in the {len(HUMAN_STORES)} human '
      f'stores carries a model suggestion', not led_bad,
      'suspect records: ' + ', '.join(led_bad[:5]))

# ── t5-llm the LLM's words are in none of them either ──────────────────────
# wrote_it() above knows the triage writer's marks. This knows the LLM
# annotator's, which are different marks on a different tier: an answer from
# a general-purpose model is below a human verdict AND below a score from a
# model trained here, and llm_annotations/ is where it lives. A record copied
# out of that store carries `unverified` and is caught by t5 already; this
# catches one whose fields were renamed on the way, because the store's
# vocabulary is prefixed for exactly that reason -- `grep llm_` separates
# every answer it has ever written from every verdict in the repo.
llm_bad = []
for _rel in HUMAN_STORES:
    for _i, _r in records(_rel):
        _hit = [k for k in LLM_KEYS if k in _r]
        _hit += [f'{k}={v!r}' for k, v in _r.items()
                 if isinstance(v, str) and v in LLM_WORDS]
        if _hit:
            llm_bad.append(f'{_rel}:{_i} ({", ".join(sorted(set(_hit))[:3])})')
check(f't5-llm none of the {n_recs:,} human records carries the LLM tier\'s '
      f'vocabulary', not llm_bad, 'suspect records: ' + ', '.join(llm_bad[:5]))

# ── t5h every flag row is one the review page could have written ───────────
# THE CHECK THAT DOES NOT DEPEND ON A MARK. Everything above recognises a
# record by something it carries, so a promotion that READS the LLM ledger and
# writes a normally-shaped flag row -- llm_says 'llm_yes' becoming label
# 'true_positive' -- passes all of them: verified on a copied store, where the
# translated row was invisible to t5 while the verbatim one failed it at
# hard_positives/labels.jsonl:126. This is the other side of that: a row must
# look like one flag_crop() wrote. The crop name is the one the review page
# mints, the image_id and the confidence are read back OUT of that name, and
# all seven fields are present. It is a floor rather than a proof -- a
# determined promoter can synthesise all of it -- and it is the shape check
# that catches the row somebody writes in a hurry.
shape_bad, n_flags = [], 0
for _rel, _label in FLAG_LEDGERS:
    for _i, _r in records(_rel):
        n_flags += 1
        _m = CROP_NAME.match(str(_r.get('crop') or ''))
        _why = ''
        if not all(k in _r for k in FLAG_FIELDS):
            _why = 'missing ' + ', '.join(k for k in FLAG_FIELDS
                                          if k not in _r)
        elif not _m:
            _why = f'crop {_r.get("crop")!r} is not a review-page crop name'
        elif str(_r.get('image_id')) != _m.group(2):
            _why = (f'image_id {_r.get("image_id")!r} is not the one in the '
                    f'crop name')
        elif abs(round(int(_m.group(3)) / 100.0, 2)
                 - float(_r.get('conf') or -1)) > 1e-9:
            _why = f'conf {_r.get("conf")!r} is not the crop name\'s score'
        elif _r.get('label') != _label:
            _why = f'label {_r.get("label")!r} in the {_label} store'
        elif not isinstance(_r.get('copied'), bool):
            _why = f'copied {_r.get("copied")!r} is not the flag writer\'s'
        if _why:
            shape_bad.append(f'{_rel}:{_i} {_why}')
check(f't5h all {n_flags:,} flag rows have the shape the review page writes',
      not shape_bad, 'rows nothing on the review page could have written: '
      + '; '.join(shape_bad[:5]))

# ── t5b the two sqlite stores hold human answers only ──────────────────────
db_bad = []
_leash = rows(LEASH_DB, 'SELECT crop, source FROM leash')
for _crop, _source in _leash or ():
    if _source not in HUMAN_SOURCES:
        db_bad.append(f'leash.db {_crop}: source {_source!r}')
# A flag whose should_be is the class the file is already filed under records
# no human decision at all -- it is the dataset's own label handed back as a
# correction, the same shape as the detector's box in the corrections ledger,
# and the training page counts it as a person disagreeing with the model.
_flags = rows(FLAGS_DB, 'SELECT file, class_was, should_be FROM flags')
for _file, _was, _be in _flags or ():
    if not _be or _be == _was:
        db_bad.append(f'label_flags.db {_file}: should_be {_be!r}')
check(f't5b the {len(_leash or ())} leash verdicts and {len(_flags or ())} '
      f'label flags on record are human answers', not db_bad,
      '; '.join(db_bad[:5]))

# ── t5c the corrections ledger still says which box, and where ─────────────
# The only thing that separates a hand-drawn box from the detector's own is
# the geometry, and comparing them needs the key to look the prediction up
# under. Drop det_idx, or start writing normalised coordinates, and t5d has
# nothing to compare -- it would pass on every record forever without ever
# reading one.
_boxes = records(BOXES)
_shape = [f'{BOXES}:{i}' for i, r in _boxes
          if not (str(r.get('image_id') or '').isdigit()
                  and isinstance(r.get('det_idx'), int)
                  and all(isinstance(r.get(k), (int, float))
                          for k in ('x1', 'y1', 'x2', 'y2')))]
check(f't5c all {len(_boxes):,} corrections carry the key and the four '
      f'numbers a comparison needs', not _shape, '; '.join(_shape[:5]))

# ── t5d a box on record by the detector's own geometry alone ───────────────
_hist = {}
for _i, _r in _boxes:
    if f'{BOXES}:{_i}' not in _shape:
        _hist.setdefault((str(_r['image_id']), int(_r['det_idx'])),
                         []).append((_r['x1'], _r['y1'], _r['x2'], _r['y2']))


def detector_boxes(keys):
    """{(image_id, det_idx): (x1, y1, x2, y2)} from the predictions store.

    Rounded the way save_box() rounds, because two decimals is the precision
    the corrections ledger speaks. Comparing at full float precision instead
    calls a byte-for-byte revert 'different' and finds two of the four
    reverts on disk rather than all four.
    """
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import duckdb
    import store as st
    det = st._sql_src(st._store_globs(st.get_detect_root(STORES), 'det'))
    ids = sorted({k[0] for k in keys})
    con = duckdb.connect()
    try:
        got = con.execute(
            'SELECT CAST(image_id AS VARCHAR), det_idx, x1, y1, x2, y2 FROM '
            + det + ' WHERE CAST(image_id AS VARCHAR) IN ('
            + ','.join('?' * len(ids)) + ')', ids).fetchall()
    finally:
        con.close()
    return {(a, int(b)): tuple(round(float(v), 2) for v in (c, d, e, f))
            for a, b, c, d, e, f in got}


_model, _why = {}, ''
if _hist:
    try:
        _model = detector_boxes(_hist)
    except Exception as _e:
        _why = str(_e)
if _why:
    # loud, and named as a skip -- a silent abort here is indistinguishable
    # from a pass, which is the failure mode two of this suite's guards were
    # already shipping
    skipped.append('t5d')
    print('SKIP t5d could not read the predictions store, so no correction '
          'was compared against the box it corrects: ' + _why)
else:
    _lone = sorted(k for k, v in _hist.items()
                   if k in _model and all(b == _model[k] for b in v))
    check('t5d no box is in the corrections store on the detector\'s own '
          'geometry alone', not _lone,
          'model geometry wearing a correction\'s clothes: '
          + ', '.join('%s#%d' % k for k in _lone[:5]))
    # Without this the whole check goes quiet the day the predictions store
    # moves: every key stops resolving, nothing is compared, and t5d prints
    # ok because it found no counterexample among the zero it looked at.
    _seen = [k for k in _hist if k in _model]
    check(f't5d-b the comparison read the detector back for {len(_seen):,} of '
          f'the {len(_hist):,} corrected boxes', bool(_seen) or not _hist,
          'not one correction resolved a prediction -- nothing was compared')

# ── t5e the review page banks no verdict a person did not make ────────────
# Paging away IS the positive verdict on everything still on screen, and that
# is the right rule for the queue. In audit mode the page walks a FIXED list
# of crops already judged, so the same code retires crops nobody answered for
# -- including one whose annotation was just removed, which the button
# promises puts it back in the queue. Both banking paths have to refuse.
if dash is not None:
    # empty when the name is gone, which fails rather than going blind on a
    # rename -- an extraction that quietly matches nothing is the shape of
    # guard this repo has been bitten by
    _ms = (dash.split('function markSeen(){', 1)[-1].split('\n}', 1)[0]
           if 'function markSeen(){' in dash else '')
    _beacon = "addEventListener('pagehide'"
    _pg = (dash.split(_beacon, 1)[-1].split('\n});', 1)[0]
           if _beacon in dash else '')
    _unguarded = [n for n, b in (('markSeen()', _ms),
                                 ('the pagehide beacon', _pg))
                  if "mode==='audit'" not in b]
    check('t5e neither banking path records a verdict in audit mode',
          not _unguarded, ' and '.join(_unguarded) + ' banks unjudged crops')

# ── t5f the audit ledgers hold answers a person could click ────────────────
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
import fn_audit as fa                                          # noqa: E402
vd_bad, ex_bad, n_vd = [], [], 0
for _stage in sorted(fa.STAGES):
    _sp = fa.STAGES[_stage]
    _rel = _sp['audit_dir'] + '/verdicts.jsonl'
    for _i, _r in records(_rel):
        n_vd += 1
        # a null verdict is read as a withdrawal, not as a third answer
        if _r.get('verdict') is not None and fa.verdict_of(
                _r.get('verdict'), _stage) is None:
            vd_bad.append(f'{_rel}:{_i} {_r.get("verdict")!r}')
    # The exported dataset carries the model's score and band beside every
    # row, as provenance. The label must still be the person's answer and
    # nothing else -- a label taken from p_dog would train the next model on
    # the current one's opinion, which is the whole thing this file is about.
    for _i, _r in records(_sp['dataset'] + '/manifest.jsonl'):
        if _r.get('label') != fa.verdict_of(_r.get('verdict'), _stage):
            ex_bad.append(f'{_sp["dataset"]}:{_i} labelled '
                          f'{_r.get("label")!r} on verdict '
                          f'{_r.get("verdict")!r}')
check(f't5f all {n_vd:,} audit verdicts are answers the page offers',
      not vd_bad, '; '.join(vd_bad[:5]))
check('t5f-b every exported label is its own verdict', not ex_bad,
      '; '.join(ex_bad[:5]))

# ── t5g every store on disk is one this file has classified ────────────────
# Two levels only: below that are the crops and the full frames, tens of
# thousands of jpgs and no ledger among them.
#
# NOT ON EXTENSION ANY MORE. It used to enumerate `*.jsonl` and `*.db` and
# then print "all N stores under data/ are classified" -- a claim about data/
# made from two suffixes. A verdict store written as .parquet, .sqlite3 or
# .csv was never enumerated and so never had to be classified, which is the
# state that let ten stores grow up unnoticed beside the two labels.jsonl this
# check exists to remember.
#
# Classified at two grains, because the two costs are different. A DIRECTORY
# of bulk model output (165 gate shards, 30 leash shards, the coverage
# parquets, the shapefiles) is named once; nothing inside it is enumerated,
# and a directory nobody has named fails. Everywhere else -- every directory a
# human decision does land in, and the top level -- every file is enumerated
# whatever its suffix, so the next verdicts.parquet or answers.sqlite3 has to
# be classified before this passes.
BULK_DIRS = ('gate',            # the gate's own shards, one model verdict each
             'leash',           # the same for the leash stage
             'detect',          # the parquet predictions store + its status
             'engines',         # tensorrt engines and their sums
             'geo',             # natural-earth shapefiles
             'grids',           # the cell grids the harvest was planned on
             'harvest',         # harvested crops
             # the 180 train-30-blind frames two detectors are compared on.
             # Symlinks into dogdet_v3 plus a dataset.yaml: derived from a
             # split, holding no decision anybody made.
             'holdout180',
             'manifests',       # coverage csv exports
             'mistakes',        # per-run mistake dumps from run_mistakes.py
             'missing_worklist', 'missing_worklist_after',
             'missing_unrecoverable')
# Files that are not anybody's verdict but sit where the ledgers do. Named one
# by one on purpose: a NEW one still has to be reviewed before this passes.
MODEL_WORKING = (
    'fn_audit/pool.parquet', 'fn_audit/pool.json', 'fn_audit/status.json',
    'leash_audit/pool.parquet', 'leash_audit/pool.json',
    'leash_audit/status.json', 'llm_annotations/status.json',
    'dashboard/board_stats.json', 'dashboard/countries.json',
    'dashboard/dataset_sizes.json', 'dashboard/distinct_counts.json',
    'dashboard/drive_smart.json', 'dashboard/map_points.json',
    'dashboard/map_points_fine.json',
    # the two model layers on the atlas, built by map_layers.py from the gate
    # and leash shards. Every count in them is a classifier's; each payload
    # names the model that produced it in its 'source' field, which is what
    # keeps them out of HUMAN_STORES.
    'dashboard/map_layer_dogs.json', 'dashboard/map_layer_leash.json',
    'dashboard/regions_status.json',
    'dashboard/sequence_cache.json', 'dashboard/triage_status.json',
    'dashboard/world.json', 'dashboard/history.duckdb',
    'dashboard/search_terms.npz', 'dashboard/triage_vecs.npz',
    'best_models.json', 'confusion.json', 'dogbin_acceptance_set.json',
    'leash_acceptance_set.json', 'dogbin_v4_clusters.json',
    'leash_v2_clusters.json', 'leash_label_conflicts.json',
    'catalog.parquet', 'catalog.duckdb', 'dead_manifest_rows.parquet',
)
# Suffixes that are never a store wherever they turn up: page assets, logs,
# plain-text config, sqlite journals, and the .bak/.tmp a rotation leaves.
NOT_A_STORE = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.txt', '.log',
               '.md', '.html', '.css', '.js', '.npz.lock', '.db-shm',
               '.db-wal', '.bak', '.tmp', '.lock')
_on_disk, _skipped, _bulk_seen = [], 0, set()
_unknown_dirs = []
for _d in sorted(os.listdir(DATA) if os.path.isdir(DATA) else ()):
    _p = os.path.join(DATA, _d)
    if os.path.isdir(_p):
        if _d in BULK_DIRS:
            _bulk_seen.add(_d)
            continue
        try:
            _names = [(_d + '/' + f, os.path.join(_p, f))
                      for f in sorted(os.listdir(_p))]
        except OSError:
            continue
    else:
        _names = [(_d, _p)]
    for _rel, _abs in _names:
        if not os.path.isfile(_abs):
            continue                      # crops and full frames live below
        if _rel.endswith(NOT_A_STORE) or '.jsonl.' in _rel:
            _skipped += 1                 # a rotated ledger keeps its suffix
            continue
        _on_disk.append(_rel)
_known = (HUMAN_STORES + MODEL_OWN + LLM_OWN + MODEL_WORKING + ACCOUNT_STORES
          + (LEASH_DB, FLAGS_DB))
_unclassified = [s for s in _on_disk if s not in _known]
check(f't5g all {len(_on_disk)} files under data/ that could be a store are '
      f'classified, whatever their suffix ({len(_bulk_seen)} bulk directories '
      f'named whole, {_skipped} assets/logs skipped)',
      not _unclassified, 'nobody has said whether these hold human '
      'decisions: ' + ', '.join(_unclassified[:5]))
# A name in BULK_DIRS that matches nothing is the ALLOWED list's old failure:
# an exemption for a directory that is not there, quietly covering nothing.
check('t5g-b every bulk directory named here exists',
      set(BULK_DIRS) <= _bulk_seen | {d for d in BULK_DIRS
                                      if not os.path.isdir(DATA)},
      'named but not on disk: ' + ', '.join(sorted(set(BULK_DIRS)
                                                   - _bulk_seen)))

# ── t7 the bucket edges still match the ImageNet class order ────────────────
if src:
    sys.path.insert(0, os.path.dirname(TRIAGE))
    # the keys are named constants, not literals, so count the class names
    # the block pins -- one per boundary the buckets depend on
    blk = src.split('EDGE = {')[1].split('}')[0] if 'EDGE = {' in src else ''
    edges = re.findall(r"'([^']+)'", blk)
    check('t7 bucket edges are asserted, not assumed', len(edges) == 4,
          f'found {len(edges)} edge assertions: {edges}')
    check('t7b the writer verifies them before writing',
          'ImageNet class order is not what the buckets' in src,
          'no runtime check of the class order')

# ── t8 both backend_of() implementations agree ─────────────────────────────
# The tool stamps each record with the backend that wrote it; the dashboard
# decides which backend a record belongs to when it reads one back. Those are
# two copies of one rule in two files, and they cannot import each other -- the
# dashboard needs duckdb, the tool needs torch. If they ever disagree, records
# are filed under one guesser and looked for under another, and 62,979 of them
# predate the field entirely and are attributed purely by this function.
#
# Compared by BEHAVIOUR, not by source text: the two are allowed to be written
# differently as long as they answer the same. Each is lifted out by AST and
# run on its own, so this stays import-free like the rest of the file.
def _lift(path, name):
    import ast as _ast
    try:
        tree = _ast.parse(read(path) or '')
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == name:
            ns = {}
            exec(compile(_ast.Module(body=[node], type_ignores=[]),
                         path, 'exec'), ns)
            return ns[name]
    return None


_a, _b = _lift(TRIAGE, 'backend_of'), _lift(DASH, 'backend_of')
if _a and _b:
    # Every model string either backend can stamp on a record. A backend the
    # tool knows and the dashboard does not files its guesses under the wrong
    # guesser silently -- which is what happened when the dog-bin gate was
    # added to one side only: 241 of its verdicts turned up in SigLIP's
    # filter, and SigLIP's own coverage absorbed them.
    probes = ['google/siglip2-so400m-patch14-384',
              'google/siglip2-base-patch16-224', 'rfdetr', 'rfdetr-large',
              'rfdetr-nano', 'imagenet', 'efficientnet_v2_s.imagenet1k_v1',
              'siglip', 'rfdetr-small', 'dogbin', 'dogbin:dogbin_008',
              'dogbin:anything', '', None, 'something-new']
    differ = [f'{p!r}: tool={_a(p)!r} dashboard={_b(p)!r}'
              for p in probes if _a(p) != _b(p)]
    check('t8 the tool and the dashboard agree which backend wrote a record',
          not differ, '; '.join(differ))
    # backend_of() recognises RF-DETR by prefix rather than by the size table,
    # so that it needs no module constants and can be lifted out and compared.
    # A size key that broke the prefix would be filed as SigLIP, silently.
    sizes = re.findall(r"'([^']+)':\s*'RFDETR",
                       (src or '').split('RFDETR_SIZES = {')[-1])
    check('t8b every RF-DETR size key starts with the prefix backend_of reads',
          bool(sizes) and all(s.startswith('rfdetr') for s in sizes),
          f'keys: {sizes}')
else:
    check('t8 both backend_of() implementations were found',
          False, f'tool={bool(_a)} dashboard={bool(_b)}')

print()
if fails:
    raise SystemExit(f'{len(fails)} isolation check(s) FAILED: '
                     + ', '.join(fails))
if skipped:
    # said again at the bottom because the one line in the middle of thirty is
    # exactly where a check that never ran gets read as one that passed
    print('NOT CHECKED: ' + ', '.join(skipped))
print('model suggestions are structurally separated from the training data')
