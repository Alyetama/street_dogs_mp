#!/usr/bin/env python3
"""An LLM's answer is a third tier, and it must not be able to become either
of the other two. Prove it from the source and from the store.

This project keeps two kinds of statement apart already, and the suite next
door guards that boundary: a human verdict is ground truth, a model's score is
for filtering the review queue and may never become a label. tools/detect/
llm_annotate.py adds a third and lower one -- a general-purpose model nobody
here trained answering "is there a dog in this crop" -- and the whole risk of
having it at all is that its answers come to look like one of the other two.

Every check below fails the build if one of these becomes true:

  L1  a write from the annotator can land outside its own store
  L2  any module but the annotator and its page names that store or its words
  L3  the page writes anything, or offers a route that could promote one
  L4  the annotator's vocabulary collides with a word a human verdict uses
  L5  a record it writes carries a human answer, or lacks its own marks
  L6  a reply that is not an answer is counted as a yes or a no
  L7  a calibration pools two prompt versions, or two models
  L8  a rate leaves the unit interval, or the two error directions are
      computed over the same denominator
  L9  fn_audit.summarise() moves when this store is written to
  L10 two runs can hold the store at once
  L11 a batch that died reads like one that finished

WHAT MAKES L9 WORTH RUNNING, because as written it looks like a tautology --
of course a number computed from one file does not move when a different file
changes. It is worth running because of what the fixture puts in that file.
The store is written with records of BOTH shapes: the ones llm_annotate really
writes, and hostile ones carrying `verdict` and `band` beside them. A future
edit that reads the ledger and translates llm_yes into a verdict moves the
number on the first set; a cruder one that simply concatenates the ledger onto
read_verdicts() moves it on the second. Either way the assertion fires. A
fixture of records that could not move the number whatever read them would be
a check certifying safety it never looked at, which is the failure mode this
suite has already shipped three times.

WHERE THE FIXTURES LIVE, and this is not a detail. Every store this file
writes is a temporary directory, reached by redirecting the module's own
paths() -- both llm_annotate and fn_audit compute their paths through a
function for exactly this reason. Nothing here writes into data/. A probe
session once pointed a guard's paths at the live gate audit and left seventeen
invented verdicts in it, and there is no command to take one back out. So
every record this file invents carries a mark no real record can hold, and the
last check greps every real store for it.

Run: python tools/detect/tests/adv_llm_tier.py
No network: the endpoint is stubbed, and a check that called it would be a
check whose result depends on somebody else's rate limit.
"""

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TOOLS = os.path.join(REPO, 'tools')
DETECT = os.path.join(REPO, 'tools', 'detect')
DASHBOARD = os.path.join(REPO, 'tools', 'dashboard')
for _p in (DETECT, DASHBOARD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llm_annotate as llm                                      # noqa: E402
import fn_audit as fa                                           # noqa: E402

# On every record this file invents, and on nothing else in the repo. The leak
# check at the end is a grep for it rather than a byte comparison, because the
# dashboard is serving while this runs and a reviewer answering a crop
# mid-check is not a fault -- a key only this file could have written is.
MARK = 'zzllmguard_'

# The words that would have to appear in any module that reads this store.
# Split in two because they are two different admissions: naming the STORE
# means reading the ledger, and naming the VOCABULARY means acting on what is
# in it. A promotion script has to do both.
STORE_WORDS = ('llm_annotations', 'llm_guesses')
VOCAB_WORDS = ('llm_says', 'llm_yes', 'llm_unparsed', 'llm_experimental')
# The annotator, its page, and the two tests that hold them to this. Anything
# else naming those words is a module that has learned about this store, which
# is the review the rule actually needs -- add it here deliberately or not at
# all.
STORE_ALLOWED = {'llm_annotate.py', 'llm_page.py', 'adv_llm_tier.py',
                 'adv_triage_isolation.py'}
# The module NAME is a laxer matter than the store: the dashboard routes to
# the page and names both files in the message it shows when they will not
# load, and that is not knowledge of the ledger.
IMPORT_ALLOWED = STORE_ALLOWED | {'dashboard.py'}

# Every word in this repo that means a person looked and said so. The point of
# the annotator's vocabulary is that it collides with none of them, so `grep
# llm_` separates every answer it has ever written from every verdict on disk.
HUMAN_WORDS = ('dog', 'not_dog', 'unsure', 'missed', 'correct',
               'true_positive', 'false_positive', 'leashed', 'not_leashed',
               'review_page', 'model_suggestion')


def _read(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return fh.read()
    except OSError:
        return ''


def write_opens(text):
    """[(first argument, mode)] for every open() in a write mode.

    Read off the parse tree rather than out of a regex, and neither of those
    is fastidiousness -- both alternatives were tried and both were blind.
    A pattern ending at the first comma reads straight past a path built by
    joining, because the comma inside the join ends the first group and the
    mode is not where the pattern then looks: a fixture write pointed at a
    real store went unnoticed. Matching parentheses instead fixed that and
    introduced the other failure, which is that prose is not code -- the
    paragraph you are reading describes such a call, and the scanner found
    it and reported this file for writing to it.

    The suite next door carries the same function for the same reason. They
    cannot import each other: that one runs its checks at import.
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
        # os.fdopen is handed a descriptor, and the question about a
        # descriptor is what opened it -- asked separately, where it is.
        if name != 'open' or len(node.args) < 2:
            continue
        mode = node.args[1]
        if not (isinstance(mode, ast.Constant)
                and isinstance(mode.value, str)
                and mode.value[:1] in ('w', 'a', 'x')):
            continue
        out.append((ast.unparse(node.args[0]), mode.value))
    return out


def _jpg(path, size=(48, 40)):
    """A real jpeg on disk, because _payload() opens it with Pillow and how
    the crop is prepared is part of the prompt version. A fake byte string
    would exercise the error path instead of the one under test."""
    from PIL import Image
    Image.new('RGB', size, (90, 110, 130)).save(path, format='JPEG')
    return path


class _Reply:
    """One canned HTTP response, shaped like the endpoint's.

    A context manager with .status and .read(), which is all urlopen's result
    is used for. Stubbed rather than called: a check whose verdict depends on
    somebody else's rate limit is a check that fails on Tuesdays.
    """

    def __init__(self, content, finish='stop', reasoning='', status=200):
        self._doc = {'choices': [{'message': {
            'content': content, 'reasoning': reasoning},
            'finish_reason': finish}],
            'usage': {'total_tokens': 271}}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._doc).encode('utf-8')


class _Store:
    """A whole llm_annotations/ store in a temp directory, and the module
    pointed at it.

    llm_annotate computes every path it owns through paths() and refuses any
    write outside what that returns, which is what makes this possible at all
    -- and is why the module was written that way. Restored on the way out,
    including the pool cache, since a fixture pool left in _POOLS would be
    read by the next check as if it were real.
    """

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix='llmguard_')
        self._paths = llm.paths
        self._pool = llm.pool
        self._pools = dict(llm._POOLS)
        self._key = llm.load_key
        # ask() reads the key out of the environment. A stub that only says
        # "there is one" leaves it to raise KeyError one line later, so the
        # fixture puts a value there that is obviously not a key and takes it
        # back out afterwards.
        self._env = os.environ.get('OPENCODE_API_KEY')
        os.environ['OPENCODE_API_KEY'] = MARK + 'not-a-key'
        self.lay = {
            'out': self.dir,
            'ledger': os.path.join(self.dir, 'llm_guesses.jsonl'),
            'status': os.path.join(self.dir, 'status.json'),
            'stop': os.path.join(self.dir, 'stop'),
            'lock': os.path.join(self.dir, 'running.lock'),
        }
        llm.paths = lambda: self.lay
        llm.load_key = lambda: True

    def crops(self, source, human, n, prefix='c'):
        """A pool of real jpegs with a human answer each, and llm.pool()
        pointed at it. The keys carry MARK so the leak check can find them."""
        got = {}
        for i in range(n):
            key = f'{MARK}{prefix}{i}'
            got[key] = (human, _jpg(os.path.join(self.dir, key + '.jpg')))
        self.pool = got
        llm.pool = lambda src=None, _g=got: _g
        return got

    def write(self, *recs):
        with open(self.lay['ledger'], 'a', encoding='utf-8') as fh:
            for r in recs:
                fh.write(json.dumps(r) + '\n')

    def lines(self):
        return [json.loads(ln) for ln in _read(self.lay['ledger']).splitlines()
                if ln.strip()]

    def close(self):
        llm.paths = self._paths
        llm.pool = self._pool
        llm.load_key = self._key
        if self._env is None:
            os.environ.pop('OPENCODE_API_KEY', None)
        else:
            os.environ['OPENCODE_API_KEY'] = self._env
        llm._POOLS.clear()
        llm._POOLS.update(self._pools)
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def rec(says=llm.LLM_YES, key='a', pool='hard_positives',
        version=None, model=None, **kw):
    """One ledger record, shaped exactly like the ones run() writes."""
    out = {'llm_says': says, 'pool': pool, 'key': MARK + key,
           'model': model or llm.MODEL,
           'prompt_version': version or llm.PROMPT_VERSION,
           'prep': llm.PREP, 'ts': time.time(), 'crop': f'tmp/{key}.jpg',
           'unverified': True, 'tier': 'llm_experimental'}
    out.update(kw)
    return out


# ── L1 a write cannot land outside the store ────────────────────────────────
def own_checks(bad):
    """_own() is the rule rather than the intention.

    Every write in the module goes through it, so the question "can an answer
    from this thing reach a human ledger" is answered by one function and by
    reading which calls go through it -- not by trusting that each of the four
    writers happens to reference the right constant.
    """
    P = llm.paths()
    for name in ('ledger', 'status', 'stop', 'lock'):
        if name not in P:
            bad.append(f'L1 paths() no longer owns {name!r}')
            continue
        if llm._own(P[name]) != P[name]:
            bad.append(f'L1 _own() would not accept the store\'s own {name}')
    root = os.path.realpath(P['out'])
    if os.path.relpath(root, REPO) != os.path.join('data', 'llm_annotations'):
        bad.append(f'L1 the store moved to {root} -- everything below and the '
                   f'suite next door name data/llm_annotations')
    # The places an answer must never reach, named one by one rather than left
    # to a prefix test that is easy to read as passing.
    forbidden = [os.path.join(REPO, 'data', p) for p in (
        'hard_positives/labels.jsonl', 'hard_negatives/labels.jsonl',
        'hard_negatives/reviewed.jsonl', 'box_corrections/boxes.jsonl',
        'fn_audit/verdicts.jsonl', 'leash_audit/verdicts.jsonl',
        'audit_finds/manifest.jsonl', 'audit_finds_leash/manifest.jsonl',
        'leash_labels/leash.db', 'label_flags/label_flags.db',
        'dashboard/triage.jsonl')]
    # and the escapes: a relative climb out, the store's parent, and the
    # sibling whose name merely starts the same way
    forbidden += [os.path.join(P['out'], '..', 'hard_positives', 'x.jsonl'),
                  os.path.join(REPO, 'data'),
                  P['out'] + '_v2/ledger.jsonl']
    try:
        import dashboard as _d
        troot = _d.training_root()
        if troot:
            forbidden.append(os.path.join(troot, 'dogdet_v3', 'labels.txt'))
    except Exception:
        pass                # no config, no dashboard import: not a failure
    for p in forbidden:
        try:
            llm._own(p)
        except RuntimeError:
            continue
        bad.append(f'L1 _own() allowed a write to {p} -- it is supposed to '
                   f'refuse every path outside the store')
    # and every write in the module goes through it. os.fdopen is excluded by
    # the lookbehind and covered on the next line instead: it is handed a
    # descriptor, and the question about a descriptor is what opened it.
    src = _read(os.path.join(DETECT, 'llm_annotate.py'))
    stray = [t for t, _ in write_opens(src) if '_own(' not in t]
    if stray:
        bad.append('L1 a write in llm_annotate.py does not go through _own(): '
                   + '; '.join(stray))
    if not (re.search(r'\block = _own\(', src)
            and re.search(r'os\.open\(\s*lock\b', src)):
        bad.append('L1 the lock file is opened on a path _own() never saw')


# ── L2 nothing else knows this store exists ─────────────────────────────────
def reader_checks(bad):
    """Who is allowed to have heard of the ledger.

    Inverted, the way t1 next door had to be rewritten to be: every python
    file under tools/ is scanned and only the annotator, its page and these
    two tests may name the store or its words. A promotion script -- the one
    thing this whole arrangement exists to make deliberate -- has to read the
    ledger, so it has to name it, so it lands here.

    This is the check that covers what a MARK on each record cannot. A record
    copied verbatim into a human ledger is caught by the suite next door,
    which recognises the marks it carries; a record TRANSLATED into the review
    page's own words carries none of them and is invisible there. Whatever
    does the translating still has to read this store to do it.
    """
    seen = {}
    for base, _, files in os.walk(TOOLS):
        for f in files:
            if not f.endswith('.py'):
                continue
            txt = _read(os.path.join(base, f))
            rel = os.path.relpath(os.path.join(base, f), REPO)
            hits = [w for w in STORE_WORDS + VOCAB_WORDS if w in txt]
            if hits and f not in STORE_ALLOWED:
                seen[rel] = hits
            if 'llm_annotate' in txt and f not in IMPORT_ALLOWED:
                seen.setdefault(rel, []).append('llm_annotate')
    for rel, hits in sorted(seen.items()):
        bad.append(f'L2 {rel} names the LLM store: {", ".join(sorted(set(hits)))}'
                   f' -- nothing but the annotator and its page may read it')
    # And the allowlist cannot rot into naming files that are gone.
    on_disk = {f for _, _, fs in os.walk(TOOLS) for f in fs}
    missing = sorted(STORE_ALLOWED - on_disk)
    if missing:
        bad.append(f'L2 the allowlist names files that do not exist: '
                   f'{missing} -- it has rotted')
    # The dashboard may route to the page. It may not read the store.
    dash = _read(os.path.join(DASHBOARD, 'dashboard.py'))
    for w in STORE_WORDS + VOCAB_WORDS:
        if w in dash:
            bad.append(f'L2 dashboard.py names {w!r} -- it routes to the page '
                       f'and has no business reading the ledger')


# ── L3 the page writes nothing, and promotes nothing ────────────────────────
def page_checks(bad):
    """The serving side is a read, twice over.

    Its docstring says THIS MODULE WRITES NOTHING, and the two POST routes
    signal the annotator rather than writing anything themselves. A page that
    grew a third POST, or an open() in write mode, or the name of a human
    store, has grown the one control this page promises not to have.
    """
    src = _read(os.path.join(DASHBOARD, 'llm_page.py'))
    if not src:
        bad.append('L3 llm_page.py is not readable')
        return
    writes = write_opens(src)
    if writes:
        bad.append(f'L3 llm_page.py opens something for writing: '
                   f'{[t for t, _ in writes]}')
    for w in ('labels.jsonl', 'hard_positives', 'hard_negatives',
              'audit_finds', 'box_corrections', 'leash.db', 'label_flags',
              'verdicts.jsonl', 'reviewed.jsonl'):
        if w in src:
            bad.append(f'L3 llm_page.py names the human store {w!r}')
    # The routes the dashboard will answer with a POST, read out of its own
    # handler rather than assumed: an added one is exactly how a promote
    # button would arrive.
    dash = _read(os.path.join(DASHBOARD, 'dashboard.py'))
    m = re.search(r"def _llm_post\(self\):.*?\n(?=    def )", dash, re.S)
    if not m:
        bad.append('L3 dashboard.py has no _llm_post -- this check is reading '
                   'nothing')
    else:
        routes = set(re.findall(r"'(/api/llm/[a-z]+)'", m.group(0)))
        if routes != {'/api/llm/run', '/api/llm/stop'}:
            bad.append(f'L3 the LLM page answers POST on {sorted(routes)}, '
                       f'not only run and stop')
    # api_run never passes the phase 2 door, and there is only one door.
    if 'allow_unjudged' in src.split('def api_run', 1)[-1].split(
            'def api_stop', 1)[0].replace('`allow_unjudged` is never passed',
                                          ''):
        bad.append('L3 api_run() mentions allow_unjudged in code -- phase 2 '
                   'is a command a person types, not a control on a page')
    # and the page says so where it cannot be missed
    html = ''
    try:
        sys.path.insert(0, DASHBOARD) if DASHBOARD not in sys.path else None
        import llm_page as lp
        html = lp.page_html()
    except Exception as e:                                  # noqa: BLE001
        bad.append(f'L3 the page would not build: {type(e).__name__}: {e}')
    if html:
        head = html[:html.find('<header')] if '<header' in html else html
        if 'experimental' not in head.lower():
            bad.append('L3 the word experimental is not above the fold')
        for phrase in ('Nothing on this page is an annotation',
                       'no control on this page that promotes'):
            if phrase not in head:
                bad.append(f'L3 the banner no longer says {phrase!r}')
        if '__BOOT__' in html or '__SIZEOPTS__' in html:
            bad.append('L3 the page has unsubstituted placeholders')


# ── L4 the vocabulary collides with nothing a person says ───────────────────
def vocabulary_checks(bad):
    """Four words, all prefixed, none of them a verdict.

    'dog' and 'not_dog' in this repo mean a person looked and said so, and
    'true_positive'/'false_positive' mean the same on the review page. The
    annotator's words have to be separable from those by a grep and, more to
    the point, by every reader that matches on them: fn_audit.verdict_of() is
    the one that turns a string into a human answer, and it must not turn any
    of these into one.
    """
    words = (llm.LLM_YES, llm.LLM_NO, llm.LLM_UNPARSED, llm.LLM_ERROR)
    if len(set(words)) != 4:
        bad.append(f'L4 the four outcomes are not four distinct words: '
                   f'{words}')
    for w in words + (llm.HUMAN_YES, llm.HUMAN_NO):
        for h in HUMAN_WORDS:
            if w.lower() == h.lower():
                bad.append(f'L4 {w!r} is also a human verdict')
    for w in words:
        if not w.startswith('llm_'):
            bad.append(f'L4 {w!r} does not start with llm_, so `grep llm_` no '
                       f'longer separates this store from every verdict')
    # The reader that decides what a string means on the human side.
    for stage in sorted(fa.STAGES):
        for w in words + (llm.HUMAN_YES, llm.HUMAN_NO, 'llm_experimental'):
            if fa.verdict_of(w, stage) is not None:
                bad.append(f'L4 fn_audit reads {w!r} as the {stage} answer '
                           f'{fa.verdict_of(w, stage)!r}')
        # and no answer a person can click is one of ours
        for a in fa.answers(stage):
            if str(a).startswith('llm_'):
                bad.append(f'L4 the {stage} audit offers {a!r} as a human '
                           f'answer')
    # the human side of the confusion matrix is named so it cannot be read as
    # a verdict either -- it is what a person said, not a word they clicked
    for w in (llm.HUMAN_YES, llm.HUMAN_NO):
        if w in HUMAN_WORDS:
            bad.append(f'L4 {w!r} is a word a ledger already uses')
    # ANSWERED is what run() will not pay for twice; an error is not in it,
    # because nothing was asked.
    if llm.LLM_ERROR in llm.ANSWERED:
        bad.append('L4 an error counts as answered, so a rate limit would '
                   'retire the crop it never asked about')
    if set(llm.ANSWERED) != {llm.LLM_YES, llm.LLM_NO, llm.LLM_UNPARSED}:
        bad.append(f'L4 ANSWERED is {llm.ANSWERED}, which is not the three '
                   f'outcomes that mean the model replied')


# ── L5 what a record carries, and what it must not ──────────────────────────
def record_checks(bad):
    """A run, end to end, against a stubbed endpoint and a temporary store.

    The record has to be the whole story of the call and none of the story of
    the reviewer: the human answer is deliberately not in it, because a store
    holding one is a store readable as a labels file, which is the one thing
    this must never be. calibration() re-reads the human answer from the human
    ledger every time instead, so a reviewer changing their mind changes the
    calibration -- and nothing downstream can pick a verdict up from here by
    accident, because there is not one here to pick up.
    """
    real = urllib_urlopen()
    with _Store() as st:
        st.crops('hard_positives', llm.HUMAN_YES, 3)
        replies = [_Reply('Yes, a dog.\nANSWER: YES'),
                   _Reply('Based on the visual characteristics, it is hard '
                          'to say. No other animals are visible.'),
                   _Reply(None, finish='length', reasoning='x' * 900)]
        set_urlopen(lambda req, timeout=None: replies.pop(0))
        try:
            out = llm.run('hard_positives', n=3, sleep=0, tries=1)
        finally:
            set_urlopen(real)
        if out['asked'] != 3:
            bad.append(f'L5 the run asked {out["asked"]} of 3')
        rows = st.lines()
        if len(rows) != 3:
            bad.append(f'L5 {len(rows)} records for three crops')
        says = [r.get('llm_says') for r in rows]
        if says != [llm.LLM_YES, llm.LLM_UNPARSED, llm.LLM_ERROR]:
            bad.append(f'L5 three replies -- an answer, prose, and a null '
                       f'content -- were recorded as {says}')
        for r in rows:
            missing = [k for k in ('pool', 'key', 'crop', 'model',
                                   'prompt_version', 'prep', 'ts', 'unverified',
                                   'tier', 'llm_says') if k not in r]
            if missing:
                bad.append(f'L5 a record is missing {missing}; a verdict made '
                           f'under a prompt that changed later has to stay '
                           f'identifiable')
            if r.get('unverified') is not True or r.get('tier') != \
                    'llm_experimental':
                bad.append(f'L5 a record does not mark itself: '
                           f'{r.get("unverified")!r}/{r.get("tier")!r}')
            if r.get('prompt_version') != llm.PROMPT_VERSION:
                bad.append('L5 a record does not carry the prompt version it '
                           'was made under')
            # the human answer is not in here, in any spelling
            for k, v in r.items():
                if k in ('label', 'verdict', 'human', 'ground_truth'):
                    bad.append(f'L5 a record carries {k!r} -- this store must '
                               f'not be readable as a labels file')
                if isinstance(v, str) and v in ('dog', 'not_dog',
                                                llm.HUMAN_YES, llm.HUMAN_NO,
                                                'true_positive',
                                                'false_positive'):
                    bad.append(f'L5 a record carries the human answer '
                               f'{k}={v!r}')
        # nothing outside the store was created
        touched = sorted(os.listdir(st.dir))
        strays = [f for f in touched
                  if not (f.startswith(MARK) or f in (
                      'llm_guesses.jsonl', 'status.json', 'stop',
                      'running.lock'))]
        if strays:
            bad.append(f'L5 the run left {strays} in the store')
        if os.path.exists(st.lay['lock']):
            bad.append('L5 the run lock outlived the run -- the next one '
                       'would be refused forever')


def urllib_urlopen():
    import urllib.request
    return urllib.request.urlopen


def set_urlopen(fn):
    import urllib.request
    urllib.request.urlopen = fn


# ── L6 an unreadable reply is its own outcome ───────────────────────────────
def unparsed_checks(bad):
    """Never a no, never re-rolled, and never dropped.

    About a fifth of this model's replies are not an answer. Coercing them
    would put that fifth on whichever side happened to be the default;
    dropping them would measure the subset of crops it happened to answer
    cleanly, which is not the subset anybody asked about. So it is a third
    outcome with a rate of its own, and the parser is deliberately narrow --
    a scan that hunts for "no" anywhere in a paragraph finds one in "no other
    animals are visible" and turns a yes into a no.
    """
    for text in ('', None, '   ', 'Based on the visual characteristics of the '
                 'image, it is difficult to determine.',
                 'There is no other animal visible in this photograph.',
                 'I cannot answer that.', 'ANSWER: MAYBE', 'ANSWER:',
                 'The answer is not clear. Possibly a dog, possibly a fox.',
                 'yes or no', 'NO DOGS OR CATS ARE VISIBLE ANYWHERE'):
        got = llm.parse(text)
        if got is not None:
            bad.append(f'L6 parse({text!r:.44}) returned {got!r} -- a reply '
                       f'that is not an answer must be its own outcome')
    for text, want in (
            ('ANSWER: YES', llm.LLM_YES),
            ('reasoning\nANSWER: NO', llm.LLM_NO),
            ('**ANSWER:** YES', llm.LLM_YES),
            ('answer: no', llm.LLM_NO),
            # the model rehearses the format mid-reasoning and commits at the
            # end; the last marker is the one it committed to
            ('I could say ANSWER: NO here.\nANSWER: YES', llm.LLM_YES),
            ('there is a dog\nYES', llm.LLM_YES),
            ('no dog here\nNO.', llm.LLM_NO)):
        got = llm.parse(text)
        if got != want:
            bad.append(f'L6 parse({text!r:.44}) returned {got!r}, not {want!r}')
    # and the calibration keeps it out of both cells
    with _Store() as st:
        st.crops('hard_positives', llm.HUMAN_YES, 4)
        st.write(rec(llm.LLM_YES, 'c0'), rec(llm.LLM_UNPARSED, 'c1'),
                 rec(llm.LLM_UNPARSED, 'c2'), rec(llm.LLM_ERROR, 'c3'))
        c = llm.calibration()
        m = c['matrix']
        cells = sum(m[h][a] for h in (llm.HUMAN_YES, llm.HUMAN_NO)
                    for a in (llm.LLM_YES, llm.LLM_NO))
        if cells != 1 or c['parsed'] != 1:
            bad.append(f'L6 one answer, two unreadable replies and one error '
                       f'put {cells} in the matrix and {c["parsed"]} in '
                       f'parsed')
        if c['unparsed'] != 2 or c['errors'] != 1:
            bad.append(f'L6 the calibration counted {c["unparsed"]} unreadable '
                       f'and {c["errors"]} failed, not 2 and 1')
        if c['unparsed_rate']['n'] != 3:
            bad.append(f'L6 the unreadable rate is over {c["unparsed_rate"]}, '
                       f'and it has to be over the replies that came back')
        if c['no_answer_rate']['k'] != 3:
            bad.append(f'L6 the no-usable-answer rate does not count both '
                       f'kinds: {c["no_answer_rate"]}')
        # it is not on the disagreements page as a disagreement either
        rows = llm.disagreements()
        if any(r['llm_says'] not in (llm.LLM_YES, llm.LLM_NO) for r in rows):
            bad.append('L6 an unreadable reply is listed as a disagreement')


# ── L7 one prompt, one model, never averaged ────────────────────────────────
def version_checks(bad):
    """A prompt is a question, and two questions do not average.

    Every record carries the version and the model it was made under, and the
    calibration counts one pair at a time and says out loud what it did not
    count. Both halves matter: without the version there is no way afterwards
    to tell which answer came from which question, and without the model the
    same words put to a different model are silently pooled into one number.
    """
    with _Store() as st:
        st.crops('hard_positives', llm.HUMAN_YES, 6)
        st.write(
            rec(llm.LLM_YES, 'c0', version='dogseen-1'),
            rec(llm.LLM_NO, 'c1', version='dogseen-1'),
            rec(llm.LLM_ERROR, 'c2', version='dogseen-1'),
            rec(llm.LLM_YES, 'c3'),
            rec(llm.LLM_NO, 'c4'),
            rec(llm.LLM_YES, 'c5', model='some-other-model'))
        c = llm.calibration(version=llm.PROMPT_VERSION, model=llm.MODEL)
        if c['parsed'] != 2:
            bad.append(f'L7 the current prompt has two answers and the '
                       f'calibration counted {c["parsed"]}')
        others = c['others']
        want = {f'dogseen-1 / {llm.MODEL}': {'records': 3, 'answered': 2},
                f'{llm.PROMPT_VERSION} / some-other-model':
                    {'records': 1, 'answered': 1}}
        if others != want:
            bad.append(f'L7 what was not counted reads {others}, not {want} '
                       f'-- a sample that got smaller must never appear '
                       f'without saying where the rest of it went')
        old = llm.calibration(version='dogseen-1', model=llm.MODEL)
        if old['parsed'] != 2 or old['errors'] != 1:
            bad.append(f'L7 asking for the older prompt gave parsed '
                       f'{old["parsed"]} errors {old["errors"]}, not 2 and 1')
        # A crop answered under both versions is not shadowed by the wrong
        # one: c0 was answered under the old prompt and is now re-asked under
        # the current one, which takes the count of noes from one to two.
        st.write(rec(llm.LLM_NO, 'c0'))
        c2 = llm.calibration()
        if (c2['parsed'], c2['matrix'][llm.HUMAN_YES][llm.LLM_NO]) != (3, 2):
            bad.append(f'L7 re-asking a crop under the current prompt gave '
                       f'parsed {c2["parsed"]} and '
                       f'{c2["matrix"][llm.HUMAN_YES][llm.LLM_NO]} noes, not '
                       f'3 and 2 -- the version filter has to run before the '
                       f'last-write-wins dedup, or one prompt shadows the '
                       f'other')
        if llm.calibration(version='dogseen-1')['parsed'] != 2:
            bad.append('L7 re-asking a crop under the new prompt changed what '
                       'the old prompt answered')
        # the footnote counts the file, not the deduplicated view of it
        v = llm.versions_on_file()
        if v[('dogseen-1', llm.MODEL)]['records'] != 3:
            bad.append(f'L7 the footnote lost a record to the dedup: {v}')
        # and resume is per model as well as per version: a crop answered by
        # another model is not retired from this model's pool
        left = {k for k, _ in llm.sample('hard_positives', n=99)}
        if MARK + 'c5' not in left:
            bad.append('L7 a crop answered by another model is treated as '
                       'done, so it can never be topped up for the model the '
                       'calibration is about')
        rows = [r for r in llm.sources() if r['source'] == 'hard_positives']
        if rows and rows[0]['annotated'] != 3:
            bad.append(f'L7 the pool row counts {rows[0]["annotated"]} '
                       f'annotated; three crops carry an answer from THIS '
                       f'model under this prompt, and counting another '
                       f'model\'s makes the page promise crops the '
                       f'calibration will never see')


# ── L8 the rates are two rates, and they are intervals ──────────────────────
def wilson_checks(bad):
    """Two error directions, two denominators, and an interval on each.

    The pilot put them at 0% and 29% on one sample of nineteen. An accuracy
    number over that pair describes neither, and moves whenever the mix of the
    sample moves -- and the mix here is chosen rather than natural, because
    these pools are the crops that already fooled a model. So the two are
    never added, never share a denominator, and never appear without the
    interval that says how little a handful of crops can settle.
    """
    for k, n in ((0, 0), (0, 1), (1, 1), (0, 14), (1, 5), (3, 400),
                 (19, 19), (7, 12)):
        p, lo, hi = fa.wilson(k, n)
        if not (0.0 <= lo <= p <= hi <= 1.0):
            bad.append(f'L8 wilson({k},{n}) = {(p, lo, hi)} left the unit '
                       f'interval or is not ordered')
        if n and abs(p - k / n) > 1e-12:
            bad.append(f'L8 wilson({k},{n}) reports {p} for {k}/{n}')
    if fa.wilson(0, 14)[2] <= 0:
        bad.append('L8 zero of fourteen reads as a zero rate with no upper '
                   'end -- nothing found is not nothing there')
    if fa.wilson(0, 14)[2] <= fa.wilson(0, 140)[2]:
        bad.append('L8 the interval does not narrow as the sample grows')
    with _Store() as st:
        # 12 crops a person called a dog, 7 they called not a dog: the pilot's
        # own shape, so the two directions cannot be confused for each other.
        dogs = {f'{MARK}d{i}': (llm.HUMAN_YES, 'x') for i in range(12)}
        nots = {f'{MARK}n{i}': (llm.HUMAN_NO, 'x') for i in range(7)}
        both = dict(dogs)
        both.update(nots)
        llm.pool = lambda src=None, _g=both: _g
        st.write(*[rec(llm.LLM_YES, f'd{i}') for i in range(12)])
        st.write(*[rec(llm.LLM_YES, f'n{i}') for i in range(2)])
        st.write(*[rec(llm.LLM_NO, f'n{i}') for i in range(2, 7)])
        c = llm.calibration()
        if (c['missed']['k'], c['missed']['n']) != (0, 12):
            bad.append(f'L8 dogs it denied reads {c["missed"]}, not 0 of 12')
        if (c['invented']['k'], c['invented']['n']) != (2, 7):
            bad.append(f'L8 dogs it invented reads {c["invented"]}, not 2 '
                       f'of 7')
        if c['missed']['n'] == c['invented']['n'] == c['parsed']:
            bad.append('L8 both error directions are computed over the whole '
                       'sample, which is one number wearing two labels')
        if c['agreement']['n'] != 19 or c['agreement']['k'] != 17:
            bad.append(f'L8 agreement reads {c["agreement"]}, not 17 of 19')
        if abs(c['missed']['rate'] + c['invented']['rate']
               - (1 - c['agreement']['rate'])) < 1e-9:
            bad.append('L8 the two errors sum to the disagreement -- they are '
                       'shares of different denominators and must not')
        for name in ('missed', 'invented', 'agreement', 'unparsed_rate',
                     'no_answer_rate'):
            r = c[name]
            if not (0.0 <= r['lo95'] <= r['rate'] <= r['hi95'] <= 1.0):
                bad.append(f'L8 {name} reads {r}, which is not an interval '
                           f'around a rate')
            if r['k'] > r['n']:
                bad.append(f'L8 {name} counts {r["k"]} of {r["n"]}')
        if c['missed']['hi95'] <= 0:
            bad.append('L8 zero dogs denied out of twelve is reported as a '
                       'certainty, and twelve crops cannot settle that')


# ── L9 the audit's number does not move ─────────────────────────────────────
def audit_checks(bad):
    """fn_audit.summarise() is a measurement of the gate against HUMAN
    answers, and it is the one number the audit page exists to produce.

    Computed over a fixture audit, then again with the LLM store written to,
    and the two have to be identical. What makes that worth asserting is the
    contents of the store between the two calls: records of the shape
    llm_annotate really writes AND hostile ones carrying `verdict` and `band`
    beside their llm_says, for the same keys and bands the audit fixture uses.
    Anything that reads that ledger -- concatenating it, or translating
    llm_yes into a verdict -- moves the number, and the check fires. A fixture
    that could not move it either way would be certifying safety it never
    looked at.

    Both stores are redirected into temp directories. The last check in this
    file greps the real ones for the mark below.
    """
    tmp = tempfile.mkdtemp(prefix='llmguard_audit_')
    real_paths = fa.paths
    lay = dict(real_paths('gate'))
    lay.update(out=tmp, pages=os.path.join(tmp, 'pages'),
               verdicts=os.path.join(tmp, 'verdicts.jsonl'),
               drawn=os.path.join(tmp, 'drawn.jsonl'),
               crops=os.path.join(tmp, 'crops'), full=os.path.join(tmp, 'full'),
               dataset=os.path.join(tmp, 'ds'),
               pool=os.path.join(tmp, 'pool.parquet'))
    fa.paths = lambda stage='gate': lay
    try:
        os.makedirs(lay['pages'])
        keys = [(0, MARK + 'a0'), (0, MARK + 'a1'), (3, MARK + 'b0'),
                (7, MARK + 'k0'), (7, MARK + 'k1')]
        with open(os.path.join(lay['pages'], '00000.json'), 'w') as fh:
            json.dump({'index': 0, 'band': None, 'n': len(keys),
                       'items': [{'key': k, 'band': b} for b, k in keys]}, fh)
        with open(lay['verdicts'], 'w') as fh:
            for key, band, v in ((MARK + 'a0', 0, 'dog'),
                                 (MARK + 'a1', 0, 'not_dog'),
                                 (MARK + 'b0', 3, 'dog'),
                                 (MARK + 'k0', 7, 'not_dog')):
                fh.write(json.dumps({'key': key, 'band': band,
                                     'verdict': v}) + '\n')
        totals = [(lo, hi, 100) for lo, hi in fa.BANDS]
        before = json.dumps(fa.summarise(totals=totals), sort_keys=True)
        with _Store() as st:
            st.crops('audit_finds', llm.HUMAN_YES, 1)
            # the shape it really writes, for the audit's own keys
            st.write(rec(llm.LLM_NO, 'a0', pool='audit_finds'),
                     rec(llm.LLM_YES, 'a1', pool='audit_finds'),
                     rec(llm.LLM_YES, 'b0', pool='audit_finds'))
            # and a hostile shape: an answer wearing a verdict's clothes, in
            # every band, so a reader that merely concatenates this file onto
            # the audit's own moves every rate on the page
            st.write(*[{'key': f'{MARK}h{i}', 'band': i, 'verdict': 'dog',
                        'llm_says': llm.LLM_YES, 'pool': 'audit_finds',
                        'unverified': True, 'tier': 'llm_experimental',
                        'prompt_version': llm.PROMPT_VERSION,
                        'model': llm.MODEL, 'ts': time.time()}
                       for i in range(len(fa.BANDS))])
            after = json.dumps(fa.summarise(totals=totals), sort_keys=True)
            keys_after = {v.get('key') for v in fa.read_verdicts(stage='gate')}
        if before != after:
            bad.append('L9 the audit summary MOVED when the LLM store was '
                       'written to -- an answer from a general-purpose model '
                       'is in the denominator of the gate\'s miss rate')
        if any(str(k).startswith(MARK + 'h') for k in keys_after):
            bad.append('L9 fn_audit.read_verdicts() returned records from the '
                       'LLM store')
        s = json.loads(before)
        if s['bands'][0]['judged'] != 2 or s['bands'][7]['judged'] != 1:
            bad.append(f'L9 the fixture audit did not read back as written '
                       f'({s["bands"][0]["judged"]}, '
                       f'{s["bands"][7]["judged"]}), so the comparison above '
                       f'compared two of nothing')
    finally:
        fa.paths = real_paths
        shutil.rmtree(tmp, ignore_errors=True)


# ── L10 one run at a time, whoever started it ───────────────────────────────
def lock_checks(bad):
    """Two runs on one store spend the same free tier twice.

    sample() is deterministic, so two runs going at once ask about the same
    crops; and they share one status file, where the first to FINISH writes
    running=False over the other's progress -- which is what re-enables the
    page's Start button while a batch is still going. The guard used to be on
    the page, which can only refuse a run of its own; it is in run() now,
    where both entry points pass.
    """
    with _Store() as st:
        free = llm._claim()
        try:
            try:
                llm._claim()
                bad.append('L10 the store was claimed twice -- two runs can '
                           'ask about the same crops on the same free tier')
            except llm.RunInProgress:
                pass
            st.crops('hard_positives', llm.HUMAN_YES, 2)
            try:
                llm.run('hard_positives', n=2, sleep=0, tries=1)
                bad.append('L10 a second run started while the store was '
                           'held')
            except llm.RunInProgress:
                pass
            if st.lines():
                bad.append('L10 the refused run wrote records anyway')
            # a dry run is a question about the pool and must still answer
            try:
                llm.run('hard_positives', n=2, dry_run=True)
            except llm.RunInProgress:
                bad.append('L10 a dry run is refused while a batch is going, '
                           'so the page cannot say what a batch would cost')
            held, who = llm.running_elsewhere()
            if held or who != os.getpid():
                bad.append(f'L10 running_elsewhere() reports {held}/{who} for '
                           f'a lock this process holds')
        finally:
            free()
        if os.path.exists(st.lay['lock']):
            bad.append('L10 releasing the lock left the file behind')
        # a lock left by a process that is gone is taken over, not obeyed
        # forever -- otherwise a killed run shuts the store until somebody
        # deletes a file they have never heard of
        with open(st.lay['lock'], 'w') as fh:
            json.dump({'pid': 2 ** 22 - 1, 'ts': time.time() - 600}, fh)
        try:
            llm._claim()()
        except llm.RunInProgress:
            bad.append('L10 a lock naming a dead process is never taken over')
        # and the page's own check-then-start is one critical section
        _page_race(bad)


def _page_race(bad):
    """Two POSTs at once must start one run.

    The dashboard is a ThreadingHTTPServer, so two tabs pressing the button
    are two threads. This is the bug the repo already measured and fixed for
    the guesser -- "two simultaneous POSTs both answered guessing started" --
    and the LLM page had the same shape: the busy check and the busy set in
    two separate lock scopes with a status read and a full dry run between
    them. The dry run is slowed here to widen that window on purpose; a guard
    that only tries it at native speed passes against the broken code most
    times it runs.
    """
    import llm_page as lp
    real_run, real_status, real_else = lp.llm.run, lp.llm.status, \
        lp.llm.running_elsewhere
    real_sources = lp.llm.sources
    saved = dict(lp._run)

    def slow(source, n=10, **kw):
        if kw.get('dry_run'):
            time.sleep(0.15)
            return {'source': source, 'planned': n, 'dry_run': True,
                    'crops': []}
        time.sleep(0.05)                       # the "batch", asking nothing
        return {'source': source, 'asked': 0, 'planned': 0, 'halted': False,
                'gave_up': '', 'counts': {}, 'seconds': 0.0}

    lp.llm.run = slow
    lp.llm.status = lambda: {}
    lp.llm.running_elsewhere = lambda: (False, None)
    lp.llm.sources = lambda model=None: [
        {'source': 'hard_positives', 'label': 'x', 'human': True,
         'enabled': True, 'note': '', 'ledger': None, 'crops': 9,
         'annotated': 0, 'unparsed': 0, 'errors': 0, 'pending': 9}]
    out = []
    try:
        with lp._lock:
            lp._run.update(busy=False, thread=None, started_at=0.0)
        ts = [threading.Thread(target=lambda: out.append(
            lp.api_run('hard_positives', 5))) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=10)
        ok = [r for r in out if r.get('ok')]
        if len(ok) != 1:
            bad.append(f'L10 two simultaneous POSTs started {len(ok)} runs: '
                       f'{out}')
        t = lp._run.get('thread')
        if t is not None:
            t.join(timeout=10)
    finally:
        lp.llm.run, lp.llm.status = real_run, real_status
        lp.llm.running_elsewhere, lp.llm.sources = real_else, real_sources
        with lp._lock:
            lp._run.update(saved)


# ── L11 a run that dies says so ─────────────────────────────────────────────
def failure_checks(bad):
    """A batch that fell over must not read as one that finished.

    Not a tier question, and it is here because the answer to "can this
    annotator be trusted yet" is computed from however many records happen to
    be on disk -- so a batch that died 23 crops in changes every rate above it
    and has to say it died. The first version wrote a status carrying the
    reason and nothing else, which walked into two of the page's own guards:
    the progress strip is hidden when a status has neither `n` nor `done`, and
    the in-process reason is merged only when the status names the run this
    process started, which needs `started`. Counters gone, bar gone, reason
    built four lines below a return that had already fired.
    """
    import llm_page as lp
    saved = dict(lp._run)
    real_run = lp.llm.run
    # inside a redirected store, because a failing run still writes a status
    with _Store():
        started = time.time()

        def boom(source, n=10, **kw):
            llm.write_status(running=True, source=source, model=llm.MODEL,
                             n=5, done=3, started=started,
                             counts={llm.LLM_YES: 2, llm.LLM_NO: 1})
            raise OSError(28, 'No space left on device')

        lp.llm.run = boom
        try:
            with lp._lock:
                lp._run.update(busy=True, started_at=started, tokens=0,
                               recent=[], error=None)
            lp._worker('hard_positives', 5)
            got = lp.api_status()
            if not got.get('failed'):
                bad.append('L11 a run that raised did not say it stopped')
            if not got.get('error'):
                bad.append('L11 the reason this process holds never reached '
                           'the page -- the status the failure wrote does not '
                           'look like this run\'s')
            if (got.get('n'), got.get('done')) != (5, 3):
                bad.append(f'L11 the failure reports {got.get("done")} of '
                           f'{got.get("n")}, not the 3 of 5 the run had '
                           f'reached; the page hides a strip with neither')

            # and a run that fell over BEFORE it started must not inherit the
            # last one's progress
            def refuse(source, n=10, **kw):
                raise SystemExit('phase 2 is off')

            lp.llm.run = refuse
            with lp._lock:
                lp._run.update(busy=True, started_at=time.time(), tokens=0,
                               recent=[], error=None)
            lp._worker('review_queue', 10)
            got2 = lp.api_status()
            if (got2.get('n'), got2.get('done')) != (10, 0):
                bad.append(f'L11 a run that never began reports '
                           f'{got2.get("done")} of {got2.get("n")} -- it is '
                           f'wearing the last run\'s counters')
        finally:
            lp.llm.run = real_run
            with lp._lock:
                lp._run.update(saved)
    # the client keeps its side of that bargain
    src = _read(os.path.join(DASHBOARD, 'llm_page.py'))
    guard = re.search(r'if\(!s\|\|\(!running[^\)]*\)\)\{prog\.hidden', src)
    if not guard:
        bad.append('L11 paintStatus no longer opens with the guard this check '
                   'reads, so nothing here is checking it')
    elif 's.failed' not in guard.group(0):
        bad.append('L11 the progress strip hides itself on a status that '
                   'carries only a failure, which is the one status whose '
                   'whole content is the message')


# ── the fixtures went nowhere near the real stores ──────────────────────────
def untouched_checks(bad):
    """Nothing this file invented is in a real store.

    The last check, and the reason every fixture key carries MARK. A guard
    whose paths pointed at the live audit put seventeen invented verdicts into
    it, and there is no command to take one back out. Grepping for the mark
    rather than comparing bytes, because the dashboard is serving while this
    runs and a reviewer answering a crop mid-check is not a fault.
    """
    stores = [os.path.join(REPO, 'data', p) for p in (
        'llm_annotations/llm_guesses.jsonl',
        'hard_positives/labels.jsonl', 'hard_negatives/labels.jsonl',
        'hard_negatives/reviewed.jsonl', 'box_corrections/boxes.jsonl',
        'fn_audit/verdicts.jsonl', 'fn_audit/drawn.jsonl',
        'leash_audit/verdicts.jsonl', 'leash_audit/drawn.jsonl',
        'audit_finds/manifest.jsonl', 'audit_finds_leash/manifest.jsonl',
        'dashboard/triage.jsonl')]
    for p in stores:
        if MARK in _read(p):
            bad.append(f'L-fixture this check wrote into the LIVE store {p} '
                       f'-- a redirect did not take')
    # and the live ledger still holds only its own words
    for i, ln in enumerate(_read(stores[0]).splitlines(), 1):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            bad.append(f'L5 llm_guesses.jsonl:{i} is not a record')
            continue
        if r.get('llm_says') not in (llm.LLM_YES, llm.LLM_NO,
                                     llm.LLM_UNPARSED, llm.LLM_ERROR):
            bad.append(f'L4 llm_guesses.jsonl:{i} says '
                       f'{r.get("llm_says")!r}, which is not one of this '
                       f'store\'s four words')
        if r.get('unverified') is not True:
            bad.append(f'L5 llm_guesses.jsonl:{i} is not marked unverified')
        for k in ('label', 'verdict', 'human'):
            if k in r:
                bad.append(f'L5 llm_guesses.jsonl:{i} carries {k!r}')


def main():
    bad = []
    for fn in (own_checks, reader_checks, page_checks, vocabulary_checks,
               record_checks, unparsed_checks, version_checks, wilson_checks,
               audit_checks, lock_checks, failure_checks, untouched_checks):
        n = len(bad)
        try:
            fn(bad)
        except Exception as e:                 # noqa: BLE001 - report, not die
            import traceback
            bad.append(f'{fn.__name__} threw {type(e).__name__}: {e}\n'
                       + traceback.format_exc(limit=4))
        print(('ok   ' if len(bad) == n else 'FAIL ') + fn.__name__)
    if bad:
        print()
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('\nan LLM answer is its own tier: it cannot be written outside its '
          'store, nothing\nelse reads it, its words are nobody else\'s, and '
          'the audit does not count it')
    return 0


if __name__ == '__main__':
    sys.exit(main())
