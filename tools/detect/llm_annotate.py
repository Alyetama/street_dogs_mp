#!/usr/bin/env python3
"""EXPERIMENTAL. Ask an LLM whether a crop holds a dog, and measure how often
it agrees with a person.

    python tools/detect/llm_annotate.py sources
    python tools/detect/llm_annotate.py run --source hard_positives --n 10
    python tools/detect/llm_annotate.py calibration
    python tools/detect/llm_annotate.py disagreements

WHAT AN ANSWER FROM HERE IS. A third tier, and the lowest one. This project
already separates two: a human verdict is ground truth, and a model's score
is for filtering the review queue and may never become a label. An answer
from a general-purpose LLM is below both. Nobody here trained it, nobody here
has measured it, and until the numbers `calibration` prints say otherwise it
is worth exactly as much as the interval beside it.

So it is kept where it cannot be mistaken for either of the other two. Its
own directory under data/, its own ledger, and its own words: every answer
recorded here is llm_yes, llm_no, llm_unparsed or llm_error. Never 'dog' and
never 'not_dog', because those two words in this repo mean a person looked
and said so, and never 'true_positive'/'false_positive', which mean the same
thing on the review page. `grep llm_` separates every record this file has
ever written from every verdict in the repo.

WHAT READS IT. Nothing that assembles a dataset, and nothing that computes a
rate about another model. fn_audit.summarise() measures the dog-bin gate
against HUMAN answers, and a single LLM answer in that denominator would
quietly corrupt the one number that page exists to produce. The traffic here
is one way and stays one way: this file READS data/fn_audit/verdicts.jsonl
and the two labels.jsonl, and writes nothing but its own ledger. _own()
below is what makes that a rule rather than an intention.

Every record also carries `unverified: true`, which is the mark the isolation
suite under tools/detect/tests/ already looks for in every human store. That
is deliberate and it is free: if one of these records is ever COPIED somewhere
it must not be, the suite's t5 fails on it without anyone having to teach the
test what this file is.

That is the limit of what a mark can do, and the limit is worth stating
plainly rather than being read as more than it is. t5 recognises a record by
the marks it carries, so it catches a record that travels. It cannot catch a
record that is TRANSLATED -- a promotion script reading llm_says == 'llm_yes'
and appending a normally-shaped {'crop': ..., 'label': 'true_positive'} row
carries none of these fields and passes t5 cleanly, which was verified rather
than assumed. tools/detect/tests/adv_llm_tier.py is what covers that: it holds
the flag ledgers to the shape the review page actually writes, and it fails
if any module outside this one and its page so much as names this store.

PHASE 1, AND IT IS THE ONLY PHASE SWITCHED ON. Calibration. Run the model on
crops that ALREADY carry a human answer and measure the disagreement. The
repo holds roughly 2,600 of them across three pools, and they are the pools
in SOURCES below.

WHY THE HEADLINE IS NEVER ONE NUMBER. A 24-crop pilot put its two errors a
long way apart: of the crops a person called a dog it called every one a dog,
and of the crops a person called not-a-dog it invented a dog in two of seven.
Strong recall, weak precision. An accuracy number averages those two into
something that describes neither, and worse, it moves when the mix of the
sample moves -- and the mix here is CHOSEN, not natural. So `calibration`
reports the two directions separately, each with a Wilson interval, and the
overall agreement only as a footnote with that warning attached.

AND WHY THE UNPARSED RATE TRAVELS WITH THEM. The model ignores the output
contract some of the time. Five of 24 pilot replies could not be read as an
answer. A calibration that drops those has measured the subset of crops the
model happened to answer cleanly, which is not the subset anyone asked about
-- so an unreadable reply is recorded as its own outcome, is never coerced
into a yes or a no, is never re-rolled until it parses, and its rate is
printed beside every other rate here.

WHY THE HUMAN ANSWER IS NOT IN THE LEDGER. It would make this file's store
readable as a labels file, which is the one thing it must never be. The
ledger holds a pointer -- which pool, which crop -- and calibration re-reads
the human answer from the human ledger every time. A person who changes their
mind changes the calibration, which is correct, and there is no copy of their
verdict here for anything downstream to pick up by accident.

PHASE 2 IS BUILT AND SWITCHED OFF. Annotating crops nobody has judged is one
entry in SOURCES with `human` false and `enabled` false, and run() refuses it
unless it is asked twice. If it is ever switched on, what it produces is a
QUEUE-ORDERING SIGNAL -- a hint about which crop to show a person next -- and
not a label, on exactly the same footing as the dog-bin's own score. Nothing
here promotes anything into a dataset, and nothing here ever will: that is a
decision for a person to make by hand, after reading these numbers.

THE THREE THINGS THAT BITE, all confirmed against the live API.
 1. Cloudflare answers 403 "error code: 1010" to Python's default User-Agent.
    Without the header below, nothing works at all.
 2. It is a reasoning model and the reply carries 'reasoning' beside
    'content'. On a small token budget the reasoning consumes all of it and
    content comes back NULL with finish_reason 'stop' -- a silent null, not
    an error. So the budget is generous and a null is a FAILED call to retry,
    never a "no".
 3. It answers in prose when asked for one word. The contract is therefore a
    last line rather than a whole reply, which survives a preamble.

HOW THIS STORE IS SIGNED OFF. The isolation suite under tools/detect/tests/
walks data/ and fails on any store it has not classified, so this ledger
failed it the day it was written -- which was the test working, and the
deliberate review the rule asks for. It is signed off now: that suite's t5g
lists this store under LLM_OWN as a third kind, neither a human ledger nor a
model's own predictions, and goes back to failing on the NEXT unclassified
store rather than standing permanently red over this one.

(That suite also fails any module here that so much as names the queue-hint
file the review page filters on -- whole-file, case-insensitively, and by the
word rather than by the path -- which is why it is described that way above,
and why neither the file nor the check is named here. Naming the check cost
one red run to find out.)
"""

import argparse
import base64
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── the endpoint ────────────────────────────────────────────────────────────
API_URL = 'https://opencode.ai/zen/v1/chat/completions'
MODEL = 'mimo-v2.5-free'

# Cloudflare in front of the endpoint answers 403 "error code: 1010" to
# urllib's default User-Agent, on every call, before the request reaches the
# model. Any ordinary browser string is accepted. This line is not politeness.
USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')

# Generous on purpose, and far more generous than the typical call needs.
# The reply carries the model's reasoning alongside its content and the
# reasoning is charged against this budget FIRST, so a budget that runs out
# returns content: null with finish_reason 'length' -- no answer, no error,
# nothing to read. Measured, on this data, three times over:
#
#     16 tokens     reasoning ate all of it, null content, finish 'stop'
#     400 tokens    1,708 characters of reasoning, null content, 'length'
#     1500 tokens   6,050 and 6,194 characters of reasoning, null, 'length'
#
# AND THE FAILURES ARE NOT SPREAD EVENLY. Both of the 1500 failures were
# hard_negatives -- crops that already fooled the detector -- because the
# crop the model thinks hardest about is the ambiguous one, which is the
# crop the calibration most needs an answer for. A budget too small does not
# lose a random sample; it loses the hard end, and quietly flatters whatever
# is measured on what is left. Hence the headroom, and hence `why` on every
# error record so the rate can be watched rather than assumed away.
MAX_TOKENS = 4000

# No machine paths in a tracked file: $OPENCODE_ENV_FILE, then the repo's own
# .env, which is gitignored. Same shape as best_models.py and fetch_confusion.
ENV_FILES = tuple(p for p in (os.environ.get('OPENCODE_ENV_FILE'),
                              os.path.join(REPO, '.env')) if p)

# ── the question ────────────────────────────────────────────────────────────
# THE PROMPT VERSION RULE, and it is a rule rather than a habit: change one
# character below -- the words, MAX_TOKENS, the temperature, or how the image
# is prepared -- and PROMPT_VERSION goes up in the same edit. Every record
# carries the version it was made under, and calibration() counts only one
# version at a time and says out loud when the ledger holds others. Two
# prompts pooled into one agreement number is the average of two experiments
# and a measurement of neither, and without a version on each record there is
# no way afterwards to tell which answer came from which question.
#
# The contract is a LAST LINE, not a whole reply. Asked for one word this
# model answers "Based on the visual characteristics..." about one time in
# five; asked to finish with a marker it can reason as much as it likes and
# still land somewhere a regex can read.
# dogseen-1 asked the model to "think for as long as you need" under a
# 400-token budget and lost whole crops to it. dogseen-2 asks for short
# reasoning and raises MAX_TOKENS. Records from the first are still on file
# and calibration() will not pool them with the second, which is the rule
# above doing its job rather than an inconvenience to work around.
PROMPT_VERSION = 'dogseen-2'
PROMPT_TEXT = (
    'Look at this photograph and decide one thing: is a dog visible in it?\n'
    '\n'
    'Count a domestic dog of any breed and any size, whether the whole '
    'animal is visible or only part of it, however small or blurry it is.\n'
    'Do not count cats, cows, goats, sheep, horses, pigs or any other '
    'animal. Do not count statues, toys, drawings, logos or pictures of '
    'dogs.\n'
    '\n'
    'Keep your reasoning short, then finish your reply with a final line '
    'that is exactly one of these two:\n'
    '\n'
    'ANSWER: YES\n'
    'ANSWER: NO\n'
    '\n'
    'The last line of your reply must be one of those two lines and '
    'nothing else.'
)
TEMPERATURE = 0

# How the picture is prepared before it is sent, which is as much a part of
# the question as the words are -- see the version rule above. The crops are
# tiny (the flag pools have a median long side of about 50px) and an
# unscaled 22px thumbnail is not a question anybody can answer, so they go up
# to a readable size with a smooth filter and are capped so a rare large one
# does not turn into a megabyte of base64.
PREP_MIN = 384
PREP_MAX = 1024
PREP = f'jpeg-lanczos-{PREP_MIN}-{PREP_MAX}'

# ── the vocabulary ──────────────────────────────────────────────────────────
# Four outcomes, all prefixed, none of them a word any human verdict in this
# repo uses. llm_unparsed and llm_error are separate on purpose: unparsed
# means the model replied and the reply was not an answer, error means there
# was no reply to read. Collapsing them would hide a rate limit inside what
# looks like the model being vague.
LLM_YES = 'llm_yes'
LLM_NO = 'llm_no'
LLM_UNPARSED = 'llm_unparsed'
LLM_ERROR = 'llm_error'
# The one `why` that is not a failure of anything: the call was abandoned
# because somebody pressed stop while it was in flight. run() recognises it
# and writes NO record, since a crop that was never answered has nothing to
# say about the model and must come back in the next sample.
WHY_STOPPED = 'stopped'
# An unparsed reply counts as asked-and-answered, so run() does not pay for it
# again. That is the honest choice: re-rolling until it parses would launder
# the model's worst failure mode straight out of the unparsed rate. An error
# does not count, because nothing was ever asked.
ANSWERED = (LLM_YES, LLM_NO, LLM_UNPARSED)

# The human side of the confusion matrix. These names live in memory and in
# what disagreements() hands to a caller; they are never written to this
# file's ledger, for the reason in the module docstring.
HUMAN_YES = 'human_says_dog'
HUMAN_NO = 'human_says_no_dog'


def paths():
    """Every file this module owns.

    Computed, never module state. fn_audit learned this the hard way: a
    module-level path constant is one import away from a test writing into
    the live store, and the store this one owns is the store whose whole
    purpose is to be somewhere else.
    """
    out = os.path.join(REPO, 'data', 'llm_annotations')
    return {
        'out': out,
        # not 'verdicts.jsonl'. That name is taken, twice, by files holding
        # what a person answered, and a store that has to be unmistakable
        # does not start by borrowing the other tier's filename.
        'ledger': os.path.join(out, 'llm_guesses.jsonl'),
        'status': os.path.join(out, 'status.json'),
        'stop': os.path.join(out, 'stop'),
        # who is running, so two processes cannot spend the same free tier on
        # the same crops -- see _claim() below
        'lock': os.path.join(out, 'running.lock'),
    }


def _own(path):
    """Refuse any write outside this module's own store.

    The rule this file exists under is that an LLM's answer never lands where
    a human decision lands. A rule kept only by which constant a function
    happens to reference is one careless refactor from being untrue, so every
    write here goes through this and a path outside data/llm_annotations/
    raises instead of appending.
    """
    root = os.path.realpath(paths()['out'])
    real = os.path.realpath(path)
    if real != root and not real.startswith(root + os.sep):
        raise RuntimeError(
            'llm_annotate may only write inside its own store; refusing '
            + os.path.relpath(real, REPO))
    return path


# ── where the human answers are ─────────────────────────────────────────────
def _audit():
    """fn_audit, imported late.

    Late because importing it at module scope would make a page that only
    wants to list the pools pay for the audit's imports, and because this
    module is imported by the dashboard. What is wanted from it is small and
    worth not copying: wilson(), and the reader that already knows the audit
    ledger's history -- 39 of its records say 'missed' rather than 'dog', and
    a null verdict there is a withdrawal rather than a third answer. A second
    copy of those rules here would drift from the first one.

    The insert is guarded, which it was not. This runs once per calibration
    and once per pool cache miss inside a dashboard that stays up for weeks,
    and an unguarded insert put a duplicate entry on sys.path on every /llm
    request -- measured going 9 to 12 over three calls to api_overview(), and
    every import in the process afterwards walks the longer list.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import fn_audit
    return fn_audit


def _audit_pool():
    """The audit's own answers: {key: (human answer, crop path)}.

    full/ rather than crops/: crops/ is downscaled for the contact sheet and
    full/ is the picture the gate itself was judged on, so this asks the
    model about the same pixels the measurement is about.
    """
    fa = _audit()
    root = os.path.join(REPO, 'data', 'fn_audit', 'full')
    out = {}
    for v in fa.read_verdicts(stage='gate'):
        ans = fa.verdict_of(v.get('verdict'), 'gate')
        # 'unsure' is an answer a person can give and is not a yes or a no,
        # so there is nothing here to agree or disagree with.
        if ans not in ('dog', 'not_dog'):
            continue
        key = str(v.get('key') or '')
        if not key:
            continue
        name = key.replace('#', '_') + '.jpg'
        out[key] = (HUMAN_YES if ans == 'dog' else HUMAN_NO,
                    os.path.join(root, name))
    return out


def _flag_pool(store, label, human):
    """One of the review page's flag ledgers: {key: (human answer, path)}.

    The ledger is append-only and holds more rows than the directory holds
    crops -- 2,653 lines against 2,239 files in hard_negatives -- because
    crops get deduplicated out from under it. Rows whose crop is gone are
    dropped here rather than counted, since a pool count that promises 2,653
    crops and can show 2,239 is a count of something else.
    """
    root = os.path.join(REPO, 'data', store)
    crops = os.path.join(root, 'crops')
    out = {}
    try:
        fh = open(os.path.join(root, 'labels.jsonl'), encoding='utf-8')
    except OSError:
        return out
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or d.get('label') != label:
                continue
            crop = d.get('crop')
            if not crop:
                continue
            p = os.path.join(crops, crop)
            # A three-byte file is on disk in hard_negatives from an old
            # probe. Size is checked as well as existence because a truncated
            # jpg costs an API call to discover.
            try:
                if os.path.getsize(p) < 512:
                    continue
            except OSError:
                continue
            out[crop[:-4] if crop.endswith('.jpg') else crop] = (human, p)
    return out


def _queue_pool():
    """The review queue: crops nobody has judged. Phase 2, switched off.

    The same two directories the review page serves from, listed directly
    rather than through the dashboard module, which pulls in duckdb and is
    not worth importing to list a directory. Nothing in here carries a human
    answer, which is exactly why it cannot be calibrated against and exactly
    why run() will not touch it without being asked twice.
    """
    out = {}
    for d in (os.path.join(REPO, 'data', 'dashboard', 'recent_crops'),
              os.path.join(REPO, 'data', 'dashboard', 'review_set')):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            if n.endswith('.jpg'):
                out[n[:-4]] = (None, os.path.join(d, n))
    return out


# One row per pool of crops, because they are the same exercise -- show the
# model a crop, ask the one question, compare to what a person said -- and
# one code path with a table is what keeps them from drifting into three
# nearly-identical functions.
#
# `human` is the only field that decides anything important. A pool with it
# false can be annotated but can never be calibrated, and produces a hint
# about queue order rather than anything resembling a label.
SOURCES = {
    'audit_finds': {
        'label': 'audit answers',
        'human': True,
        'enabled': True,
        'reader': _audit_pool,
        'ledger': 'data/fn_audit/verdicts.jsonl',
        'note': 'crops drawn evenly across the gate’s ten score bands, '
                'so near-threshold cases are hugely over-represented',
    },
    'hard_positives': {
        'label': 'confirmed dogs',
        'human': True,
        'enabled': True,
        'reader': lambda: _flag_pool('hard_positives', 'true_positive',
                                     HUMAN_YES),
        'ledger': 'data/hard_positives/labels.jsonl',
        'note': 'dogs a reviewer picked out of the queue; median long side '
                'about 50px, so this is the hard end of yes',
    },
    'hard_negatives': {
        'label': 'confirmed not-dogs',
        'human': True,
        'enabled': True,
        'reader': lambda: _flag_pool('hard_negatives', 'false_positive',
                                     HUMAN_NO),
        'ledger': 'data/hard_negatives/labels.jsonl',
        'note': 'the detector’s false positives -- things that already '
                'fooled one model, so the hard end of no',
    },
    'review_queue': {
        'label': 'unjudged queue (phase 2)',
        'human': False,
        'enabled': False,
        'reader': _queue_pool,
        'ledger': None,
        'note': 'nobody has judged these, so nothing here can be checked '
                'against anything; a queue-ordering hint at most',
    },
}
DEFAULT_SOURCE = 'audit_finds'


def spec(source=DEFAULT_SOURCE):
    if source not in SOURCES:
        raise KeyError(f'unknown source {source!r}')
    return SOURCES[source]


# Pools already read, one entry per source, keyed on the mtime of the ledger
# it came from. A page polling sources() would otherwise re-read 2,653 ledger
# lines and stat 2,239 files on every request. Same shape as fn_audit's sheet
# cache and for the same reason; the entry is replaced rather than added to,
# so a reviewer's next verdict invalidates it and the cache cannot grow.
_POOLS = {}


def pool(source=DEFAULT_SOURCE):
    """{key: (human answer or None, crop path)} for one source."""
    sp = spec(source)
    led = sp.get('ledger')
    if not led:
        # The review queue has no ledger to date it by and rotates hourly, so
        # there is nothing to key a cache on that would not go stale inside
        # the hour. Listing a directory is cheap; a wrong listing is not.
        return sp['reader']()
    try:
        stamp = os.path.getmtime(os.path.join(REPO, led))
    except OSError:
        stamp = 0
    got = _POOLS.get(source)
    if got is None or got[0] != stamp:
        got = (stamp, sp['reader']())
        _POOLS[source] = got
    return got[1]


# ── the key ─────────────────────────────────────────────────────────────────
def load_key():
    """OPENCODE_API_KEY from the environment or a .env. Never logged.

    Read at call time and held in the process. It is never put on a command
    line, never printed, and never written into a record -- the ledger says
    which model answered, not who paid for it.
    """
    if os.environ.get('OPENCODE_API_KEY'):
        return True
    for p in ENV_FILES:
        try:
            with open(p, encoding='utf-8') as fh:
                for ln in fh:
                    m = re.match(
                        r'^\s*(?:export\s+)?OPENCODE_API_KEY\s*=(.+)$',
                        ln.rstrip('\n'))
                    if m:
                        os.environ['OPENCODE_API_KEY'] = \
                            m.group(1).strip().strip('"').strip("'")
                        return True
        except OSError:
            continue
    return False


# ── the picture ─────────────────────────────────────────────────────────────
def _payload(path):
    """(data URI, width, height, bytes) for one crop.

    Pillow is required rather than optional. It could be made optional by
    sending the file untouched when it is missing, and that would silently
    put two different questions under one prompt version -- a 22px thumbnail
    and a 384px upscale of it are not the same question. One contract.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            'Pillow is needed to prepare the crop, and how the crop is '
            'prepared is part of the prompt version -- sending the file '
            'untouched instead would quietly ask a different question'
        ) from e
    with Image.open(path) as src:
        im = src.convert('RGB')
        w, h = im.size
        long_side = max(w, h)
        if long_side < PREP_MIN:
            k = PREP_MIN / float(long_side)
        elif long_side > PREP_MAX:
            k = PREP_MAX / float(long_side)
        else:
            k = 1.0
        if k != 1.0:
            w, h = max(1, int(round(w * k))), max(1, int(round(h * k)))
            im = im.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=92)
    raw = buf.getvalue()
    uri = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')
    return uri, w, h, len(raw)


# ── reading the reply ───────────────────────────────────────────────────────
# The contract's own marker, anywhere in the reply, last match wins -- the
# model sometimes rehearses the format mid-reasoning before committing to it
# at the end. Markdown bold around the answer is common enough to allow for.
_MARKED = re.compile(r'ANSWER\s*[:\-]?\s*\**\s*(YES|NO)\b', re.I)
# The fallback, and it is deliberately narrow: the LAST line of the reply, if
# that whole line is the bare word. A looser scan that hunts for a "no"
# anywhere in a paragraph finds one in "no other animals are visible" and
# turns a yes into a no, which is worse than recording the reply as unparsed.
_BARE = re.compile(r'^\**\s*(YES|NO)\s*[.!]?\s*\**$', re.I)


def parse(content):
    """A reply -> llm_yes, llm_no, or None if it is not an answer.

    None is not a "no". It is the third outcome, and the caller records it as
    one; about a fifth of replies land here and pretending otherwise would
    put that fifth on whichever side happened to be the default.
    """
    if not content:
        return None
    text = content.strip()
    if not text:
        return None
    hits = _MARKED.findall(text)
    if hits:
        return LLM_YES if hits[-1].upper() == 'YES' else LLM_NO
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        m = _BARE.match(lines[-1])
        if m:
            return LLM_YES if m.group(1).upper() == 'YES' else LLM_NO
    return None


# HTTP statuses worth trying again. 429 is the rate limit, the 5xx are the
# endpoint having a moment, 408 is its own timeout. Everything else -- 400,
# 401, 404 -- is a request that will fail identically forever, and retrying
# it four times only spends four times as long finding that out.
RETRY_CODES = (408, 429, 500, 502, 503, 504)

# When to stop asking at all. These are the failures that are about the
# endpoint and not about one crop, so the next crop gets the same answer and
# pays the same retry chain to hear it -- four attempts against a 120 second
# timeout plus the backoff is 501 seconds a crop, and a batch of fifty in that
# state runs for about seven hours to record fifty identical errors. Five in a
# row is the endpoint being down; one is a blip worth riding out.
GIVE_UP_WHY = ('rate_limit', 'network', 'http', 'setup')
GIVE_UP_AFTER = 5


def _backoff(attempt, retry_after=None):
    """Seconds to wait. The endpoint publishes no rate limit, so this is
    deliberately slow: 3, 6, 12, 24, with jitter so several runs do not
    synchronise onto the same second. A Retry-After header wins outright."""
    if retry_after:
        try:
            return min(120.0, max(1.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
    jitter = 0.8 + 0.4 * random.random()
    return min(60.0, 3.0 * (2 ** (attempt - 1))) * jitter


def _wait(seconds, stop=None):
    """Sleep, in slices, looking up between them. True if a stop arrived.

    A backoff is where a run spends its longest uninterrupted stretch -- three
    waits of up to 120 s each when the endpoint hands back a big Retry-After
    -- and a single time.sleep() through one of those is dead to the stop
    signal. Measured against a black-holed endpoint: a stop pressed half a
    second in was not seen for 19 s of a 2-try chain, and with the shipped
    defaults it would have been 501 s.
    """
    end = time.time() + max(0.0, seconds)
    while True:
        left = end - time.time()
        if left <= 0:
            return False
        if stop and stop():
            return True
        time.sleep(min(0.5, left))


def ask(path, model=MODEL, tries=4, timeout=120, stop=None):
    """One crop -> one record. Never raises for anything the network did.

    The record is the whole story of the call: what was asked (model, prompt
    version, how the image was prepared), what came back, how long it took
    and what it cost. A failure is a record too -- llm_error with the reason
    -- because a run that quietly annotates 8 of 10 crops and says nothing
    about the other two is how a rate limit turns into a biased sample.

    `stop` is checked between attempts and inside the backoff, not only by the
    caller between crops. Without it the retry chain for the crop in flight is
    uninterruptible -- four attempts against a timeout plus three waits, 501
    seconds on the shipped defaults -- while the page says a stop lands within
    about one call. A stop seen here abandons the crop: the record comes back
    with `why` WHY_STOPPED and run() writes nothing down for it, because a
    crop nobody got an answer about is not a fact about the model.
    """
    if not load_key():
        return {'llm_says': LLM_ERROR, 'why': 'setup',
                'error': 'no OPENCODE_API_KEY in the environment or .env',
                'model': model, 'prompt_version': PROMPT_VERSION,
                'prep': PREP, 'ts': time.time(), 'ms': 0, 'attempts': 0}
    try:
        uri, w, h, nbytes = _payload(path)
    except Exception as e:
        # An unreadable crop is not the model's fault and must not be counted
        # against it, so it is an error like any other rather than an answer.
        return {'llm_says': LLM_ERROR, 'why': 'crop',
                'error': f'crop unreadable: {e}',
                'model': model, 'prompt_version': PROMPT_VERSION,
                'prep': PREP, 'ts': time.time(), 'ms': 0, 'attempts': 0}

    body = json.dumps({
        'model': model,
        'max_tokens': MAX_TOKENS,
        'temperature': TEMPERATURE,
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': PROMPT_TEXT},
            {'type': 'image_url', 'image_url': {'url': uri}}]}],
    }).encode('utf-8')

    started = time.time()
    # `why` is the error's KIND, kept as a field rather than left to be
    # recovered by grepping the message later. A rate limit, a budget that
    # ran out and a crop the endpoint refused are three different problems
    # and only one of them is fixed by waiting.
    err, code, made, why = 'no attempt made', None, 0, 'none'
    for attempt in range(1, max(1, tries) + 1):
        if attempt > 1 and stop and stop():
            # Between attempts, so a stop does not have to wait out the rest
            # of a retry chain that is already failing. Nothing is recorded
            # for this crop -- see the docstring.
            return {'llm_says': LLM_ERROR, 'why': WHY_STOPPED,
                    'error': 'stopped before attempt %d' % attempt,
                    'model': model, 'prompt_version': PROMPT_VERSION,
                    'prep': PREP, 'ts': time.time(),
                    'ms': int((time.time() - started) * 1000),
                    'attempts': made}
        # counted here rather than reported as `tries` at the end: a 400 or a
        # blown token budget stops on the first call, and a record claiming
        # four attempts on one call makes a rate limit and a bad request look
        # like the same thing in the ledger
        made = attempt
        req = urllib.request.Request(API_URL, data=body, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('User-Agent', USER_AGENT)
        # Built here and nowhere else. It is not stored on the object, not
        # logged, and not put in the record.
        req.add_header('Authorization',
                       'Bearer ' + os.environ['OPENCODE_API_KEY'])
        wait = None
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                code = r.status
                doc = json.loads(r.read().decode('utf-8', 'replace'))
        except urllib.error.HTTPError as e:
            code, why = e.code, ('rate_limit' if e.code == 429 else 'http')
            wait = e.headers.get('Retry-After') if e.headers else None
            # The body can carry a reason worth keeping, and can equally be a
            # Cloudflare page. Truncated hard either way.
            try:
                err = f'HTTP {code}: ' + e.read().decode(
                    'utf-8', 'replace')[:200].replace('\n', ' ')
            except Exception:
                err = f'HTTP {code}'
            if code not in RETRY_CODES:
                break
        except Exception as e:
            code, why = None, 'network'
            err = f'{type(e).__name__}: {e}'[:200]
        else:
            ch = ((doc.get('choices') or [{}])[0]) or {}
            msg = ch.get('message') or {}
            content = msg.get('content')
            reasoning = msg.get('reasoning') or msg.get('reasoning_content')
            usage = doc.get('usage') or {}
            if content and content.strip():
                says = parse(content)
                return {
                    'llm_says': says or LLM_UNPARSED,
                    # kept for every record, not only the unreadable ones:
                    # the disagreements page is where a person finds out
                    # whether the model or the reviewer was wrong, and the
                    # model's own words are most of that.
                    'reply': content.strip()[:400],
                    'model': model, 'prompt_version': PROMPT_VERSION,
                    'prep': PREP, 'ts': time.time(),
                    'ms': int((time.time() - started) * 1000),
                    'tokens': usage.get('total_tokens'),
                    'attempts': attempt, 'http': code,
                    'w': w, 'h': h, 'bytes': nbytes,
                    'finish': ch.get('finish_reason'),
                    # how much of the budget the reasoning took. When content
                    # comes back null this is the explanation, and it is the
                    # number to look at before touching MAX_TOKENS.
                    'reason_chars': len(reasoning or ''),
                }
            # A null content is the silent failure in the docstring: the
            # reasoning spent the budget and there was nothing left to answer
            # with. It is a failed call, never a "no".
            finish = ch.get('finish_reason')
            why = 'budget' if finish == 'length' else 'empty'
            err = (f'empty content (finish {finish!r}, '
                   f'{len(reasoning or "")} reasoning chars)')
            if finish == 'length':
                # But do not retry THIS one. 'length' says the budget ran out
                # rather than that anything went wrong on the wire, and an
                # identical request gets an identical answer -- the first
                # version of this file spent four calls and 47 seconds
                # discovering that on one crop. The fix is MAX_TOKENS, and it
                # is a person's decision, so the record says so and stops.
                err += ' -- the token budget is too small for this crop'
                break
        if attempt < max(1, tries):
            if _wait(_backoff(attempt, wait), stop):
                return {'llm_says': LLM_ERROR, 'why': WHY_STOPPED,
                        'error': 'stopped during the backoff after attempt '
                                 '%d' % attempt,
                        'model': model, 'prompt_version': PROMPT_VERSION,
                        'prep': PREP, 'ts': time.time(),
                        'ms': int((time.time() - started) * 1000),
                        'attempts': made}
    return {'llm_says': LLM_ERROR, 'why': why, 'error': err, 'http': code,
            'model': model, 'prompt_version': PROMPT_VERSION, 'prep': PREP,
            'ts': time.time(), 'ms': int((time.time() - started) * 1000),
            'attempts': made, 'w': w, 'h': h, 'bytes': nbytes}


# ── the ledger ──────────────────────────────────────────────────────────────
def _append(rec):
    """One record, opened and closed. Appending through a held handle is
    faster and loses everything buffered when a run is killed; at six seconds
    a call the open costs nothing measurable, and the promise that a kill
    costs one record is worth more."""
    P = paths()
    os.makedirs(P['out'], exist_ok=True)
    with open(_own(P['ledger']), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec) + '\n')
        fh.flush()


def read_ledger(version=None, model=None, source=None):
    """{(pool, key): record} -- append-only, last write wins.

    Keyed on the pool and the crop so a crop asked about twice counts once,
    as its latest answer. Filtering by version here rather than at every call
    site is what makes "calibration never mixes two prompts" a property of
    the reader instead of a thing each caller has to remember.
    """
    out = {}
    try:
        fh = open(paths()['ledger'], encoding='utf-8')
    except OSError:
        return out
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or not d.get('key'):
                continue
            if version and d.get('prompt_version') != version:
                continue
            if model and d.get('model') != model:
                continue
            if source and d.get('pool') != source:
                continue
            out[(d.get('pool'), d['key'])] = d
    return out


def versions_on_file():
    """{(prompt version, model): {'records', 'answered'}} across the ledger.

    So a page can say "there are answers here from a prompt you have since
    changed" rather than presenting a smaller sample with no explanation of
    where the rest went.

    Counted off the FILE's lines rather than through read_ledger(), which is
    keyed on (pool, crop) and keeps only the last write. That dedup is right
    for a calibration and wrong for this: re-asking a pool under a new prompt
    made the old version's tally count DOWN as each of its records was
    shadowed by a new one, so the line whose whole job is to say "there is
    more on file than is counted here" under-reported the thing it names. Ten
    dogseen-1 records were on disk and it reported three.

    `answered` is separated from `records` because the three it did report
    were all llm_error, printed as "3 answers" -- a footnote describing
    answers nobody ever got.
    """
    out = {}
    try:
        fh = open(paths()['ledger'], encoding='utf-8')
    except OSError:
        return out
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or not d.get('key'):
                continue
            row = out.setdefault((d.get('prompt_version'), d.get('model')),
                                 {'records': 0, 'answered': 0})
            row['records'] += 1
            if d.get('llm_says') in ANSWERED:
                row['answered'] += 1
    return out


# ── one run at a time, whoever started it ───────────────────────────────────
class RunInProgress(RuntimeError):
    """Another run holds this store. Raised by run(), before anything is asked.

    Its own type so a page can tell "somebody else is running" from a run that
    fell over, and answer the first as a fact rather than as a failure.
    """


def _alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _claim():
    """Take the store's run lock, or raise RunInProgress. Returns a release.

    THE GUARD BELONGS HERE, not on the page that usually starts a run. The
    page can only refuse a second run of ITS OWN: a `llm_annotate.py run` typed
    in a terminal beside a dashboard batch was refused by nothing at all, and
    two processes then asked the same crops on the same free tier -- measured,
    10 records over 8 distinct crops, neither process complaining. Worse, the
    two share one status file, and the first to FINISH is not the first to
    start: the short run's final running=False landed on top of the long run's
    progress, which is exactly the state that re-enables the page's button.
    Both entry points come through run(), so this is the one door both pass.

    An O_EXCL file rather than a flock, and it carries the pid: a lock left
    behind by a killed run is then recognisable as stale rather than jamming
    the store shut until somebody deletes a file they have never heard of --
    the same "is that pid still alive" question status() already asks. An
    unreadable lock is only treated as stale once it is old, because for a
    moment after O_EXCL succeeds it is legitimately empty, and stealing it in
    that window would let both racers through.
    """
    P = paths()
    os.makedirs(P['out'], exist_ok=True)
    lock = _own(P['lock'])
    for attempt in (1, 2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pass
        except OSError as e:
            # A store that cannot be locked is a store that cannot be written
            # either, so this is not a reason to go ahead unguarded.
            raise RunInProgress(f'could not take the run lock: {e}') from e
        else:
            with os.fdopen(fd, 'w') as fh:
                fh.write(json.dumps({'pid': os.getpid(), 'ts': time.time()}))
            return lambda: _release(lock)
        held, age = None, 0.0
        try:
            with open(lock, encoding='utf-8') as fh:
                held = (json.loads(fh.read() or '{}') or {}).get('pid')
        except (OSError, ValueError):
            held = None
        try:
            age = time.time() - os.path.getmtime(lock)
        except OSError:
            age = 0.0
        if attempt == 2 or (held is not None and _alive(held)):
            raise RunInProgress(
                'a run is already going in another process'
                + (f' (pid {held})' if held else '')
                + ' -- two runs share this store, ask the same crops and '
                  'write over each other\'s progress')
        if held is None and age < 30:
            raise RunInProgress('another process is starting a run right now')
        # a lock whose owner is gone: take it over, once
        try:
            os.remove(lock)
        except OSError:
            pass
    raise RunInProgress('a run is already going in another process')


def _release(lock):
    """Drop the lock, and only if it is still ours. A stale lock that was
    stolen while we ran belongs to whoever stole it."""
    try:
        with open(lock, encoding='utf-8') as fh:
            if (json.loads(fh.read() or '{}') or {}).get('pid') != os.getpid():
                return
    except (OSError, ValueError):
        return
    try:
        os.remove(lock)
    except OSError:
        pass


def running_elsewhere():
    """(True, pid) if another live process holds the lock.

    For a page that wants to say so before offering a button, rather than
    starting a thread to find out.
    """
    try:
        with open(paths()['lock'], encoding='utf-8') as fh:
            held = (json.loads(fh.read() or '{}') or {}).get('pid')
    except (OSError, ValueError):
        return False, None
    if held is not None and held != os.getpid() and _alive(held):
        return True, held
    return False, held


# ── the stop signal ─────────────────────────────────────────────────────────
# A file rather than a signal, because the page that starts a run is not
# necessarily the process that runs it, and a file works whether the run is a
# thread of the dashboard or a detached command. run() checks it between
# crops and ask() checks it between attempts and inside the backoff, so a stop
# lands within one HTTP attempt: about six seconds while the endpoint is
# answering, and at worst the socket timeout while it is not. It used to be
# checked between crops only, and the promise of "about six seconds" was then
# true of a healthy endpoint and 501 seconds out of date against a dead one.
def request_stop():
    P = paths()
    os.makedirs(P['out'], exist_ok=True)
    with open(_own(P['stop']), 'w', encoding='utf-8') as fh:
        fh.write(str(time.time()))


def clear_stop():
    try:
        os.remove(_own(paths()['stop']))
    except OSError:
        pass


def stop_requested():
    return os.path.exists(paths()['stop'])


def write_status(**kw):
    """Publish where a run has got to. Atomic, because a page polls it while
    it is being written and half a JSON document on the wire reads as 'no
    run', which is a lie about a run that is going fine."""
    kw.setdefault('schema', 1)
    kw.setdefault('pid', os.getpid())
    kw.setdefault('prompt_version', PROMPT_VERSION)
    kw['updated'] = time.time()
    P = paths()
    try:
        os.makedirs(P['out'], exist_ok=True)
        tmp = _own(P['status']) + '.tmp'
        with open(_own(tmp), 'w', encoding='utf-8') as fh:
            json.dump(kw, fh)
        os.replace(tmp, P['status'])
    except OSError:
        pass          # progress reporting must never break a run


def status():
    try:
        with open(paths()['status'], encoding='utf-8') as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict):
        return {}
    if doc.get('running'):
        # A status file outliving the process that wrote it says a run is in
        # progress forever. Cheap to check, and the alternative is a page
        # with a spinner nobody can stop.
        try:
            os.kill(int(doc.get('pid') or 0), 0)
        except (OSError, TypeError, ValueError):
            doc['running'] = False
            doc['stale'] = True
    return doc


# ── the pools, and what has been asked of them ──────────────────────────────
def sources(model=MODEL):
    """One row per pool: how many crops carry a human answer, and how many of
    them this prompt version has already been asked about.

    `pending` is what a run would actually cost, which is the number a page
    needs before it offers a button. Errors are not counted as done, so a
    pool with a rate limit in its history shows the work still outstanding.

    One MODEL as well as one prompt version, and for the same reason. This
    counted every model's records as done while calibration() counted only
    one model's, so a single `run --model something-else` from the command
    line took those crops out of the pending count for good: the page said
    annotated, sample() never re-drew them, and the calibration they were
    missing from had no way to ask for them back. "Left" now means left for
    the model the numbers on the page are about.
    """
    led = read_ledger(version=PROMPT_VERSION, model=model)
    out = []
    for name, sp in SOURCES.items():
        try:
            got = pool(name)
        except Exception:
            got = {}
        mine = {k: r for (p, k), r in led.items() if p == name}
        done = sum(1 for r in mine.values()
                   if r.get('llm_says') in ANSWERED)
        out.append({
            'source': name, 'label': sp['label'], 'human': sp['human'],
            'enabled': sp['enabled'], 'note': sp['note'],
            'ledger': sp['ledger'], 'crops': len(got),
            'annotated': done,
            'unparsed': sum(1 for r in mine.values()
                            if r.get('llm_says') == LLM_UNPARSED),
            'errors': sum(1 for r in mine.values()
                          if r.get('llm_says') == LLM_ERROR),
            'pending': max(0, len(got) - done),
        })
    return out


def sample(source=DEFAULT_SOURCE, n=10, seed=7, version=PROMPT_VERSION,
           model=MODEL):
    """[(key, path)] -- the next n crops to ask about, deterministically.

    Sorted then shuffled under a named seed, so the same seed picks the same
    crops on a second machine and a run that was killed resumes on the same
    order rather than re-drawing a fresh sample every time.

    Already-asked means asked UNDER THIS PROMPT AND THIS MODEL. Filtering on
    the version alone made a record from any other model mark its crop as done
    forever while counting towards no rate anywhere -- see sources().
    """
    got = pool(source)
    done = {k for (p, k), r in read_ledger(version=version, model=model).items()
            if p == source and r.get('llm_says') in ANSWERED}
    keys = sorted(k for k in got if k not in done)
    random.Random(seed).shuffle(keys)
    if n and n > 0:
        keys = keys[:n]
    return [(k, got[k][1]) for k in keys]


def run(source=DEFAULT_SOURCE, n=10, seed=7, sleep=0.5, model=MODEL,
        stop=None, on_record=None, allow_unjudged=False, tries=4,
        dry_run=False):
    """Annotate n crops from one pool. Resumable, interruptible, append-only.

    Resumable in the only sense that matters on a metered API: a crop this
    prompt version and this model have already been asked about is not paid
    for twice. Each record is appended the moment it exists, so a kill costs
    the call in flight and nothing else.

    `stop` is a callable for a caller that has its own idea of when to quit;
    the default watches the stop file. It is checked before the first call,
    between crops, and inside ask() between attempts and during a backoff, so
    a stop lands within one HTTP attempt rather than at the end of whatever
    retry chain the crop in flight is halfway through.

    ONE RUN AT A TIME, ACROSS PROCESSES. The lock is taken here rather than by
    the page, because the page is only one of the two ways in -- see _claim().
    A dry run takes nothing: it is a question about the pool, asks nothing and
    must not refuse to answer while a batch is going.

    AND IT GIVES UP. A dead endpoint used to cost the same full retry chain on
    every crop in turn -- 501 seconds each, about seven hours for a batch of
    fifty -- for fifty identical failures. GIVE_UP_AFTER consecutive failures
    of the kinds that are about the endpoint rather than about one crop end
    the batch and say so.
    """
    sp = spec(source)
    if (not sp['enabled'] or not sp['human']) and not allow_unjudged:
        # Phase 2, and it is off. Asked twice on purpose: the pool is marked
        # off in the table AND the caller has to say allow_unjudged, so no
        # page can start one by passing a source name through from a form.
        # What comes back from a pool with no human answers cannot be checked
        # against anything, which makes it a hint about which crop to put in
        # front of a person next -- the same footing as the dog-bin's own
        # score -- and not a label.
        raise SystemExit(
            f"'{source}' holds crops nobody has judged, so nothing from it "
            f'can be checked against anything. It is phase 2 and it is '
            f'switched off. Pass allow_unjudged only if you understand that '
            f'what it produces orders a queue and is not a label.')
    todo = sample(source, n, seed, model=model)
    if dry_run:
        return {'source': source, 'planned': len(todo), 'dry_run': True,
                'crops': [os.path.relpath(p, REPO) for _, p in todo]}
    stop = stop or stop_requested
    # NOT cleared at the start, which is what this did first and it was
    # wrong. Clearing meant a stop signalled in the moment before a run began
    # was thrown away and the run went ahead -- caught here by a stop that
    # was ignored and cost three calls to a metered endpoint. So a pending
    # stop is honoured on entry: the run halts having asked nothing, says so
    # in what it returns and in the status file, and clears the signal on its
    # way out. The worst a stale signal can now do is cost one visible no-op
    # start, where before it could silently spend a whole batch.
    counts = {LLM_YES: 0, LLM_NO: 0, LLM_UNPARSED: 0, LLM_ERROR: 0}
    started, done, halted, gave_up = time.time(), 0, False, ''
    free = _claim()
    try:
        write_status(running=True, source=source, model=model, n=len(todo),
                     done=0, started=started, counts=counts)
        streak = 0
        for key, path in todo:
            if stop():
                halted = True
                break
            rec = ask(path, model=model, tries=tries, stop=stop)
            if rec.get('why') == WHY_STOPPED:
                # The stop arrived mid-retry and the crop was abandoned. No
                # record: nobody got an answer about it, and an llm_error
                # here would count a click of the Stop button as a failure of
                # the endpoint in every rate errors_why feeds.
                halted = True
                break
            rec.update({
                'pool': source,
                'key': key,
                # relative, because this ledger gets read on other machines
                # and an absolute path is true of exactly one of them
                'crop': os.path.relpath(path, REPO),
                # The mark the isolation suite's wrote_it() already hunts for
                # in every human store, carried so that a record which
                # escapes into one is caught by a test that has never heard
                # of this file. It catches a record COPIED there; a row
                # translated into the review page's own words carries none of
                # this, which is what adv_llm_tier.py is for.
                'unverified': True,
                'tier': 'llm_experimental',
            })
            _append(rec)
            counts[rec['llm_says']] = counts.get(rec['llm_says'], 0) + 1
            done += 1
            if on_record:
                on_record(rec)
            # Consecutive, and only the kinds that are about the endpoint. A
            # crop that will not open and a budget that ran out are facts
            # about one crop and the next one may be fine; a rate limit and a
            # dead socket are not, and paying the full retry chain fifty
            # times over to learn the same thing fifty times is how a batch
            # runs for seven hours and annotates nothing.
            streak = (streak + 1) if rec.get('why') in GIVE_UP_WHY else 0
            if streak >= GIVE_UP_AFTER:
                gave_up = rec.get('why') or 'error'
                break
            write_status(running=True, source=source, model=model,
                         n=len(todo), done=done, started=started,
                         counts=counts, last=rec['llm_says'])
            if sleep and done < len(todo):
                if _wait(sleep, stop):
                    halted = True
                    break
        if halted:
            clear_stop()
        write_status(running=False, source=source, model=model, n=len(todo),
                     done=done, started=started, counts=counts, halted=halted,
                     gave_up=gave_up)
    finally:
        free()
    return {'source': source, 'asked': done, 'planned': len(todo),
            'halted': halted, 'gave_up': gave_up, 'counts': counts,
            'seconds': round(time.time() - started, 1)}


# ── what it is worth ────────────────────────────────────────────────────────
def calibration(source=None, version=PROMPT_VERSION, model=MODEL):
    """Agreement against the human answers, in both directions separately.

    A confusion matrix, and the two error rates that matter read off it:

        missed    of the crops a person called a dog, the share this said no
        invented  of the crops a person called not-a-dog, the share it said yes

    Each with a Wilson interval, because at the rates worth caring about --
    a handful of errors in a few dozen crops -- the normal approximation runs
    off the end of the scale.

    They are never added together. The pilot put them at 0% and 29% on the
    same 19 crops; one accuracy number over those two is 89.5%, describes
    neither error, and moves whenever the mix of the sample moves. The mix
    here is chosen rather than natural -- these pools are the crops that
    already fooled a detector or sat on a threshold -- so `agreement` is
    reported last, with that caveat attached, and is not the headline.

    One prompt version and one model, and both defaulted rather than left
    open. Version because a prompt is a question and two questions do not
    average; model because the same words put to a different model are a
    different experiment just as surely. Everything on file that is not the
    counted pair is reported in `others`, so a sample that got smaller never
    appears without saying where the rest of it went.
    """
    led = read_ledger(version=version, model=model, source=source)
    per = {}
    n_unparsed = n_error = 0
    why = {}
    for (p, key), rec in led.items():
        sp = SOURCES.get(p)
        if not sp or not sp['human']:
            continue          # nothing to agree with
        says = rec.get('llm_says')
        if says == LLM_ERROR:
            n_error += 1
            # Counted by kind, and reported. A crop lost to a budget that ran
            # out is not a crop lost at random -- see MAX_TOKENS -- so a
            # calibration that mentions only its successes is describing the
            # easy end of whatever it sampled.
            why[rec.get('why') or 'unknown'] = \
                why.get(rec.get('why') or 'unknown', 0) + 1
            continue
        human = (pool(p).get(key) or (None, None))[0]
        if human not in (HUMAN_YES, HUMAN_NO):
            # the human answer has been withdrawn or changed to 'unsure'
            # since; there is no longer anything to compare against
            continue
        if says == LLM_UNPARSED:
            n_unparsed += 1
        row = per.setdefault(p, {'source': p, 'label': sp['label'],
                                 'note': sp['note'], 'unparsed': 0,
                                 HUMAN_YES: {LLM_YES: 0, LLM_NO: 0},
                                 HUMAN_NO: {LLM_YES: 0, LLM_NO: 0}})
        if says == LLM_UNPARSED:
            row['unparsed'] += 1
        elif says in (LLM_YES, LLM_NO):
            row[human][says] += 1

    wilson = _audit().wilson
    m = {HUMAN_YES: {LLM_YES: 0, LLM_NO: 0},
         HUMAN_NO: {LLM_YES: 0, LLM_NO: 0}}
    for row in per.values():
        for h in (HUMAN_YES, HUMAN_NO):
            for a in (LLM_YES, LLM_NO):
                m[h][a] += row[h][a]

    dogs = m[HUMAN_YES][LLM_YES] + m[HUMAN_YES][LLM_NO]
    nots = m[HUMAN_NO][LLM_YES] + m[HUMAN_NO][LLM_NO]
    parsed = dogs + nots

    def rate(k, n):
        p, lo, hi = wilson(k, n)
        return {'k': k, 'n': n, 'rate': p, 'lo95': lo, 'hi95': hi}

    return {
        'prompt_version': version, 'model': model,
        'source': source, 'matrix': m,
        'parsed': parsed, 'unparsed': n_unparsed, 'errors': n_error,
        'errors_why': why,
        # every crop asked about that yielded no usable answer, either way
        'no_answer_rate': rate(n_unparsed + n_error,
                               parsed + n_unparsed + n_error),
        # the share of everything the model was asked that it did not answer
        # in the shape it was asked to. Never dropped quietly.
        'unparsed_rate': rate(n_unparsed, parsed + n_unparsed),
        # dogs it denied -- the direction that loses data
        'missed': rate(m[HUMAN_YES][LLM_NO], dogs),
        # dogs it invented -- the direction the pilot says is the weak one
        'invented': rate(m[HUMAN_NO][LLM_YES], nots),
        # last, and only with its warning: this moves with the mix of the
        # sample, and the mix here was chosen
        'agreement': rate(m[HUMAN_YES][LLM_YES] + m[HUMAN_NO][LLM_NO],
                          parsed),
        'by_source': sorted(per.values(), key=lambda r: r['source']),
        # {'prompt / model': {'records', 'answered'}} for everything on file
        # that this pair does not count. Two numbers because they are two
        # different reassurances: how much is on disk, and how much of it was
        # ever an answer. The version that reported one number reported the
        # smaller one under the word "answers".
        'others': {f'{v} / {mo}': c
                   for (v, mo), c in sorted(versions_on_file().items(),
                                            key=lambda kv: str(kv[0]))
                   if (v, mo) != (version, model)},
    }


def disagreements(source=None, version=PROMPT_VERSION, limit=0):
    """Every crop where the model and the person differ, newest first.

    The most useful surface here by a distance. An agreement number tells you
    how often they differ; this tells you WHO WAS RIGHT, which is the only
    way to find out whether the model is unreliable or the reviewer clicked
    the wrong tile -- and both have turned up in this project's stores
    before. The model's own words come with each row for the same reason.
    """
    out = []
    for (p, key), rec in read_ledger(version=version, source=source).items():
        sp = SOURCES.get(p)
        if not sp or not sp['human']:
            continue
        says = rec.get('llm_says')
        if says not in (LLM_YES, LLM_NO):
            continue
        human = (pool(p).get(key) or (None, None))[0]
        if human not in (HUMAN_YES, HUMAN_NO):
            continue
        agrees = ((human == HUMAN_YES and says == LLM_YES)
                  or (human == HUMAN_NO and says == LLM_NO))
        if agrees:
            continue
        out.append({'source': p, 'key': key, 'crop': rec.get('crop'),
                    'human': human, 'llm_says': says,
                    'reply': rec.get('reply'), 'ts': rec.get('ts'),
                    'model': rec.get('model'),
                    'prompt_version': rec.get('prompt_version'),
                    # which way round it went, so a page can group them
                    'direction': ('invented' if says == LLM_YES
                                  else 'missed')})
    out.sort(key=lambda r: r.get('ts') or 0, reverse=True)
    return out[:limit] if limit and limit > 0 else out


def unparsed(source=None, version=PROMPT_VERSION, limit=0):
    """The replies that were not an answer, so the contract can be improved.

    Kept as a first-class listing because the fix for a 20% unparsed rate is
    in these strings and nowhere else -- and because improving the prompt on
    the back of them is exactly the edit that has to bump PROMPT_VERSION.
    """
    out = [{'source': p, 'key': k, 'crop': r.get('crop'),
            'reply': r.get('reply'), 'ts': r.get('ts'),
            'finish': r.get('finish'), 'reason_chars': r.get('reason_chars')}
           for (p, k), r in read_ledger(version=version,
                                        source=source).items()
           if r.get('llm_says') == LLM_UNPARSED]
    out.sort(key=lambda r: r.get('ts') or 0, reverse=True)
    return out[:limit] if limit and limit > 0 else out


# ── command line ────────────────────────────────────────────────────────────
def _cmd_sources(args):
    print(f'prompt {PROMPT_VERSION}   model {MODEL}\n')
    print(f"  {'source':<16} {'crops':>7} {'asked':>7} {'left':>7}  what it "
          f"is")
    for r in sources():
        tag = '' if r['enabled'] else '   [OFF -- phase 2]'
        print(f"  {r['source']:<16} {r['crops']:>7,} {r['annotated']:>7,} "
              f"{r['pending']:>7,}  {r['label']}{tag}")
        print(f"  {'':<16} {'':>7} {'':>7} {'':>7}  {r['note']}")
    print('\nEXPERIMENTAL. Nothing here is a label and nothing reads it.')
    return 0


def _cmd_run(args):
    if args.dry_run:
        r = run(args.source, args.n, args.seed, dry_run=True,
                allow_unjudged=args.allow_unjudged)
        print(f"{r['planned']} crops would be asked about:")
        for c in r['crops']:
            print('  ' + c)
        return 0
    n = args.n
    print(f'{n} crops from {args.source}, prompt {PROMPT_VERSION}, '
          f'model {args.model}')
    print(f'  ~6 s a call, so this is roughly {n * 6 // 60} min '
          f'{n * 6 % 60} s. Ctrl-C or `llm_annotate.py stop` to halt; '
          f'records are on disk as they are made.')
    try:
        r = run(args.source, n, args.seed, sleep=args.sleep, model=args.model,
                allow_unjudged=args.allow_unjudged, tries=args.tries)
    except RunInProgress as e:
        # A refusal, not a crash. Somebody else -- most likely the dashboard
        # -- is spending the same free tier on the same store, and a traceback
        # for that reads like a bug in the tool.
        print(str(e))
        return 1
    c = r['counts']
    print(f"\nasked {r['asked']} of {r['planned']} in {r['seconds']}s"
          + ('  (halted)' if r['halted'] else '')
          + (f"  (gave up after {GIVE_UP_AFTER} {r['gave_up']} failures in a "
             f'row)' if r['gave_up'] else ''))
    print(f"  {c[LLM_YES]} {LLM_YES}   {c[LLM_NO]} {LLM_NO}   "
          f"{c[LLM_UNPARSED]} {LLM_UNPARSED}   {c[LLM_ERROR]} {LLM_ERROR}")
    return 0


def _pct(d):
    return (f"{d['rate']:.1%} [{d['lo95']:.1%}, {d['hi95']:.1%}]  "
            f"{d['k']}/{d['n']}") if d['n'] else 'nothing to measure yet'


def _others(others):
    """The one-line "and this is what is on file that I did not count"."""
    return ', '.join(
        f"{k}: {v['records']} record{'' if v['records'] == 1 else 's'}, "
        f"{v['answered']} answered" for k, v in others.items())


def _cmd_calibration(args):
    s = calibration(args.source, version=args.version, model=args.model)
    m = s['matrix']
    if not s['parsed'] and not s['unparsed']:
        print('nothing answered under prompt ' + s['prompt_version'])
        if s['others']:
            print('  the ledger also holds ' + _others(s['others']))
        return 0
    print(f"prompt {s['prompt_version']}   model {s['model']}"
          + (f"   source {s['source']}" if s['source'] else ''))
    if s['others']:
        print('  NOTE: not counted here -- ' + _others(s['others'])
              + '\n        a prompt is a question, and two questions do not '
                'average')
    print('\n                       said dog   said no dog')
    print(f"  person: dog          {m[HUMAN_YES][LLM_YES]:>8}   "
          f"{m[HUMAN_YES][LLM_NO]:>11}")
    print(f"  person: not a dog    {m[HUMAN_NO][LLM_YES]:>8}   "
          f"{m[HUMAN_NO][LLM_NO]:>11}")
    print(f"\ndogs it denied      {_pct(s['missed'])}")
    print(f"dogs it invented    {_pct(s['invented'])}")
    print(f"replies unreadable  {_pct(s['unparsed_rate'])}")
    print(f"no usable answer    {_pct(s['no_answer_rate'])}")
    if s['errors_why']:
        print('  ' + ', '.join(f'{k} {v}' for k, v in
                               sorted(s['errors_why'].items())))
        if s['errors_why'].get('budget'):
            print('  a crop lost to the token budget is not a crop lost at '
                  'random: the model\n  thinks longest about the ambiguous '
                  'ones, so this loses the hard end')
    print(f"\n  agreement {_pct(s['agreement'])}")
    print('  -- read that last. It mixes the two errors above into one '
          'number and\n     moves with the mix of the sample, and these '
          'pools were chosen, not\n     sampled: they are the crops that '
          'already fooled a model.')
    if len(s['by_source']) > 1:
        print('\n  by pool')
        for r in s['by_source']:
            print(f"    {r['source']:<16} dog->yes {r[HUMAN_YES][LLM_YES]:>4} "
                  f"dog->no {r[HUMAN_YES][LLM_NO]:>4}   "
                  f"notdog->yes {r[HUMAN_NO][LLM_YES]:>4} "
                  f"notdog->no {r[HUMAN_NO][LLM_NO]:>4}   "
                  f"unreadable {r['unparsed']:>3}")
    return 0


def _cmd_disagreements(args):
    rows = disagreements(args.source, version=args.version, limit=args.limit)
    if not rows:
        print('none under prompt ' + args.version)
        return 0
    print(f'{len(rows)} crops where the model and a person differ\n')
    for r in rows:
        print(f"  {r['direction']:<9} {r['source']:<16} {r['crop']}")
        if r.get('reply'):
            print('             ' + r['reply'].replace('\n', ' ')[:120])
    return 0


def _cmd_unparsed(args):
    rows = unparsed(args.source, version=args.version, limit=args.limit)
    print(f'{len(rows)} replies that were not an answer\n')
    for r in rows:
        print(f"  {r['source']:<16} {r['crop']}")
        print('    ' + (r.get('reply') or '').replace('\n', ' ')[:160])
    return 0


def _cmd_ask(args):
    """One crop, one call. For checking the endpoint still answers, without
    starting a run and without writing anything down."""
    rec = ask(args.path, model=args.model)
    if args.quiet:
        rec.pop('reply', None)
    print(json.dumps(rec, indent=2))
    return 0


def _cmd_stop(args):
    request_stop()
    print('stop requested; a run halts after the call in flight')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='EXPERIMENTAL LLM annotator. Its answers are not labels.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('sources', help='the pools and what has been asked')
    s.set_defaults(fn=_cmd_sources)

    r = sub.add_parser('run', help='annotate n crops from one pool')
    r.set_defaults(fn=_cmd_run)
    r.add_argument('--source', default=DEFAULT_SOURCE, choices=sorted(SOURCES))
    r.add_argument('--n', type=int, default=10)
    r.add_argument('--seed', type=int, default=7)
    r.add_argument('--sleep', type=float, default=0.5,
                   help='pause between calls; the endpoint publishes no rate '
                        'limit, so err slow')
    r.add_argument('--tries', type=int, default=4)
    r.add_argument('--model', default=MODEL)
    r.add_argument('--dry-run', action='store_true',
                   help='print the crops it would ask about and spend nothing')
    r.add_argument('--allow-unjudged', action='store_true',
                   help='phase 2: a pool with no human answers, which yields '
                        'a queue-ordering hint and never a label')

    c = sub.add_parser('calibration', help='agreement, both directions apart')
    c.set_defaults(fn=_cmd_calibration)
    c.add_argument('--source', default=None, choices=sorted(SOURCES))
    c.add_argument('--model', default=MODEL)
    c.add_argument('--version', default=PROMPT_VERSION,
                   help='count an older prompt instead; it will not be '
                        'pooled with the current one either way')

    d = sub.add_parser('disagreements', help='where it and a person differ')
    d.set_defaults(fn=_cmd_disagreements)
    d.add_argument('--source', default=None, choices=sorted(SOURCES))
    d.add_argument('--version', default=PROMPT_VERSION)
    d.add_argument('--limit', type=int, default=0)

    u = sub.add_parser('unparsed', help='replies that were not an answer')
    u.set_defaults(fn=_cmd_unparsed)
    u.add_argument('--source', default=None, choices=sorted(SOURCES))
    u.add_argument('--version', default=PROMPT_VERSION)
    u.add_argument('--limit', type=int, default=0)

    a = sub.add_parser('ask', help='one crop, one call, nothing written')
    a.set_defaults(fn=_cmd_ask)
    a.add_argument('path')
    a.add_argument('--model', default=MODEL)
    a.add_argument('--quiet', action='store_true')

    p = sub.add_parser('stop', help='halt a run')
    p.set_defaults(fn=_cmd_stop)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
