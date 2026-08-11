#!/usr/bin/env python3
"""The /llm page: what an LLM said about a crop, and whether it can be trusted.

Serving side of tools/detect/llm_annotate.py. That module asks a
general-purpose model whether a crop holds a dog and keeps its answers in a
store of its own; everything here turns those answers into a page, runs a
batch in the background so the handler is free, and serves the crops a
disagreement is about.

WHAT THIS PAGE IS FOR, AND IT IS ONE QUESTION. Can this annotator be trusted
yet? So the calibration is the hero and the run controls are a toolbar. A
page built the other way round -- a big Start button and the numbers folded
away underneath -- would be an interface for spending a free tier, and the
only reason to spend it is to make those numbers mean something.

THE THIRD TIER, AND WHY IT IS EVERYWHERE IN HERE. This project already keeps
two apart: a human verdict is ground truth, and a model's score is for
filtering the review queue and may never become a label. An answer from an
LLM is below both, and the whole risk of putting one on a screen is that it
comes to look like the other two. So on this page it never wears their
clothes. What a person said is drawn in the palette the review and audit
pages use for a verdict; what the model said is drawn in violet, which
appears nowhere else in the dashboard, and is spelled with the store's own
words -- `llm_yes`, `llm_no`, `llm_unparsed` -- rather than translated into
'dog' and 'not a dog', which are two words that in this repo mean a person
looked and said so.

THIS MODULE WRITES NOTHING. Not a ledger, not a thumbnail, not a cache. Every
route here is a read, except the two that start and stop a run -- and those
call llm_annotate, which owns the only file that is ever appended to and
refuses any path outside its own store. There is no route that promotes
anything into a dataset and there is no disabled button implying one click
would do it: promoting an LLM answer is a decision a person makes by hand,
later, and the page says so in words instead of gesturing at it with a
control.

THE CROPS ARE SERVED THROUGH THE POOL, NOT THROUGH A PATH. A tile asks for a
source and a crop key. The source is checked against llm_annotate.SOURCES and
the key is LOOKED UP in that pool's dictionary -- nothing a client sends is
ever joined onto a directory, which is the same door audit.py uses and for
the same reason. The pools are the tiny cut crops (a median of about 1 KB in
the flag stores, 7 KB in the audit's), so they are served as they sit on disk
rather than thumbnailed: a cache directory would be a new store under data/
for pictures that are already smaller than the thumbnails would be.

A RUN IS A THREAD, AND THE THREAD IS JOINED. Fifty calls is five minutes, so
a run cannot happen inside the handler; it happens on a background thread and
the page polls. The thread is a daemon AND is signalled and joined at exit,
because this repo has already shipped the other version of that: a daemon
thread still holding files when the interpreter tears down took duckdb's
static teardown with it, and a script that rendered one page and returned
died of SIGABRT reporting failure about work that had succeeded.
"""

import atexit
import json
import os
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DETECT = os.path.join(REPO, 'tools', 'detect')
if _DETECT not in sys.path:
    sys.path.insert(0, _DETECT)
import llm_annotate as llm                                      # noqa: E402

# The batch sizes the page offers, and a ceiling that does not come off the
# interface. This is a free tier nobody has published a rate limit for, so
# the select is the polite half and RUN_MAX is the half that holds: `n`
# arrives off a request and a hand-made one asking for five thousand would
# otherwise be honoured, quietly, for nine hours.
RUN_SIZES = (5, 10, 25, 50)
RUN_DEFAULT = 10
RUN_MAX = 50
# Roughly what one call costs, measured over the records on file. Only used to
# tell a reader how long a batch will take before they start it -- a five
# minute wait that announces itself is a choice, and one that does not is a
# hang.
SECONDS_PER_CALL = 6.5

# The seed is not a control. sample() already excludes every crop this prompt
# version has answered, so a second seed would only reorder what is left, and
# a knob whose two settings cannot be told apart from the outside is a knob
# that gets pressed to see what it does -- on a metered endpoint.
RUN_SEED = 7

# How many disagreement tiles one answer carries. Every row holds the model's
# reply, so this is the size of the response as much as the size of the grid;
# 2,600 crops all disagreeing would be a megabyte of prose the page draws
# below the fold and nobody scrolls to.
GRID_MAX = 120
UNPARSED_MAX = 60
# The last few calls, kept in memory so a run has something to show between
# the counters moving. Anything longer is a log, and the ledger is the log.
RECENT_MAX = 8

# A crop is a crop: about 1 KB in the flag pools and 7 KB in the audit's, and
# 980 px on the longest side anything on disk here reaches. The ceiling is for
# the day something else lands in one of those directories -- a whole frame
# read into memory before it is written to the socket is how a dashboard
# serving pictures loses its memory to one file.
CROP_MAX = 8 << 20


def run_size(v, default=RUN_DEFAULT):
    """One of the offered batch sizes, or the default. Never a free number."""
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return v if v in RUN_SIZES else default


def _runnable(row):
    """True if a page may start a run on this pool.

    Both flags, because they mean different things and only one of them is
    about phase 2: `human` says the pool carries answers to compare against,
    `enabled` says the table has switched it on. llm_annotate.run() refuses
    the same pair unless it is asked twice; this is the same refusal made one
    layer earlier so the message can be about the page rather than a
    SystemExit raised on a worker thread where nobody would see it.
    """
    return bool(row.get('human')) and bool(row.get('enabled'))


# ── the picture ─────────────────────────────────────────────────────────────

def crop(source, key):
    """(jpeg bytes, content type) for one crop, or (None, None).

    The one door to the filesystem on this page. The source has to be a pool
    this repo knows about and the key has to already be IN that pool -- it is
    a dictionary lookup, so no string a client sends is ever joined onto a
    directory, and a key naming a file that is not one of these crops cannot
    be built. audit.py reaches the same conclusion by matching the key against
    the shape it is minted in; a lookup is the same rule where a dictionary
    already exists.

    (None, None) covers every way this fails -- an unknown pool, a key that
    has left it, a crop deduplicated off the disk since the ledger recorded
    it -- and the caller answers 404 to all of them. The tile then draws as a
    missing picture, which is a fact about the store and belongs on the page.
    """
    if source not in llm.SOURCES:
        return None, None
    try:
        got = llm.pool(source).get(str(key or ''))
    except Exception:
        return None, None       # a pool that will not read is a missing tile
    if not got:
        return None, None
    path = got[1]
    # The pools are cut as jpeg by the tools that made them, but checked
    # rather than assumed: a pool that one day holds a png should be a missing
    # tile, not a png body served under a jpeg header.
    if not str(path).lower().endswith(('.jpg', '.jpeg')):
        return None, None
    try:
        if os.path.getsize(path) > CROP_MAX:
            return None, None
        with open(path, 'rb') as fh:
            return fh.read(), 'image/jpeg'
    except OSError:
        return None, None


# ── a run, on a thread of its own ───────────────────────────────────────────
# llm_annotate.run() blocks for six seconds a crop and is the whole point of
# the page, so it cannot happen inside the handler. The state below is this
# PROCESS's memory of the run it started; the run's own progress lives in
# llm_annotate's status file, which survives a restart and is the authority
# for what is happening. The two are merged in api_status(), carefully.

_lock = threading.Lock()
# Only what the status file cannot answer. The pool, the counts, how far it
# has got and whether it halted are all in there and are read from there --
# a second copy here would be a second answer to the same question, and the
# one this process happens to remember is the one that is wrong after a
# restart.
_run = {
    'thread': None,
    'busy': False,
    'started_at': 0.0,      # when this process launched it, not when it began
    'tokens': 0,
    'recent': [],
    'error': None,
}
# Set by the exit hook so a run in flight stops between crops instead of at
# the end of the batch.
_exiting = threading.Event()


def _stop_now():
    """The run's stop condition: the user asked, or the process is going.

    llm_annotate.run() checks this between crops, so either lands within about
    one call. Passed in rather than left to the default, which watches only
    the stop file -- an interpreter shutting down would otherwise wait out
    however much of a fifty-crop batch was left.
    """
    return _exiting.is_set() or llm.stop_requested()


def _shutdown():
    """Ask a run to stop, and wait for the call in flight to land.

    Bounded, because the call in flight can be a 120 second HTTP timeout and
    no exit should wait that long -- but long enough to usually win: the calls
    measured here took 2.9, 5.4 and 9.9 seconds, and a 5 second join lost the
    record every time it was tested. The wait only happens when a run is
    actually going, which is a state somebody deliberately started.

    If the call does outlast the join the thread is a daemon and the
    interpreter goes on without it. That was tested too, and it costs exactly
    one thing: the answer in flight is not written, so the crop is not marked
    as asked and comes back in the next sample. The ledger is opened and
    closed around every single record, so what is on disk is whole either way
    -- which is the half of this that matters, since the version this replaces
    left a daemon thread holding files through interpreter teardown and took
    duckdb's with it.
    """
    _exiting.set()
    t = _run.get('thread')
    if t is not None and t.is_alive():
        t.join(timeout=8.0)


atexit.register(_shutdown)


def _worker(source, n):
    """One batch, on the thread. Everything it learns goes through the lock."""

    def on_record(rec):
        # Tokens are the one cost the status file does not carry -- run()
        # counts outcomes, and the token count lives on each record. This is
        # the only place it can be added up without re-reading the ledger
        # after every call.
        with _lock:
            _run['tokens'] += int(rec.get('tokens') or 0)
            _run['recent'].insert(0, {
                'key': rec.get('key'), 'source': rec.get('pool'),
                'llm_says': rec.get('llm_says'), 'ms': rec.get('ms'),
                'tokens': rec.get('tokens'), 'why': rec.get('why'),
            })
            del _run['recent'][RECENT_MAX:]

    try:
        llm.run(source, n=n, seed=RUN_SEED, on_record=on_record,
                stop=_stop_now)
    except BaseException as e:                             # noqa: BLE001
        # SystemExit included, deliberately: that is what run() raises for a
        # pool it will not touch, and on a worker thread it would otherwise
        # end the thread in silence. Whatever the reason, the status file
        # still says running -- and status() only clears that when the PID is
        # gone, which is the dashboard and is very much alive -- so a page
        # would spin on a spinner nobody could stop. Say it stopped, and say
        # why.
        #
        # AND SAY IT IN A DOCUMENT THE PAGE WILL DRAW. The first version of
        # this wrote running=False with the reason and nothing else, which
        # took two of the page's own guards head on: paintStatus hides the
        # whole progress strip -- the `failed` line included -- when a status
        # carries neither `n` nor `done`, and api_status only merges the
        # in-process error when the status names the run this process
        # started, which needs `started`. So a batch that died 23 crops in
        # left the counters gone, the bar gone and not one word anywhere. The
        # run's shape is carried over from the live status when that status
        # is THIS run's -- not when it is the last run's, which would report
        # somebody else's 23 of 50 for a run that never began.
        with _lock:
            _run['error'] = f'{type(e).__name__}: {e}'
            since = _run['started_at']
        try:
            live = llm.status()
            # RUNNING, and ours, and not older than this run. All three: the
            # status file outlives every run, so the document sitting there
            # when a batch fails before it ever starts -- a refused pool, a
            # store somebody else holds -- is the LAST run's, and reporting
            # its "23 of 50" for a run that asked nothing is a worse lie than
            # saying nothing. Caught by driving exactly that: two failures a
            # few milliseconds apart, where the second inherited the first's
            # counters through the two-second slack below.
            mine = (live.get('running')
                    and live.get('pid') == os.getpid()
                    and (live.get('started') or 0) >= since - 2)
            llm.write_status(
                running=False, source=source, model=llm.MODEL,
                n=live.get('n') if mine else n,
                done=live.get('done') if mine else 0,
                started=(live.get('started') if mine else since) or since,
                counts=(live.get('counts') if mine else None) or {},
                failed=str(e)[:300])
        except Exception:
            pass
    finally:
        with _lock:
            _run['busy'] = False


def api_run(source=llm.DEFAULT_SOURCE, n=RUN_DEFAULT):
    """Start a batch. Answers immediately; the page polls for the rest.

    Everything a client can choose is checked here and nothing it sends
    reaches a path, a prompt or a command line: the source is one of four
    names, `n` is clamped to a size the interface offers, and the seed, the
    model and the prompt are this module's and llm_annotate's business.

    `allow_unjudged` is never passed, at all. It is the second of the two
    locks on phase 2 and it is not something a page should be able to turn --
    if the day comes that the unjudged queue is worth annotating, it is worth
    a person typing the command.
    """
    source = str(source or '')
    if source not in llm.SOURCES:
        return {'ok': False, 'msg': f'unknown pool {source!r}'}
    row = [r for r in llm.sources() if r['source'] == source][0]
    if not _runnable(row):
        return {'ok': False, 'msg':
                'Nobody has judged the crops in this pool, so nothing from '
                'it can be checked against anything. It is phase 2 and it is '
                'switched off. What it would produce orders a review queue; '
                'it is not a label.'}
    n = run_size(n)
    if n > RUN_MAX:
        n = RUN_MAX
    # THE CLAIM AND THE CHECK ARE ONE CRITICAL SECTION. They were two, with a
    # status read and a full dry run in between -- and a dry run on a cold
    # cache reads 2,653 ledger lines and stats 2,239 files, all of it with
    # busy still False. This is a ThreadingHTTPServer: two tabs pressing the
    # button land in two threads, both pass a check nothing has answered yet,
    # and both start a batch. The dashboard already fixed this exact bug once
    # for the guesser, and the comment on _SPAWN_LOCK records what it cost --
    # "two simultaneous POSTs both answered guessing started". Here it would
    # also leave _run['thread'] naming only the second thread, so the exit
    # hook joins one and abandons the other mid-call, which is the teardown
    # the join exists to prevent.
    with _lock:
        t = _run['thread']
        if _run['busy'] or (t is not None and t.is_alive()):
            return {'ok': False, 'running': True,
                    'msg': 'a run is already going'}
        _run.update(busy=True, started_at=time.time(),
                    tokens=0, recent=[], error=None)
    started = False
    try:
        # A run started from the command line is a different process holding
        # the same free tier, and two of them interleaving would double the
        # rate nobody has published a limit for. Asked twice: the status file
        # says what a run has published, and the store's lock says who holds
        # it -- a run that has claimed the store but not yet written a status
        # is invisible to the first question and not to the second.
        st = llm.status()
        if st.get('running'):
            return {'ok': False, 'running': True,
                    'msg': 'a run is already going somewhere else '
                           f"(pid {st.get('pid')})"}
        held, who = llm.running_elsewhere()
        if held:
            return {'ok': False, 'running': True,
                    'msg': f'another process holds the store (pid {who})'}
        # What the batch would actually cost, before a thread is started for
        # it. sample() drops every crop this prompt version and this model
        # have already answered, so an exhausted pool is worth saying out loud
        # rather than showing a progress bar that goes from 0 of 0 to done.
        try:
            plan = llm.run(source, n=n, dry_run=True)
        except Exception as e:
            return {'ok': False, 'msg': f'could not read the pool: {e}'}
        if not plan.get('planned'):
            return {'ok': False, 'msg':
                    f'every crop in {source} has already been asked about '
                    f'under prompt {llm.PROMPT_VERSION}'}
        with _lock:
            _run['thread'] = threading.Thread(
                target=_worker, args=(source, n), daemon=True,
                name='llm-annotate-run')
            _run['thread'].start()
        started = True
        return {'ok': True, 'source': source, 'n': n,
                'planned': plan['planned'],
                'seconds': round(plan['planned'] * SECONDS_PER_CALL)}
    finally:
        # Every path out of here that did not start a thread gives the claim
        # back, including the ones that return a refusal and the ones that
        # raise. A claim left set is a Start button that never comes back
        # until the dashboard is restarted.
        if not started:
            with _lock:
                _run['busy'] = False


def api_stop():
    """Halt a run after the call in flight.

    Refused when nothing is running, and that refusal is the point rather than
    tidiness. The stop signal is a FILE, and llm_annotate deliberately does
    not clear it when a run starts -- a stop that arrives in the moment before
    a batch begins is honoured instead of thrown away. The other side of that
    is that a stop written while nothing is running sits there and costs the
    NEXT run a no-op start, so a page that wrote one on every click of a
    disabled-looking button would arm a trap for the button beside it.
    """
    if not llm.status().get('running'):
        with _lock:
            busy = _run['busy']
        if not busy:
            return {'ok': False, 'msg': 'nothing is running'}
    llm.request_stop()
    return {'ok': True, 'msg': 'stopping after the call in flight'}


def api_status():
    """Where a run has got to: the status file, plus what this process knows.

    The status file is the authority -- it survives a restart and it carries
    the PID check that stops a killed run from claiming to be alive forever.
    The token count and the last few answers are memory, and memory is only
    true of the run THIS process started: merging them unconditionally would
    credit a batch run last night from the command line with the token count
    of one run here, which is a number that looks measured and is invented.
    So they are merged only when the file's run is the one we launched.
    """
    st = dict(llm.status())
    with _lock:
        ours = (_run['started_at']
                and st.get('pid') == os.getpid()
                and (st.get('started') or 0) >= _run['started_at'] - 2)
        if ours:
            st['tokens'] = _run['tokens']
            st['recent'] = list(_run['recent'])
            st['error'] = _run['error']
        st['busy'] = _run['busy']
    # 'stopping' rather than a stopped flag: the run halts between crops, so
    # for up to one call the page is showing a live run that has already been
    # told to stop, and a Stop button that goes back to looking pressable is
    # how somebody presses it twice.
    st['stopping'] = bool(st.get('running')) and llm.stop_requested()
    # A stop signal on file with nothing running, which llm_annotate honours on
    # the next start: that run halts having asked nothing and clears it. That
    # is deliberate over there -- a stop clicked in the moment before a batch
    # begins must not be thrown away -- and it is worth naming here, because
    # the way the file gets left behind is not a click at all. atexit does not
    # run on SIGTERM, and SIGTERM is how this dashboard is stopped, so a
    # restart during a run leaves one. Seen doing exactly that, and without a
    # word for it the next press of the button looks like a broken button.
    st['stop_pending'] = bool(llm.stop_requested()) and not st.get('running')
    st['prompt_version'] = st.get('prompt_version') or llm.PROMPT_VERSION
    st['model'] = st.get('model') or llm.MODEL
    return st


# ── what the page reads ─────────────────────────────────────────────────────

def api_overview():
    """Everything the page draws at rest: the pools, the calibration, the run.

    `error` rather than an exception. calibration() reaches into fn_audit for
    the Wilson interval and re-reads the human ledgers to look every verdict
    up again, so it has more ways to fail than the rest of the page put
    together -- and the banner, the pools and the run controls are all still
    true when it does. A 500 here would take the one paragraph that says
    these are not annotations off the screen along with the numbers.
    """
    rows = llm.sources()
    for r in rows:
        r['runnable'] = _runnable(r)
    out = {
        'prompt': {
            'version': llm.PROMPT_VERSION,
            'text': llm.PROMPT_TEXT,
            'model': llm.MODEL,
            'max_tokens': llm.MAX_TOKENS,
            'temperature': llm.TEMPERATURE,
            'prep': llm.PREP,
            'endpoint': llm.API_URL,
        },
        'sources': rows,
        'store': os.path.relpath(llm.paths()['ledger'], REPO),
        'run': {'sizes': list(RUN_SIZES), 'default': RUN_DEFAULT,
                'max': RUN_MAX, 'seconds_per_call': SECONDS_PER_CALL},
        'status': api_status(),
        'words': {'yes': llm.LLM_YES, 'no': llm.LLM_NO,
                  'unparsed': llm.LLM_UNPARSED, 'error': llm.LLM_ERROR,
                  'human_yes': llm.HUMAN_YES, 'human_no': llm.HUMAN_NO},
    }
    try:
        out['calibration'] = llm.calibration()
    except Exception as e:
        out['calibration'] = None
        out['error'] = f'the calibration could not be computed: {e}'
    return out


def api_disagreements(source=None, direction=None, limit=GRID_MAX):
    """The crops where the model and a person differ, newest first.

    Filtered to the model the calibration counts. disagreements() takes a
    prompt version and not a model, and the two numbers have to describe the
    same crops: a grid showing seven tiles beside a rate computed over five is
    a page arguing with itself, and the reader has no way to tell which half
    is wrong.
    """
    rows = [r for r in llm.disagreements(source or None)
            if r.get('model') == llm.MODEL]
    if direction in ('missed', 'invented'):
        rows = [r for r in rows if r.get('direction') == direction]
    try:
        limit = max(1, min(int(limit), GRID_MAX))
    except (TypeError, ValueError):
        limit = GRID_MAX
    return {'items': rows[:limit], 'total': len(rows), 'limit': limit,
            'prompt_version': llm.PROMPT_VERSION, 'model': llm.MODEL}


def api_unparsed(source=None, limit=UNPARSED_MAX):
    """The replies that were not an answer, so the contract can be improved.

    Same one-model rule as the grid above, reached a different way: unparsed()
    rows do not carry the model, so the ledger is read once under both filters
    and the two are intersected on (pool, key). That is one small ledger read
    rather than a second implementation of what counts as unreadable, which
    would be the copy that drifts.
    """
    mine = llm.read_ledger(version=llm.PROMPT_VERSION, model=llm.MODEL)
    rows = [r for r in llm.unparsed(source or None)
            if (r.get('source'), r.get('key')) in mine]
    try:
        limit = max(1, min(int(limit), UNPARSED_MAX))
    except (TypeError, ValueError):
        limit = UNPARSED_MAX
    return {'items': rows[:limit], 'total': len(rows), 'limit': limit}


# ── the page ────────────────────────────────────────────────────────────────
# A sibling of /audit and /datasets, wearing the same clothes: same palette,
# same cards, same controls, same type scale, numbers in the monospaced face
# for the same reason -- every figure here is read by comparison.
#
# With one addition, and it is the whole point of the page. Violet belongs to
# the LLM and to nothing else in this dashboard. What a person answered is
# drawn the way a verdict is drawn everywhere else; what the model said is
# violet, is spelled with the store's own words, and never appears without
# something naming it as the model's. A reader who learns one thing from
# this page should learn that those are two different kinds of statement.
LLM_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM annotator &mdash; experimental</title><style>
:root{--bg:#13151a;--panel:#1b2027;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;
--red:#ef5350;
/* Numbers get their own face, as on the audit and datasets pages: every rate,
   interval and count here is read against the one above it, and in a
   proportional face the digits move under the eye between rows. */
--num:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
/* The LLM's colour, and nothing else in this dashboard uses it. A verdict is
   amber or green wherever a person gives one; an answer from the model must
   never be able to pass for one at a glance. */
--llm:#9a8de0;--llmbg:rgba(154,141,224,.13);--llmbd:rgba(154,141,224,.42)}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);-webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  line-height:1.5;padding:0 22px 80px}
.wrap{max-width:1560px;margin:0 auto}
a{color:inherit}
/* ── the banner ──
   First thing on the page, above the title, and it does not fold, collapse,
   dismiss or scroll away behind anything. The one mistake this page could
   make is being read as a page of annotations. */
.xban{display:flex;gap:14px;align-items:flex-start;margin-top:18px;
  background:var(--llmbg);border:1px solid var(--llmbd);
  border-left:4px solid var(--llm);border-radius:12px;padding:13px 16px}
.xtag{flex:none;font-size:10.5px;font-weight:720;letter-spacing:.1em;
  text-transform:uppercase;background:var(--llm);color:#14121c;
  border-radius:6px;padding:3px 8px;margin-top:1px}
.xtxt{font-size:13px;color:var(--tx);max-width:104ch}
.xtxt b{font-weight:660}
.xtxt .x2{display:block;margin-top:4px;font-size:12.5px;color:var(--mut)}
.xtxt code{font-family:var(--num);font-size:11.5px;color:var(--llm)}
/* ── header ── */
header{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;
  padding:20px 0 16px;border-bottom:1px solid var(--bd);margin-bottom:16px}
h1{font-size:20px;font-weight:660;letter-spacing:-.3px}
.sub{color:var(--dim);font-size:12.5px;margin-top:3px;max-width:74ch}
.back{font-size:12px;color:var(--mut);text-decoration:none;margin-left:auto;
  border:1px solid var(--bd);border-radius:8px;padding:6px 11px}
.back:hover{color:var(--tx);border-color:rgba(130,140,150,.3)}
.card{background:var(--panel);border:1px solid var(--bd);border-radius:14px;
  margin-bottom:16px}
.chead{display:flex;gap:10px;align-items:baseline;padding:11px 15px;
  border-bottom:1px solid var(--bd);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.07em;color:var(--dim)}
.chead .n{margin-left:auto;font-family:var(--num);letter-spacing:0;
  text-transform:none;font-size:11.5px;color:var(--mut)}
.chead .n b{color:var(--tx);font-weight:640}
/* ── calibration ──
   The hero. Three blocks, and the first two are the two error directions,
   never added together: the pilot put them at 0% and 29% on one sample, and
   an accuracy number over that pair describes neither and moves whenever the
   mix of the sample moves -- and these pools were chosen, not sampled. */
.three{display:grid;grid-template-columns:repeat(auto-fit,minmax(268px,1fr));
  gap:0}
.mb{padding:17px 20px 18px;border-right:1px solid var(--bd)}
.mb:last-child{border-right:0}
.mlab{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
  color:var(--dim)}
.mbig{font-size:34px;font-weight:680;letter-spacing:-1.2px;line-height:1.15;
  margin-top:6px;font-variant-numeric:tabular-nums;font-family:var(--num)}
.mbig.none{color:var(--dim);font-size:22px;letter-spacing:-.4px}
.mfrac{font-family:var(--num);font-size:12.5px;color:var(--mut);
  font-variant-numeric:tabular-nums;margin-top:2px}
.msay{font-size:12.5px;color:var(--mut);margin-top:9px;max-width:42ch}
.msay b{color:var(--tx);font-weight:640;font-family:var(--num);
  font-variant-numeric:tabular-nums}
.mwhy{font-size:11.5px;color:var(--dim);margin-top:7px;max-width:44ch}
/* The estimate, drawn the way the audit page draws one: the 95% interval as a
   segment on a 0-100% axis and the point estimate as a tick inside it. A bare
   percentage off six crops and one off six hundred look identical, and only
   one of them is a measurement. */
.track{position:relative;height:15px;margin-top:11px}
.track::before{content:'';position:absolute;left:0;right:0;top:50%;height:1px;
  background:rgba(130,140,150,.16)}
.ci{position:absolute;top:50%;transform:translateY(-50%);height:7px;
  border-radius:4px;background:rgba(232,166,69,.3);min-width:2px}
.dot{position:absolute;top:50%;transform:translate(-50%,-50%);width:3px;
  height:13px;border-radius:2px;background:var(--acc)}
.zero .ci{background:rgba(67,181,129,.26)}
.zero .dot{background:var(--green)}
.iax{display:flex;justify-content:space-between;font-size:10px;
  color:var(--dim);font-family:var(--num);margin-top:2px}
.rests{padding:13px 20px;border-top:1px solid var(--bd);font-size:12.5px;
  color:var(--mut);display:flex;gap:22px;flex-wrap:wrap;align-items:baseline}
.rests b{color:var(--tx);font-weight:640;font-family:var(--num);
  font-variant-numeric:tabular-nums}
.warnpill{color:var(--acc);border:1px solid rgba(232,166,69,.36);
  border-radius:7px;padding:2px 8px;font-size:11.5px}
.foot2{padding:13px 20px;border-top:1px solid var(--bd);font-size:12px;
  color:var(--dim);max-width:96ch}
.foot2 b{color:var(--mut);font-weight:600;font-family:var(--num)}
.foot2+.foot2{border-top:1px solid rgba(130,140,150,.07)}
/* ── the confusion matrix ── */
.mxwrap{padding:15px 20px 17px;border-top:1px solid var(--bd);
  display:flex;gap:30px;flex-wrap:wrap;align-items:flex-start}
.mx{border-collapse:collapse;font-size:12.5px}
.mx th{font-weight:560;color:var(--dim);font-size:10.5px;
  text-transform:uppercase;letter-spacing:.07em;padding:0 10px 7px;
  text-align:center}
.mx th.rh{text-align:left;text-transform:none;letter-spacing:0;font-size:12px;
  color:var(--mut);padding:7px 14px 7px 0}
.mx th.llmh{color:var(--llm)}
.mx td{font-family:var(--num);font-variant-numeric:tabular-nums;font-size:15px;
  text-align:center;padding:7px 10px;min-width:74px;border:1px solid var(--bd);
  color:var(--mut)}
.mx td.agree{color:var(--green)}
.mx td.wrong{color:var(--acc);font-weight:640}
.mx td.nil{color:var(--dim)}
.mxnote{font-size:12px;color:var(--dim);max-width:40ch}
.mxnote b{color:var(--mut);font-weight:600}
/* by pool, because the three pools are three different questions: the audit's
   crops sit on the threshold, the flag pools are the crops that already
   fooled a model */
/* The name column is capped rather than left as a fraction. On a 1500 px card
   a 1.4fr name pushed all five counts to the far right edge, a hand's width
   from the pool they belong to, and the row read as two unrelated things. */
.pools{padding:0 20px 16px;max-width:760px}
.prow{display:grid;grid-template-columns:minmax(0,1fr) repeat(4,56px) 76px;
  gap:10px;font-size:11.5px;color:var(--mut);padding:6px 0;
  font-variant-numeric:tabular-nums;align-items:baseline}
.prow+.prow{border-top:1px solid rgba(130,140,150,.07)}
.prow.h{color:var(--dim);font-size:10px;text-transform:uppercase;
  letter-spacing:.07em;border-bottom:1px solid var(--bd);padding-bottom:7px}
/* the heading sits over its own column, right-aligned like the count beneath
   it -- left-aligned they were a few pixels off every number they name */
.prow.h span+span{text-align:right}
.prow b{color:var(--tx);font-weight:600}
.prow .num{text-align:right;font-family:var(--num)}
.prow .pn{color:var(--tx);overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
/* ── what it was asked ── */
.ask{border-top:1px solid var(--bd);padding:11px 20px 13px}
.ask summary{list-style:none;cursor:pointer;font-size:11.5px;color:var(--dim)}
.ask summary::-webkit-details-marker{display:none}
.ask summary::after{content:' \25b8'}
.ask[open] summary::after{content:' \25be'}
.ask summary:hover{color:var(--tx)}
.ask pre{margin-top:9px;background:var(--panel2);border:1px solid var(--bd);
  border-radius:10px;padding:11px 13px;font-family:var(--num);font-size:11.5px;
  color:var(--mut);white-space:pre-wrap;max-width:88ch;line-height:1.55}
.askkv{display:flex;gap:18px;flex-wrap:wrap;margin-top:9px;font-size:11.5px;
  color:var(--dim);font-family:var(--num)}
.askkv b{color:var(--mut);font-weight:600}
/* ── the run strip ──
   A toolbar, not a hero. Spending a free tier is the means; the numbers above
   are the end, and a big Start button at the top of the page would have said
   the opposite. */
.rrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px 15px}
.btn{background:var(--panel2);border:1px solid var(--bd);color:var(--mut);
  border-radius:9px;padding:7px 13px;font-size:12.5px;cursor:pointer;
  font-family:inherit}
.btn:hover:not(:disabled){color:var(--tx);border-color:rgba(130,140,150,.32)}
.btn:disabled{opacity:.4;cursor:default}
.btn.go{color:var(--llm);border-color:var(--llmbd)}
.btn.go:hover:not(:disabled){color:var(--llm);
  border-color:rgba(154,141,224,.7);background:var(--llmbg)}
.pick{display:inline-flex;align-items:center;gap:7px;font-size:11.5px;
  color:var(--dim)}
.pick select{background:var(--panel2);border:1px solid var(--bd);
  color:var(--mut);border-radius:9px;padding:7px 9px;font-size:12.5px;
  font-family:inherit;cursor:pointer;max-width:290px}
.pick select:hover{color:var(--tx)}
.rnote{font-size:11.5px;color:var(--dim);max-width:56ch}
.rnote.bad{color:var(--acc)}
.spacer{margin-left:auto}
.prog{padding:0 15px 14px}
.prog[hidden]{display:none}
.pbar{height:5px;border-radius:3px;background:rgba(130,140,150,.14);
  overflow:hidden}
.pbar i{display:block;height:100%;border-radius:3px;background:var(--llm);
  transition:width .3s ease}
/* A finished batch leaves a full bar, and a full violet bar reads at a glance
   like one that is going. Dimmed, it reads as the report it is -- the line
   under it already says "last run". */
.pbar i.over{background:rgba(154,141,224,.34)}
.pline{font-size:12px;color:var(--mut);margin-top:8px;display:flex;gap:16px;
  flex-wrap:wrap;font-variant-numeric:tabular-nums}
.pline b{color:var(--tx);font-weight:640;font-family:var(--num)}
.pline .bad{color:var(--acc)}
.recent{margin-top:8px;display:flex;gap:7px;flex-wrap:wrap}
.rchip{font-size:10.5px;font-family:var(--num);border:1px solid var(--bd);
  border-radius:6px;padding:2px 7px;color:var(--dim);background:var(--panel2)}
.rchip.y,.rchip.n{color:var(--llm);border-color:var(--llmbd)}
.rchip.u{color:var(--acc);border-color:rgba(232,166,69,.4)}
.rchip.e{color:var(--red);border-color:rgba(239,83,80,.4)}
.phase2{padding:0 15px 14px;font-size:11.5px;color:var(--dim);max-width:96ch}
.phase2 b{color:var(--mut);font-weight:600}
/* ── the disagreements ── */
.dbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:11px 15px;
  border-bottom:1px solid var(--bd)}
.views{display:inline-flex;gap:2px;padding:2px;border:1px solid var(--bd);
  border-radius:10px}
.viewbtn{appearance:none;background:transparent;border:0;color:var(--dim);
  border-radius:8px;padding:6px 11px;font-size:12px;cursor:pointer;
  font-family:inherit}
.viewbtn:hover{color:var(--tx)}
.viewbtn.on{background:rgba(232,166,69,.15);color:var(--acc);font-weight:640}
.viewbtn b{font-family:var(--num);font-weight:640;opacity:.75;margin-left:5px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));
  gap:12px;padding:14px 15px}
.tile{background:var(--panel2);border:1px solid var(--bd);border-radius:11px;
  overflow:hidden}
.shot{position:relative;background:#0e1014;aspect-ratio:1;display:flex;
  align-items:center;justify-content:center;cursor:zoom-in}
/* Filled rather than shown at natural size, and image-rendering is left
   alone. These crops have a median long side of about 50 px: at natural size
   the tile is a thumbnail floating in a black square, and nobody can tell
   from it whether the model was right. The model was not shown them at
   natural size either -- llm_annotate upscales to 384 px with a smooth filter
   before sending -- so a smooth upscale here is the nearer picture of the
   question that was actually asked. */
.shot img{width:100%;height:100%;object-fit:contain;display:block}
.dirtag{position:absolute;left:6px;top:6px;font-size:10px;border-radius:6px;
  padding:2px 7px;letter-spacing:.03em;background:rgba(10,12,16,.86);
  border:1px solid rgba(232,166,69,.42);color:var(--acc)}
.verd{display:grid;grid-template-columns:1fr;gap:3px;padding:7px 8px 8px}
.who{font-size:10.5px;border-radius:6px;padding:2px 7px;
  border:1px solid var(--bd);color:var(--mut);white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.who i{font-style:normal;color:var(--dim);margin-right:5px}
.who.hy{border-color:rgba(232,166,69,.4);color:var(--acc)}
.who.hn{border-color:rgba(67,181,129,.36);color:var(--green)}
/* every one of these carries the store's own word, in the monospaced face,
   in the colour that belongs to the model and to nothing else */
.who.mm{border-color:var(--llmbd);color:var(--llm);background:var(--llmbg);
  font-family:var(--num);font-size:10px}
.empty{color:var(--dim);font-size:12.5px;padding:34px 18px;text-align:center;
  line-height:1.6;max-width:70ch;margin:0 auto}
.empty b{color:var(--mut);font-weight:600}
.more{padding:0 15px 14px;font-size:11.5px;color:var(--dim)}
/* ── the unreadable replies ── */
.ulist{padding:4px 15px 14px}
.urow{display:grid;grid-template-columns:56px minmax(0,1fr);gap:12px;
  padding:9px 0;align-items:start}
.urow+.urow{border-top:1px solid rgba(130,140,150,.07)}
.urow img{width:56px;height:56px;object-fit:contain;background:#0e1014;
  border-radius:8px;border:1px solid var(--bd)}
.utxt{font-size:12px;color:var(--mut);white-space:pre-wrap;
  word-break:break-word}
.umeta{font-size:10.5px;color:var(--dim);font-family:var(--num);margin-top:4px}
/* ── the closing sentence ── */
.close{font-size:12.5px;color:var(--mut);padding:15px 18px;max-width:96ch;
  background:var(--panel);border:1px solid var(--bd);
  border-left:4px solid var(--llm);border-radius:12px}
.close b{color:var(--tx);font-weight:640}
/* ── lightbox ── */
.lb{position:fixed;inset:0;background:rgba(0,0,0,.9);display:flex;
  align-items:center;justify-content:center;flex-direction:column;gap:14px;
  z-index:50;padding:24px;overflow:auto}
.lb[hidden]{display:none}
/* A BOX, not a ceiling. max-width/max-height left a 46 px crop at 46 px in
   the middle of a black screen, and the only reason to open this is to look
   at the crop and decide whether the model was wrong or you were. It is
   upscaled smoothly because that is how the model was shown it: llm_annotate
   sends every crop resampled up to 384 px. */
.lb img{width:min(52vh,560px);height:min(52vh,560px);object-fit:contain;
  background:#0e1014;border:1px solid var(--bd);border-radius:12px}
.lbcard{background:var(--panel);border:1px solid var(--bd);border-radius:12px;
  padding:14px 16px;max-width:min(760px,92vw);font-size:12.5px;
  color:var(--mut)}
.lbv{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:9px}
.lbreply{color:var(--tx);white-space:pre-wrap;font-size:12.5px;
  line-height:1.55;border-left:2px solid var(--llmbd);padding-left:11px}
.lbmeta{font-size:11px;color:var(--dim);font-family:var(--num);margin-top:10px;
  display:flex;gap:14px;flex-wrap:wrap}
.lbcap{font-size:12px;color:var(--mut);display:flex;gap:10px;
  align-items:center}
.lbcap button{background:var(--panel2);border:1px solid var(--bd);
  color:var(--mut);border-radius:7px;padding:4px 9px;font-size:11.5px;
  cursor:pointer;font-family:inherit}
.lbcap button:hover{color:var(--tx)}
.toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);
  background:var(--panel2);border:1px solid var(--bd);border-radius:9px;
  padding:8px 14px;font-size:12.5px;color:var(--tx);z-index:60}
.toast[hidden]{display:none}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
@media(max-width:900px){
  .mb{border-right:0;border-bottom:1px solid var(--bd)}
  .mb:last-child{border-bottom:0}
}
</style></head><body><div class="wrap">

<div class="xban">
  <span class="xtag">experimental</span>
  <div class="xtxt"><b>Nothing on this page is an annotation.</b> These are
    guesses made by a general-purpose LLM that nobody here trained and nobody
    here has measured, and they are worth exactly as much as the intervals
    below say they are. They are not human answers, they do not enter any
    dataset, they are not training labels, and no number the audit reports is
    computed from them.
    <span class="x2">Everything the model says is kept in
      <code id="storepath">data/llm_annotations/llm_guesses.jsonl</code> and
      nowhere else. There is no control on this page that promotes any of it
      into anything.</span></div>
</div>

<header>
  <div><h1>LLM annotator</h1>
    <div class="sub">Phase 1, and it is the only phase switched on: show the
      model crops you have <em>already</em> judged and measure how often it
      agrees with you. Until these numbers say otherwise, that is all this
      is.</div></div>
  <a class="back" href="/">&larr; dashboard</a>
</header>

<section class="card">
  <div class="chead"><span>can it be trusted yet?</span>
    <span class="n" id="calwho">&mdash;</span></div>
  <div id="calbody">
    <div class="empty">Reading the store&hellip;</div>
  </div>
  <details class="ask" id="askfold">
    <summary>the exact question it was asked</summary>
    <pre id="prompttext"></pre>
    <div class="askkv" id="promptkv"></div>
  </details>
</section>

<section class="card">
  <div class="chead"><span>ask it about more crops</span>
    <span class="n" id="runwho">&mdash;</span></div>
  <div class="rrow">
    <label class="pick">pool <select id="src"></select></label>
    <label class="pick">crops <select id="n">__SIZEOPTS__</select></label>
    <button class="btn go" id="start">ask the model</button>
    <button class="btn" id="stop" disabled>stop</button>
    <span class="rnote" id="rnote"></span>
    <span class="rnote bad" id="stopnote" hidden></span>
  </div>
  <div class="prog" id="prog" hidden>
    <div class="pbar"><i id="pfill" style="width:0"></i></div>
    <div class="pline" id="pline"></div>
    <div class="recent" id="recent"></div>
  </div>
  <div class="phase2" id="phase2"></div>
</section>

<section class="card">
  <div class="chead"><span>where it and you differ</span>
    <span class="n" id="dwho">&mdash;</span></div>
  <div class="dbar">
    <div class="views" id="dirsel">
      <button class="viewbtn on" data-dir="">both</button>
      <button class="viewbtn" data-dir="invented">dogs it invented</button>
      <button class="viewbtn" data-dir="missed">dogs it denied</button>
    </div>
    <label class="pick">pool <select id="dsrc"></select></label>
    <span class="spacer"></span>
    <span class="rnote">click a crop for the model&rsquo;s own words</span>
  </div>
  <div class="grid" id="dgrid"></div>
  <div class="empty" id="dempty" hidden></div>
  <div class="more" id="dmore" hidden></div>
</section>

<section class="card" id="ucard" hidden>
  <div class="chead"><span>replies that were not an answer</span>
    <span class="n" id="uwho">&mdash;</span></div>
  <div class="ulist" id="ulist"></div>
  <div class="more">The fix for these is in the wording, and editing the
    wording is the edit that bumps the prompt version &mdash; after which
    these answers stop being counted with the new ones.</div>
</section>

<div class="close"><b>Promotion is a decision you make by hand.</b> If the
  agreement above ever looks good enough, moving any of this into a dataset is
  something you do deliberately, in a commit, on a day you choose. There is no
  button for it here and there is not going to be one &mdash; a control that
  did it in a click is exactly how an experimental guess ends up in a training
  set, and the whole point of keeping this store separate is that it cannot.
</div>

</div>

<div class="lb" id="lb" hidden>
  <img id="lbimg" alt="">
  <div class="lbcard">
    <div class="lbv" id="lbv"></div>
    <div class="lbreply" id="lbreply"></div>
    <div class="lbmeta" id="lbmeta"></div>
  </div>
  <div class="lbcap"><span id="lbtxt"></span>
    <button id="lbclose">close</button></div>
</div>
<div class="toast" id="toast" hidden></div>

<script>
var BOOT=__BOOT__;
var over=null,dis=null,dir='',dsrc='',poll=null,wasRunning=false;
var $=function(id){return document.getElementById(id)};

function esc(s){var d=document.createElement('div');d.textContent=s==null?'':s;
  return d.innerHTML}
/* esc() is a text-node round trip and a text node has no quotes to escape, so
   it is safe between tags and unsafe inside an attribute. Crop keys come off
   a ledger and a reply is 400 characters the model wrote; both end up in a
   title="" here. Same helper, same reason, as the review and datasets pages. */
function att(s){return esc(s).replace(/"/g,'&quot;').replace(/'/g,'&#39;')}
function fmtn(n){return (+n||0).toLocaleString('en-US')}
function pct(v){var x=(+v||0)*100;
  return (x>=99.95||x<=0.05?Math.round(x):x.toFixed(1))+'%'}
function q(o){var p=[],k;
  for(k in o)p.push(k+'='+encodeURIComponent(o[k]));return p.join('&')}
function cropSrc(r){return '/llm/crop?'+q({source:r.source,key:r.key})}
function toast(t){var e=$('toast');e.textContent=t;e.hidden=false;
  clearTimeout(e._t);e._t=setTimeout(function(){e.hidden=true},2600)}
function plural(n,one,many){return (+n===1)?one:many}
/* minutes, for a wait somebody is about to agree to */
function mins(s){s=Math.round(+s||0);
  if(s<90)return s+' s';
  return Math.round(s/60)+' min'}

/* ── the calibration ──────────────────────────────────────────────────────
   Three blocks and never one accuracy number. The two errors are different
   errors: denying a dog loses the crop, inventing one costs a click, and the
   pilot put them at 0% and 29% on the same sample. */
function bar(r){
  if(!r||!r.n)return '';
  function at(v){var x=(+v||0)*100;
    return Math.max(0,Math.min(100,x!==x?0:x))}
  var lo=at(r.lo95),hi=at(r.hi95),m=at(r.rate);
  return '<div class="track'+(r.k?'':' zero')+'" title="the bar is the 95% '+
    'interval, the tick is the estimate — '+r.k+' of '+r.n+'">'+
    '<div class="ci" style="left:'+lo.toFixed(2)+'%;width:'+
      Math.max(0.7,hi-lo).toFixed(2)+'%"></div>'+
    '<div class="dot" style="left:'+m.toFixed(2)+'%"></div></div>'+
    '<div class="iax"><span>0%</span><span>'+pct(r.lo95)+' &ndash; '+
    pct(r.hi95)+'</span><span>100%</span></div>';
}
/* `nothing` is what the block says with no crops behind it, `why` is the line
   under the sentence when there are. Both are arguments rather than something
   spliced into the returned string afterwards: the first version appended the
   second one with a replace() on '</div></div>', which matched inside the
   interval axis instead of at the end and drew that paragraph over the
   sentence it was meant to follow. */
function block(lab,r,say,nothing,why){
  if(!r||!r.n)return '<div class="mb"><div class="mlab">'+lab+'</div>'+
    '<div class="mbig none">nothing to measure</div>'+
    '<div class="mwhy">'+nothing+'</div></div>';
  return '<div class="mb"><div class="mlab">'+lab+'</div>'+
    '<div class="mbig">'+pct(r.rate)+'</div>'+
    '<div class="mfrac">'+fmtn(r.k)+' of '+fmtn(r.n)+'</div>'+
    bar(r)+
    '<div class="msay">'+say+'</div>'+
    (why?'<div class="mwhy">'+why+'</div>':'')+'</div>';
}
/* What is on file under some other prompt or model, and it is two numbers on
   purpose. Records is how much is in the ledger; answered is how much of it
   was ever an answer -- the version of this line that printed one number
   printed "3 answers" over three records that were all llm_error. */
function othertxt(o){
  if(o==null)return '0 records';
  if(typeof o!=='object')return fmtn(o)+' '+plural(o,'record','records');
  return fmtn(o.records)+' '+plural(o.records,'record','records')+', '+
    fmtn(o.answered)+' answered';
}
function matrix(c){
  var m=c.matrix,hy=BOOT.words.human_yes,hn=BOOT.words.human_no,
      ly=BOOT.words.yes,ln=BOOT.words.no;
  function cell(v,ok){return '<td class="'+(v?(ok?'agree':'wrong'):'nil')+
    '">'+fmtn(v)+'</td>'}
  return '<div class="mxwrap"><table class="mx">'+
    '<tr><th></th><th class="llmh">'+esc(ly)+'</th>'+
      '<th class="llmh">'+esc(ln)+'</th></tr>'+
    '<tr><th class="rh">you said <b>dog</b></th>'+
      cell(m[hy][ly],true)+cell(m[hy][ln],false)+'</tr>'+
    '<tr><th class="rh">you said <b>not a dog</b></th>'+
      cell(m[hn][ly],false)+cell(m[hn][ln],true)+'</tr>'+
    '</table>'+
    '<div class="mxnote">Down the side is what a person answered, which is the '+
    'ground truth here. Across the top is what the model replied, in the '+
    'store&rsquo;s own words &mdash; <b>'+esc(ly)+'</b> and <b>'+esc(ln)+
    '</b>, never &ldquo;dog&rdquo; and &ldquo;not_dog&rdquo;, because those '+
    'two words in this repo mean a person looked and said so.</div></div>';
}
function pools(c){
  var hy=BOOT.words.human_yes,hn=BOOT.words.human_no,
      ly=BOOT.words.yes,ln=BOOT.words.no,rows=c.by_source||[];
  if(rows.length<2)return '';
  return '<div class="pools">'+
    '<div class="prow h"><span>pool</span><span class="num">dog&rarr;yes</span>'+
    '<span class="num">dog&rarr;no</span><span class="num">not&rarr;yes</span>'+
    '<span class="num">not&rarr;no</span><span class="num">unreadable</span>'+
    '</div>'+
    rows.map(function(r){
      return '<div class="prow" title="'+att(r.note||'')+'">'+
        '<span class="pn">'+esc(r.label||r.source)+'</span>'+
        '<span class="num">'+fmtn(r[hy][ly])+'</span>'+
        '<span class="num">'+fmtn(r[hy][ln])+'</span>'+
        '<span class="num">'+fmtn(r[hn][ly])+'</span>'+
        '<span class="num">'+fmtn(r[hn][ln])+'</span>'+
        '<span class="num">'+fmtn(r.unparsed)+'</span></div>';
    }).join('')+'</div>';
}
function paintCal(){
  var c=over&&over.calibration,body=$('calbody');
  $('calwho').innerHTML='prompt <b>'+esc(BOOT.prompt.version)+'</b> &middot; '+
    esc(BOOT.prompt.model);
  if(over&&over.error){
    body.innerHTML='<div class="empty">'+esc(over.error)+'</div>';return;
  }
  if(!c||(!c.parsed&&!c.unparsed&&!c.errors)){
    var others='';
    if(c&&c.others&&Object.keys(c.others).length)
      others=' The ledger does hold records from '+
        Object.keys(c.others).map(function(k){
          return esc(k)+' ('+othertxt(c.others[k])+')'}).join(', ')+
        ', which are not counted here: a prompt is a question, and two '+
        'questions do not average.';
    body.innerHTML='<div class="empty">Nothing has been answered under '+
      'prompt <b>'+esc(BOOT.prompt.version)+'</b> yet. Ask it about a few '+
      'dozen crops you have already judged and the two error rates fill in '+
      'here.'+others+'</div>';
    return;
  }
  var m=c.matrix,hy=BOOT.words.human_yes,hn=BOOT.words.human_no,
      ly=BOOT.words.yes,ln=BOOT.words.no,
      dogs=m[hy][ly]+m[hy][ln],nots=m[hn][ly]+m[hn][ln];
  var missed=block('dogs it denied',c.missed,
    'Of the <b>'+fmtn(dogs)+'</b> '+plural(dogs,'crop','crops')+
    ' you called a dog, it agreed with <b>'+fmtn(m[hy][ly])+'</b> and said '+
    'there was no dog in <b>'+fmtn(m[hy][ln])+'</b>.',
    'You have not shown it a crop you called a dog yet, under this prompt.',
    /* The two errors do not cost the same thing, which is the whole reason
       they are two blocks. The audit page draws the same distinction one
       model down the pipeline. */
    'A dog it denies is a dog thrown away: nothing downstream would ever '+
    'see that crop again.');
  var inv=block('dogs it invented',c.invented,
    'Of the <b>'+fmtn(nots)+'</b> '+plural(nots,'crop','crops')+
    ' you called not a dog, it invented a dog in <b>'+fmtn(m[hn][ly])+
    '</b>.',
    'You have not shown it a crop you called not-a-dog yet, under this '+
    'prompt.',
    'A dog it invents costs somebody a click. That is the cheaper error, '+
    'and the pilot said it is the common one.');
  /* The unparsed rate is a headline and not a footnote. A fifth of the
     pilot's replies were not an answer, and an agreement number computed
     after dropping those is a number about the crops the model happened to
     answer cleanly -- which is not the set anybody asked about. */
  var un=c.unparsed_rate,unsay='';
  if(un&&un.n)unsay='<b>'+fmtn(un.k)+'</b> of <b>'+fmtn(un.n)+'</b> replies '+
    'could not be read as an answer'+
    (c.errors?', and <b>'+fmtn(c.errors)+'</b> '+plural(c.errors,'call','calls')+
      ' never got a reply at all':'')+'.';
  var unwhy=un&&un.k
    ? 'These are not noes. Every rate beside this one is computed over the '+
      'replies that <em>were</em> answers, so read it knowing that share was '+
      'dropped.'
    : (c.errors?'':'Every reply so far parsed. That is what the answer '+
       'contract is for, and it is the first thing to check after any edit '+
       'to the prompt.');
  var unblock=block('replies it did not answer',un,unsay,
    'Nothing has been asked under this prompt yet.',unwhy);
  var why=c.errors_why||{},whytxt=Object.keys(why).map(function(k){
    return esc(k)+' '+fmtn(why[k])}).join(', ');
  var small=(c.parsed||0)<100;
  var rests='<div class="rests">'+
    '<span>this rests on <b>'+fmtn(c.parsed)+'</b> '+
      plural(c.parsed,'crop','crops')+' carrying a human answer</span>'+
    (small?'<span class="warnpill">at this size an interval moves a long '+
      'way on a single answer</span>':'')+
    (whytxt?'<span>failures: '+whytxt+'</span>':'')+'</div>';
  var budget=why.budget
    ? '<div class="foot2">'+fmtn(why.budget)+' '+
      plural(why.budget,'call was','calls were')+' lost to the token budget, '+
      'and that is not a random loss: the model thinks longest about the '+
      'ambiguous crop, so a budget that runs out drops the hard end of the '+
      'sample and flatters whatever is measured on the rest.</div>'
    : '';
  var ag=c.agreement,agree=ag&&ag.n
    ? '<div class="foot2">Overall it agreed with you on <b>'+pct(ag.rate)+
      '</b> of the '+fmtn(ag.n)+' answered crops &mdash; read that last, and '+
      'not on its own. It averages the two errors above into a number that '+
      'describes neither, and it moves with the mix of the sample. This mix '+
      'was chosen rather than sampled: these pools are the crops that already '+
      'fooled a model or sat on a threshold.</div>'
    : '';
  var oth=c.others&&Object.keys(c.others).length
    ? '<div class="foot2">Also on file, and not counted here: '+
      Object.keys(c.others).map(function(k){
        return '<b>'+esc(k)+'</b> ('+othertxt(c.others[k])+')'}).join(', ')+
      '. A prompt is a question, and two questions do not average.</div>'
    : '';
  body.innerHTML='<div class="three">'+missed+inv+unblock+'</div>'+rests+
    matrix(c)+pools(c)+budget+agree+oth;
}

/* ── what it was asked ── */
function paintPrompt(){
  var p=BOOT.prompt;
  $('prompttext').textContent=p.text;
  $('promptkv').innerHTML=
    '<span>model <b>'+esc(p.model)+'</b></span>'+
    '<span>version <b>'+esc(p.version)+'</b></span>'+
    '<span>max_tokens <b>'+fmtn(p.max_tokens)+'</b></span>'+
    '<span>temperature <b>'+esc(String(p.temperature))+'</b></span>'+
    /* how the crop was prepared belongs with the words: a 46 px thumbnail and
       its 384 px upscale are not the same question, which is why changing it
       changes the version */
    '<span>image <b>'+esc(p.prep)+'</b></span>'+
    (p.endpoint?'<span>endpoint <b>'+esc(p.endpoint)+'</b></span>':'');
}

/* ── the run strip ──────────────────────────────────────────────────────── */
function poolRows(){return (over&&over.sources)||[]}
function fillSources(){
  var sel=$('src'),rows=poolRows().filter(function(r){return r.runnable});
  /* The pool you picked survives the refill. Every option is rewritten here
     because the counts on them change, and a <select> whose options are
     replaced falls back to the first one -- so finishing a run on the
     confirmed dogs silently moved the control to the audit pool, and the next
     press of the button would have spent the next batch somewhere nobody
     chose. Caught by running five crops and reading the line under it. */
  var want=sel.value;
  sel.innerHTML=rows.map(function(r){
    return '<option value="'+att(r.source)+'"'+
      (r.source===want?' selected':'')+'>'+esc(r.label)+' &middot; '+
      fmtn(r.pending)+' left</option>'}).join('');
  var d=$('dsrc');
  d.innerHTML='<option value="">every pool</option>'+
    poolRows().filter(function(r){return r.human}).map(function(r){
      return '<option value="'+att(r.source)+'"'+
        (r.source===dsrc?' selected':'')+'>'+esc(r.label)+'</option>'}).join('');
  var off=poolRows().filter(function(r){return !r.runnable});
  $('phase2').innerHTML=off.map(function(r){
    return '<b>'+esc(r.label)+'</b> ('+fmtn(r.crops)+' crops) is switched '+
      'off: '+esc(r.note)+'. It is phase 2. What it would produce orders a '+
      'review queue and is not a label, and starting one is a command a '+
      'person types, not a control on a page.'}).join(' ');
  noteCost();
}
function picked(){
  var v=$('src').value,rows=poolRows();
  for(var i=0;i<rows.length;i++)if(rows[i].source===v)return rows[i];
  return null;
}
function noteCost(){
  var r=picked(),n=+$('n').value||0,el=$('rnote');
  if(!r){el.className='rnote bad';
    el.textContent='no pool with human answers is switched on';return}
  var will=Math.min(n,r.pending);
  el.className='rnote';
  el.innerHTML=will
    ? fmtn(will)+' '+plural(will,'crop','crops')+' &middot; about '+
      mins(will*BOOT.run.seconds_per_call)+' at roughly '+
      BOOT.run.seconds_per_call+' s a call &middot; '+fmtn(r.pending)+
      ' in this pool have never been asked about'
    : 'every crop in this pool has already been asked about under prompt '+
      esc(BOOT.prompt.version);
}
function chipClass(w){
  if(w===BOOT.words.yes)return 'y';
  if(w===BOOT.words.no)return 'n';
  if(w===BOOT.words.unparsed)return 'u';
  return 'e';
}
function paintStatus(s){
  var running=!!(s&&s.running),prog=$('prog');
  $('start').disabled=running||!picked();
  $('stop').disabled=!running||!!(s&&s.stopping);
  $('runwho').innerHTML=running
    ? '<b>running</b> &middot; '+esc(s.source||'')
    : 'store <b>'+esc(BOOT.store)+'</b>';
  var sn=$('stopnote');
  sn.hidden=!(s&&s.stop_pending);
  if(!sn.hidden)sn.textContent='A stop is still on file from an earlier run '+
    '— usually the dashboard restarting while one was going. The next '+
    'start halts without asking anything and clears it; press it again after '+
    'that.';
  /* A run that fell over has a reason to show and often nothing else: the
     strip is hidden only when there is genuinely nothing to say. Without the
     last two terms a batch that died mid-way took its own error message off
     the screen with it -- counters gone, bar gone, and the one line naming
     the reason built four lines below a return that had already fired. */
  if(!s||(!running&&!s.done&&!s.n&&!s.failed&&!s.error)){prog.hidden=true;return}
  prog.hidden=false;
  var n=+s.n||0,done=+s.done||0,c=s.counts||{};
  var fill=$('pfill');
  fill.style.width=(n?Math.min(100,done/n*100):0).toFixed(1)+'%';
  fill.className=running?'':'over';
  var secs=s.started?Math.max(0,(s.updated||Date.now()/1000)-s.started):0;
  /* The status file outlives the run, so at rest this row is a report on the
     LAST one and says so. Unlabelled, a halted batch from yesterday reads as
     a run that is going nowhere right now. */
  var bits=[
    '<span>'+(running?'':'last run &middot; ')+'<b>'+fmtn(done)+
    '</b> of <b>'+fmtn(n)+'</b> asked</span>',
    '<span>'+esc(BOOT.words.yes)+' <b>'+fmtn(c[BOOT.words.yes])+'</b></span>',
    '<span>'+esc(BOOT.words.no)+' <b>'+fmtn(c[BOOT.words.no])+'</b></span>'];
  /* the unreadable and the failed always show, at zero as much as at nine.
     They are the two counts a run has every reason to hide and every reason
     to be judged on. */
  bits.push('<span class="'+(c[BOOT.words.unparsed]?'bad':'')+'">'+
    esc(BOOT.words.unparsed)+' <b>'+fmtn(c[BOOT.words.unparsed])+'</b></span>');
  bits.push('<span class="'+(c[BOOT.words.error]?'bad':'')+'">'+
    esc(BOOT.words.error)+' <b>'+fmtn(c[BOOT.words.error])+'</b></span>');
  if(s.tokens!=null)bits.push('<span><b>'+fmtn(s.tokens)+'</b> tokens</span>');
  if(secs>=1)bits.push('<span>'+mins(secs)+'</span>');
  if(s.stopping)bits.push('<span class="bad">stopping after the call in '+
    'flight</span>');
  else if(s.halted&&!running)bits.push('<span class="bad">halted</span>');
  /* A batch that abandoned itself is not a batch that finished, and the bar
     it leaves behind is the same full bar either way. The reason is the kind
     of failure that repeated, because a rate limit and a dead socket are
     different mornings. */
  if(s.gave_up)bits.push('<span class="bad">gave up &mdash; '+esc(s.gave_up)+
    ' failures one after another, so the rest of the batch was not asked'+
    '</span>');
  if(s.stale)bits.push('<span class="bad">the process that started this is '+
    'gone</span>');
  if(s.error)bits.push('<span class="bad">'+esc(s.error)+'</span>');
  if(s.failed)bits.push('<span class="bad">'+esc(s.failed)+'</span>');
  $('pline').innerHTML=bits.join('');
  $('recent').innerHTML=(s.recent||[]).map(function(r){
    return '<span class="rchip '+chipClass(r.llm_says)+'" title="'+
      att((r.source||'')+' / '+(r.key||'')+
        (r.why?' — '+r.why:'')+(r.tokens?' — '+r.tokens+' tokens':''))+'">'+
      esc(r.llm_says)+(r.ms?' &middot; '+(r.ms/1000).toFixed(1)+'s':'')+
      '</span>'}).join('');
}

/* ── the disagreements ──────────────────────────────────────────────────── */
function paintDis(){
  var g=$('dgrid'),e=$('dempty'),rows=(dis&&dis.items)||[];
  $('dwho').innerHTML=dis
    ? '<b>'+fmtn(dis.total)+'</b> '+plural(dis.total,'crop','crops')
    : '&mdash;';
  var counts={invented:0,missed:0};
  g.innerHTML=rows.map(function(r,i){
    counts[r.direction]=(counts[r.direction]||0)+1;
    var hum=r.human===BOOT.words.human_yes;
    return '<div class="tile" data-i="'+i+'">'+
      '<div class="shot" data-zoom="'+i+'">'+
        '<img loading="lazy" src="'+cropSrc(r)+'" alt="'+att(r.key)+'">'+
        '<span class="dirtag">'+(r.direction==='invented'
          ? 'invented' : 'denied')+'</span></div>'+
      '<div class="verd">'+
        '<span class="who '+(hum?'hy':'hn')+'"><i>you</i>'+
          (hum?'dog':'not a dog')+'</span>'+
        '<span class="who mm"><i>llm</i>'+esc(r.llm_says)+'</span>'+
      '</div></div>';
  }).join('');
  var db=$('dirsel').children;
  for(var i=0;i<db.length;i++){
    var d=db[i].getAttribute('data-dir');
    db[i].classList.toggle('on',d===dir);
  }
  if(!rows.length){
    e.hidden=false;
    /* "it agreed with you every time" is a claim about answers, and with
       nothing asked there are none to agree about. Asked and never wrong is
       a finding; never asked is an empty store, and they must not read the
       same. */
    var asked=over&&over.calibration&&over.calibration.parsed;
    e.innerHTML=!(dis&&dis.total===0&&!dir&&!dsrc)
      ? 'Nothing under this filter.'
      : (asked
        ? 'It has not contradicted you yet under prompt <b>'+
          esc(BOOT.prompt.version)+'</b> &mdash; on the '+fmtn(asked)+
          ' crops asked about so far it answered the way you did every time.'
        : 'Nothing has been asked under prompt <b>'+esc(BOOT.prompt.version)+
          '</b> yet, so there is nothing to disagree about.');
  }else e.hidden=true;
  var mo=$('dmore');
  if(dis&&dis.total>rows.length){
    mo.hidden=false;
    mo.textContent='showing the '+fmtn(rows.length)+' most recent of '+
      fmtn(dis.total)+'.';
  }else mo.hidden=true;
}
function zoom(i){
  var r=(dis&&dis.items||[])[i];if(!r)return;
  var hum=r.human===BOOT.words.human_yes;
  $('lbimg').src=cropSrc(r);
  $('lbv').innerHTML='<span class="who '+(hum?'hy':'hn')+'"><i>you said</i>'+
    (hum?'dog':'not a dog')+'</span>'+
    '<span class="who mm"><i>llm said</i>'+esc(r.llm_says)+'</span>';
  $('lbreply').textContent=r.reply||'(no reply on file)';
  $('lbmeta').innerHTML=
    '<span>'+esc(r.model||'')+'</span><span>'+esc(r.prompt_version||'')+
    '</span><span>'+esc(r.source||'')+'</span>'+
    '<span>'+(r.ts?new Date(r.ts*1000).toLocaleString():'')+'</span>';
  $('lbtxt').textContent=r.key||'';
  $('lb').hidden=false;
}

/* ── the unreadable replies ── */
function paintUn(u){
  var rows=(u&&u.items)||[];
  $('ucard').hidden=!rows.length;
  if(!rows.length)return;
  $('uwho').innerHTML='<b>'+fmtn(u.total)+'</b> '+
    plural(u.total,'reply','replies');
  $('ulist').innerHTML=rows.map(function(r){
    return '<div class="urow">'+
      '<img loading="lazy" src="'+cropSrc(r)+'" alt="">'+
      '<div><div class="utxt">'+esc(r.reply||'(empty reply)')+'</div>'+
      '<div class="umeta">'+esc(r.source||'')+' &middot; '+esc(r.key||'')+
      (r.finish?' &middot; finish '+esc(r.finish):'')+
      (r.reason_chars?' &middot; '+fmtn(r.reason_chars)+
        ' reasoning chars':'')+'</div></div></div>';
  }).join('');
}

/* ── loading ────────────────────────────────────────────────────────────── */
function loadOver(){
  return fetch('/api/llm').then(function(r){return r.json()})
    .then(function(j){
      over=j;
      /* BOOT is what the page was built with and the answer is what the
         server has NOW. They differ across a restart with an edited prompt,
         and every label on this page would then be describing a version the
         numbers below it are not from. */
      if(j.prompt)BOOT.prompt=j.prompt;
      if(j.words)BOOT.words=j.words;
      if(j.store)BOOT.store=j.store;
      $('storepath').textContent=j.store||BOOT.store;
      paintPrompt();paintCal();fillSources();paintStatus(j.status);
    })
    .catch(function(){toast('could not read the store')});
}
function loadDis(){
  return fetch('/api/llm/disagreements?'+q({source:dsrc,direction:dir}))
    .then(function(r){return r.json()})
    .then(function(j){dis=j;paintDis()})
    .catch(function(){});
}
function loadUn(){
  return fetch('/api/llm/unparsed?'+q({source:dsrc}))
    .then(function(r){return r.json()}).then(paintUn).catch(function(){});
}
function loadStatus(){
  return fetch('/api/llm/status').then(function(r){return r.json()})
    .then(function(s){
      paintStatus(s);
      var running=!!s.running;
      /* A finished run has changed every number on the page, so the reload
         is hung off the edge rather than off a timer: polling the whole
         overview every two seconds would re-read 2,600 ledger lines and
         re-stat the pools for a page nobody is touching. */
      if(wasRunning&&!running){loadOver();loadDis();loadUn()}
      wasRunning=running;
    }).catch(function(){});
}
/* Polls only while the tab is visible and only every two seconds, the way the
   dashboard's own run strips do. A run is six seconds a crop; nothing here
   changes faster than that. */
function tick(){
  if(document.hidden)return;
  loadStatus();
}
function startRun(){
  var r=picked();if(!r)return;
  $('start').disabled=true;
  fetch('/api/llm/run',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({source:r.source,n:+$('n').value})})
    .then(function(x){return x.json()})
    .then(function(j){
      if(!j||!j.ok){toast((j&&j.msg)||'could not start');
        $('start').disabled=false;return}
      wasRunning=true;
      toast('asking about '+fmtn(j.planned)+' '+
        plural(j.planned,'crop','crops')+' — about '+mins(j.seconds));
      loadStatus();
    })
    .catch(function(){$('start').disabled=false;toast('could not start')});
}
function stopRun(){
  $('stop').disabled=true;
  fetch('/api/llm/stop',{method:'POST'})
    .then(function(x){return x.json()})
    .then(function(j){toast((j&&j.msg)||'stopping');loadStatus()})
    .catch(function(){toast('could not stop')});
}

$('start').addEventListener('click',startRun);
$('stop').addEventListener('click',stopRun);
$('n').addEventListener('change',noteCost);
$('src').addEventListener('change',noteCost);
$('dsrc').addEventListener('change',function(){
  dsrc=this.value;loadDis();loadUn()});
$('dirsel').addEventListener('click',function(e){
  var b=e.target.closest&&e.target.closest('.viewbtn');
  if(!b)return;
  dir=b.getAttribute('data-dir')||'';loadDis();
});
$('dgrid').addEventListener('click',function(e){
  var z=e.target.closest&&e.target.closest('[data-zoom]');
  if(z)zoom(+z.getAttribute('data-zoom'));
});
$('lbclose').addEventListener('click',function(){$('lb').hidden=true});
$('lb').addEventListener('click',function(e){
  if(e.target===this)this.hidden=true});
document.addEventListener('keydown',function(e){
  if(e.key==='Escape'&&!$('lb').hidden){$('lb').hidden=true}});

loadOver();loadDis();loadUn();
poll=setInterval(tick,2000);
</script></body></html>
"""


def page_html():
    """The whole /llm page.

    Nothing about the store is baked in: the page arrives with the prompt
    version, the batch sizes and the store's vocabulary, and asks /api/llm for
    everything else. The vocabulary travels rather than being written into the
    markup on purpose -- 'llm_yes' appears on this page because it is the word
    in the ledger, and a copy of it here would go on saying so after somebody
    changed the one over there.
    """
    opts = ''.join(
        f'<option value="{n}"{" selected" if n == RUN_DEFAULT else ""}>'
        f'{n}</option>' for n in RUN_SIZES)
    boot = {
        'prompt': {'version': llm.PROMPT_VERSION, 'model': llm.MODEL,
                   'text': llm.PROMPT_TEXT, 'max_tokens': llm.MAX_TOKENS,
                   'temperature': llm.TEMPERATURE, 'prep': llm.PREP},
        'words': {'yes': llm.LLM_YES, 'no': llm.LLM_NO,
                  'unparsed': llm.LLM_UNPARSED, 'error': llm.LLM_ERROR,
                  'human_yes': llm.HUMAN_YES, 'human_no': llm.HUMAN_NO},
        'store': os.path.relpath(llm.paths()['ledger'], REPO),
        'run': {'sizes': list(RUN_SIZES), 'default': RUN_DEFAULT,
                'max': RUN_MAX, 'seconds_per_call': SECONDS_PER_CALL},
    }
    out = LLM_HTML
    for k, v in (('__BOOT__', json.dumps(boot)), ('__SIZEOPTS__', opts)):
        out = out.replace(k, v)
    return out
