#!/usr/bin/env python3
"""Read ultralytics training runs off disk, for the dashboard's tracker section.

The training runs live in a SEPARATE repo from this one, so nothing here may
assume a path: the root comes from the ``training_root`` config key (or
$TRAINING_ROOT) and an unset root is a normal, reportable state rather than an
error.

What this answers, in the order the questions actually get asked:

  1. Is something training right now, and how close is it to stopping?
     ultralytics stops when ``epoch - best_epoch >= patience``, so the number
     that matters is epochs since the best -- not epochs elapsed.
  2. Is this run beating the last one in the same project?
  3. What has this project trained, and which one got promoted?

Two details that are easy to get wrong and silently produce a wrong answer:

  FITNESS TIES BREAK TOWARD THE FIRST EPOCH. Ultralytics keeps the earlier
  epoch when fitness ties; Python's max() keeps the later one. On a metric that
  saturates -- a classifier at top5 = 1.0 for 200 epochs -- that disagreement
  moves the reported best epoch by tens of epochs and makes a converged run
  look like it is still improving. This module reproduces ultralytics.

  RESULTS.CSV HEADERS ARE PADDED IN OLDER RUNS. Some ultralytics versions
  right-align the header row ("     epoch,   train/loss"). A parser that does
  not strip finds no columns at all and reports an empty run, which looks
  exactly like a run that has not started yet.

Standalone:

    python tools/dashboard/training_tracker.py            # every run found
    python tools/dashboard/training_tracker.py --live     # only what is running
"""

import csv
import json
import os
import re
import sys
import time

# ── ultralytics fitness ─────────────────────────────────────────────────────
# The single number early stopping actually watches. Anything else on screen is
# context; this is what decides when the run ends.
CLS_FITNESS = ('metrics/accuracy_top1', 'metrics/accuracy_top5')
DET_FITNESS = ('metrics/mAP50(B)', 'metrics/mAP50-95(B)')


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_results(path):
    """[{col: float}] from a results.csv, headers stripped of padding."""
    rows = []
    try:
        with open(path, newline='') as fh:
            rd = csv.reader(fh)
            try:
                head = [h.strip() for h in next(rd)]
            except StopIteration:
                return []
            for raw in rd:
                if not raw:
                    continue
                r = {}
                for k, v in zip(head, raw):
                    n = _num(v.strip())
                    if n is not None:
                        r[k] = n
                if r:
                    rows.append(r)
    except OSError:
        return []
    return rows


def task_of(rows):
    """'classify' | 'detect' | None -- from the columns, not the folder name."""
    if not rows:
        return None
    cols = set(rows[0])
    if CLS_FITNESS[0] in cols:
        return 'classify'
    if DET_FITNESS[1] in cols or 'metrics/mAP50-95(B)' in cols:
        return 'detect'
    return None


# The detect fitness formula CHANGED between ultralytics releases, and both
# releases are installed on this machine:
#   <= 8.3  Metric.fitness w = [0, 0, 0.1, 0.9]  -> 0.1*mAP50 + 0.9*mAP50-95
#   >= 8.4  Metric.fitness w = [0, 0, 0.0, 1.0]  -> mAP50-95 alone
# Hardcoding the 8.3 form made three runs disagree with what actually happened
# on disk: train-22 was reported early-stopped at best@248 when it ran its full
# 300-epoch budget and peaked at 262. Nothing looked wrong -- the epoch was a
# plausible number in the right range.
DET_W_LEGACY = (0.1, 0.9)
DET_W_84 = (0.0, 1.0)
_VER_RE = re.compile(rb'8\.\d+\.\d+')


def ultra_version(run_dir):
    """The ultralytics version that wrote this run, from its checkpoint.

    The version is in the checkpoint's pickle. Read with zipfile alone -- no
    torch, no unpickling of untrusted data, and only the small data.pkl member
    rather than the 100MB of weights beside it.
    """
    for w in ('weights/best.pt', 'weights/last.pt'):
        p = os.path.join(run_dir, w)
        if not os.path.exists(p):
            continue
        try:
            import zipfile
            with zipfile.ZipFile(p) as z:
                names = [n for n in z.namelist() if n.endswith('data.pkl')]
                if not names:
                    continue
                m = _VER_RE.search(z.read(names[0]))
            if m:
                return m.group(0).decode()
        except Exception:
            pass
    return None


def det_weights(version):
    """(w_mAP50, w_mAP50-95) for a version string, or None if unknown."""
    if not version:
        return None
    try:
        major, minor = (int(x) for x in version.split('.')[:2])
    except ValueError:
        return None
    return DET_W_84 if (major, minor) >= (8, 4) else DET_W_LEGACY


def fitness_of(row, task, det_w=DET_W_84):
    """Ultralytics' fitness: what early stopping compares, epoch to epoch."""
    if task == 'classify':
        a = row.get(CLS_FITNESS[0])
        b = row.get(CLS_FITNESS[1])
        if a is None:
            return None
        return (a + (b if b is not None else a)) / 2
    if task == 'detect':
        m50 = row.get('metrics/mAP50(B)')
        m95 = row.get('metrics/mAP50-95(B)')
        if m95 is None:
            return None
        return det_w[0] * (m50 if m50 is not None else 0.0) + det_w[1] * m95
    return None


def best_index(rows, task, det_w=DET_W_84):
    """Index of the best epoch, exactly as ultralytics' EarlyStopping tracks it.

    Two behaviours that are easy to miss and both change the answer:
      - a tie keeps the EARLIER epoch (strict >), where Python's max() keeps
        the later one;
      - `or self.best_fitness == 0` means that while fitness is still zero,
        EVERY epoch replaces the best -- so a run that never leaves 0.0 has its
        best at the LAST epoch, and never early-stops.
    """
    best, at = 0.0, None
    for i, r in enumerate(rows):
        f = fitness_of(r, task, det_w)
        if f is None:                      # val=False epochs do not count
            continue
        if f > best or best == 0.0:
            best, at = f, i
    return at


# ── the deciding metric to PLOT (fitness is a blend; readers want the metric)─
HEADLINE = {
    'classify': ('metrics/accuracy_top1', 'top-1 accuracy'),
    'detect': ('metrics/mAP50-95(B)', 'mAP50-95'),
}
# What the LATEST epoch reported, beyond the one metric early stopping
# watches. Keys are the glossary names the dashboard already uses, so the
# hovers come from one place. Ordered: the deciding metric first.
# Short on purpose -- top-5 on a two-class problem is 1.0 forever, so the
# classifiers get one card. The two mAPs sit together because they answer the
# same question at different strictness, and the gap between them IS the
# reading: mAP50 high with mAP50-95 low means the boxes are found but loose.
LATEST = {
    'detect': (('metrics/mAP50-95(B)', 'mAP50-95'),   # what decides the run
               ('metrics/mAP50(B)', 'mAP50'),         # same, forgiving overlap
               ('metrics/recall(B)', 'recall'),       # the unrecoverable error
               ('metrics/precision(B)', 'precision')),
    'classify': (('metrics/accuracy_top1', 'accuracy_top1'),),
}


def latest_metrics(rows, task, best_index=None):
    """[{key, latest, peak, peak_epoch, at_best, series}] for the newest epoch.

    `at_best` is the value this metric held at the run's BEST-FITNESS epoch --
    the checkpoint that actually gets promoted, so it is the number that
    describes the model you would ship. It is not the same as `peak`, and on
    a recall-first project the difference is the whole argument.

    `peak` is the running max of THAT metric, which is not the value at the
    best-fitness epoch -- reporting the latter as "best recall" would
    attribute one metric's peak to another metric's epoch, and the two agree
    on every run where they happen to coincide.

    The series comes back with it so the card can draw the metric's own shape
    rather than describing it in words.
    """
    if not rows:
        return []
    last = rows[-1]
    out = []
    for col, key in LATEST.get(task, ()):
        v = last.get(col)
        if v is None:
            continue
        series = [r.get(col) for r in rows]
        seen = [(x, i) for i, x in enumerate(series) if x is not None]
        if not seen:
            continue
        pk, pi = max(seen)                      # ties keep the EARLIER epoch
        pk, pi = max(seen, key=lambda t: (t[0], -t[1]))
        e = rows[pi].get('epoch')
        at_best = None
        if best_index is not None and 0 <= best_index < len(series):
            at_best = series[best_index]
        # `col` travels with `key`: the csv column is what identifies a metric
        # across tables, and HEADLINE and LATEST label the same classify
        # metric two different ways ('top-1 accuracy' vs 'accuracy_top1'), so
        # a consumer matching on the label finds nothing for a classifier.
        out.append({'key': key, 'col': col, 'latest': v, 'peak': pk,
                    'peak_epoch': int(e) if e is not None else pi + 1,
                    'peak_index': pi, 'at_best': at_best, 'series': series})
    return out


def loss_keys(rows):
    """(train keys, val keys) taken from the FILE, not from a fixed list.

    The detect heads are not stable across ultralytics versions: this project
    has runs with train/dfl_loss and runs with train/l1_loss. A hardcoded
    triple silently drops the term it does not know about and still produces a
    plausible curve -- while the caption goes on naming a loss that is not in
    the sum.
    """
    if not rows:
        return ((), ())
    cols = [c for c in rows[0] if c.endswith('loss')]
    return (tuple(c for c in cols if c.startswith('train/')),
            tuple(c for c in cols if c.startswith('val/')))


def loss_label(rows):
    """What the summed curve actually contains, in the file's own words."""
    tr, _ = loss_keys(rows)
    parts = [c.split('/', 1)[1].replace('_loss', '') for c in tr]
    if not parts:
        return 'loss'
    if len(parts) == 1:
        return 'train/loss vs val/loss'
    return ' + '.join(parts) + ' loss, summed'


def loss_series(rows, task=None):
    """(train, val) totals per epoch. Detect has several loss heads; their sum
    is what the reader wants -- one idea should not be six overlaid series."""
    tr_k, va_k = loss_keys(rows)

    def tot(r, keys):
        vals = [r[k] for k in keys if r.get(k) is not None]
        return sum(vals) if vals else None

    return ([tot(r, tr_k) for r in rows], [tot(r, va_k) for r in rows])


# ── args.yaml ───────────────────────────────────────────────────────────────
_ARG = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$')


def read_args(path):
    """Flat key: value from an ultralytics args.yaml, without a yaml import.

    ultralytics writes a flat mapping of scalars, so a line parser is enough --
    and keeps the dashboard free of a dependency the rest of it does not need.
    """
    out = {}
    try:
        with open(path) as fh:
            for ln in fh:
                m = _ARG.match(ln.rstrip('\n'))
                if not m:
                    continue
                k, v = m.group(1), m.group(2)
                if v in ('null', '', '~'):
                    out[k] = None
                elif v in ('true', 'True'):
                    out[k] = True
                elif v in ('false', 'False'):
                    out[k] = False
                else:
                    n = _num(v)
                    out[k] = n if n is not None else v.strip('"\'')
    except OSError:
        pass
    return out


# ── which run is training RIGHT NOW ─────────────────────────────────────────
def _cmdlines():
    """[(pid, [argv...])] for every process this user can see."""
    out = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as fh:
                argv = fh.read().decode('utf-8', 'replace').split('\0')
        except OSError:
            continue
        if argv and any(a for a in argv):
            out.append((int(pid), [a for a in argv if a]))
    return out


def _open_dirs(pid):
    """Directories under which this pid holds an open file."""
    out = set()
    try:
        for fd in os.listdir(f'/proc/{pid}/fd'):
            try:
                out.add(os.path.dirname(os.path.realpath(f'/proc/{pid}/fd/{fd}')))
            except OSError:
                pass
    except OSError:
        pass
    return out


def _boot_time():
    try:
        with open('/proc/stat') as fh:
            for ln in fh:
                if ln.startswith('btime '):
                    return float(ln.split()[1])
    except OSError:
        pass
    return None


def proc_start(pid, boot=None):
    """Unix time this pid began, from /proc/<pid>/stat field 22.

    The directory mtime of /proc/<pid> is NOT this -- it tracks the last change
    to the proc entry. Getting it wrong matters here: the start time is what
    rules out a run directory that existed before the process did.
    """
    boot = _boot_time() if boot is None else boot
    if boot is None:
        return None
    try:
        with open(f'/proc/{pid}/stat') as fh:
            s = fh.read()
        # comm can contain spaces AND ')', so split after the LAST ')'
        fields = s[s.rindex(')') + 1:].split()
        return boot + float(fields[19]) / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError, IndexError):
        return None


def live_trainings():
    """[{pid, argv, project, name, data, started}] for running yolo trainings.

    Matched on the command line rather than on a pid file, because the user
    starts these by hand from a shell and no pid file exists.

    One training is several processes -- ultralytics forks dataloader workers
    that carry the parent's argv verbatim. Counting them separately would let
    one training claim two run directories, so identical command lines collapse
    to the earliest pid.
    """
    boot = _boot_time()
    seen = {}
    for pid, argv in _cmdlines():
        joined = ' '.join(argv)
        if 'yolo' not in joined or ' train' not in f' {joined}':
            continue
        if 'sweep.py' in joined:      # the detection sweep is not a training
            continue
        kv = {}
        for a in argv:
            if '=' in a and not a.startswith('-'):
                k, _, v = a.partition('=')
                kv[k] = v
        # `yolo train resume=path/to/last.pt` is the documented resume form
        # and carries neither model= nor data=. Dropping it made a resumed
        # training invisible: no live card, and the run listed as interrupted.
        if not ({'model', 'data', 'resume'} & set(kv)):
            continue
        rec = {'pid': pid, 'argv': argv, 'project': kv.get('project'),
               'name': kv.get('name'), 'data': kv.get('data'),
               'epochs': _num(kv.get('epochs')),
               'patience': _num(kv.get('patience')),
               'started': proc_start(pid, boot)}
        prev = seen.get(joined)
        if prev is None or (rec['started'] or 0) < (prev['started'] or 0):
            seen[joined] = rec
    return sorted(seen.values(), key=lambda r: r['started'] or 0)


def _incremented(requested, actual):
    """Is `actual` ultralytics' auto-increment of `requested`?

    It appends an integer to the requested name when the directory already
    exists: dogbin_009 -> dogbin_0092. Only trailing DIGITS count, so an
    unrelated run someone named dogbin_009_retry does not match.
    """
    if not requested or not actual or requested == actual:
        return False
    a, r = str(actual), str(requested)
    return a.startswith(r) and a[len(r):].isdigit() and bool(a[len(r):])


def _same_path(a, b):
    return (a and b
            and os.path.abspath(str(a)) == os.path.abspath(str(b)))


def _proj_name(p):
    """The name of a project, given whatever was passed as ``project=``.

    Ultralytics treats project= as a directory, so it is equally happy with
    a bare `dog-bin` and an absolute path ending in `/dog-bin`, and writes back
    exactly what it was given. Both mean the same project to a reader, so both must reduce to the
    same key here.
    """
    p = str(p or '').strip()
    if not p:
        return p
    return os.path.basename(os.path.normpath(p)) or p


CLAIM_FLOOR = 40   # a shared project name alone proves nothing


def _live_score(run, lv):
    """How strongly ``lv`` claims ``run``. None = not a candidate at all.

    Deliberately NOT gated on the process start time. It is tempting -- a run
    directory is created by the process that trains into it, so the directory
    should never be older. It routinely is: ultralytics re-execs for DDP, so the
    surviving pid started 14 minutes after the directory it owns. Vetoing on
    that marks a live training as finished, which is the worse error: it is the
    one state a reader would act on.

    What actually separates the live run from an abandoned earlier attempt with
    the same project and dataset is the run NAME plus exclusivity -- one process
    claims one directory, the best-scoring one.
    """
    args = run.get('args') or {}
    if any(d == run['dir'] or d.startswith(run['dir'] + os.sep)
           for d in _open_dirs(lv['pid'])):
        return 300                      # an open fd inside it is proof
    score = 0
    if lv.get('name') and lv['name'] == args.get('name'):
        score += 100
    elif _incremented(lv.get('name'), args.get('name')):
        # ultralytics renames rather than overwrite: ask for dogbin_009 when
        # that directory exists and it writes dogbin_0092, rewriting name and
        # save_dir in the NEW run's args.yaml while the command line keeps the
        # name you typed. Without this the abandoned directory took the whole
        # +100 for matching a name the live process is no longer writing to,
        # and the panel reported the husk as the running one -- measured 230
        # against 130 for the run that was actually training.
        score += 90
    if _same_path(lv.get('data'), args.get('data')):
        score += 40
    # compared by NAME: the command line and the run's own args.yaml can spell
    # the same project differently (bare name vs absolute path) whenever a run
    # is cancelled and relaunched with the path edited
    if (lv.get('project')
            and _proj_name(lv['project']) == _proj_name(args.get('project'))):
        score += 10
    # A run records where it writes. When that is an absolute path, it settles
    # which of two identically-named directories is the real one: the leftover
    # skeleton at <root>/<project>/<run> carries a RELATIVE save_dir from a
    # different working directory and never matches itself.
    sd = args.get('save_dir')
    if sd and os.path.isabs(str(sd)):
        score += 80 if _same_path(sd, run['dir']) else -60
    if score < CLAIM_FLOOR:
        return None
    # A directory that has recorded epochs is where training is happening.
    # Exact rather than heuristic, and it is what finally separates the husk
    # from its increment: the husk never gets a results.csv. It does NOT help
    # in the seconds before the real run writes epoch 1 -- both have none then,
    # and the husk's other points still win -- but it flips as soon as one
    # lands, and a run that has trained for an hour can never lose to a
    # directory that never trained at all.
    try:
        if read_results(os.path.join(run['dir'], 'results.csv')):
            score += 40
    except Exception:
        pass
    # a directory created at or after the process is the likelier output; worth
    # a nudge between otherwise equal candidates, never a veto
    started, written = lv.get('started'), None
    try:
        written = os.path.getmtime(os.path.join(run['dir'], 'args.yaml'))
    except OSError:
        pass
    if started and written and written >= started - 5:
        score += 5
    return score


def attach_live(runs, lives):
    """{run dir: live record} -- each process claims AT MOST ONE run.

    Without the exclusivity a single training lights up every directory that
    resembles it; this project already had three such directories from one
    afternoon of restarts.
    """
    claims = {}
    for lv in lives:
        best, at = None, None
        for run in runs:
            if run['dir'] in claims:
                continue
            s = _live_score(run, lv)
            if s is not None and (best is None or s > best):
                best, at = s, run['dir']
        if at:
            claims[at] = lv
    return claims


# ── discovery ───────────────────────────────────────────────────────────────
# Names worth not descending into -- but ONLY when the directory is not itself
# a run. 'train' is on this list because a dataset split is called train and is
# full of images; it is ALSO ultralytics' default run name (cfg: name = name or
# f"{args.mode}"), so pruning it by name alone hid every run started without
# name= -- including, on this machine, runs/detect/train and
# runs/detect/DogDetection/train.
SKIP_DIRS = {'weights', 'node_modules', '__pycache__', 'images', 'labels',
             'train', 'val', 'test', 'archived', 'archived_datasets'}
MAX_DEPTH = 5


def _prunable(cur, d):
    """A directory holding an args.yaml is a run, whatever it is called."""
    if d.startswith('.'):
        return True
    if d not in SKIP_DIRS:
        return False
    return not os.path.isfile(os.path.join(cur, d, 'args.yaml'))


def discover(root, projects=None):
    """[run] under ``root`` -- any directory holding an args.yaml.

    A fixed <root>/<project>/<run> walk is wrong, and wrong in the worst way:
    ultralytics honours its own `runs_dir` setting, so `project=dogdetection`
    can land at <root>/runs/detect/dogdetection/<run> while a stale directory
    of the SAME name sits at <root>/dogdetection/<run> from an earlier attempt.
    The two-level walk found only the stale one -- an args.yaml, no results,
    reported as "no epoch finished" while the real run was eight epochs in.

    The project name comes from args.yaml, not from the parent directory: runs
    get moved and folders get renamed, but a run's own record of what project
    it belongs to does not drift.
    """
    runs = []
    if not root or not os.path.isdir(root):
        return runs
    base = os.path.abspath(root)
    for cur, dirs, files in os.walk(base):
        depth = cur[len(base):].count(os.sep)
        if depth >= MAX_DEPTH:
            dirs[:] = []
        dirs[:] = sorted(d for d in dirs if not _prunable(cur, d))
        if 'args.yaml' not in files:
            continue
        dirs[:] = []                      # a run holds no nested runs
        args = read_args(os.path.join(cur, 'args.yaml'))
        # ultralytics with no project= writes to <runs_dir>/<task>/<name>, so
        # the parent is the TASK ("detect"), not a project. Calling that a
        # project puts eleven stray experiments under a heading that looks
        # like one and is not.
        proj = args.get('project')
        if not proj:
            parent = os.path.basename(os.path.dirname(cur))
            proj = ('(no project)' if parent in ('detect', 'classify',
                                                 'segment', 'pose', 'obb')
                    else parent or '(no project)')
        else:
            # `project=` is a DIRECTORY, so ultralytics takes an absolute path
            # and records it verbatim. Grouping on that raw string split one
            # project in two: the same dog-bin, reached once by absolute path
            # and once by bare name, became separate headings -- and since the
            # star marks the best run WITHIN a project, each half got its own
            # "best", so dog-bin showed two. The last component is the name.
            proj = _proj_name(proj)
        if projects and proj not in projects:
            continue
        runs.append({'project': str(proj), 'name': os.path.basename(cur),
                     'dir': cur, 'args': args})
    runs.sort(key=lambda r: (r['project'], r['name']))
    return runs


def summarize(run, live=None, registry=None):
    """Everything the tracker needs about one run, computed once."""
    rows = read_results(os.path.join(run['dir'], 'results.csv'))
    task = task_of(rows) or ('classify' if 'cls' in str(
        (run.get('args') or {}).get('model', '')) else None)
    version = ultra_version(run['dir']) if task == 'detect' else None
    det_w = det_weights(version) or DET_W_84
    bi = best_index(rows, task, det_w)
    # An unknown version only matters if the two formulas actually disagree on
    # THIS run. Saying "uncertain" when both agree would be noise; staying
    # silent when they differ would be the wrong kind of quiet.
    formula_uncertain = False
    if task == 'detect' and not det_weights(version):
        formula_uncertain = best_index(rows, task, DET_W_LEGACY) != bi
    args = run.get('args') or {}

    epochs_planned = int(args.get('epochs') or 0) or None
    patience = int(args.get('patience') or 0) or None
    done = len(rows)
    since_best = (done - 1 - bi) if bi is not None else None

    # The epoch NUMBER comes from the file, not from the row index: a resumed
    # run's results.csv does not start at 1, and "best @1" on a run that
    # resumed at epoch 180 is a number with no relation to anything.
    def epoch_at(i):
        e = rows[i].get('epoch') if 0 <= i < len(rows) else None
        return int(e) if e is not None else (i + 1)

    last_epoch = epoch_at(done - 1) if done else None

    # seconds per epoch from the cumulative 'time' column, over the last 10
    # epochs -- an average over the whole run understates a slowdown
    secs = None
    tcol = [r.get('time') for r in rows if r.get('time') is not None]
    if len(tcol) >= 2:
        k = min(10, len(tcol) - 1)
        secs = (tcol[-1] - tcol[-1 - k]) / k
        if secs <= 0:
            # the 'time' column restarts at 0 on resume, so a window spanning
            # the restart yields a negative rate -- and a negative ETA
            secs = None

    head_key, head_label = HEADLINE.get(task, (None, ''))
    curve = [r.get(head_key) for r in rows] if head_key else []
    tr, va = loss_series(rows)

    # Five states, not two. "finished" for a directory that holds an args.yaml
    # and nothing else would be a lie -- three of those exist here from one
    # afternoon of restarts, and calling them finished runs puts them in the
    # history beside runs that actually trained.
    if live:
        status = 'running'
    elif done == 0:
        status = 'never_started'
    elif epochs_planned and (last_epoch or done) >= epochs_planned:
        # Checked BEFORE patience: a run that reached its epoch budget ran to
        # the end by definition. With this test second, train-22 -- 300 rows of
        # a 300-epoch budget -- was labelled early-stopped.
        status = 'completed'
    elif patience and since_best is not None and since_best >= patience:
        status = 'early_stopped'
    else:
        status = 'interrupted'
    stopped_early = status == 'early_stopped'

    promoted = None
    if registry:
        for proj, d in (registry.get('projects') or {}).items():
            b = d.get('best') or {}
            if b.get('run') == run['name']:
                promoted = {'project': proj, 'key': b.get('key'),
                            'deployed': bool(b.get('deployed'))}
            elif any(c.get('run') == run['name']
                     for c in (d.get('candidates') or [])):
                promoted = promoted or {'project': proj, 'candidate': True}

    return {
        'project': run['project'], 'name': run['name'], 'dir': run['dir'],
        'task': task, 'epochs_done': done, 'epochs_planned': epochs_planned,
        'patience': patience,
        'best_epoch': epoch_at(bi) if bi is not None else None,
        'last_epoch': last_epoch,
        'best_fitness': fitness_of(rows[bi], task) if bi is not None else None,
        'best_headline': (curve[bi] if bi is not None and bi < len(curve)
                          else None),
        'last_headline': curve[-1] if curve else None,
        'since_best': since_best, 'secs_per_epoch': secs,
        'headline_key': head_key, 'headline_label': head_label,
        'wall_clock_s': (tcol[-1] if tcol else None),
        'latest': latest_metrics(rows, task, bi),
        'latest_train_loss': (tr[-1] if tr else None),
        'latest_val_loss': (va[-1] if va else None),
        'lr': rows[-1].get('lr/pg0') if rows else None,
        'ultralytics': version, 'fitness_uncertain': formula_uncertain,
        'fitness_formula': ('mAP50-95' if det_w == DET_W_84 else
                            '0.1*mAP50 + 0.9*mAP50-95') if task == 'detect'
                           else '(top1 + top5) / 2',
        'curve': curve, 'train_loss': tr, 'val_loss': va,
        'loss_label': loss_label(rows),
        'live': bool(live), 'pid': live['pid'] if live else None,
        'started': live.get('started') if live else None,
        'stopped_early': stopped_early, 'status': status,
        'promoted': promoted,
        'mtime': _mtime(run['dir']),
        'imgsz': args.get('imgsz'), 'batch': args.get('batch'),
        'model': args.get('model'), 'data': args.get('data'),
        'optimizer': args.get('optimizer'), 'single_cls': args.get('single_cls'),
    }


def _mtime(d):
    """Newest mtime inside the run dir -- when it last did anything."""
    best = 0.0
    for base, _, files in os.walk(d):
        for f in files:
            try:
                best = max(best, os.path.getmtime(os.path.join(base, f)))
            except OSError:
                pass
        if base.count(os.sep) - d.count(os.sep) > 2:
            break
    return best or (os.path.getmtime(d) if os.path.isdir(d) else 0.0)


def canon_projects(runs, registry=None):
    """Fold project names that differ only in case onto one spelling.

    `DogDetection` and `dogdetection` are one project that ultralytics wrote
    two ways, and left alone they split the history in half and put the same
    work under two headings. The surviving spelling is the registry's when it
    knows the project, so the panel and data/best_models.json agree; otherwise
    the one more runs actually use.
    """
    reg = {k.lower(): k for k in ((registry or {}).get('projects') or {})}
    counts = {}
    for r in runs:
        counts.setdefault(r['project'].lower(), {}).setdefault(r['project'], 0)
        counts[r['project'].lower()][r['project']] += 1
    winner = {}
    for low, spellings in counts.items():
        winner[low] = reg.get(low) or max(sorted(spellings),
                                          key=lambda n: spellings[n])
    for r in runs:
        r['project'] = winner[r['project'].lower()]
    return runs


def collect(root, registry=None, projects=None):
    """[summary] newest first."""
    runs = canon_projects(discover(root, projects), registry)
    claims = attach_live(runs, live_trainings())
    out = [summarize(r, claims.get(r['dir']), registry) for r in runs]
    out.sort(key=lambda s: (s['live'], s['mtime']), reverse=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.environ.get('TRAINING_ROOT', ''))
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    if not a.root:
        print('no training root: pass --root or set $TRAINING_ROOT',
              file=sys.stderr)
        return 2
    runs = collect(a.root)
    if a.live:
        runs = [r for r in runs if r['live']]
    if a.json:
        for r in runs:
            r.pop('curve', None), r.pop('train_loss', None)
            r.pop('val_loss', None)
        print(json.dumps(runs, indent=1, default=str))
        return 0
    for r in runs:
        tag = r['status']
        eta = ''
        if r['live'] and r['secs_per_epoch'] and r['patience'] \
                and r['since_best'] is not None:
            left = r['patience'] - r['since_best']
            eta = f'  ~{left * r["secs_per_epoch"] / 3600:.1f}h to patience'
        print(f'{r["project"]:>14}/{r["name"]:<16} {tag:<14} '
              f'{r["epochs_done"]:>4} ep  best@{r["best_epoch"]}  '
              f'{r["headline_label"]}={r["best_headline"]}{eta}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
