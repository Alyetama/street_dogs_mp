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


def fitness_of(row, task):
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
        return 0.1 * (m50 if m50 is not None else 0.0) + 0.9 * m95
    return None


def best_index(rows, task):
    """Index of the best epoch, ties broken toward the FIRST -- as ultralytics
    does. Python's max() breaks toward the last, which is the opposite."""
    best, at = None, None
    for i, r in enumerate(rows):
        f = fitness_of(r, task)
        if f is None:
            continue
        if best is None or f > best:   # strict >, so a tie keeps the earlier
            best, at = f, i
    return at


# ── the deciding metric to PLOT (fitness is a blend; readers want the metric)─
HEADLINE = {
    'classify': ('metrics/accuracy_top1', 'top-1 accuracy'),
    'detect': ('metrics/mAP50-95(B)', 'mAP50-95'),
}
LOSSES = {
    'classify': (('train/loss',), ('val/loss',)),
    'detect': (('train/box_loss', 'train/cls_loss', 'train/dfl_loss'),
               ('val/box_loss', 'val/cls_loss', 'val/dfl_loss')),
}


def loss_series(rows, task):
    """(train, val) totals per epoch. Detect has three loss heads; their sum is
    what the reader wants -- three overlaid pairs is six series for one idea."""
    tr_k, va_k = LOSSES.get(task, ((), ()))

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
        if 'model' not in kv and 'data' not in kv:
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


def _same_path(a, b):
    return (a and b
            and os.path.abspath(str(a)) == os.path.abspath(str(b)))


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
    if _same_path(lv.get('data'), args.get('data')):
        score += 40
    if lv.get('project') and lv['project'] == args.get('project'):
        score += 10
    if score < CLAIM_FLOOR:
        return None
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
def discover(root, projects=None):
    """[run] under ``root``: <root>/<project>/<run>/args.yaml.

    Only directories holding an args.yaml count as runs, so dataset folders and
    scratch directories sitting beside the projects are skipped without needing
    a list of names to exclude.
    """
    runs = []
    if not root or not os.path.isdir(root):
        return runs
    for proj in sorted(os.listdir(root)):
        pdir = os.path.join(root, proj)
        if not os.path.isdir(pdir) or (projects and proj not in projects):
            continue
        for name in sorted(os.listdir(pdir)):
            d = os.path.join(pdir, name)
            ay = os.path.join(d, 'args.yaml')
            if not os.path.isfile(ay):
                continue
            runs.append({'project': proj, 'name': name, 'dir': d,
                         'args': read_args(ay)})
    return runs


def summarize(run, live=None, registry=None):
    """Everything the tracker needs about one run, computed once."""
    rows = read_results(os.path.join(run['dir'], 'results.csv'))
    task = task_of(rows) or ('classify' if 'cls' in str(
        (run.get('args') or {}).get('model', '')) else None)
    bi = best_index(rows, task)
    args = run.get('args') or {}

    epochs_planned = int(args.get('epochs') or 0) or None
    patience = int(args.get('patience') or 0) or None
    done = len(rows)
    since_best = (done - 1 - bi) if bi is not None else None

    # seconds per epoch from the cumulative 'time' column, over the last 10
    # epochs -- an average over the whole run understates a slowdown
    secs = None
    tcol = [r.get('time') for r in rows if r.get('time') is not None]
    if len(tcol) >= 2:
        k = min(10, len(tcol) - 1)
        secs = (tcol[-1] - tcol[-1 - k]) / k

    head_key, head_label = HEADLINE.get(task, (None, ''))
    curve = [r.get(head_key) for r in rows] if head_key else []
    tr, va = loss_series(rows, task)

    # Five states, not two. "finished" for a directory that holds an args.yaml
    # and nothing else would be a lie -- three of those exist here from one
    # afternoon of restarts, and calling them finished runs puts them in the
    # history beside runs that actually trained.
    if live:
        status = 'running'
    elif done == 0:
        status = 'never_started'
    elif patience and since_best is not None and since_best >= patience:
        status = 'early_stopped'
    elif epochs_planned and done >= epochs_planned:
        status = 'completed'
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
        'patience': patience, 'best_epoch': (bi + 1) if bi is not None else None,
        'best_fitness': fitness_of(rows[bi], task) if bi is not None else None,
        'best_headline': (curve[bi] if bi is not None and bi < len(curve)
                          else None),
        'last_headline': curve[-1] if curve else None,
        'since_best': since_best, 'secs_per_epoch': secs,
        'headline_key': head_key, 'headline_label': head_label,
        'curve': curve, 'train_loss': tr, 'val_loss': va,
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


def collect(root, registry=None, projects=None):
    """[summary] newest first."""
    runs = discover(root, projects)
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
