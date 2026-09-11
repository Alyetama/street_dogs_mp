#!/usr/bin/env python3
"""Long work that outlives the page that started it.

    python tools/dashboard/jobs.py list
    python tools/dashboard/jobs.py show <job-id>
    python tools/dashboard/jobs.py tail <job-id>
    python tools/dashboard/jobs.py cancel <job-id>

Building a dataset takes minutes and training takes hours, and neither has any
business being tied to a browser tab or to the dashboard process. So a job is
a DIRECTORY, not an object: everything about it is on disk, nothing is held in
memory, and the dashboard learns what is happening by reading rather than by
remembering. Restart the dashboard mid-training and it picks the run back up,
because there was never anything to lose.

...AND NOT TIED TO THE DASHBOARD'S SERVICE EITHER. Orphaning a process is not
enough on a machine running systemd: a child keeps its parent's CONTROL GROUP,
and `systemctl restart` kills a unit's whole control group by default. Three
trainings died that way -- two at epochs 79 and 80, one at epoch 12 -- with
setsid faithfully doing what it promises and the cgroup taking them anyway.
So a job is started inside a transient scope of its own, under the USER
manager (`systemd-run --user --scope`), which puts it in
user@<uid>.service/app.slice/<job>.scope. Nothing the system unit does can
reach it: it can be restarted, stopped, killed or crash, and the run carries
on writing its log and its exit status. Where there is no user manager to ask,
the plain orphan is used and the record says so, so the page can be honest
about which kind of job somebody is looking at.

    data/dashboard/jobs/<id>/job.json   what it is, and what happened
                            /log        stdout and stderr, appended
                            /exit       the exit status, written by the shell

WHY THE SHELL WRITES THE EXIT STATUS. The child is deliberately orphaned --
start_new_session=True puts it in its own session so that killing, restarting
or crashing the dashboard leaves it running. That also means nobody is left to
wait() for it, so its exit code would be lost to the reaper. The command is
therefore run under `sh -c` with the status written to a file afterwards: the
one place that always knows how a process ended is the shell that ran it.

WHY THE PID ALONE IS NOT AN ANSWER. A recorded pid is a claim about a moment.
After a reboot -- or after enough process churn -- that number belongs to
somebody else, and asking os.kill(pid, 0) about it cheerfully reports the job
as still running for ever. Two facts are recorded with it: the boot id, which
changes when the numbering starts again, and the process's own start time out
of /proc, which is what tells one pid 4021 from the next. Both have to match
before a live pid is believed.

LANES. One GPU, so one training run: a lane is a name that at most one job may
hold at a time, and submitting into a busy lane is refused with the job that
holds it rather than queued behind it. Refusing is the honest answer -- a queue
implies something will come along later to run the next one, and nothing here
would.

NOTHING SECRET GOES IN job.json. The argv is recorded because reproducing a
run means knowing exactly what was run, and it is shown on a page; a token
passed on a command line would be recorded and shown with it. Pass secrets by
environment, which is not written down.
"""
import argparse
import errno
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOBS_DIR = os.path.join(REPO, 'data', 'dashboard', 'jobs')

# A job id is a sortable timestamp, the kind, and enough randomness that two
# submitted in the same second cannot collide. Readable on purpose: it is the
# directory name, and it is what somebody greps for a week later.
ID_RE = re.compile(r'^[0-9]{8}-[0-9]{6}-[a-z0-9_]{1,24}-[0-9a-f]{6}$')
KIND_RE = re.compile(r'^[a-z0-9_]{1,24}$')

# Every lane this module knows about. A job in an unknown lane is refused
# rather than silently running unlaned, because the lane is the only thing
# stopping two trainings from sharing one GPU.
LANES = ('build', 'train')

# How long a cancelled job gets to stop politely before it is killed.
TERM_GRACE_S = 8.0

# The user manager, and how to reach it from a SYSTEM service. A unit running
# as User=biodiv under the system manager has no session bus in its
# environment, so the two variables that address the user bus are supplied
# here when they are missing -- this is the ordinary way a system service
# talks to a user manager, not a trick.
_SYSTEMD = {'at': 0.0, 'ok': None}
SYSTEMD_PROBE_S = 300.0


def _user_bus_env():
    """{XDG_RUNTIME_DIR, DBUS_SESSION_BUS_ADDRESS} for this user, or {}."""
    run = os.environ.get('XDG_RUNTIME_DIR') or '/run/user/%d' % (os.getuid(),)
    bus = os.path.join(run, 'bus')
    if not os.path.exists(bus):
        return {}
    return {'XDG_RUNTIME_DIR': run,
            'DBUS_SESSION_BUS_ADDRESS': 'unix:path=%s' % (bus,)}


def systemd_ready(env=None, ttl=SYSTEMD_PROBE_S):
    """Can this process put a job in a scope of its own?

    Asked of the machine rather than assumed: there is no user manager in a
    container, on a mac, or for a user without lingering enabled, and a job
    that cannot be started is worse than a job that is merely killable.
    Memoised, because it is a subprocess and the answer changes about as often
    as the machine boots.
    """
    now = time.time()
    if _SYSTEMD['ok'] is not None and now - _SYSTEMD['at'] < ttl:
        return _SYSTEMD['ok']
    ok = False
    bus = _user_bus_env()
    if bus and _which('systemd-run') and _which('systemctl'):
        try:
            run_env = dict(os.environ, **bus)
            got = subprocess.run(['systemctl', '--user', 'show', '-p',
                                  'Version', '--value'],
                                 capture_output=True, text=True, timeout=10,
                                 env=run_env)
            ok = got.returncode == 0 and bool((got.stdout or '').strip())
        except (OSError, subprocess.SubprocessError):
            ok = False
    _SYSTEMD.update(at=now, ok=ok)
    return ok


def _which(name):
    for d in (os.environ.get('PATH') or '/usr/bin:/bin').split(os.pathsep):
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def unit_active(unit):
    """Is this job's scope still running? False for anything unanswerable."""
    if not unit or not _which('systemctl'):
        return False
    try:
        got = subprocess.run(['systemctl', '--user', 'is-active', unit],
                             capture_output=True, text=True, timeout=10,
                             env=dict(os.environ, **_user_bus_env()))
    except (OSError, subprocess.SubprocessError):
        return False
    return (got.stdout or '').strip() == 'active'

# A job that has only just started, whose process is already gone and whose
# exit file has not appeared yet, is a RACE, not a job that vanished. The
# window is milliseconds -- the wrapper writes the status and exits -- but a
# job read inside it used to be written off as `lost` permanently, and `lost`
# is a final state that nothing revisits.
LOST_GRACE_S = 5.0

# What `state` can be. `lost` is its own answer and not a failure: the process
# is gone and left no exit status, which means it was killed from outside or
# the machine went down under it -- a different thing from a run that failed,
# and the only honest label for "nobody knows".
STATES = ('running', 'done', 'failed', 'cancelled', 'lost')


# ── the file the state lives in ─────────────────────────────────────────────

def _boot_id():
    """This boot's id, so a pid from a previous one is never believed."""
    try:
        with open('/proc/sys/kernel/random/boot_id') as fh:
            return fh.read().strip()
    except OSError:
        return ''


def _stat(pid):
    """(state, starttime) out of /proc/<pid>/stat, or (None, None).

    The comm field can contain spaces and brackets -- a process really can be
    called `(sleep) 60` -- so the parse starts after the LAST ')' rather than
    splitting the line.
    """
    try:
        with open('/proc/%d/stat' % (int(pid),)) as fh:
            raw = fh.read()
        rest = raw[raw.rindex(')') + 2:].split()
        return rest[0], int(rest[19])
    except (OSError, ValueError, IndexError):
        return None, None


def _proc_start(pid):
    """The process's own start time, which tells one pid 4021 from the next."""
    return _stat(pid)[1]


def _alive_pid(pid):
    """Bare "is there a process with this number", with no identity check."""
    try:
        os.kill(int(pid), 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM


def _write(path, obj):
    """One job record, atomically. A half-written job.json is a job whose
    state cannot be read, and this file is read by another process while it
    is being written."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path) as fh:
            got = json.load(fh)
        return got if isinstance(got, dict) else None
    except (OSError, ValueError):
        return None


def job_dir(job_id):
    return os.path.join(JOBS_DIR, job_id)


def _paths(job_id):
    d = job_dir(job_id)
    return d, os.path.join(d, 'job.json'), os.path.join(d, 'log'), \
        os.path.join(d, 'exit')


def _pid_file(job_id):
    """Where the wrapper writes its own pid, as its first act.

    `echo $!` reports whatever the outer shell backgrounded, and once the work
    is started through systemd-run that is not necessarily the process holding
    it: systemd-run may exec into the wrapper or fork it, and either way the
    pid worth recording is the one INSIDE the scope. The wrapper knows it, so
    the wrapper says it.
    """
    return os.path.join(job_dir(job_id), 'pid')


# ── is it still going ───────────────────────────────────────────────────────

def alive(job):
    """Whether this job's process is still the process it started.

    Three things must agree: the machine has not rebooted since, a process
    with that pid exists, and it is the SAME process -- same start time. Any
    one of them failing means the pid is a number somebody else may be using.
    """
    if not job or not job.get('pid'):
        return False
    if job.get('boot_id') and job['boot_id'] != _boot_id():
        return False
    pid = int(job['pid'])
    try:
        os.kill(pid, 0)
    except OSError as e:
        if e.errno == errno.ESRCH:
            return False
        if e.errno != errno.EPERM:      # EPERM: alive, and not ours to signal
            return False
    state, start = _stat(pid)
    # A ZOMBIE IS NOT ALIVE. os.kill(pid, 0) succeeds against a process that
    # has exited and not yet been reaped, so a killed job read as running for
    # ever -- which is exactly what cancel then waited on. It cannot happen to
    # a job started through submit() any more, because the shell that holds
    # the work is orphaned deliberately and reaped by init, but a pid is a
    # pid and this is the honest test.
    if state == 'Z':
        return False
    want = job.get('pid_start')
    if want is not None and start != want:
        return False
    return True


def group_alive(pgid):
    """Whether ANY process is still in a job's process group.

    The recorded pid is the shell that holds the work, and killing that shell
    does not kill work that ignored the signal -- so "the pid is gone" is not
    "the job is gone". The group is: signal 0 to the group succeeds while a
    single member of it is left.
    """
    if pgid is None:
        return False
    try:
        os.killpg(int(pgid), 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM     # alive, and not ours to signal


def _running(job):
    """Is there still work behind this job, by any witness available?

    Three, in cost order, and any one of them is enough:

      * the recorded pid is still the process it was;
      * something is left in its process group -- the pid recorded is a
        wrapper, and a trainer that outlives it is still the job;
    The scope is NOT one of them. It answers about a control group, and an
    emptied group can outlive the last process in it by a moment -- long
    enough for a cancelled job to still read as running, which holds the lane
    against the next one. The scope's job is to keep the work alive across a
    restart of this process; the pid then answers for it, because the wrapper
    is inside the scope too and its pid is still its own.
    """
    return alive(job) or group_alive(job.get('pgid'))


def _settle(job, path, exit_path):
    """Fill in how a job ended, once it has. Returns the job either way.

    The exit file is written by the shell after the command returns, so its
    presence is the end of the job and its contents are the status. A process
    that is gone with no exit file ended in a way it could not record.
    """
    if job.get('state') != 'running':
        return job
    raw = None
    try:
        with open(exit_path) as fh:
            raw = fh.read().strip()
    except OSError:
        pass
    if raw is not None and raw != '':
        try:
            code = int(raw)
        except ValueError:
            code = -1
        job['exit_code'] = code
        # 143 is SIGTERM, 130 is SIGINT: a job that was asked to stop did
        # stop, and calling that a failure sends somebody looking for a bug
        # in a run they cancelled themselves.
        job['state'] = ('done' if code == 0 else
                        'cancelled' if job.get('cancel_at') or code in (130, 143)
                        else 'failed')
    elif _running(job):
        return job
    else:
        # ONE MORE LOOK BEFORE WRITING IT OFF. Between the liveness test above
        # and this line the wrapper may have finished and written its status;
        # `lost` is final, so it is worth the second read.
        try:
            with open(exit_path) as fh:
                raw = fh.read().strip()
        except OSError:
            raw = ''
        if raw:
            return _settle(dict(job, state='running'), path, exit_path)
        started = job.get('started_at') or job.get('created_at') or 0
        # ...unless somebody asked for it to stop, in which case there is no
        # mystery about where the process went and no reason to hold the lane
        # shut for another five seconds.
        if not job.get('cancel_at') and time.time() - started < LOST_GRACE_S:
            return job                    # too soon to call it anything
        job['state'] = 'cancelled' if job.get('cancel_at') else 'lost'
        job['exit_code'] = None
    job['ended_at'] = int(time.time())
    _write(path, job)
    return job


def read(job_id, settle=True):
    """One job, with its state brought up to date. None if there is no such
    job, which is also the answer for an id somebody made up."""
    if not isinstance(job_id, str) or not ID_RE.match(job_id):
        return None
    _d, path, _log, exit_path = _paths(job_id)
    job = _read_json(path)
    if job is None:
        return None
    return _settle(job, path, exit_path) if settle else job


def listing(limit=60, kind=None, settle=True):
    """Every job, newest first. The id sorts by time, so the directory does."""
    try:
        names = sorted(os.listdir(JOBS_DIR), reverse=True)
    except OSError:
        return []
    out = []
    for name in names:
        if not ID_RE.match(name):
            continue
        job = read(name, settle=settle)
        if job is None:
            continue
        if kind and job.get('kind') != kind:
            continue
        out.append(job)
        if len(out) >= limit:
            break
    return out


def lane_holder(lane):
    """The job holding a lane, or None. Settles as it goes, so a lane is
    never held by a run that has already finished."""
    for job in listing(limit=200, settle=True):
        if job.get('lane') == lane and job.get('state') == 'running':
            return job
    return None


# ── starting one ────────────────────────────────────────────────────────────

def _new_id(kind, now=None):
    stamp = time.strftime('%Y%m%d-%H%M%S',
                          time.localtime(now if now else time.time()))
    return '%s-%s-%s' % (stamp, kind, os.urandom(3).hex())


def submit(kind, argv, lane, label='', by='', cwd=None, env=None, meta=None,
           now=None):
    """Start something that will outlive this process.

        {'ok', 'job', 'message'}

    argv is a list, never a string: a command assembled by joining strings is
    a command somebody can put a semicolon into, and the values here come from
    a web form.
    """
    if not KIND_RE.match(str(kind or '')):
        return {'ok': False, 'job': None, 'message': 'That is not a job kind.'}
    if lane not in LANES:
        return {'ok': False, 'job': None,
                'message': 'There is no %r lane to run that in.' % (lane,)}
    if not argv or not isinstance(argv, (list, tuple)) or \
            not all(isinstance(a, str) for a in argv):
        return {'ok': False, 'job': None,
                'message': 'A job is a list of arguments, not a command line.'}
    held = lane_holder(lane)
    if held is not None:
        return {'ok': False, 'job': held,
                'message': '%s is already running (%s). Wait for it or cancel '
                           'it first.' % (held.get('label') or held['id'],
                                          held['id'])}
    ts = int(time.time() if now is None else now)
    job_id = _new_id(kind, ts)
    d, path, log, exit_path = _paths(job_id)
    os.makedirs(d, exist_ok=True)

    job = {'id': job_id, 'kind': kind, 'lane': lane,
           'label': str(label or ''), 'by': str(by or ''),
           'argv': list(argv), 'cwd': cwd or REPO,
           'created_at': ts, 'started_at': None, 'ended_at': None,
           'pid': None, 'boot_id': _boot_id(), 'pid_start': None,
           'exit_code': None, 'state': 'running', 'cancel_at': None,
           # The scope this job runs in, or None when there was no user
           # manager to make one. Recorded because it is the difference
           # between a run that survives `systemctl restart` and one that
           # does not, and the page says which.
           'unit': None, 'detached': False,
           'meta': meta or {}}
    _write(path, job)

    # THE SHELL RECORDS THE STATUS. Nobody wait()s for an orphan, so without
    # this the exit code dies with the process. shlex.quote on every argument,
    # because this string is handed to sh.
    #
    # AND THE WORK IS BACKGROUNDED INSIDE THAT SHELL, so it is orphaned the
    # moment the shell exits and init adopts it. Run as a direct child instead,
    # the dashboard would have to wait() for every job it ever started: a
    # finished run stays a zombie until somebody reaps it, os.kill(pid, 0)
    # answers yes to a zombie, and a cancelled job therefore read as running
    # for ever. `echo $!` hands back the pid of the thing that is actually
    # doing the work.
    #
    # `} >/dev/null 2>&1 &` matters as much as the `&`. Without it the
    # backgrounded group inherits this pipe and holds it open for the life of
    # the job, so reading the pid back would block until the job finished --
    # which for a training run is hours, and for a fast one meant the pid was
    # already gone by the time it arrived.
    pid_path = _pid_file(job_id)
    inner = '{ printf %%s "$$" >%s; %s >>%s 2>&1; printf %%s "$?" >%s; }' % (
        shlex.quote(pid_path),
        ' '.join(shlex.quote(a) for a in argv),
        shlex.quote(log), shlex.quote(exit_path))
    run_env = dict(os.environ)
    run_env.update({str(k): str(v) for k, v in (env or {}).items()})
    run_env.setdefault('PYTHONUNBUFFERED', '1')
    # THE WHOLE WRAPPER GOES IN THE SCOPE, not just the work. systemd-run
    # --scope EXECS the command it is given, so the shell that records the
    # exit status is itself inside the new control group -- which is what
    # keeps a restart of the dashboard from taking the status file with it,
    # and what keeps the recorded pid a pid that still exists afterwards.
    unit = None
    if systemd_ready():
        unit = '%s.scope' % (job_id,)
        run_env.update(_user_bus_env())
        # stdout to /dev/null, stderr to the LOG. The first of those keeps
        # the backgrounded job off this pipe, which is what lets the pid be
        # read back at once; the second is why a job that never started says
        # so. It used to send both to /dev/null, and a scope that could not be
        # created left a job in state `lost` with an empty log and nothing
        # anywhere to say why.
        line = ('systemd-run --user --scope --collect --quiet '
                '--unit=%s -- /bin/sh -c %s >/dev/null 2>>%s & echo $!'
                % (shlex.quote(unit), shlex.quote(inner), shlex.quote(log)))
    else:
        line = '%s >/dev/null 2>>%s & echo $!' % (inner, shlex.quote(log))
    try:
        # start_new_session puts it in its own session AND its own process
        # group. The first is what makes it survive this process; the second
        # is what lets cancel take the whole tree, which matters because a
        # training run is a parent with a dozen dataloader workers under it.
        proc = subprocess.Popen(
            ['/bin/sh', '-c', line], cwd=job['cwd'], env=run_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True)
        # The outer shell exits as soon as it has backgrounded the work, so
        # this returns at once -- and reaps it, which is the point.
        out, _err = proc.communicate(timeout=20)
        pid = int((out or b'').strip() or 0)
        if pid <= 0:
            raise OSError('the shell did not report a pid')
        # ...and then the wrapper's own pid, which is the one inside the
        # scope. Waited for briefly: it is written before the work starts, so
        # this is microseconds in practice, and a job whose wrapper never got
        # far enough to write it is a job whose `$!` is the best there is.
        end = time.time() + 2.0
        while time.time() < end:
            try:
                with open(pid_path) as fh:
                    got = int((fh.read() or '').strip() or 0)
                if got > 0:
                    pid = got
                    break
            except (OSError, ValueError):
                pass
            if not _alive_pid(pid):
                break                     # it is over; nothing more is coming
            time.sleep(0.01)
        # The group, recorded now while there is certainly something to ask.
        # Later it is the only way to tell a job whose shell was killed from
        # one whose work is really over: the trainer under it carries on,
        # reparented, with the GPU still busy.
        try:
            pgid = os.getpgid(pid)
        except OSError:                   # it finished before we looked
            pgid = None
    except Exception as e:                # noqa: BLE001 - report, never raise
        job.update(state='failed', ended_at=ts, exit_code=None,
                   error='%s: %s' % (type(e).__name__, e))
        _write(path, job)
        return {'ok': False, 'job': job,
                'message': 'That would not start (%s).' % (type(e).__name__,)}
    job.update(pid=pid, pid_start=_proc_start(pid), pgid=pgid,
               unit=unit, detached=bool(unit),
               started_at=int(time.time()))
    # DID IT ACTUALLY START? The wrapper writes its pid before it runs
    # anything, so no pid file and no wrapper left means the work never began
    # -- a scope that could not be created, a binary that is not there. That
    # used to be reported as a job that started fine and then, five seconds
    # later, as one that was `lost`; the lane was held for those five seconds
    # and the reason was nowhere. Said now, at the click, with whatever the
    # shell complained about.
    if not os.path.exists(pid_path) and not _alive_pid(pid) \
            and not os.path.exists(exit_path):
        why = ''
        try:
            with open(log) as fh:
                why = plain(fh.read())[-300:].strip().splitlines()[-1]
        except (OSError, IndexError):
            why = ''
        job.update(state='failed', ended_at=int(time.time()),
                   error=why or 'the job did not start')
        _write(path, job)
        return {'ok': False, 'job': job,
                'message': 'That did not start: %s'
                           % (why or 'no reason was recorded.')}
    _write(path, job)
    return {'ok': True, 'job': job, 'message': ''}


# ── stopping one ────────────────────────────────────────────────────────────

def cancel(job_id, grace=TERM_GRACE_S, now=None):
    """Ask a job to stop, then insist.

    Signals the process GROUP, not the process: the pid recorded is the shell,
    and the work is its child with its own children under that. TERM to the
    group first so ultralytics can close its writers, KILL after the grace
    period for whatever ignored it, and -- for a job in a scope of its own --
    `systemctl --user stop` after that, which empties the control group
    whatever is left in it.
    """
    job = read(job_id)
    if job is None:
        return {'ok': False, 'job': None, 'message': 'No such job.'}
    if job.get('state') != 'running':
        # Not an error. Two clicks, or a job that finished while the page was
        # open, is a race rather than a mistake worth a message.
        return {'ok': True, 'job': job, 'message': ''}
    job['cancel_at'] = int(time.time() if now is None else now)
    _d, path, _log, _exit = _paths(job['id'])
    _write(path, job)
    if not alive(job):
        return {'ok': True, 'job': read(job_id), 'message': ''}
    try:
        pgid = os.getpgid(int(job['pid']))
    except OSError:
        pgid = None

    def stopped():
        # THE GROUP, NOT THE PID. TERM kills the recorded shell whether or not
        # the work under it ignored the signal, so waiting on the pid stopped
        # the moment the shell died -- and returned success with the training
        # still on the GPU. Then the lane came free and a second run could
        # start on a card that was still busy.
        return not group_alive(pgid) and not alive(job)

    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                os.kill(int(job['pid']), sig)
        except OSError:
            break
        end = time.time() + wait
        while time.time() < end:
            if stopped():
                break
            time.sleep(0.1)
        if stopped():
            break
    # ...AND THE SCOPE, ALWAYS. Signalling the group is what gives ultralytics
    # its chance to close its writers, and it is enough for every run that
    # keeps the group it was started in. A worker that made its own session
    # would survive it and keep the GPU -- systemd owns the whole control
    # group and empties it. Unconditional, not just when something is left:
    # an emptied scope that nobody tore down lingers in `systemctl --user
    # list-units` and is one more thing to explain later.
    if job.get('unit'):
        try:
            subprocess.run(['systemctl', '--user', 'stop', job['unit']],
                           capture_output=True, timeout=20,
                           env=dict(os.environ, **_user_bus_env()))
        except (OSError, subprocess.SubprocessError):
            pass
    return {'ok': True, 'job': read(job_id), 'message': ''}


# ── watching one ────────────────────────────────────────────────────────────

# \x1b[K and friends: any CSI sequence. tqdm writes one per redraw.
_ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b')


def plain(text):
    """Terminal output as text: ANSI stripped, \r redraws collapsed.

    A tqdm bar is one logical line REDRAWN hundreds of times, each redraw
    separated by a carriage return -- a single epoch of training is a 45,000
    character "line" of which only the last 150 characters are the state a
    reader wants. On a terminal the \r overwrites in place; in a log box it
    wrapped into thirty lines of bar fragments. Keeping the text after the
    last \r per line is exactly what the terminal would have shown.
    """
    text = _ANSI_RE.sub('', text or '')
    # rstrip first: a CRLF line ENDS in \r, and taking the text after the
    # last one turned every Windows-ended line into nothing at all
    return '\n'.join(line.rstrip('\r').rsplit('\r', 1)[-1]
                     for line in text.split('\n'))


def tail(job_id, nbytes=16000):
    """The end of a job's output, as a terminal would have shown it.

    Read from the end rather than loaded whole: a training log is megabytes
    by the time it matters, and this is polled. The RAW window read is many
    times the text returned, because redraw storms inflate the bytes: one
    epoch is ~45KB raw and ~150 characters once collapsed, so a 24KB raw
    window could land entirely inside one bar -- the old readline()-past-
    the-partial-line trick then consumed everything and returned an EMPTY
    tail for a job that was mid-epoch and healthy.
    """
    if not isinstance(job_id, str) or not ID_RE.match(job_id):
        return ''
    _d, _p, log, _e = _paths(job_id)
    raw_window = max(nbytes * 16, 1 << 19)
    try:
        size = os.path.getsize(log)
        with open(log, 'rb') as fh:
            if size > raw_window:
                fh.seek(size - raw_window)
            raw = fh.read()
    except OSError:
        return ''
    text = plain(raw.decode('utf-8', 'replace'))
    if size > raw_window:
        # drop the line the seek landed inside -- AFTER collapsing, when a
        # "line" is once again something a person would call one
        text = text.partition('\n')[2]
    return text[-nbytes:]


def progress(job_id):
    """How far along, when the job says so.

    A job writes `PROGRESS <done> <total> <what>` on its own line and this
    reports the last one. Deliberately a convention rather than an interface:
    a builder that never prints one is not broken, it just cannot be drawn as
    a bar, and nothing here parses ultralytics' output pretending to know it.
    """
    last = None
    for line in tail(job_id, 4000).splitlines():
        if line.startswith('PROGRESS '):
            bits = line.split(None, 3)
            if len(bits) >= 3:
                try:
                    last = {'done': int(bits[1]), 'total': int(bits[2]),
                            'what': bits[3] if len(bits) > 3 else ''}
                except ValueError:
                    pass
    return last


# ── housekeeping ────────────────────────────────────────────────────────────

def forget(job_id):
    """Drop one finished job's record, log and all.

    Refuses a running one: the directory is where the shell writes its exit
    status, so deleting it under a live job loses the only evidence of how it
    ended and leaves work running that nothing on the page can still reach.
    """
    if not isinstance(job_id, str) or not ID_RE.match(job_id):
        return {'ok': False, 'message': 'that is not a job id'}
    job = read(job_id)
    if job is None:
        return {'ok': False, 'message': 'no such job'}
    if job.get('state') == 'running':
        return {'ok': False, 'message': 'that job is still running -- stop it '
                                        'first'}
    # ...and a job whose recorded process is gone is not necessarily a job
    # whose WORK is gone: kill the shell and the trainer under it carries on,
    # reparented, with the GPU still busy. Clearing the record there leaves
    # work nothing on the page can reach.
    if job.get('pgid') and group_alive(job['pgid']):
        return {'ok': False,
                'message': 'the work this job started is still running -- '
                           'stop it first'}
    import shutil
    try:
        shutil.rmtree(job_dir(job_id))
    except OSError as e:
        return {'ok': False, 'message': 'could not remove it: %s' % (e,)}
    return {'ok': True, 'message': ''}


def prune(keep=200, older_than_s=30 * 86400, now=None):
    """Drop the records of finished jobs nobody will read again.

    Never touches a running one, and never touches the newest `keep` -- the
    log of a build is how somebody works out what a dataset is made of.
    """
    ts = int(time.time() if now is None else now)
    import shutil
    gone = []
    for i, job in enumerate(listing(limit=10000)):
        if i < keep or job.get('state') == 'running':
            continue
        end = job.get('ended_at') or job.get('created_at') or ts
        if ts - end < older_than_s:
            continue
        try:
            shutil.rmtree(job_dir(job['id']))
            gone.append(job['id'])
        except OSError:
            pass
    return gone


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    for name in ('show', 'tail', 'cancel', 'forget'):
        p = sub.add_parser(name)
        p.add_argument('job_id')
    a = ap.parse_args(argv)
    if a.cmd == 'list':
        for job in listing():
            print('%-42s %-8s %-7s %s' % (job['id'], job.get('lane', ''),
                                          job.get('state', ''),
                                          job.get('label', '')))
        return 0
    if a.cmd == 'show':
        job = read(a.job_id)
        if job is None:
            print('no such job', file=sys.stderr)
            return 1
        print(json.dumps(job, indent=1, sort_keys=True))
        return 0
    if a.cmd == 'tail':
        sys.stdout.write(tail(a.job_id, 64000))
        return 0
    if a.cmd == 'forget':
        got = forget(a.job_id)
        if not got['ok']:
            print(got['message'], file=sys.stderr)
            return 1
        print('forgotten')
        return 0
    got = cancel(a.job_id)
    print(got['message'] or (got['job'] or {}).get('state', ''))
    return 0 if got['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
