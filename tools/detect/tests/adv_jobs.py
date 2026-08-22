#!/usr/bin/env python3
"""The job runner: work that outlives whatever started it.

Everything here is driven against real processes, because every defect this
module can have is a defect about processes and none of it can be reasoned
about from the source.

Four things have to hold, and three of them fail silently.

IT MUST OUTLIVE ITS PARENT. That is the whole feature: a training run started
from a browser tab has to survive the tab, the dashboard restarting, and the
dashboard crashing. The check spawns from a process that then exits, and reads
the outcome from a third process that never met the job.

SUBMIT MUST RETURN AT ONCE. It reads the pid back through a pipe, and the
first version of that pipe was inherited by the backgrounded work -- so
reading the pid blocked until the job ENDED. A dataset build made the page
hang for four minutes and a training run would have hung it for hours.

A ZOMBIE IS NOT ALIVE. os.kill(pid, 0) answers yes to a process that has
exited and not been reaped, so a cancelled job read as running for ever and
cancel() waited on a corpse. The job is orphaned to init now so it can never
be an unreaped child of anybody's, and the state is checked as well.

A PID IS NOT AN IDENTITY. Recorded across a reboot, or after enough churn, it
belongs to somebody else -- and a liveness check that trusts it reports a job
as running when the number is now the display manager's.

Run: python tools/detect/tests/adv_jobs.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DASH = os.path.join(REPO, 'tools', 'dashboard')
sys.path.insert(0, DASH)


def _settle(jobs, job_id, want=('done', 'failed', 'cancelled', 'lost'),
            timeout=20.0):
    """Wait for a job to reach a final state, or give up and say so."""
    end = time.time() + timeout
    job = jobs.read(job_id)
    while time.time() < end:
        job = jobs.read(job_id)
        if job and job.get('state') in want:
            return job
        time.sleep(0.05)
    return job


def basic_checks(bad, jobs):
    """A job runs, ends, and says how it ended."""
    t0 = time.time()
    got = jobs.submit('probe', ['/bin/sh', '-c', 'echo hi; sleep 2; echo bye'],
                      lane='build', label='a probe', by='admin')
    took = time.time() - t0
    if not got['ok']:
        bad.append('a plain job would not start: %s' % (got['message'],))
        return
    job = got['job']
    # SUBMIT RETURNS AT ONCE. Blocking here is invisible until the job is a
    # long one, and then it is a page that never loads.
    if took > 3.0:
        bad.append('submit took %.1fs for a job that runs for 2 -- it is '
                   'waiting for the work instead of starting it' % (took,))
    if not jobs.alive(job):
        bad.append('a job that has just started does not read as alive')
    if jobs.read(job['id'], settle=True)['state'] != 'running':
        bad.append('a running job does not read as running')
    # ...AND IT IS NOT A CHILD OF THIS PROCESS. An unreaped child becomes a
    # zombie, and a zombie answers yes to os.kill(pid, 0) for ever.
    try:
        with open('/proc/%d/stat' % (job['pid'],)) as fh:
            raw = fh.read()
        ppid = int(raw[raw.rindex(')') + 2:].split()[1])
        if ppid == os.getpid():
            bad.append('the job is a direct child of the process that '
                       'started it, so it becomes a zombie nobody reaps')
    except (OSError, ValueError, IndexError):
        bad.append('the job process could not be inspected right after start')
    done = _settle(jobs, job['id'])
    if not done or done['state'] != 'done' or done['exit_code'] != 0:
        bad.append('a job that succeeds reads as %r/%r'
                   % (done and done['state'], done and done['exit_code']))
    if jobs.tail(job['id']).split() != ['hi', 'bye']:
        bad.append('the log did not capture stdout: %r'
                   % (jobs.tail(job['id']),))

    # a failure is a failure, with its status
    f = jobs.submit('probe', ['/bin/sh', '-c', 'echo no >&2; exit 7'],
                    lane='build')['job']
    got = _settle(jobs, f['id'])
    if not got or got['state'] != 'failed' or got['exit_code'] != 7:
        bad.append('a job that exits 7 reads as %r/%r'
                   % (got and got['state'], got and got['exit_code']))
    if 'no' not in jobs.tail(f['id']):
        bad.append('stderr is not in the log, so a failure explains nothing')


def redraw_checks(bad, jobs):
    """The log box shows what a terminal would have shown.

    tqdm redraws one logical line hundreds of times, each redraw behind a
    carriage return with an ANSI erase code -- one epoch of training is a
    45,000-character "line" whose last 150 characters are the state. The old
    tail seeked into the middle of that, readline()d past everything, and
    returned an EMPTY tail for a job that was mid-epoch and healthy; what it
    did return elsewhere wrapped into thirty lines of bar fragments.
    """
    job = jobs.submit('probe', ['/bin/sh', '-c', (
        'awk \'BEGIN{for(i=0;i<4000;i++)'
        'printf "\\rprogress %d/4000 \\033[K", i;'
        'print "\\rprogress 4000/4000 done"}\'; echo tail line')],
        lane='build')['job']
    got = _settle(jobs, job['id'])
    if not got or got['state'] != 'done':
        bad.append('the redraw fixture did not run, so this proves nothing')
        return
    t = jobs.tail(job['id'], 2000)
    if not t.strip():
        bad.append('a log that is one long redrawn line tails EMPTY -- the '
                   'page shows nothing for a job that is mid-epoch and fine')
        return
    if '\r' in t or '\x1b' in t:
        bad.append('the tail still carries carriage returns or ANSI codes, '
                   'which the log box renders as text all over the place')
    if 'progress 4000/4000 done' not in t:
        bad.append('the final state of the redrawn line is missing: %r'
                   % (t[-120:],))
    if t.count('progress ') > 2:
        bad.append('%d redraws of one line survive in the tail; a terminal '
                   'would have shown one' % (t.count('progress '),))
    if 'tail line' not in t:
        bad.append('the line after the bar is missing')


def survives_its_parent(bad, jobs_dir):
    """THE FEATURE. Spawn from a process that exits, read from a third.

    Driven as three separate interpreters on purpose: a check that keeps the
    module loaded proves nothing about a dashboard that has been restarted.
    """
    spawn = os.path.join(jobs_dir, '_spawn.py')
    with open(spawn, 'w') as fh:
        fh.write(
            'import sys\n'
            'sys.path.insert(0, %r)\n'
            'import jobs\n'
            'jobs.JOBS_DIR = %r\n'
            "got = jobs.submit('probe', ['/bin/sh','-c',"
            "'for i in 1 2 3 4 5 6; do echo tick; sleep 1; done'],"
            " lane='train', label='outlives its parent')\n"
            "print(got['job']['id'] if got['ok'] else 'FAILED')\n"
            % (DASH, jobs_dir))
    out = subprocess.run([sys.executable, spawn], capture_output=True,
                         text=True, timeout=60)
    job_id = (out.stdout or '').strip().splitlines()[-1:] or ['']
    job_id = job_id[0]
    if not job_id or job_id == 'FAILED':
        bad.append('the spawning process could not start a job: %r'
                   % (out.stderr[-300:],))
        return
    # the parent is gone; a second interpreter asks whether the work is not
    probe = (
        'import sys; sys.path.insert(0, %r)\n'
        'import jobs; jobs.JOBS_DIR = %r\n'
        'j = jobs.read(%r)\n'
        "print(j['state'], jobs.alive(j), len(jobs.tail(%r).split()))\n"
        % (DASH, jobs_dir, job_id, job_id))
    mid = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                         text=True, timeout=60)
    say = (mid.stdout or '').strip().split()
    if say[:2] != ['running', 'True']:
        bad.append('a job did not survive the process that started it: %r %r'
                   % (mid.stdout.strip(), mid.stderr[-200:]))
        return
    # ...and a THIRD interpreter reads how it ended, having never met it
    end = time.time() + 40
    while time.time() < end:
        fin = subprocess.run([sys.executable, '-c', probe],
                             capture_output=True, text=True, timeout=60)
        got = (fin.stdout or '').strip().split()
        if got and got[0] != 'running':
            if got[0] != 'done':
                bad.append('an orphaned job ended as %r, not done' % (got[0],))
            elif int(got[2]) != 6:
                bad.append('an orphaned job logged %s lines, not 6 -- it was '
                           'cut short when its parent went' % (got[2],))
            return
        time.sleep(0.5)
    bad.append('an orphaned job never finished')


def cancel_checks(bad, jobs):
    """Stopping one, including the half that will not stop politely."""
    s = jobs.submit('probe', ['/bin/sh', '-c', 'trap "" TERM; sleep 60'],
                    lane='build')['job']
    time.sleep(0.4)
    pgid = os.getpgid(s['pid'])
    t0 = time.time()
    got = jobs.cancel(s['id'], grace=1.0)
    took = time.time() - t0
    if jobs.alive(got['job']):
        bad.append('a job that ignores TERM survived cancel (%.1fs) -- the '
                   'KILL never landed, or a zombie read as alive' % (took,))
    # ...AND SO DID EVERYTHING UNDER IT. alive() follows the shell that was
    # recorded, and TERM kills that shell whether or not the work under it
    # ignored the signal -- so "the job is gone" can be true while the thing
    # actually burning the GPU is still burning it. The group is the answer.
    time.sleep(0.4)
    left = subprocess.run(['pgrep', '-g', str(pgid)], capture_output=True,
                          text=True).stdout.split()
    if left:
        bad.append('cancel reported success with %d process(es) still in the '
                   "job's group -- the shell died and the work did not"
                   % (len(left),))
    if got['job']['state'] != 'cancelled':
        bad.append('a cancelled job reads as %r -- a run somebody stopped on '
                   'purpose is not a failure to go looking for'
                   % (got['job']['state'],))
    if took > 6:
        bad.append('cancel took %.1fs to insist' % (took,))

    # THE WHOLE TREE. A training run is a parent with a dozen dataloader
    # workers under it; signalling the pid alone leaves them on the GPU.
    c = jobs.submit('probe',
                    ['/bin/sh', '-c', 'sleep 120 & sleep 120 & wait'],
                    lane='build')['job']
    time.sleep(0.5)
    try:
        pgid = os.getpgid(c['pid'])
    except OSError:
        bad.append('a running job has no process group to signal')
        return
    before = subprocess.run(['pgrep', '-g', str(pgid)], capture_output=True,
                            text=True).stdout.split()
    jobs.cancel(c['id'], grace=1.0)
    time.sleep(0.6)
    after = subprocess.run(['pgrep', '-g', str(pgid)], capture_output=True,
                           text=True).stdout.split()
    if len(before) < 3:
        bad.append('the tree check never got a tree to kill (%d processes)'
                   % (len(before),))
    elif after:
        bad.append('%d of %d processes outlived cancel -- the children were '
                   'left behind' % (len(after), len(before)))
    # cancelling something already finished is a race, not an error
    fin = jobs.submit('probe', ['/bin/true'], lane='build')['job']
    _settle(jobs, fin['id'])
    if not jobs.cancel(fin['id'])['ok']:
        bad.append('cancelling a finished job is reported as a failure')


def lane_checks(bad, jobs):
    """One GPU, one training run."""
    a = jobs.submit('probe', ['/bin/sh', '-c', 'sleep 30'], lane='train',
                    label='holds the lane')
    if not a['ok']:
        bad.append('the first job in a lane was refused')
        return
    b = jobs.submit('probe', ['/bin/true'], lane='train')
    if b['ok']:
        bad.append('two jobs took the same lane at once — two trainings on '
                   'one GPU')
    elif 'holds the lane' not in (b['message'] or ''):
        bad.append('a refused submit does not name what is holding the lane: '
                   '%r' % (b['message'],))
    if not jobs.submit('probe', ['/bin/true'], lane='build')['ok']:
        bad.append('a busy lane blocked a different lane')
    holder = jobs.lane_holder('train')
    if not holder or holder['id'] != a['job']['id']:
        bad.append('the lane holder is not the job holding it')
    jobs.cancel(a['job']['id'], grace=0.5)
    if jobs.lane_holder('train') is not None:
        bad.append('the lane stayed held by a job that has stopped')
    if not jobs.submit('probe', ['/bin/true'], lane='train')['ok']:
        bad.append('the lane never came free')


def refusal_checks(bad, jobs):
    """What must never be accepted, and what must never be believed."""
    mark = '/tmp/adv_jobs_should_not_exist'
    try:
        os.unlink(mark)
    except OSError:
        pass
    # ARGV IS A LIST. The values reaching this come off a web form, and a
    # command assembled by joining strings is a command with a semicolon in it.
    got = jobs.submit('probe', ['/bin/echo', 'x; touch %s' % (mark,)],
                      lane='build')
    _settle(jobs, got['job']['id'])
    if os.path.exists(mark):
        bad.append('an argument was run as a command — the job line is built '
                   'by joining strings')
        os.unlink(mark)
    if 'x; touch' not in jobs.tail(got['job']['id']):
        bad.append('the argument did not arrive whole: %r'
                   % (jobs.tail(got['job']['id']),))
    if jobs.submit('probe', '/bin/true; rm -rf /', lane='build')['ok']:
        bad.append('a command STRING was accepted as a job')
    for lane in ('nope', '', None, 'BUILD'):
        if jobs.submit('probe', ['/bin/true'], lane=lane)['ok']:
            bad.append('a job ran in lane %r' % (lane,))
    for kind in ('../etc', 'has space', 'A' * 40, ''):
        if jobs.submit(kind, ['/bin/true'], lane='build')['ok']:
            bad.append('a job kind of %r was accepted' % (kind,))
    # a made-up id is not a file read
    for bogus in ('../../etc/passwd', 'nope', '', '/etc/passwd', None, 7):
        if jobs.read(bogus) is not None or jobs.tail(bogus) != '':
            bad.append('an id of %r resolved to something' % (bogus,))
    # A PID IS NOT AN IDENTITY.
    live = jobs.submit('probe', ['/bin/sh', '-c', 'sleep 20'],
                       lane='build')['job']
    if jobs.alive(dict(live, boot_id='not-this-boot')):
        bad.append('a pid recorded before a reboot is believed')
    if jobs.alive(dict(live, pid_start=(live['pid_start'] or 0) + 999)):
        bad.append('a pid is believed without checking it is the same '
                   'process — after a reboot that number is somebody else')
    if jobs.alive(dict(live, pid=None)) or jobs.alive({}) or jobs.alive(None):
        bad.append('a job with no pid reads as alive')
    jobs.cancel(live['id'], grace=0.5)
    # A ZOMBIE IS NOT ALIVE. Unreachable through submit() now -- the work is
    # orphaned to init, which reaps it -- but alive() is handed a pid and a
    # pid can be anything, so it is asked about a real one. Made here rather
    # than described: os.kill(pid, 0) answers YES to a zombie, which is how a
    # cancelled job read as running for ever.
    z = subprocess.Popen(['/bin/true'])
    z.wait_for_zombie = None
    end = time.time() + 5
    while time.time() < end:
        state, start = jobs._stat(z.pid)
        if state == 'Z':
            if jobs.alive({'pid': z.pid, 'pid_start': start,
                           'boot_id': jobs._boot_id()}):
                bad.append('a zombie reads as alive -- os.kill(pid, 0) says '
                           'yes to a process that has already exited')
            break
        time.sleep(0.05)
    else:
        bad.append('the zombie check never got a zombie to ask about')
    z.poll()
    # ...and nothing secret is written down
    src = open(os.path.join(DASH, 'jobs.py'), encoding='utf-8').read()
    if 'run_env' not in src or "'env'" in src.split('_write(path, job)')[0]:
        bad.append('the environment is written into the job record, which is '
                   'shown on a page')


def progress_checks(bad, jobs):
    """A job that says how far along it is, and one that does not."""
    j = jobs.submit('probe', ['/bin/sh', '-c',
                              'echo PROGRESS 1 10 crops; '
                              'echo noise; echo PROGRESS 7 10 crops'],
                    lane='build')['job']
    _settle(jobs, j['id'])
    got = jobs.progress(j['id'])
    if not got or (got['done'], got['total']) != (7, 10):
        bad.append('the last progress line is not what is reported: %r'
                   % (got,))
    q = jobs.submit('probe', ['/bin/sh', '-c', 'echo PROGRESS nonsense'],
                    lane='build')['job']
    _settle(jobs, q['id'])
    if jobs.progress(q['id']) is not None:
        bad.append('an unparseable progress line is reported as progress')
    r = jobs.submit('probe', ['/bin/echo', 'quiet'], lane='build')['job']
    _settle(jobs, r['id'])
    if jobs.progress(r['id']) is not None:
        bad.append('a job that says nothing is reported as having progress')


def forget_checks(bad, jobs):
    """Clearing a record, and the one record that may not be cleared."""
    done = jobs.submit('probe', ['/bin/echo', 'over'], lane='build')['job']
    _settle(jobs, done['id'])
    got = jobs.forget(done['id'])
    if not got['ok']:
        bad.append('a finished job could not be cleared: %r' % (got,))
    if jobs.read(done['id']) is not None:
        bad.append('a cleared job is still on record')
    if os.path.isdir(jobs.job_dir(done['id'])):
        bad.append('a cleared job still has its directory, so its log is '
                   'still on disk')
    if any(x['id'] == done['id'] for x in jobs.listing(limit=200)):
        bad.append('a cleared job is still in the listing')

    # ...and the live one is not clearable, because the directory it would
    # take with it is where the shell writes how the job ended
    live = jobs.submit('probe', ['/bin/sh', '-c', 'sleep 30'],
                       lane='build')['job']
    try:
        got = jobs.forget(live['id'])
        if got['ok']:
            bad.append('A RUNNING JOB WAS CLEARED -- its work is still on the '
                       'GPU and nothing can reach it now')
        if jobs.read(live['id']) is None:
            bad.append('a running job lost its record')
    finally:
        jobs.cancel(live['id'], grace=0.5)

    # ...and a job whose recorded process was killed but whose WORK carries
    # on -- the trainer reparents and keeps the GPU -- is not clearable
    # either. The record is the only way back to it.
    orphan = jobs.submit('probe', ['/bin/sh', '-c', 'sleep 25 & wait'],
                         lane='build')['job']
    time.sleep(1)
    rec = jobs.read(orphan['id'])
    if not rec.get('pgid'):
        bad.append('a job does not record its process group, so nothing can '
                   'tell work that is still running from work that is over')
    else:
        try:
            os.kill(rec['pid'], 9)
            time.sleep(1)
            after = jobs.read(orphan['id'])
            if not jobs.group_alive(after.get('pgid')):
                bad.append('the probe could not keep the group alive, so this '
                           'check proves nothing')
            elif jobs.forget(orphan['id'])['ok']:
                bad.append('A JOB WAS CLEARED WHILE ITS WORK WAS STILL '
                           'RUNNING -- the GPU is busy and nothing on the '
                           'page can reach it any more')
        finally:
            try:
                os.killpg(rec['pgid'], 9)
            except OSError:
                pass
            time.sleep(0.4)
            jobs.forget(orphan['id'])

    for bogus, why in (('../../etc', 'a path'), ('', 'nothing'),
                       ('no-such-job-at-all', 'a name that is not a job')):
        got = jobs.forget(bogus)
        if got['ok']:
            bad.append('clearing accepted %s' % (why,))


def main():
    import jobs
    tmp = tempfile.mkdtemp(prefix='adv_jobs_')
    jobs.JOBS_DIR = tmp
    bad = []
    try:
        for fn, args in ((basic_checks, (jobs,)), (redraw_checks, (jobs,)),
                         (cancel_checks, (jobs,)),
                         (lane_checks, (jobs,)), (refusal_checks, (jobs,)),
                         (progress_checks, (jobs,)),
                         (forget_checks, (jobs,)),
                         (survives_its_parent, (tmp,))):
            try:
                fn(bad, *args)
            except Exception as e:        # noqa: BLE001 - report, not die
                bad.append('%s threw %s: %s'
                           % (fn.__name__, type(e).__name__, e))
            # nothing may be left running between checks
            for job in jobs.listing(limit=200):
                if job.get('state') == 'running':
                    jobs.cancel(job['id'], grace=0.5)
    finally:
        for job in jobs.listing(limit=500):
            if job.get('state') == 'running':
                jobs.cancel(job['id'], grace=0.5)
        shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        for b in bad:
            print('FAIL ' + b)
        return 1
    print('a job outlives the process that started it, says how it ended to '
          'anybody who asks later, stops with its whole tree when told to, '
          'and never runs an argument as a command')
    return 0


if __name__ == '__main__':
    sys.exit(main())
