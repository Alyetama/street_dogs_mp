#!/usr/bin/env python3
"""Delegated work: the target, the count behind it, and who may see either.

Three things can go wrong here and only one of them is loud.

The loud one is the arithmetic -- a bar that says 137 of 500 when the ledgers
hold something else. That is checked against ledgers this file writes itself,
in a temporary directory, so the numbers are known rather than sampled.

The quiet one is SCOPE. A target is a measurement of a person. The route that
answers "how am I doing" must read the username off the session the gate
resolved and never off a query string, or every signed-in reader has a
scoreboard of their colleagues; and the roster must answer a member with the
same empty 404 an address that does not exist gets.

The quietest one is a read that writes. Progress is counted on demand, and
counting is allowed to STAMP a target as reached -- so a bug in which store it
stamps means an admin glancing at the roster writes into the live accounts
database. That one leaves no trace at all until somebody notices a row they
did not touch, which is why it is pinned here twice.

Run: python tools/detect/tests/adv_delegation.py
"""
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DASH = os.path.join(REPO, 'tools', 'dashboard')
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
sys.path.insert(0, DASH)


def _md5(path):
    """The ledger exactly as it is, so "untouched" is a fact not a hope."""
    import hashlib
    try:
        with open(path, 'rb') as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except OSError:
        return None


def load_dashboard():
    """dashboard.py as a module, without running its CLI."""
    spec = importlib.util.spec_from_file_location(
        'dashboard', os.path.join(DASH, 'dashboard.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['dashboard'] = mod
    spec.loader.exec_module(mod)
    return mod


def schema_checks(bad, A):
    """The table, and the rule the DATABASE keeps rather than a page.

    One open target per person per surface is a constraint, not a convention.
    Two admins clicking Delegate at the same moment both pass a read-then-
    write check; only a unique index makes the second one lose.
    """
    tmp = tempfile.mkdtemp(prefix='adv_del_schema_')
    try:
        p = os.path.join(tmp, 'a.db')
        A.create_user('boss', 'a-password-long-enough', role='admin', path=p)
        A.create_user('sam', 'another-good-password', path=p)
        con = A.connect(p)
        try:
            names = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if 'assignments' not in names:
                bad.append('no assignments table after a migration')
                return
            idx = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='index'")}
            if 'assignments_one_open' not in idx:
                bad.append('the one-open-target rule is not an index, so two '
                           'admins clicking at once both win')
            # a target under a name nobody holds is not a target
            cols = {r[1] for r in con.execute(
                'PRAGMA table_info(assignments)')}
            for need in ('user_id', 'surface', 'target', 'start_at',
                         'created_by', 'due_at', 'done_at', 'cancelled_at'):
                if need not in cols:
                    bad.append('the assignments row has no %s' % (need,))
        finally:
            con.close()
        # THE RULE, exercised rather than read off the schema
        ok = A.create_assignment('sam', 500, surface='review',
                                 created_by='boss', path=p)
        if not ok['ok']:
            bad.append('a first target was refused: %s' % ok['message'])
        again = A.create_assignment('sam', 9, surface='review',
                                    created_by='boss', path=p)
        if again['ok']:
            bad.append('a second open target on the same surface was '
                       'accepted — two bars over one pile of work')
        other = A.create_assignment('sam', 9, surface='gate',
                                    created_by='boss', path=p)
        if not other['ok']:
            bad.append('a target on a DIFFERENT surface was refused: %s'
                       % other['message'])
        # ...and the rule frees up once the first is finished or called off
        A.cancel_assignment(ok['assignment']['id'], path=p)
        if not A.create_assignment('sam', 9, surface='review',
                                   created_by='boss', path=p)['ok']:
            bad.append('cancelling a target does not free the surface')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def validation_checks(bad, A):
    """What may be delegated, and to whom."""
    tmp = tempfile.mkdtemp(prefix='adv_del_valid_')
    try:
        p = os.path.join(tmp, 'a.db')
        A.create_user('boss', 'a-password-long-enough', role='admin', path=p)
        A.create_user('sam', 'another-good-password', path=p)
        now = int(time.time())
        for target in (0, -5, 'lots', None, A.MAX_TARGET + 1):
            got = A.create_assignment('sam', target, created_by='boss',
                                      path=p)
            if got['ok']:
                bad.append('a target of %r was accepted' % (target,))
        for surface in ('nope', '', None, 'REVIEW; DROP TABLE users'):
            got = A.create_assignment('sam', 5, surface=surface,
                                      created_by='boss', path=p)
            if got['ok']:
                bad.append('a target on surface %r was accepted'
                           % (surface,))
        past = A.create_assignment('sam', 5, due_at=now - 10,
                                   created_by='boss', now=now, path=p)
        if past['ok']:
            bad.append('a deadline that has already gone was accepted')
        # A RETIRED ACCOUNT. A target nobody can sign in to work on reads as
        # unmet for ever, and blames a person who was never given the chance.
        A.set_active('sam', False, path=p)
        got = A.create_assignment('sam', 5, created_by='boss', path=p)
        if got['ok']:
            bad.append('work was delegated to a retired account')
        A.set_active('sam', True, path=p)
        if not A.create_assignment('sam', 5, created_by='boss', path=p)['ok']:
            bad.append('a plain target was refused')
        # the names come from a JOIN, so renaming somebody keeps the record
        rows = A.list_assignments(who='sam', path=p)
        if not rows or rows[0]['username'] != 'sam' or \
                rows[0]['created_by_name'] != 'boss':
            bad.append('an assignment does not carry who it is for and who '
                       'set it: %r' % (rows[:1],))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _ledger(d, tmp, rows, stage_rows):
    """Point the module at ledgers this file owns, and fill them.

    The ledger FILENAMES come from the module, never spelt here. A guard that
    names a store is a guard that will one day be read as writing into human
    data, and the isolation checks next door are watching for exactly that.
    """
    hn, hp = os.path.join(tmp, 'hn'), os.path.join(tmp, 'hp')
    os.makedirs(hn, exist_ok=True)
    os.makedirs(hp, exist_ok=True)
    d.HN_DIR, d.HP_DIR = hn, hp
    d.HN_CROPS = os.path.join(hn, 'crops')
    d.HN_FULL = os.path.join(hn, 'full')
    d.HN_LABELS = os.path.join(hn, os.path.basename(d.HN_LABELS))
    d._flagged = None
    for label, recs in rows.items():
        with open(d._store_for(label)['labels'], 'w') as fh:
            for r in recs:
                fh.write(json.dumps(r) + '\n')
    import fn_audit as fa
    lay = {}
    for stage in fa.STAGES:
        sd = os.path.join(tmp, stage)
        os.makedirs(sd, exist_ok=True)
        base = dict(fa.paths(stage))
        base.update(out=sd, verdicts=os.path.join(sd, 'v.jsonl'),
                    drawn=os.path.join(sd, 'drawn.jsonl'),
                    pages=os.path.join(sd, 'pages'),
                    pool=os.path.join(sd, 'pool.parquet'))
        lay[stage] = base
        with open(base['verdicts'], 'w') as fh:
            for r in stage_rows.get(stage, []):
                fh.write(json.dumps(r) + '\n')
    return lay


def counting_checks(bad, d, A):
    """How many, since when, on which surface -- against known ledgers."""
    import fn_audit as fa
    tmp = tempfile.mkdtemp(prefix='adv_del_count_')
    keep = {k: getattr(d, k) for k in
            ('HN_DIR', 'HN_CROPS', 'HN_FULL', 'HN_LABELS', 'HP_DIR')}
    kept_flags, real_paths = d._flagged, fa.paths
    T0 = 1_800_000_000                          # the moment work was handed out

    def crop(i, at, who):
        return {'crop': '%d_p%03d_%03d.jpg' % (1_700_000_000_000 + i, i, 50),
                'flagged_at': at, 'by': who}
    try:
        lay = _ledger(d, tmp, {
            d.POS_LABEL: [crop(1, T0 + 10, 'sam'), crop(2, T0 + 20, 'sam'),
                          crop(3, T0 - 50, 'sam'),      # before the target
                          crop(4, T0 + 30, 'ana')],     # somebody else
            d.FLAG_LABEL: [crop(5, T0 + 40, 'sam'),
                           crop(6, T0 + 50, None)],     # legacy -> admin
        }, {
            'gate': [{'key': 'g1#0', 'verdict': 'dog', 'ts': T0 + 5,
                      'by': 'sam'},
                     {'key': 'g2#0', 'verdict': 'not_dog', 'ts': T0 + 6,
                      'by': 'sam'},
                     {'key': 'g3#0', 'verdict': 'dog', 'ts': T0 - 9,
                      'by': 'sam'},
                     # answered, then withdrawn: read_verdicts drops it, so
                     # nobody is credited for a verdict that is not there
                     {'key': 'g4#0', 'verdict': 'dog', 'ts': T0 + 7,
                      'by': 'sam'},
                     {'key': 'g4#0', 'verdict': None, 'ts': T0 + 8,
                      'by': 'sam'},
                     # the same crop answered twice counts ONCE
                     {'key': 'g5#0', 'verdict': 'dog', 'ts': T0 + 9,
                      'by': 'sam'},
                     {'key': 'g5#0', 'verdict': 'not_dog', 'ts': T0 + 11,
                      'by': 'sam'}],
            'leash': [{'key': 'l1#0', 'verdict': 'leashed', 'ts': T0 + 12,
                       'by': 'sam'}],
        })
        fa.paths = lambda stage='gate': lay[stage]
        for surface, want in (('review', 3), ('gate', 3), ('leash', 1),
                              ('any', 7)):
            got = d.done_by('sam', surface, T0)
            if got != want:
                bad.append('done_by(sam, %r) = %s, want %s'
                           % (surface, got, want))
        # ...and the work before the target does not count towards it
        if d.done_by('sam', 'any', 0) != 9:
            bad.append('counting from the beginning of time does not see the '
                       'earlier work (%s) — start_at is doing nothing'
                       % d.done_by('sam', 'any', 0))
        if d.done_by('ana', 'review', T0) != 1:
            bad.append('one annotator\'s work is credited to another')
        if d.done_by(d.LEGACY_AUTHOR, 'review', T0) != 1:
            bad.append('a row written before accounts existed is not the '
                       'admin\'s — the legacy author stopped applying')
        for nobody in ('', None, 'nobody'):
            if d.done_by(nobody, 'any', 0) != 0:
                bad.append('%r is credited with work' % (nobody,))

        # ── the record a page reads ──
        p = os.path.join(tmp, 'a.db')
        A.create_user('boss', 'a-password-long-enough', role='admin', path=p)
        A.create_user('sam', 'another-good-password', path=p)
        row = A.create_assignment('sam', 4, surface='any', created_by='boss',
                                  now=T0, path=p)['assignment']
        got = d.assignment_progress(row, now=T0 + 100, mark_done=False)
        if (got['done'], got['target'], got['left']) != (7, 4, 0):
            bad.append('progress reads %s/%s (%s left), want 7/4 (0 left)'
                       % (got['done'], got['target'], got['left']))
        # PER CENT NEVER ROUNDS UP TO A HUNDRED. 499 of 500 on a bar
        # somebody is being measured against must not read as finished.
        near = dict(row, target=500)
        d_near = d.assignment_progress(near, now=T0, mark_done=False)
        if d_near['pct'] >= 100:
            bad.append('7 of 500 reads as %s%%' % d_near['pct'])
        far = dict(row, target=8)
        if d.assignment_progress(far, now=T0, mark_done=False)['pct'] != 87:
            bad.append('7 of 8 does not read as 87%%: %s'
                       % d.assignment_progress(far, now=T0,
                                               mark_done=False)['pct'])
        # THE DUE DAY, not the morning after it
        due = dict(row, due_at=int(time.mktime(
            (2026, 8, 25, 0, 0, 0, 0, 0, -1))) + 86400)
        txt = d.assignment_progress(due, now=T0, mark_done=False)['due_txt']
        if '25' not in txt:
            bad.append('a deadline of the 25th prints as %r — off by the '
                       'day the reader still has' % (txt,))
    finally:
        fa.paths = real_paths
        for k, v in keep.items():
            setattr(d, k, v)
        d._flagged = kept_flags
        shutil.rmtree(tmp, ignore_errors=True)


def stamp_checks(bad, d, A, auth):
    """Counting may finish a target. It may not touch the wrong database.

    assignment_progress() stamps done_at when the number is met, and that
    stamp is what makes "reached it on the 14th" survive somebody undoing an
    annotation afterwards. It writes -- so which file it writes to is the
    whole question, and the answer has to be the store the GATE is using.
    """
    tmp = tempfile.mkdtemp(prefix='adv_del_stamp_')
    real_count = None
    try:
        auth.bootstrap(db_path=os.path.join(tmp, 'a.db'),
                       key_path=os.path.join(tmp, 'k'),
                       env={'DASHBOARD_USER': 'boss',
                            'DASHBOARD_PASSWORD': 'a-password-long-enough'})
        p = d._accounts_db()
        if not p or not p.startswith(tmp):
            bad.append('the module counts against %r rather than the store '
                       'the gate opened — a progress read would write to the '
                       'live database' % (p,))
            return
        now = int(time.time())
        row = A.create_assignment('boss', 1, surface='any', created_by='boss',
                                  now=now, path=p)['assignment']
        # nothing has been judged in this fixture, so the target is not met
        got = d.assignments_for({'username': 'boss'}, now=now)
        if not got or got[0]['state'] != 'open':
            bad.append('an unmet target does not read as open: %r'
                       % (got[:1],))
        # Meet it by making the COUNT say so, rather than by writing a
        # target of nothing: zero is refused by the table, and the thing
        # under test is what happens when the number is reached.
        real_count = d.done_by
        d.done_by = lambda *a, **k: 5
        row0 = A.get_assignment(row['id'], path=p)
        d.assignment_progress(row0, now=now)
        if not A.get_assignment(row['id'], path=p)['done_at']:
            bad.append('a met target is never stamped, so "finished on the '
                       '14th" cannot survive an annotation being undone')
        # ...and a met target READS as met even where nothing stamped it,
        # or an admin sees a full green bar beside the word "open".
        con = A.connect(p)
        con.execute('UPDATE assignments SET done_at = NULL WHERE id = ?',
                    (row['id'],))
        con.commit()
        con.close()
        seen = d.assignment_roster(now=now)
        if not seen or seen[0]['state'] != 'done':
            bad.append('the roster calls a met target %r — the table argues '
                       'with the bar in its own row'
                       % (seen[0]['state'] if seen else None,))
        # A ROSTER READ IS A PURE READ.
        con = A.connect(p)
        con.execute('UPDATE assignments SET done_at = NULL WHERE id = ?',
                    (row['id'],))
        con.commit()
        con.close()
        d.assignment_roster(now=now)
        if A.get_assignment(row['id'], path=p)['done_at']:
            bad.append('looking at the roster finished somebody\'s target — '
                       'when they reached it is now when an admin opened a '
                       'page')
        # A GATE THAT CANNOT NAME ITS STORE DOES NOT WRITE ONE. Passing no
        # path falls through to the DEFAULT accounts.db, so a request served
        # while the gate is half-loaded would write to the live file -- and
        # with a colliding row id it would finish the wrong person's target.
        con = A.connect(p)
        con.execute('UPDATE assignments SET done_at = NULL WHERE id = ?',
                    (row['id'],))
        con.commit()
        con.close()
        # Watched at the SOURCE rather than by looking for damage: a stamp
        # aimed at the default store lands on whatever row happens to hold
        # that id there, which on this machine is usually none -- so "the
        # live file did not change" would pass a build that is one colliding
        # id away from finishing a stranger's target.
        calls = []
        real_stamp = A.complete_assignment
        A.complete_assignment = lambda *a, **k: calls.append(k.get('path'))
        real_state = auth._state
        auth._state = lambda: {}
        try:
            out = d.assignment_progress(A.get_assignment(row['id'], path=p),
                                        now=now)
        finally:
            auth._state = real_state
            A.complete_assignment = real_stamp
        if not out or out.get('done') is None:
            bad.append('a count with no store to record it in reported '
                       'nothing — the number is still knowable')
        if calls:
            bad.append('a target was stamped into %r while the gate could '
                       'not name its own store — with no path that is the '
                       'LIVE accounts database' % (calls[0],))
        if A.get_assignment(row['id'], path=p)['done_at']:
            bad.append('a target was stamped while the gate could not name '
                       'its own store')
        # ...and a finished one lingers rather than vanishing at the moment
        # it is reached, which is the one moment it was for
        d.assignment_progress(A.get_assignment(row['id'], path=p), now=now)
        seen = d.assignments_for({'username': 'boss'}, now=now)
        if not seen or seen[0]['state'] != 'done':
            bad.append('a target disappears the instant it is met: %r'
                       % (seen,))
        if d.assignments_for({'username': 'boss'},
                             now=now + d.ASSIGN_DONE_LINGER_S + 10):
            bad.append('a finished target is still on screen weeks later')
        d.done_by = real_count
        # A FINISHED TARGET CANNOT BE UN-FINISHED. Cancelling one is a
        # no-op by design, so this un-stamps it FIRST -- cancelling the done
        # row and expecting it to take would be this file testing its own
        # misunderstanding.
        con = A.connect(p)
        con.execute('UPDATE assignments SET done_at = NULL WHERE id = ?',
                    (row['id'],))
        con.commit()
        con.close()
        A.cancel_assignment(row['id'], path=p)
        if A.get_assignment(row['id'], path=p)['state'] != 'cancelled':
            bad.append('an open target could not be called off')
        # cancelled work never draws a bar
        if d.assignments_for({'username': 'boss'}, now=now):
            bad.append('a target that was called off still asks for work')
        # nobody signed in is nobody's target
        for who in (None, {}, {'username': ''}):
            if d.assignments_for(who, now=now):
                bad.append('a session of %r was shown somebody\'s target'
                           % (who,))
    finally:
        if real_count is not None:
            d.done_by = real_count
        shutil.rmtree(tmp, ignore_errors=True)


def route_checks(bad, d, A, auth):
    """Who may ask about whose progress, over real HTTP."""
    tmp = tempfile.mkdtemp(prefix='adv_del_route_')
    srv = None
    try:
        got = auth.bootstrap(
            db_path=os.path.join(tmp, 'a.db'), key_path=os.path.join(tmp, 'k'),
            env={'DASHBOARD_USER': 'boss',
                 'DASHBOARD_PASSWORD': 'a-password-long-enough'})
        if not got.get('ok'):
            bad.append('the gate would not start: %r' % (got,))
            return
        p, key = d._accounts_db(), os.path.join(tmp, 'k')
        A.create_user('sam', 'another-good-password', path=p)
        A.create_assignment('sam', 500, surface='review', created_by='boss',
                            path=p)
        A.create_assignment('boss', 300, surface='gate', created_by='boss',
                            path=p)

        def cookie(name):
            user = A.get_user(name, path=p)
            value, _ = auth.mint(user, key_path=key)
            return {'Cookie': auth.COOKIE + '=' + value}

        class Quiet(d.BoardHandler):
            def log_message(self, *a):
                pass
        srv = ThreadingHTTPServer(('127.0.0.1', 0), Quiet)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = 'http://127.0.0.1:%d' % srv.server_port

        def get(path, who, timeout=20):
            r = urllib.request.urlopen(
                urllib.request.Request(base + path, headers=cookie(who)),
                timeout=timeout)
            return r.status, json.loads(r.read() or b'{}')

        # ── your own, and only your own ──
        st, body = get('/api/assignment', 'sam')
        names = {a['username'] for a in body.get('assignments', [])}
        if names != {'sam'}:
            bad.append('/api/assignment answered sam with %r' % (names,))
        st, body = get('/api/assignment', 'boss')
        names = {a['username'] for a in body.get('assignments', [])}
        if names != {'boss'}:
            bad.append('/api/assignment answered boss with %r' % (names,))
        # A USERNAME IN THE QUERY STRING CHANGES NOTHING. If it did, every
        # signed-in reader would have a scoreboard of their colleagues.
        st, body = get('/api/assignment?who=boss&username=boss&user=boss',
                       'sam')
        names = {a['username'] for a in body.get('assignments', [])}
        if names != {'sam'}:
            bad.append('naming somebody else in the query string returned '
                       'their targets: %r' % (names,))

        # ── the roster is the admin's ──
        st, body = get('/api/assignments', 'boss')
        if {a['username'] for a in body.get('assignments', [])} != \
                {'sam', 'boss'}:
            bad.append('the roster does not list everybody: %r' % (body,))
        try:
            st, body = get('/api/assignments', 'sam')
            bad.append('a member read the roster: %d %r' % (st, body))
        except urllib.error.HTTPError as e:
            if e.code != 404:
                bad.append('a member asking for the roster got %d, not the '
                           'same empty 404 a dead address gets' % e.code)
        # A ROUTE ANSWERS. Counting reads ledgers off six drives and can
        # fail for reasons that have nothing to do with the request; every
        # other route in that file answers as JSON, and a dropped connection
        # reaches the page as the same nothing an empty target would.
        real_count = d.done_by

        def boom(*a, **k):
            raise RuntimeError('a drive went away')
        d.done_by = boom
        try:
            for path in ('/api/assignment', '/api/assignments'):
                try:
                    st, body = get(path, 'boss')
                    if st != 200 or 'assignments' not in body:
                        bad.append('%s answered %s %r when the count threw'
                                   % (path, st, body))
                except Exception as e:
                    bad.append('%s dropped the connection when the count '
                               'threw (%s) — every route beside it answers'
                               % (path, type(e).__name__))
        finally:
            d.done_by = real_count

        # signed out is nothing at all
        try:
            r = urllib.request.urlopen(base + '/api/assignment', timeout=20)
            if r.status == 200 and json.loads(r.read() or b'{}').get(
                    'assignments'):
                bad.append('a request with no cookie was told about '
                           'somebody\'s targets')
        except urllib.error.HTTPError as e:
            if e.code not in (302, 401, 403, 404):
                bad.append('an unauthenticated /api/assignment answered %d'
                           % e.code)
    finally:
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        shutil.rmtree(tmp, ignore_errors=True)


def admin_checks(bad, A, auth, d):
    """Delegating is an admin action, and calling work off keeps the work."""
    tmp = tempfile.mkdtemp(prefix='adv_del_admin_')
    try:
        auth.bootstrap(db_path=os.path.join(tmp, 'a.db'),
                       key_path=os.path.join(tmp, 'k'),
                       env={'DASHBOARD_USER': 'boss',
                            'DASHBOARD_PASSWORD': 'a-password-long-enough'})
        p = auth._state()['db']
        A.create_user('sam', 'another-good-password', path=p)
        boss = A.get_user('boss', path=p)
        admin = {'id': boss['id'], 'username': 'boss', 'csrf': 'T',
                 'role': 'admin', 'active': True}
        now = int(time.time())

        def req(**kw):
            class _R:
                path = '/admin'

                def arg(self, k, dflt=''):
                    return dflt

                def one(self, k):
                    return kw.get(k, '')
            return _R()

        got = auth.admin_action('assign', req(do='new', who='sam',
                                              target='500', surface='review',
                                              due='2099-01-01', note='pass'),
                                admin, now=now, path=p)
        if not got['ok']:
            bad.append('an admin could not delegate: %s' % got['message'])
        rows = A.list_assignments(path=p, now=now)
        if not rows or rows[0]['target'] != 500 or rows[0]['note'] != 'pass':
            bad.append('the delegated target was not stored: %r' % (rows[:1],))
        # A MEMBER CANNOT, even holding the endpoint name.
        member = dict(admin, role='member')
        if auth.admin_action('assign', req(do='new', who='sam', target='9'),
                             member, now=now, path=p)['ok']:
            bad.append('a member delegated work to somebody')
        if auth.admin_action('assign',
                             req(do='cancel', id=str(rows[0]['id'])),
                             member, now=now, path=p)['ok']:
            bad.append('a member called off an admin\'s target')
        # a date that is not one is refused rather than silently dropped: a
        # deadline nobody set is not the same as a deadline that failed to
        # parse, and only one of them is worth telling somebody about
        if auth.admin_action('assign', req(do='new', who='sam', target='9',
                                           surface='gate',
                                           due='2026-02-31'),
                             admin, now=now, path=p)['ok']:
            bad.append('the 31st of February was accepted as a deadline')
        for good in ('2099-01-01',):
            if auth._day_end(good) is None:
                bad.append('%r is not read as a date' % (good,))
        # CALLING WORK OFF KEEPS THE WORK. The row goes to cancelled; nothing
        # reaches into a ledger.
        auth.admin_action('assign', req(do='cancel', id=str(rows[0]['id'])),
                          admin, now=now, path=p)
        after = A.get_assignment(rows[0]['id'], path=p)
        if after['state'] != 'cancelled':
            bad.append('calling a target off did not cancel it: %r'
                       % (after['state'],))
        if after['target'] != 500:
            bad.append('cancelling rewrote the record of what was asked for')

        # ── DELETE: the record goes, the work does not ──
        # Calling off and deleting answer different questions. Cancelling
        # says "we stopped wanting this" and keeps what was asked for and
        # where it got to; deleting is for a row that should never have
        # existed -- the wrong person, or 5000 typed for 500.
        n_before = d.done_by('boss', 'any', 0)
        led = [d._store_for(lb)['labels'] for lb in d.FLAG_LABELS]
        before = [_md5(f) for f in led]
        keep = A.create_assignment('sam', 77, surface='leash',
                                   created_by='boss', now=now,
                                   path=p)['assignment']
        gone = A.create_assignment('sam', 88, surface='gate',
                                   created_by='boss', now=now,
                                   path=p)['assignment']
        n_rows = len(A.list_assignments(path=p))
        got = auth.admin_action('assign', req(do='delete',
                                              id=str(gone['id'])),
                                admin, now=now, path=p)
        if not got['ok'] or got['notice'] != 'assign_deleted':
            bad.append('an admin could not delete a target: %r' % (got,))
        if A.get_assignment(gone['id'], path=p) is not None:
            bad.append('a deleted target is still on record')
        if A.get_assignment(keep['id'], path=p) is None:
            bad.append('deleting one target took another with it')
        # counted, not spot-checked: a DELETE with a loose WHERE takes the
        # rows on one side of the id and leaves whichever neighbour this
        # check happened to look at
        if len(A.list_assignments(path=p)) != n_rows - 1:
            bad.append('deleting one target removed %d rows'
                       % (n_rows - len(A.list_assignments(path=p)),))
        # ...AND NOT ONE ANNOTATION MOVED. The ledgers and this database do
        # not point at each other: progress is counted by asking the ledgers
        # who wrote what, so removing the asking removes no answer. Somebody
        # who judged four hundred crops towards a target deleted by mistake
        # has still judged four hundred crops.
        if [_md5(f) for f in led] != before:
            bad.append('deleting a target rewrote an annotation ledger')
        if d.done_by('boss', 'any', 0) != n_before:
            bad.append('deleting a target changed what somebody had judged')
        # deleting frees the surface, because there is no row left to clash
        if not A.create_assignment('sam', 9, surface='gate',
                                   created_by='boss', now=now, path=p)['ok']:
            bad.append('the surface stayed claimed by a target that is gone')
        # a second click, or two admins on one row, is a race not a mistake
        again = auth.admin_action('assign', req(do='delete',
                                                id=str(gone['id'])),
                                  admin, now=now, path=p)
        if not again['ok'] or again['notice']:
            bad.append('deleting an already-deleted target is reported as '
                       'something happening: %r' % (again,))
        for junk in ('', 'not-a-number', '-1', '999999'):
            if not auth.admin_action('assign', req(do='delete', id=junk),
                                     admin, now=now, path=p)['ok']:
                bad.append('a delete of %r threw rather than doing nothing'
                           % (junk,))
        # A MEMBER CANNOT, even holding the endpoint name and a real id
        left = A.create_assignment('sam', 5, surface='review',
                                   created_by='boss', now=now,
                                   path=p)['assignment']
        if auth.admin_action('assign', req(do='delete', id=str(left['id'])),
                             member, now=now, path=p)['ok']:
            bad.append('a member deleted an admin\'s target')
        if A.get_assignment(left['id'], path=p) is None:
            bad.append('a member\'s refused delete removed the row anyway')
        # and the control asks before it acts
        page = auth.admin_page(admin, req(), now=now)
        if 'data-confirm=' not in page:
            bad.append('delete does not ask first — it is the one control '
                       'here that cannot be undone')
        if 'window.confirm' not in page:
            bad.append('nothing on the page acts on data-confirm, so the '
                       'attribute is decoration')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def strip_checks(bad, d):
    """One strip, three pages, and nothing at all when nothing is delegated."""
    import audit
    import work_strip as W
    pages = {'review': d.REVIEW_HTML,
             'gate': audit.page_html('gate', account=('', '')),
             'leash': audit.page_html('leash', account=('', ''))}
    for name, html in pages.items():
        for need, why in (
                ('.asg1{', 'the strip has no styling'),
                ('id="asg"', 'the strip is not on the page'),
                ('data-surface="%s"' % (name,),
                 'the strip does not say which surface this page is, so a '
                 'leash target would draw a bar over the dog-bin sheet'),
                ('refreshWorkStrip', 'the bar never moves as work is done'),
                ('/api/assignment', 'the strip asks nobody how it is going')):
            if need not in html:
                bad.append('%s: %s (%r)' % (name, why, need))
        for slot in ('__WORKCSS__', '__WORKSTRIP__', '__WORKJS__',
                     '__DATECSS__'):
            if slot in html:
                bad.append('%s: %s was never substituted' % (name, slot))
        # ONE SPELLING. The tab strip above it is duplicated between two files
        # and pinned byte for byte; this one is not duplicated at all, and
        # that is what the check is for -- both pages must carry the SAME
        # bytes, out of work_strip.py.
        if W.STRIP_CSS not in html:
            bad.append('%s carries its own copy of the strip CSS' % (name,))
        if W.STRIP_JS not in html:
            bad.append('%s carries its own copy of the strip script' % (name,))
        # ...and the date pair beside it, for the same reason: it lived in
        # two files that had to stay identical with nothing making them, and
        # they came apart by 2px of padding and a border radius.
        if W.DATE_CSS not in html:
            bad.append('%s carries its own copy of the date-pair styling'
                       % (name,))
    # ONE VOCABULARY. The admin page names the surfaces, the annotator's bar
    # names them again, and the script inside the bar names them a third
    # time. They said "any surface" and "every surface" for the same target.
    import auth as _auth
    if _auth.SURFACE_WORDS is not W.SURFACE_WORDS:
        bad.append('the admin page keeps its own names for the surfaces, so '
                   'the page that sets a target can disagree with the bar '
                   'it is measured on')
    got = re.search(r'var WORDS=(\{.*?\});', W.STRIP_JS)
    if not got:
        bad.append('the script has no surface vocabulary at all')
    elif json.loads(got.group(1)) != W.SURFACE_WORDS:
        bad.append('the script names the surfaces differently from the '
                   'pages: %s' % (got.group(1),))
    for s in W.SURFACE_WORDS:
        if s not in __import__('accounts').SURFACES:
            bad.append('%r is named but cannot be delegated' % (s,))
    for s in __import__('accounts').SURFACES:
        if s not in W.SURFACE_WORDS:
            bad.append('%r can be delegated but has no name on the bar'
                       % (s,))
    # hidden until something is delegated: a dashboard nobody delegates on
    # must look exactly as it did
    if 'hidden' not in W.strip_html('review'):
        bad.append('the strip ships visible, so every page grows a "your '
                   'target" label for people who have none')
    if W.strip_html('review').strip().endswith('</div>') is False:
        bad.append('the strip is not an empty element the script fills')


def main():
    bad = []
    try:
        import accounts as A
        import auth
        d = load_dashboard()
    except Exception as e:                # noqa: BLE001 - report, not die
        print('FAIL could not load the modules: %s: %s'
              % (type(e).__name__, e))
        return 1
    for fn, args in ((schema_checks, (A,)), (validation_checks, (A,)),
                     (counting_checks, (d, A)), (stamp_checks, (d, A, auth)),
                     (route_checks, (d, A, auth)), (admin_checks, (A, auth, d)),
                     (strip_checks, (d,))):
        try:
            fn(bad, *args)
        except Exception as e:            # noqa: BLE001 - report, not die
            bad.append('%s threw %s: %s' % (fn.__name__, type(e).__name__, e))
    if bad:
        for b in bad:
            print('FAIL ' + b)
        return 1
    print('a target is one open job per surface, counted from the ledgers '
          'since it was set, visible only to the person it is for and to an '
          'admin, and looking at it never writes anywhere it should not')
    return 0


if __name__ == '__main__':
    sys.exit(main())
