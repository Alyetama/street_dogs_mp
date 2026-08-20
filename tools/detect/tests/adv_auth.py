#!/usr/bin/env python3
"""The gate is the only thing between the tailnet and everything behind it.

    python tools/detect/tests/adv_auth.py

tools/dashboard/accounts.py holds the passwords; tools/dashboard/auth.py is
what decides whether a request gets an answer. Everything here defends one of
the claims it makes:

  * a cookie is a signature over EVERY field, so no value inside one can
    forge another, and a token from another key, another version or another
    century is not a session;
  * expiry is enforced from the signed payload, never from the Max-Age the
    client was asked to honour;
  * the session_epoch in the cookie is compared against the users row on
    every request, so disabling an account, changing its password or signing
    it out ends its live sessions -- and a demotion takes effect on the next
    click, because the ROW is what authorises, not the signed role;
  * the cookie is HttpOnly, SameSite=Lax, Path=/ and deliberately NOT Secure,
    because Secure on a plain-HTTP origin is a login that can never succeed;
  * the gate is an ALLOW-list: a path nobody has thought about is closed, so
    the page somebody adds next month is behind the login by default;
  * with no usable admin the ONLY thing served anywhere is the login page and
    the sentence naming the variable to set -- no page, no image, no /api;
  * a wrong password and a username nobody has produce the same sentence and
    the same amount of work, and neither one gets to keep guessing;
  * a state-changing admin action carries a token bound to the session, and
    the admin page is gated on the ROLE in the database, answering a member
    with the same empty 404 an unknown address gets;
  * an invite link is validated before its form is drawn, says plainly why it
    is dead without naming who issued it, and cannot be spent twice;
  * the token never reaches a Location header, a log line or a cached page.

The fixtures are temp directories. Nothing here reads or writes
data/dashboard, and the only file it opens out of the repo is dashboard.py --
as text, to check what the static handler is allowed to hand out and whether
the source watcher has been told about these modules.

Every check is written to fail if the defect it names comes back; a check that
cannot be made to fail is a certificate of nothing.
"""

import ast
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DASH = os.path.join(REPO, 'tools', 'dashboard')
if DASH not in sys.path:
    sys.path.insert(0, DASH)

# The suite must not inherit the operator's shell. A DASHBOARD_SECRET exported
# in a terminal would mean the key file is never written and the check that it
# is 0600 would pass by never running.
for _v in ('DASHBOARD_SECRET', 'DASHBOARD_SESSION_HOURS',
           'DASHBOARD_INVITE_TTL_HOURS', 'DASHBOARD_USER',
           'DASHBOARD_PASSWORD', 'DASHBOARD_PASSWORD_HASH'):
    os.environ.pop(_v, None)

import accounts as A                                          # noqa: E402
import auth as U                                              # noqa: E402

DASHBOARD_PY = os.path.join(DASH, 'dashboard.py')
AUTH_PY = os.path.join(DASH, 'auth.py')

PW = 'correct-horse-battery'
PW2 = 'a-completely-different-one'
# Distinctive enough that "is this name on the page" is a real question. The
# admin is not called 'admin' anywhere in this suite: the word appears in the
# markup as a ROLE, and a leak check keyed on it would pass on the wrong text.
ADMIN = 'issuer.zeta'
ADMIN_ENV = {'DASHBOARD_USER': ADMIN, 'DASHBOARD_PASSWORD': PW}


class Fixture:
    """A temp directory, a database and a key in it. Never data/dashboard."""

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix='adv_auth_')
        self.db = os.path.join(self.dir, 'accounts.db')
        self.key = os.path.join(self.dir, 'session.key')
        self.use()
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.dir, ignore_errors=True)
        return False

    def use(self):
        """Point the module at this fixture. Called again by any check that
        borrowed the module state for a second deployment."""
        return U.bootstrap(db_path=self.db, key_path=self.key,
                           env=dict(ADMIN_ENV))

    def admin(self):
        return A.get_user(ADMIN, path=self.db)

    def member(self, name='field.phone', pw=PW2):
        u = A.get_user(name, path=self.db)
        return u or A.create_user(name, pw, path=self.db)

    def req(self, method='GET', path='/', query=None, form=None, cookie=None,
            remote='10.0.0.9', host='dash.example:8080'):
        return U.Request(method=method, path=path, query=_qs(query),
                         form=_qs(form),
                         cookies={U.COOKIE: cookie} if cookie else {},
                         remote=remote, host=host)

    def cookie_for(self, user):
        """A live cookie for one account, the way a login would mint it."""
        return U.mint(user, key_path=self.key)[0]

    def session_for(self, user):
        return U.resolve(self.cookie_for(user), path=self.db, key_path=self.key)


def _qs(d):
    """{'k': 'v'} written the way parse_qs hands it over."""
    if not d:
        return {}
    return {k: (v if isinstance(v, list) else [v]) for k, v in d.items()}


def _cookie_value(reply):
    """The session value out of a reply's Set-Cookie, '' if it clears it."""
    raw = reply.header('Set-Cookie')
    if not raw:
        return ''
    return raw.split(';', 1)[0].split('=', 1)[1]


def _literal(src, name):
    """The value of a module-level literal assignment, or None."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == name:
                try:
                    return ast.literal_eval(node.value)
                except ValueError:
                    if (isinstance(node.value, ast.Call)
                            and getattr(node.value.func, 'id', '')
                            in ('frozenset', 'set', 'tuple', 'list')):
                        try:
                            return ast.literal_eval(node.value.args[0])
                        except (ValueError, IndexError):
                            return None
    return None


# ── the signing key ─────────────────────────────────────────────────────────

def key_checks(bad, fx):
    """The key is 0600, it is stable across restarts, and .env wins.

    A key that changes on every start signs everyone out on every start --
    and this process re-execs itself whenever a source file is edited, which
    is the whole reason the session is a cookie and not a table.
    """
    p = os.path.join(fx.dir, 'made.key')
    U._KEYS.pop(p, None)
    was = os.umask(0o000)                  # the hostile case: nothing masked
    try:
        first = U.secret(p)
    finally:
        os.umask(was)
    if not os.path.exists(p):
        bad.append('no key file was written, so nothing about its mode was '
                   'checked and sessions cannot survive a restart')
        return
    mode = oct(os.stat(p).st_mode & 0o777)
    if mode != '0o600':
        bad.append(f'the session key is {mode}, not 0o600: anyone on this box '
                   f'can mint a cookie for any account')
    if len(first) < 32:
        bad.append(f'the session key is {len(first)} bytes; a short key is a '
                   f'forgeable signature')
    U._KEYS.pop(p, None)
    if U.secret(p) != first:               # the restart
        bad.append('the key changed when it was read again: every open '
                   'session dies on every restart, and the process restarts '
                   'itself whenever a source file is edited')

    # DASHBOARD_SECRET is how you rotate: it has to win over the file.
    U._KEYS.pop(p, None)
    os.environ['DASHBOARD_SECRET'] = 'x' * U.SECRET_MIN
    try:
        if U.secret(p) != b'x' * U.SECRET_MIN:
            bad.append('DASHBOARD_SECRET did not override the key file, so '
                       'there is no way to invalidate every session at once')
        # ... and a short one is refused, without the value being printed.
        U._KEYS.pop(p, None)
        os.environ['DASHBOARD_SECRET'] = 'zz9pza'
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            got = U.secret(p)
        noise = out.getvalue() + err.getvalue()
        if got == b'zz9pza':
            bad.append('a six-character DASHBOARD_SECRET was accepted as the '
                       'signing key; that is a cookie anybody can forge')
        if 'zz9pza' in noise:
            bad.append('the value of DASHBOARD_SECRET was printed while it '
                       'was being refused')
        if U.ENV_SECRET not in noise:
            bad.append('a refused DASHBOARD_SECRET said nothing about which '
                       'variable was wrong')
    finally:
        os.environ.pop('DASHBOARD_SECRET', None)
        U._KEYS.pop(p, None)

    # And it is not something the static handler will hand out.
    try:
        src = open(DASHBOARD_PY, encoding='utf-8').read()
    except OSError as e:
        bad.append(f'could not read dashboard.py: {e}')
        return
    files = _literal(src, 'STATIC_FILES') or ()
    dirs = _literal(src, 'STATIC_DIRS') or ()
    if not files:
        bad.append('STATIC_FILES could not be read out of dashboard.py, so '
                   'the check that the key is unreachable is asleep')
    for name in sorted(U.PRIVATE_FILES):
        url = '/' + name
        if url in files or any(url.startswith(d) for d in dirs):
            bad.append(f'{url} is in dashboard.py\'s static allow-list: the '
                       f'signing key is downloadable, and a downloadable key '
                       f'is a session for any account')


# ── the signature ───────────────────────────────────────────────────────────

def signature_checks(bad, fx):
    """A cookie is only a session if this server signed exactly these bytes.

    The named failure is the one that costs everything: a payload whose
    fields are joined with a delimiter, signed, and split apart again --
    where a username containing the delimiter writes its own role.
    """
    admin = fx.admin()
    value = fx.cookie_for(admin)
    if U.read_session(value, key_path=fx.key) is None:
        bad.append('a cookie this server just minted does not verify')
        return
    body, _, sig = value.rpartition('.')

    # Every field is under the signature: change one and it stops verifying.
    payload = U.read_session(value, key_path=fx.key)
    for field, new in (('role', 'member'), ('uid', 999), ('epoch', 99),
                       ('exp', payload['exp'] + 86400), ('name', 'somebody'),
                       ('nonce', 'aaaaaaaaaaaaaaaa')):
        forged = dict(payload)
        forged[field] = new
        if forged[field] == payload[field]:
            bad.append(f'the tamper check for {field} rewrites it to the '
                       f'value it already had, so it proves nothing')
            continue
        import json as _json
        tampered = (U.SESSION_VERSION + '.'
                    + U._b64(_json.dumps(forged, sort_keys=True,
                                         separators=(',', ':')).encode())
                    + '.' + sig)
        if U.read_session(tampered, key_path=fx.key) is not None:
            bad.append(f'a cookie with {field} rewritten still verified: the '
                       f'signature does not cover it')

    for label, broken in (
            ('a signature from another key',
             body + '.' + U._b64(U._mac(U.SIGN_LABEL, body.encode(),
                                        os.path.join(fx.dir, 'other.key')))),
            ('a truncated signature', body + '.' + sig[:-4]),
            ('an empty signature', body + '.'),
            ('no signature at all', body),
            ('junk', 'not-a-cookie-at-all'),
            ('an empty cookie', ''),
            ('a cookie from another version', 's9.' + body.split('.', 1)[1]
             + '.' + sig),
            ('four megabytes of it', 'x' * (U.MAX_COOKIE + 1))):
        if U.read_session(broken, key_path=fx.key) is not None:
            bad.append(f'{label} was accepted as a session')

    # The delimiter cannot be injected, because the signed part is base64url
    # and base64url has no '.' in it. A username full of the characters that
    # would end a field must still come back as itself.
    tricky = A.create_user('a.b-c_d', PW2, path=fx.db)
    v2 = fx.cookie_for(tricky)
    got = U.read_session(v2, key_path=fx.key)
    if got is None or got['name'] != 'a.b-c_d' or got['role'] != 'member':
        bad.append('a username containing the token delimiter did not survive '
                   'a round trip -- the payload is not encoded before it is '
                   'signed')
    middle = v2.split('.')[1]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', middle):
        bad.append('the signed payload is not base64url, so a field value can '
                   'contain the delimiter that separates fields')

    # A session for one deployment is not a session for another.
    other = os.path.join(fx.dir, 'second.key')
    U._KEYS.pop(other, None)
    if U.read_session(value, key_path=other) is not None:
        bad.append('a cookie verified under a different signing key')


def expiry_checks(bad, fx):
    """Expiry comes out of the signed payload, not out of the cookie's age.

    Max-Age is a request to the client. A cookie saved off a machine and
    replayed a year later arrives looking exactly like a fresh one, and the
    only field that cannot be lied about is the signed 'exp'.
    """
    admin = fx.admin()
    now = int(time.time())
    value, ttl = U.mint(admin, now=now, ttl=600, key_path=fx.key)
    if U.read_session(value, now=now + 599, key_path=fx.key) is None:
        bad.append('a session expired before its own expiry')
    if U.read_session(value, now=now + 601, key_path=fx.key) is not None:
        bad.append('an expired cookie was accepted: expiry is not being read '
                   'from the signed payload')
    if U.resolve(value, now=now + 601, path=fx.db, key_path=fx.key) is not None:
        bad.append('resolve() accepted an expired cookie')
    if ttl != 600:
        bad.append(f'mint() reported a lifetime of {ttl}s for a 600s session, '
                   f'so the Max-Age it sets does not match what it signed')
    # A cookie stamped in the future is a clock that moved, not a licence.
    ahead, _ = U.mint(admin, now=now + 4000, ttl=600, key_path=fx.key)
    if U.read_session(ahead, now=now, key_path=fx.key) is not None:
        bad.append('a cookie issued in the future was accepted, so a wrong '
                   'clock extends a session by however wrong it was')
    # The default has to be a real number of seconds, whatever the environment
    # says -- a session that lasts 0 seconds is a login loop.
    for value_, why in (('', 'unset'), ('junk', 'not a number'), ('0', 'zero'),
                        ('999999', 'longer than the maximum')):
        env = {'DASHBOARD_SESSION_HOURS': value_}
        got = U.session_ttl(env)
        if got != U.SESSION_TTL_DEFAULT:
            bad.append(f'DASHBOARD_SESSION_HOURS {why} gave a session of '
                       f'{got}s instead of the default')
    if U.session_ttl({'DASHBOARD_SESSION_HOURS': '2'}) != 7200:
        bad.append('DASHBOARD_SESSION_HOURS is not honoured')


def epoch_checks(bad, fx):
    """Ending a session has to end it, and there is no session table to walk.

    Every one of these is a live cookie that must stop working the moment the
    account behind it changes. Without the epoch comparison the "disable"
    button on the admin page is decoration.
    """
    for label, act in (
            ('bump_session_epoch()', lambda u: A.bump_session_epoch(
                u['id'], path=fx.db)),
            ('a password change', lambda u: A.set_password(
                u['id'], PW2 + '-x', path=fx.db)),
            ('disabling the account', lambda u: A.set_active(
                u['id'], False, path=fx.db))):
        u = A.create_user('epoch.%d' % (len(label),), PW2, path=fx.db)
        value = fx.cookie_for(u)
        if U.resolve(value, path=fx.db, key_path=fx.key) is None:
            bad.append(f'a fresh cookie did not resolve before testing '
                       f'{label}')
            continue
        act(u)
        if U.resolve(value, path=fx.db, key_path=fx.key) is not None:
            bad.append(f'{label} left the account\'s open sessions working: '
                       f'there is no session table, so the epoch comparison '
                       f'is the only thing that revokes anything')

    # active=0 without a bump, which is what a hand-edited database or a
    # future code path that forgets one looks like. The flag has to be read
    # in its own right, not inferred from the epoch having moved.
    u = A.create_user('switched.off', PW2, path=fx.db)
    value = fx.cookie_for(u)
    con = A.connect(fx.db)
    try:
        con.execute('UPDATE users SET active = 0 WHERE id = ?', (u['id'],))
    finally:
        con.close()
    if U.resolve(value, path=fx.db, key_path=fx.key) is not None:
        bad.append('a disabled account kept its session because the epoch '
                   'happened not to move; active is not being checked')

    # A demotion does NOT bump the epoch -- deliberately, since it is not a
    # reason to throw somebody out of the review queue -- so reading the role
    # from the row is the only thing that makes it take effect.
    u = A.create_user('demoted.one', PW2, role='admin', path=fx.db)
    value = fx.cookie_for(u)
    ses = U.resolve(value, path=fx.db, key_path=fx.key)
    if not ses or ses['role'] != 'admin':
        bad.append('an admin cookie did not resolve as an admin')
    A.set_role(u['id'], 'member', path=fx.db)
    ses = U.resolve(value, path=fx.db, key_path=fx.key)
    if ses is None:
        bad.append('demoting an account invalidated its session; it is meant '
                   'to keep working as a member')
    elif ses['role'] != 'member':
        bad.append('a demoted account still reads as an admin from its old '
                   'cookie: the signed role is being trusted instead of the '
                   'users row')
    if U.serve_request(fx.req(path=U.ADMIN_PATH, cookie=value)).status != 404:
        bad.append('a demoted admin could still open the admin page with the '
                   'cookie they already had')


def cookie_checks(bad, fx):
    """The flags, including the one that must NOT be there."""
    header = U.set_cookie('abc', 600)
    if header[0] != 'Set-Cookie':
        bad.append('set_cookie() is not producing a Set-Cookie header')
    v = header[1]
    # 'Path=/;' with the semicolon: 'Path=/' alone is a prefix of
    # 'Path=/admin', and a cookie scoped to one route is a session the rest of
    # the dashboard never sees.
    for flag in ('HttpOnly', 'SameSite=Lax', 'Path=/;'):
        if flag not in v:
            bad.append(f'the session cookie is missing {flag}')
    if 'secure' in v.lower():
        bad.append('the session cookie is marked Secure. This server speaks '
                   'plain HTTP on a tailnet, so the browser would withhold it '
                   'from every request and the login would loop forever')
    if 'Max-Age=600' not in v:
        bad.append('the cookie carries no Max-Age, so it survives the browser '
                   'being closed for longer than the session it stands for')
    gone = U.clear_cookie()[1]
    if 'Max-Age=0' not in gone or '%s=;' % (U.COOKIE,) not in gone:
        bad.append('clear_cookie() does not actually clear it')

    # The parser takes the pairs it understands and is not taken down by a
    # neighbour's malformed cookie.
    got = U.parse_cookie('a=1; %s=xyz; broken; b="2"' % (U.COOKIE,))
    if got.get(U.COOKIE) != 'xyz' or got.get('b') != '2':
        bad.append(f'the cookie parser lost a value in ordinary company: '
                   f'{got}')


# ── cross-site forms ────────────────────────────────────────────────────────

def csrf_checks(bad, fx):
    """A state-changing admin action carries a token bound to its session.

    SameSite=Lax stops the cross-site POST in every browser that implements
    it, and the token is the server's own check for the ones that do not.
    The failure it names: a page on another origin auto-submitting a form to
    /admin/user and disabling an account with the admin's own cookie.
    """
    admin = fx.admin()
    cookie = fx.cookie_for(admin)
    ses = U.resolve(cookie, path=fx.db, key_path=fx.key)
    victim = A.create_user('csrf.victim', PW2, path=fx.db)

    def disable_attempt(token):
        return U.serve_request(fx.req(
            'POST', U.ADMIN_PATH + '/user', cookie=cookie,
            form={'csrf': token, 'id': str(victim['id']), 'do': 'disable'}))

    for label, token in (('no token', ''), ('a made-up token', 'x' * 43),
                         ('another session\'s token',
                          U.csrf_for({'nonce': 'other', 'id': 4321},
                                     key_path=fx.key))):
        r = disable_attempt(token)
        if r.status not in (403, 400):
            bad.append(f'an admin POST with {label} answered {r.status}')
        if not A.get_user(victim['id'], path=fx.db)['active']:
            bad.append(f'an admin POST with {label} disabled an account '
                       f'anyway -- the token is not being checked before the '
                       f'action runs')
            A.set_active(victim['id'], True, path=fx.db)

    r = disable_attempt(ses['csrf'])
    if r.status not in (302, 303):
        bad.append(f'a properly signed admin POST answered {r.status}')
    if A.get_user(victim['id'], path=fx.db)['active']:
        bad.append('the real CSRF token did not get the action done')
    A.set_active(victim['id'], True, path=fx.db)

    # One token per SESSION, not per account: two browsers signed into one
    # account must not be able to submit each other's forms, and signing out
    # of one must not leave a usable token behind in the other.
    a = U.resolve(fx.cookie_for(admin), path=fx.db, key_path=fx.key)
    b = U.resolve(fx.cookie_for(admin), path=fx.db, key_path=fx.key)
    if a['csrf'] == b['csrf']:
        bad.append('two sessions for one account share a CSRF token, so the '
                   'token is not bound to the session -- the nonce is not '
                   'random or is not in the signature')
    if U.csrf_ok(a, b['csrf'], key_path=fx.key):
        bad.append('one session accepted another session\'s CSRF token')
    if not U.csrf_ok(a, a['csrf'], key_path=fx.key):
        bad.append('a session rejected its own CSRF token')
    if U.csrf_ok(a, '', key_path=fx.key) or U.csrf_ok(None, a['csrf'],
                                                      key_path=fx.key):
        bad.append('csrf_ok() accepted an empty token or a missing session')
    # The two HMACs are different keys. Otherwise a value handed out in a
    # form is a value that verifies as a signature.
    if a['csrf'] in fx.cookie_for(admin):
        bad.append('the CSRF token appears inside the session cookie: the '
                   'session signature and the form token are the same HMAC')


# ── the gate ────────────────────────────────────────────────────────────────

# Representative of everything behind the gate: a page, the API, the image
# routes and something nobody has written yet.
GATED = ('/', '/index.html', '/api/board', '/api/review', '/audit/review',
         '/audit/gate', '/datasets', '/llm', '/hq/x.jpg', '/orig/x.jpg',
         '/flagged/x.jpg', '/audit/crop/gate/x.jpg', '/datasets/thumb',
         '/world.json', '/map_points.json', '/echarts.min.js', '/favicon.ico',
         '/a-page-nobody-has-written-yet')


def gate_checks(bad, fx):
    """Nothing gets an answer without a session. Including what does not exist
    yet.

    An ALLOW-list, because the failure mode of the other kind is a page added
    next month that nobody remembered to protect -- and this server grows a
    page most months.
    """
    for p in GATED:
        r = U.serve_request(fx.req(path=p))
        if r is None:
            bad.append(f'{p} was served to a request with no cookie at all')
            continue
        if p.startswith('/api/'):
            if r.status != 401:
                bad.append(f'{p} answered {r.status} to a stranger; a page\'s '
                           f'fetch() needs a 401 it can read, not a redirect '
                           f'into the login HTML')
        elif r.status != 302 or not r.header('Location').startswith(
                U.LOGIN_PATH):
            bad.append(f'{p} answered {r.status} to a stranger instead of '
                       f'sending them to the login page')
        if r.body and b'dog' in r.body.lower():
            bad.append(f'{p} put something that looks like data in front of a '
                       f'stranger')

    # A forged or expired cookie is exactly a stranger.
    for label, cookie in (('a forged cookie', 's1.abc.def'),
                          ('somebody else\'s junk', 'x' * 200)):
        r = U.serve_request(fx.req(path='/', cookie=cookie))
        if r is None or r.status != 302:
            bad.append(f'{label} got past the gate')
        elif 'm=session_over' not in r.header('Location'):
            bad.append('a request that arrived with a cookie was not told '
                       'that its session had ended')

    # With a session, the gate steps out of the way and hands the router the
    # user it already looked up.
    cookie = fx.cookie_for(fx.admin())
    for p in GATED:
        req = fx.req(path=p, cookie=cookie)
        if U.serve_request(req) is not None:
            bad.append(f'{p} was refused to a signed-in admin')
        elif not req.session or req.session['username'] != ADMIN:
            bad.append(f'{p} passed the gate without req.session being set, '
                       f'so the router has to look the user up again')

    # The bookmark survives the detour.
    r = U.serve_request(fx.req(path='/audit/review',
                               query={'country': 'JPN', 'sort': 'new'}))
    where = r.header('Location')
    if 'country' not in where or 'JPN' not in where:
        bad.append(f'the query string was dropped on the way to the login '
                   f'page ({where}), so a filtered bookmark comes back '
                   f'unfiltered')

    # The three public addresses, and only those three.
    for p in sorted(U.PUBLIC_PATHS):
        r = U.serve_request(fx.req(path=p))
        if r is None:
            bad.append(f'{p} is public and returned None, which the router '
                       f'reads as "carry on" -- there is nothing behind it')
        elif r.status not in (200, 302, 303, 401):
            bad.append(f'{p} answered {r.status} to a stranger')
    if U.is_public('/api/board') or U.is_public('/'):
        bad.append('is_public() calls the dashboard public')


def locked_checks(bad, fx):
    """With no usable admin: the login page, the explanation, and nothing else.

    Fail closed on data, fail open on uptime. The source watcher re-execs this
    process unattended, so refusing to start is an outage nobody is watching;
    serving one page is not.
    """
    d = tempfile.mkdtemp(prefix='adv_auth_locked_')
    try:
        got = U.bootstrap(db_path=os.path.join(d, 'accounts.db'),
                          key_path=os.path.join(d, 'session.key'), env={})
        if got['ok']:
            bad.append('a deployment with no credential and no accounts '
                       'reported itself usable')
        if U.usable():
            bad.append('usable() is True with nothing to sign in to')
        detail = got.get('detail', '')
        if A.ENV_PASSWORD not in detail:
            bad.append('the locked state does not name the variable to set, '
                       'which is the only thing the page can say')
        if PW in detail:
            bad.append('the locked state printed a password')
        for p in GATED + (U.SIGNUP_PATH, U.ADMIN_PATH, U.LOGOUT_PATH):
            r = U.serve_request(fx.req(path=p))
            if r is None:
                bad.append(f'{p} was served while no account exists')
                continue
            if p.startswith('/api/') and r.status != 503:
                bad.append(f'{p} answered {r.status} while locked')
            if r.body and (b'<table' in r.body or b'audit' in r.body):
                bad.append(f'{p} rendered something other than the '
                           f'explanation while locked')
        r = U.serve_request(fx.req(path=U.LOGIN_PATH))
        if r.status != 200 or A.ENV_PASSWORD.encode() not in r.body:
            bad.append('the login page does not carry the explanation while '
                       'locked, so there is nothing on screen to act on')
        if b'<form' in r.body:
            bad.append('the locked login page offers a form there is no '
                       'account to use it with')
        # A POST is no different: no route, no data, no exception.
        r = U.serve_request(fx.req('POST', U.LOGIN_PATH,
                                   form={'username': ADMIN, 'password': PW}))
        if r is None or r.status != 200 or b'<form' in r.body:
            bad.append('a login POST while locked was answered with something '
                       'other than the explanation')

        # And an admin that an EARLIER run created still works after the
        # credential is tidied out of .env -- locking the dashboard over a
        # variable nobody needs any more is an outage of our own making.
        p2 = os.path.join(d, 'had_one.db')
        A.ensure_admin(path=p2, env=dict(ADMIN_ENV))
        got = U.bootstrap(db_path=p2, key_path=os.path.join(d, 'k2'), env={})
        if got['action'] != 'unset' or not got['ok'] or not U.usable():
            bad.append('an admin created by an earlier run stopped being able '
                       'to sign in when DASHBOARD_PASSWORD was removed from '
                       '.env')
    finally:
        shutil.rmtree(d, ignore_errors=True)
        fx.use()

    # A database that cannot be opened at all is a page, never a traceback:
    # this process is re-exec'd unattended and a crash loop has nobody
    # watching it.
    d2 = tempfile.mkdtemp(prefix='adv_auth_broken_')
    try:
        wall = os.path.join(d2, 'no-entry')
        os.makedirs(wall, mode=0o500)
        got = U.bootstrap(db_path=os.path.join(wall, 'accounts.db'),
                          key_path=os.path.join(d2, 'k'), env=dict(ADMIN_ENV))
        if got['ok']:
            bad.append('an unopenable database reported itself usable')
        r = U.serve_request(fx.req(path='/'))
        if r is None or r.status not in (302, 200):
            bad.append('an unopenable database did not leave a servable login '
                       'page behind')
    except Exception as e:                          # noqa: BLE001
        bad.append(f'a broken database raised out of the gate instead of '
                   f'becoming a page: {type(e).__name__}: {e}')
    finally:
        os.chmod(os.path.join(d2, 'no-entry'), 0o700)
        shutil.rmtree(d2, ignore_errors=True)
        fx.use()

    # And one that breaks AFTER the process is up -- the drive holding
    # data/dashboard unmounts under a running server. Every request opens the
    # store for itself, so this is not hypothetical, and a traceback per
    # request is a 500 on a process nobody is watching.
    cookie = fx.cookie_for(fx.admin())
    U._STATE['db'] = fx.dir                     # a directory is not a database
    try:
        for method, path, form in (
                ('GET', '/', None), ('GET', U.LOGIN_PATH, None),
                ('POST', U.LOGIN_PATH, {'username': ADMIN, 'password': PW}),
                ('GET', U.SIGNUP_PATH, None), ('GET', U.ADMIN_PATH, None),
                ('POST', U.SIGNUP_PATH, {'t': 'x', 'username': 'a.b',
                                         'password': 'a-long-enough-password',
                                         'confirm': 'a-long-enough-password'})):
            r = U.serve_request(fx.req(method, path, form=form, cookie=cookie))
            if r is not None and r.status >= 500:
                bad.append(f'{method} {path} answered {r.status} once the '
                           f'store stopped answering')
    except Exception as e:                          # noqa: BLE001
        bad.append(f'a store that stopped answering raised out of the gate '
                   f'({type(e).__name__}: {e}) instead of becoming a page')
    finally:
        fx.use()


# ── logging in ──────────────────────────────────────────────────────────────

def login_checks(bad, fx):
    """One answer for both wrong halves, and a limit on how often you may
    ask."""
    src = 'ip:test-login'
    A.clear_failures(src, path=fx.db)
    good = U.attempt_login(ADMIN, PW, src, path=fx.db)
    if not good['ok'] or good['user']['username'] != ADMIN:
        bad.append('the right password did not sign the admin in')
    wrong = U.attempt_login(ADMIN, 'not-the-password', src, path=fx.db)
    unknown = U.attempt_login('nobody.here', PW, src, path=fx.db)
    if wrong['ok'] or unknown['ok']:
        bad.append('a wrong password or an unknown name signed somebody in')
    if wrong['message'] != unknown['message']:
        bad.append(f'a wrong password says {wrong["message"]!r} and an '
                   f'unknown name says {unknown["message"]!r}: the login form '
                   f'is a user directory')
    disabled = A.create_user('gone.away', PW2, path=fx.db)
    A.set_active(disabled['id'], False, path=fx.db)
    off = U.attempt_login('gone.away', PW2, src, path=fx.db)
    if off['ok'] or off['message'] != wrong['message']:
        bad.append('a disabled account is answered differently from a wrong '
                   'password, which tells a stranger the account exists')
    A.clear_failures(src, path=fx.db)

    # The same work either way. A fresh source per attempt, so the throttle
    # never engages and the only thing being measured is the hashing.
    def spend(name, pw, n=5):
        out = []
        for i in range(n):
            t0 = time.perf_counter()
            U.attempt_login(name, pw, 'ip:timing-%s-%d' % (name, i),
                            path=fx.db)
            out.append(time.perf_counter() - t0)
        return sorted(out)[n // 2]
    miss = spend('nobody.at.all', PW)
    hit = spend(ADMIN, 'wrong-password-entirely')
    if miss <= 0 or hit <= 0:
        bad.append('a failed login took no measurable time at all')
    elif not 0.5 <= miss / hit <= 2.0:
        bad.append(f'an unknown username takes {miss * 1000:.1f}ms and a '
                   f'wrong password {hit * 1000:.1f}ms; the difference is '
                   f'readable over a network and turns the form into a user '
                   f'directory')

    # Guessing has a price, and paying it does not buy a correct answer.
    src2 = 'ip:test-lockout'
    A.clear_failures(src2, path=fx.db)
    for _ in range(A.THROTTLE_FREE + 1):
        U.attempt_login(ADMIN, 'wrong', src2, path=fx.db)
    st = A.throttle_state(src2, path=fx.db)
    if not st['locked']:
        bad.append(f'{A.THROTTLE_FREE + 1} wrong passwords from one address '
                   f'bought no lockout at all')
    now = U.attempt_login(ADMIN, PW, src2, path=fx.db)
    if now['ok']:
        bad.append('a correct password was accepted during a lockout, so the '
                   'lockout is a delay on the wrong guesses only')
    if 'ip:' in now['message'] or PW in now['message']:
        bad.append('the lockout message quotes something it should not')
    # ... and the wait is over when it is over.
    later = U.attempt_login(ADMIN, PW, src2,
                            now=int(time.time()) + A.THROTTLE_MAX + 1,
                            path=fx.db)
    if not later['ok']:
        bad.append('the lockout never expires, so one afternoon of wrong '
                   'guesses locks the account out for good')
    if A.throttle_state(src2, path=fx.db)['fails'] != 0:
        bad.append('a successful login left the failure count standing, so '
                   'the next mistyped password locks the account out')

    # A lockout is per DEVICE. Counting per account hands anybody who knows
    # the admin's name a way to lock the admin out of their own dashboard.
    a = U.throttle_source(fx.req('POST', U.LOGIN_PATH, remote='10.0.0.1',
                                 form={'username': ADMIN}))
    b = U.throttle_source(fx.req('POST', U.LOGIN_PATH, remote='10.0.0.1',
                                 form={'username': 'someone.else'}))
    c = U.throttle_source(fx.req('POST', U.LOGIN_PATH, remote='10.0.0.2',
                                 form={'username': ADMIN}))
    if a != b:
        bad.append('the throttle key changes with the username submitted, so '
                   'six wrong guesses at a name lock its owner out')
    if a == c:
        bad.append('the throttle key is the same from two different '
                   'addresses, so one guesser locks out everybody')
    if ADMIN in a:
        bad.append('the throttle key carries the submitted username')

    # The credentials come out of the BODY. A URL is history, a bookmark and
    # a Referer; a body is none of those.
    r = U.serve_request(fx.req('POST', U.LOGIN_PATH,
                               query={'username': ADMIN, 'password': PW}))
    if _cookie_value(r):
        bad.append('a username and password in the QUERY STRING signed '
                   'somebody in, which writes the password to browser history')

    # The real thing, end to end.
    r = U.serve_request(fx.req('POST', U.LOGIN_PATH,
                               form={'username': ADMIN, 'password': PW,
                                     'next': '/audit/review'}))
    if r.status != 303 or r.header('Location') != '/audit/review':
        bad.append(f'a good login answered {r.status} to '
                   f'{r.header("Location")!r} instead of the page that was '
                   f'asked for')
    value = _cookie_value(r)
    if not value or U.resolve(value, path=fx.db, key_path=fx.key) is None:
        bad.append('a good login did not set a working session cookie')
    r = U.serve_request(fx.req('POST', U.LOGIN_PATH,
                               form={'username': ADMIN, 'password': 'nope'}))
    if r.status != 401 or _cookie_value(r):
        bad.append('a bad login handed out a cookie or answered 200')
    if PW.encode() in r.body:
        bad.append('the login page printed the password that was submitted')

    # Signing out is its own check; see logout_checks.


def logout_checks(bad, fx):
    """Sign-out has to END the session, not just forget it here.

    IT USED TO DO NOTHING ON THE SERVER. Set-Cookie with Max-Age=0 and
    nothing else: the browser dropped its copy while the cookie stayed valid
    for the rest of its seven-day signed life. This module says the cookie
    travels in the clear and offers the epoch as the answer -- and the epoch
    moved for a password change, a disable and bump_session_epoch, for every
    revoking action except the one somebody who thinks their cookie was read
    off the wire actually clicks. Replaying the captured cookie after a
    sign-out is the whole check.

    AND IT HAS TO BE A POST WITH A TOKEN, precisely because it now revokes:
    on GET, one <img src="/logout"> on any page an annotator visits signs
    them out of every device they own, repeatedly.
    """
    def signed_in(who=ADMIN, pw=PW):
        r = U.serve_request(fx.req('POST', U.LOGIN_PATH,
                                   form={'username': who, 'password': pw}))
        return _cookie_value(r)

    ck = signed_in()
    if not ck or U.resolve(ck, path=fx.db, key_path=fx.key) is None:
        bad.append('logout_checks could not sign in to begin with')
        return

    # A GET may not act. It renders the button, so a bookmark still works and
    # somebody else's page cannot press it.
    r = U.serve_request(fx.req(path=U.LOGOUT_PATH, cookie=ck))
    if r.status != 200 or b'method="post"' not in r.body.lower():
        bad.append(f'GET {U.LOGOUT_PATH} answered {r.status} without a form '
                   f'to submit — a sign-out that a GET performs is a '
                   f'sign-out any page the reader visits can perform for '
                   f'them with one <img src>')
    if U.resolve(ck, path=fx.db, key_path=fx.key) is None:
        bad.append(f'GET {U.LOGOUT_PATH} ended the session by itself, which '
                   f'is exactly what an <img src="{U.LOGOUT_PATH}"> on any '
                   f'other site would then do')
    if U.CSRF_FIELD.encode() not in r.body:
        bad.append(f'the sign-out page carries no {U.CSRF_FIELD} field, so '
                   f'its own form cannot be submitted')

    # A POST without the session's token may not act either.
    r = U.serve_request(fx.req('POST', U.LOGOUT_PATH, cookie=ck,
                               form={U.CSRF_FIELD: 'not-the-token'}))
    if U.resolve(ck, path=fx.db, key_path=fx.key) is None:
        bad.append('a POST carrying the wrong CSRF token signed the account '
                   'out anyway — the token is not being checked')

    # The real thing: the token off the page it was drawn on.
    ses = U.resolve(ck, path=fx.db, key_path=fx.key)
    r = U.serve_request(fx.req('POST', U.LOGOUT_PATH, cookie=ck,
                               form={U.CSRF_FIELD: ses['csrf']}))
    if 'Max-Age=0' not in r.header('Set-Cookie'):
        bad.append('signing out did not clear the cookie in this browser')
    if not r.header('Location').startswith(U.LOGIN_PATH):
        bad.append(f'signing out went to {r.header("Location")!r} instead of '
                   f'the login form')
    # THE REPLAY. This is the finding.
    if U.resolve(ck, path=fx.db, key_path=fx.key) is not None:
        bad.append('the cookie still resolves AFTER signing out: sign-out '
                   'deleted the browser\'s copy and changed nothing on the '
                   'server, so a cookie already read off the wire goes on '
                   'working for the rest of its seven days and the only real '
                   'remedy is a password change')
    for path, want in (('/', 302), ('/api/board', 401),
                       ('/audit/review', 302)):
        r = U.serve_request(fx.req(path=path, cookie=ck))
        # None is the gate saying "carry on, this one is signed in", which on
        # a cookie that has just been signed out is the whole defect. Named
        # rather than dereferenced: r.status on None is an AttributeError
        # that reads as a broken test instead of a live session.
        got = 'let straight through' if r is None else r.status
        if got != want:
            bad.append(f'{path} answered {got} to the signed-out cookie, not '
                       f'{want} — the session outlived the sign-out on the '
                       f'routes that matter')

    # It ends the sessions of THAT account, and of no other.
    other = fx.member('logout.other', PW2)
    oval, _ = U.mint(other, key_path=fx.key)
    ck2 = signed_in()
    ses2 = U.resolve(ck2, path=fx.db, key_path=fx.key)
    U.serve_request(fx.req('POST', U.LOGOUT_PATH, cookie=ck2,
                           form={U.CSRF_FIELD: ses2['csrf']}))
    if U.resolve(oval, path=fx.db, key_path=fx.key) is None:
        bad.append('signing one account out ended another account\'s session '
                   'too — sign-out is scoped to the account that clicked it')

    # ...and the password still works afterwards. A revocation that locked
    # the owner out would be a worse defect than the one it fixed.
    if not U.attempt_login(ADMIN, PW, 'ip:logout-after', path=fx.db)['ok']:
        bad.append('the account could not sign in again after signing out')

    # With no session at all it is a redirect, not a crash and not a page
    # asking a stranger to confirm something they are not holding.
    r = U.serve_request(fx.req(path=U.LOGOUT_PATH))
    if r.status not in (301, 302, 303):
        bad.append(f'{U.LOGOUT_PATH} answered {r.status} to a caller with no '
                   f'session at all')


def burst_checks(bad, fx):
    """The free-attempt budget must be the same all at once as one at a time.

    THIS IS THE ONE A SEQUENTIAL TEST CANNOT SEE. The throttle used to read
    the counter, hand the password to scrypt for ~40ms and only then write
    the failure -- and the dashboard serves on a thread per request, so every
    attempt that arrived inside that window read the same pre-failure counter
    and was let through. One at a time it allowed the documented 6 checks;
    fired together it allowed 30 to 37, and since the lockout caps at
    THROTTLE_MAX a burst per window turned ~96 guesses a day into ~3,500.

    Counted the way an attacker counts: an answer carrying the wrong-password
    sentence cost a real scrypt derivation, an answer carrying the lockout
    sentence cost nothing.
    """
    import threading as _t
    wrong = U.attempt_login(ADMIN, 'not-the-password', 'ip:burst-probe',
                            path=fx.db)['message']
    A.clear_failures('ip:burst-probe', path=fx.db)

    def budget(n, concurrent):
        src = 'ip:burst-%s-%d' % ('all' if concurrent else 'one', n)
        A.clear_failures(src, path=fx.db)
        seen = []
        lock = _t.Lock()

        def one():
            got = U.attempt_login(ADMIN, 'definitely-not-it', src,
                                  path=fx.db)['message']
            with lock:
                seen.append(got)
        if concurrent:
            ts = [_t.Thread(target=one) for _ in range(n)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        else:
            for _ in range(n):
                one()
        return sum(1 for m in seen if m == wrong)

    want = A.THROTTLE_FREE + 1
    one_at_a_time = budget(40, False)
    if one_at_a_time != want:
        bad.append(f'{one_at_a_time} passwords were checked one at a time '
                   f'against a THROTTLE_FREE of {A.THROTTLE_FREE}; the '
                   f'concurrent budget below is measured against this one, '
                   f'so a drift here makes that check meaningless')
    for n in (40, 120):
        got = budget(n, True)
        if got > want:
            bad.append(f'{n} logins fired at once bought {got} real password '
                       f'checks where {want} were bought one at a time — the '
                       f'throttle reads its counter before the 40ms hash and '
                       f'writes the failure after it, so every attempt inside '
                       f'that window sees a counter nobody has incremented '
                       f'yet and the lockout is worth {got / want:.0f}x less '
                       f'than it reads')

    # Counting the attempt BEFORE checking it is the fix, and it has one trap:
    # branch on the lock the increment just produced and the account can never
    # be checked again -- every later attempt extends its own lockout ahead of
    # the password, including the right one. login_checks proves the wait ends;
    # this proves the counting itself did not close the door.
    src = 'ip:burst-recovery'
    A.clear_failures(src, path=fx.db)
    for _ in range(A.THROTTLE_FREE + 3):
        U.attempt_login(ADMIN, 'wrong', src, path=fx.db)
    later = U.attempt_login(ADMIN, PW, src,
                            now=int(time.time()) + A.THROTTLE_MAX + 1,
                            path=fx.db)
    if not later['ok']:
        bad.append('after the lockout expired the CORRECT password was still '
                   'refused: counting an attempt before checking it has '
                   'locked the real owner out permanently')


def redirect_checks(bad, fx):
    """?next= may only ever be a path on this server.

    An open redirect on a login page is a phishing primitive: a link that
    starts with the address somebody trusts and lands on one they do not,
    immediately after they have typed a password.
    """
    for hostile in ('//evil.example/x', 'https://evil.example',
                    'http://evil.example', '/\\evil.example',
                    '\\\\evil.example', 'javascript:alert(1)',
                    '//evil.example\r\nSet-Cookie: a=b', 'evil.example',
                    ''):
        got = U.safe_next(hostile)
        if got != '/':
            bad.append(f'safe_next({hostile!r}) returned {got!r}: the login '
                       f'page will send somebody off this server')
    for ok in ('/', '/audit/review', '/audit/review?country=JPN',
               '/datasets/thumb?name=x'):
        if U.safe_next(ok) != ok:
            bad.append(f'safe_next({ok!r}) refused a path on this server')
    if U.safe_next(U.LOGIN_PATH) != '/':
        bad.append('the login page redirects to itself after a login')
    r = U.serve_request(fx.req('POST', U.LOGIN_PATH,
                               form={'username': ADMIN, 'password': PW,
                                     'next': '//evil.example/x'}))
    if r.header('Location') != '/':
        bad.append(f'a login carried a hostile ?next= all the way to the '
                   f'Location header: {r.header("Location")!r}')


# ── invites and signup ──────────────────────────────────────────────────────

def signup_checks(bad, fx):
    """A link is checked before its form is drawn, and it works once."""
    admin = fx.admin()
    inv = A.create_invite(admin['id'], ttl=600, note='for bob only',
                          path=fx.db)
    tok = inv['token']

    st = U.peek_invite(tok, path=fx.db)
    if st['state'] != 'open':
        bad.append(f'a fresh invite peeks as {st["state"]}')
    if A._token_hash(tok) != A._token_hash(tok):
        bad.append('the token hash is not stable')          # paranoia, cheap
    for key, value in st.items():
        if isinstance(value, str) and (ADMIN in value or 'bob' in value):
            bad.append(f'peek_invite leaks {key}: whoever holds a link learns '
                       f'who issued it and what it was for')

    r = U.serve_request(fx.req(path=U.SIGNUP_PATH, query={'t': tok}))
    if r.status != 200 or b'<form' not in r.body:
        bad.append('a valid invite link did not render a form')
    body = r.body.decode()
    for leak in (ADMIN, 'for bob only'):
        if leak in body:
            bad.append(f'the signup page names {leak!r} to whoever opened the '
                       f'link')
    if r.header('Referrer-Policy') != 'no-referrer':
        bad.append('the signup page does not set Referrer-Policy: no-referrer, '
                   'so the token in the URL walks out in the next request\'s '
                   'Referer header')
    if 'no-store' not in r.header('Cache-Control'):
        bad.append('the signup page is cacheable, so the token sits in a disk '
                   'cache')

    # Every dead link says which kind of dead, and offers no form.
    dead = {}
    for state in ('expired', 'revoked', 'used'):
        iv = A.create_invite(admin['id'], ttl=600, path=fx.db)
        if state == 'expired':
            st2 = U.peek_invite(iv['token'], now=int(time.time()) + 3600,
                                path=fx.db)
            dead[state] = (iv['token'], st2, int(time.time()) + 3600)
            continue
        if state == 'revoked':
            A.revoke_invite(iv['id'], path=fx.db)
        else:
            A.redeem_invite(iv['token'], 'taken.%s' % (state,), PW2,
                            path=fx.db)
        dead[state] = (iv['token'], U.peek_invite(iv['token'], path=fx.db),
                       None)
    for state, (token, st2, when) in dead.items():
        if st2['state'] != state:
            bad.append(f'an invite that is {state} peeks as {st2["state"]}')
        # 'when' is set for the expired one only: the page has to be rendered
        # at the same clock the peek was taken at, or it is still open and the
        # check quietly tests nothing.
        r = U.serve_request(fx.req(path=U.SIGNUP_PATH, query={'t': token}),
                            now=when)
        if b'<form' in r.body and b'method="post"' in r.body:
            bad.append(f'a {state} invite still drew a signup form')
        if U.INVITE_WORDS[state].encode() not in r.body:
            bad.append(f'a {state} invite did not say so')
    r = U.serve_request(fx.req(path=U.SIGNUP_PATH, query={'t': 'made-up'}))
    if b'<form' in r.body and b'method="post"' in r.body:
        bad.append('a made-up token drew a signup form')

    # The real thing. One account, then nothing.
    before = len(A.list_users(path=fx.db))
    r = U.serve_request(fx.req('POST', U.SIGNUP_PATH,
                               form={'t': tok, 'username': 'new.person',
                                     'password': 'a-long-enough-password',
                                     'confirm': 'a-long-enough-password'}))
    value = _cookie_value(r)
    if r.status != 303 or not value:
        bad.append(f'a good signup answered {r.status} without a session; the '
                   f'invite was the proof of who they are and they have just '
                   f'chosen the password')
    ses = U.resolve(value, path=fx.db, key_path=fx.key)
    if not ses or ses['username'] != 'new.person' or ses['role'] != 'member':
        bad.append('the account a signup created is not the one it signed in')
    if tok in r.header('Location'):
        bad.append('the invite token was put in a Location header, which is a '
                   'URL, which is browser history')
    r = U.serve_request(fx.req('POST', U.SIGNUP_PATH,
                               form={'t': tok, 'username': 'second.person',
                                     'password': 'a-long-enough-password',
                                     'confirm': 'a-long-enough-password'}))
    if _cookie_value(r):
        bad.append('the same invite link made a second account')
    if len(A.list_users(path=fx.db)) != before + 1:
        bad.append('one invite produced more than one account')

    # A mistake must not spend the link.
    iv = A.create_invite(admin['id'], ttl=600, path=fx.db)
    for label, form in (
            ('a mistyped confirmation',
             {'username': 'careful.one', 'password': 'a-long-enough-password',
              'confirm': 'a-long-enough-passwerd'}),
            ('a password that is too short',
             {'username': 'careful.one', 'password': 'short',
              'confirm': 'short'}),
            ('a username somebody already has',
             {'username': 'new.person', 'password': 'a-long-enough-password',
              'confirm': 'a-long-enough-password'}),
            ('a username with a space in it',
             {'username': 'not a name', 'password': 'a-long-enough-password',
              'confirm': 'a-long-enough-password'})):
        form['t'] = iv['token']
        r = U.serve_request(fx.req('POST', U.SIGNUP_PATH, form=form))
        if _cookie_value(r):
            bad.append(f'{label} created an account')
        st2 = U.peek_invite(iv['token'], path=fx.db)
        if st2['state'] != 'open':
            bad.append(f'{label} spent the invite link, so somebody has to '
                       f'ask for a new one over a typo')
        if r.status != 400 or b'msg bad' not in r.body:
            bad.append(f'{label} was not explained on the page')

    # A guessed token costs the guesser something; a typo does not.
    src = 'ip:signup-guess'
    A.clear_failures(src, path=fx.db)
    U.attempt_signup('not-a-real-token', 'x.y', 'a-long-enough-password',
                     'a-long-enough-password', src, path=fx.db)
    if A.throttle_state(src, path=fx.db)['fails'] != 1:
        bad.append('guessing an invite token is free, so the signup route is '
                   'an unauthenticated way to hammer the database')
    A.clear_failures(src, path=fx.db)
    U.attempt_signup(iv['token'], 'new.person', 'a-long-enough-password',
                     'a-long-enough-password', src, path=fx.db)
    if A.throttle_state(src, path=fx.db)['fails'] != 0:
        bad.append('a real invite holder who chose a taken username was '
                   'counted as an attacker')


# ── the admin page ──────────────────────────────────────────────────────────

def admin_checks(bad, fx):
    """Gated on the role in the database, and answering a member with nothing.

    "Exactly what a stranger gets" is the requirement, and the empty 404 is
    how it is met: a page that says "admins only" tells a member the address
    exists and is worth attacking.
    """
    admin = fx.admin()
    cookie = fx.cookie_for(admin)
    ses = U.resolve(cookie, path=fx.db, key_path=fx.key)
    member = fx.member('plain.member')
    mcookie = fx.cookie_for(member)
    mses = U.resolve(mcookie, path=fx.db, key_path=fx.key)

    for label, ck in (('a member', mcookie), ('a stranger', None)):
        r = U.serve_request(fx.req(path=U.ADMIN_PATH, cookie=ck))
        if label == 'a member':
            if r.status != 404 or r.body:
                bad.append(f'{label} at {U.ADMIN_PATH} got {r.status} with '
                           f'{len(r.body)} bytes; they are meant to get the '
                           f'same empty 404 as an address that does not exist')
        elif r.status != 302:
            bad.append(f'{label} at {U.ADMIN_PATH} got {r.status}')

    # And the endpoints under it, which is where a member would actually try.
    before = len(A.list_invites(path=fx.db))
    for path, form in ((U.ADMIN_PATH + '/invite', {'hours': '2'}),
                       (U.ADMIN_PATH + '/user',
                        {'id': str(admin['id']), 'do': 'disable'}),
                       (U.ADMIN_PATH + '/revoke', {'id': '1'})):
        form['csrf'] = mses['csrf']
        r = U.serve_request(fx.req('POST', path, cookie=mcookie, form=form))
        if r.status != 404 or r.body:
            bad.append(f'a member POSTing to {path} got {r.status}')
    if len(A.list_invites(path=fx.db)) != before:
        bad.append('a member minted an invite')
    if not A.get_user(admin['id'], path=fx.db)['active']:
        bad.append('a member disabled the admin')
    # Called directly, with an action the STORE does not gate for itself:
    # accounts.set_active() does not ask who is calling, so this is the one
    # place the role has to be checked and the check has to live in the
    # function, not only in the route that reaches it today.
    spare = A.create_user('spare.account', PW2, path=fx.db)
    if U.admin_action('user', fx.req('POST', form={'id': str(spare['id']),
                                                   'do': 'disable'}),
                      mses, path=fx.db)['ok']:
        bad.append('admin_action() itself does not check the role, so a route '
                   'that forgets to is a hole')
    if not A.get_user(spare['id'], path=fx.db)['active']:
        bad.append('a member disabled an account through admin_action()')

    # What the page has to show.
    inv = A.create_invite(admin['id'], ttl=600, note='night <shift>',
                          path=fx.db)
    A.revoke_invite(inv['id'], path=fx.db)
    r = U.serve_request(fx.req(path=U.ADMIN_PATH, cookie=cookie))
    body = r.body.decode()
    if r.status != 200:
        bad.append(f'the admin page answered {r.status} to its admin')
    for want, why in ((ADMIN, 'who issued an invite'),
                      ('revoked', 'the state of a withdrawn invite'),
                      ('plain.member', 'the accounts table'),
                      ('disable', 'a way to retire an account'),
                      ('make admin', 'a way to change a role'),
                      (ses['csrf'], 'a CSRF token in its forms')):
        if want not in body:
            bad.append(f'the admin page does not show {why}')
    if 'night &lt;shift&gt;' not in body or '<shift>' in body:
        bad.append('a note typed by an admin is rendered as markup: a note is '
                   'the one field on this page somebody types freely')

    # Minting: the token appears once, in the page, and never in a redirect.
    r = U.serve_request(fx.req('POST', U.ADMIN_PATH + '/invite', cookie=cookie,
                               form={'csrf': ses['csrf'], 'hours': '3',
                                     'note': 'field team', 'role': 'member'},
                               host='dash.example:8080'))
    body = r.body.decode()
    if r.status != 200:
        bad.append(f'minting an invite answered {r.status}; a redirect would '
                   f'have to carry the token in its Location')
    m = re.search(r'value="(https?://[^"]*/signup\?t=[^"]+)"', body)
    if not m:
        bad.append('the minted link is not on the page in a form anybody can '
                   'copy')
    else:
        token = m.group(1).split('t=')[1]
        if U.peek_invite(token, path=fx.db)['state'] != 'open':
            bad.append('the link on the page is not a working invite')
        if r.header('Location'):
            bad.append('the mint answered with a redirect carrying the token')
        if 'no-store' not in r.header('Cache-Control'):
            bad.append('the page showing a fresh invite token is cacheable')
        r2 = U.serve_request(fx.req(path=U.ADMIN_PATH, cookie=cookie))
        if token in r2.body.decode():
            bad.append('the token is still on the page after a reload: it is '
                       'meant to be shown once and never stored')
    if 'only time it is shown' not in body:
        bad.append('the page does not say the link will not be shown again, '
                   'which is the one thing the admin has to know before they '
                   'close it')

    # The store's refusals reach the page instead of becoming a traceback.
    only = A.count_admins(path=fx.db)
    if only == 1:
        r = U.serve_request(fx.req('POST', U.ADMIN_PATH + '/user',
                                   cookie=cookie,
                                   form={'csrf': ses['csrf'],
                                         'id': str(admin['id']),
                                         'do': 'disable'}))
        if A.get_user(admin['id'], path=fx.db)['active'] is False:
            bad.append('the last admin disabled themselves out of their own '
                       'dashboard')
        if r.status != 400 or b'msg bad' not in r.body:
            bad.append('the refusal was not shown on the page')

    # Even with a second admin: the account you are holding is not one you
    # may switch off from here.
    second = A.create_user('other.admin', PW2, role='admin', path=fx.db)
    r = U.serve_request(fx.req('POST', U.ADMIN_PATH + '/user', cookie=cookie,
                               form={'csrf': ses['csrf'],
                                     'id': str(admin['id']), 'do': 'disable'}))
    if not A.get_user(admin['id'], path=fx.db)['active']:
        bad.append('an admin disabled the account they were signed in with '
                   'and landed on the login page')
    # ... but somebody else's, they may.
    r = U.serve_request(fx.req('POST', U.ADMIN_PATH + '/user', cookie=cookie,
                               form={'csrf': ses['csrf'],
                                     'id': str(second['id']),
                                     'do': 'disable'}))
    if A.get_user(second['id'], path=fx.db)['active']:
        bad.append('an admin could not disable another account')
    if U.resolve(fx.cookie_for(second), path=fx.db, key_path=fx.key):
        bad.append('a disabled account still resolves a session')
    A.set_active(second['id'], True, path=fx.db)

    # An unknown action is refused, not guessed at.
    r = U.serve_request(fx.req('POST', U.ADMIN_PATH + '/nonsense',
                               cookie=cookie, form={'csrf': ses['csrf']}))
    if r.status not in (400, 404):
        bad.append(f'an unknown admin action answered {r.status}')


# ── the pages themselves ────────────────────────────────────────────────────

def page_checks(bad, fx):
    """Every page is self-contained, escaped, and says nothing extra.

    The login and signup pages are the whole unauthenticated surface. One
    <script src> or one stylesheet on them is another route that has to be
    public, and the day it is added it will be public for everybody.
    """
    admin = fx.admin()
    cookie = fx.cookie_for(admin)
    inv = A.create_invite(admin['id'], ttl=600, path=fx.db)
    pages = {
        'login': U.serve_request(fx.req(path=U.LOGIN_PATH)),
        'signup': U.serve_request(fx.req(path=U.SIGNUP_PATH,
                                         query={'t': inv['token']})),
        'admin': U.serve_request(fx.req(path=U.ADMIN_PATH, cookie=cookie)),
    }
    d = tempfile.mkdtemp(prefix='adv_auth_locked2_')
    try:
        U.bootstrap(db_path=os.path.join(d, 'a.db'),
                    key_path=os.path.join(d, 'k'), env={})
        pages['locked'] = U.serve_request(fx.req(path=U.LOGIN_PATH))
    finally:
        shutil.rmtree(d, ignore_errors=True)
        fx.use()

    for name, r in pages.items():
        body = r.body.decode('utf-8', 'replace')
        if not body.startswith('<!doctype html>'):
            bad.append(f'the {name} page is not a document')
        for pat, why in (
                (r'<script[^>]+src=', 'a script file'),
                (r'<link[^>]+stylesheet', 'a stylesheet'),
                (r'<img[^>]+src="(?!data:)', 'an image'),
                (r'@import', 'an imported stylesheet'),
                (r'<link[^>]+href="(?!data:)', 'a linked file')):
            if re.search(pat, body, re.I):
                bad.append(f'the {name} page loads {why}: the unauthenticated '
                           f'surface is meant to be one document with nothing '
                           f'behind it')
        for header, want in (('Cache-Control', 'no-store'),
                             ('Referrer-Policy', 'no-referrer'),
                             ('X-Content-Type-Options', 'nosniff'),
                             ('X-Frame-Options', 'DENY'),
                             ('Content-Security-Policy', "default-src 'none'")):
            if want not in r.header(header):
                bad.append(f'the {name} page is missing {header}: {want}')
        if 'charset=utf-8' not in r.header('Content-Type'):
            bad.append(f'the {name} page does not declare its encoding')

    # Anything a person typed is escaped on the way back out, including the
    # username of a failed attempt, which is the one field a stranger controls.
    hostile = '"><script>alert(1)</script>'
    r = U.serve_request(fx.req('POST', U.LOGIN_PATH,
                               form={'username': hostile, 'password': 'x'}))
    body = r.body.decode()
    if '<script>alert' in body:
        bad.append('the login page echoes a submitted username as markup')
    if '&lt;script&gt;' not in body:
        bad.append('the login page did not echo the username at all, so the '
                   'escaping check is asleep')
    r = U.serve_request(fx.req('POST', U.SIGNUP_PATH,
                               form={'t': inv['token'], 'username': hostile,
                                     'password': 'x', 'confirm': 'x'}))
    if '<script>alert' in r.body.decode():
        bad.append('the signup page echoes a submitted username as markup')

    # The pages do not name the password, the token or anything else secret.
    r = pages['admin']
    if inv['token'] in r.body.decode():
        bad.append('the admin page lists a live invite token; only the moment '
                   'it is minted may show one')
    if PW in pages['login'].body.decode():
        bad.append('the login page contains the admin password')


def silence_checks(bad, fx):
    """Nothing on these paths prints a secret.

    .env holds a hundred Mapillary keys next to the dashboard password, and
    the dashboard's own log_message is a no-op precisely so that a request
    line carrying an invite token never reaches disk. Anything this module
    printed would land in the same journal.
    """
    admin = fx.admin()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        fx.use()
        cookie = fx.cookie_for(admin)
        ses = U.resolve(cookie, path=fx.db, key_path=fx.key)
        U.attempt_login(ADMIN, PW, 'ip:quiet', path=fx.db)
        U.attempt_login(ADMIN, 'wrong', 'ip:quiet', path=fx.db)
        r = U.serve_request(fx.req('POST', U.ADMIN_PATH + '/invite',
                                   cookie=cookie,
                                   form={'csrf': ses['csrf'], 'hours': '2'}))
        token = re.search(r'/signup\?t=([^"&]+)"', r.body.decode()).group(1)
        U.serve_request(fx.req(path=U.SIGNUP_PATH, query={'t': token}))
        U.serve_request(fx.req('POST', U.SIGNUP_PATH,
                               form={'t': token, 'username': 'quiet.one',
                                     'password': 'a-long-enough-password',
                                     'confirm': 'a-long-enough-password'}))
    noise = out.getvalue() + err.getvalue()
    for needle, what in ((PW, 'the admin password'),
                         (token, 'an invite token'),
                         (cookie, 'a session cookie'),
                         (U.secret(fx.key).hex(), 'the signing key')):
        if needle and needle in noise:
            bad.append(f'{what} was printed to the journal')
    if noise.strip():
        bad.append(f'the gate printed something on an ordinary request: '
                   f'{noise.strip()[:120]!r}')


# ── the source itself ───────────────────────────────────────────────────────

def source_checks(bad, fx):
    """Two things no behavioural test can see.

    A signature compared with == passes every functional check ever written
    and leaks how much of a forgery was right, one byte at a time. And an
    X-Forwarded-For read anywhere in here is a throttle key the caller
    chooses -- which is no throttle at all.
    """
    try:
        src = open(AUTH_PY, encoding='utf-8').read()
        tree = ast.parse(src)
    except (OSError, SyntaxError) as e:
        bad.append(f'could not parse auth.py: {e}')
        return
    if 'compare_digest' not in src:
        bad.append('nothing in auth.py uses hmac.compare_digest')
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(o, (ast.Eq, ast.NotEq)) for o in node.ops):
            continue
        names = {n.id.lower() for n in ast.walk(node)
                 if isinstance(n, ast.Name)}
        names |= {n.attr.lower() for n in ast.walk(node)
                  if isinstance(n, ast.Attribute)}
        hit = names & {'sig', 'mac', 'want', 'given', 'digest', 'signature',
                       'csrf', 'token'}
        if hit:
            bad.append(f'auth.py:{node.lineno}: {"/".join(sorted(hit))} is '
                       f'compared with ==; a byte-at-a-time comparison of a '
                       f'signature tells an attacker how much of it was right')
    # String CONSTANTS, not the source text: the module comments explain why
    # a forwarded address is not trusted, and a check that reads its own
    # documentation as the defect fires on the file that gets it right.
    # Short ones only, for the same reason: a docstring that uses the word
    # "forwarded" in a sentence is prose, and a header name is never 32
    # characters long.
    literals = {n.value.lower() for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and len(n.value) <= 32}
    for header in ('x-forwarded-for', 'x-real-ip', 'forwarded'):
        if any(header in s for s in literals):
            bad.append(f'auth.py reads {header}: nothing proxies this server, '
                       f'so that header is a string the caller chose and the '
                       f'lockout it keys becomes somebody else\'s')
    # Cookies are parsed, not evaluated, and nothing here builds SQL.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = getattr(node.func, 'id', '') or getattr(node.func, 'attr', '')
            if fn in ('eval', 'exec', 'pickle', 'loads') and fn != 'loads':
                bad.append(f'auth.py:{node.lineno}: {fn}() on request data')
            if fn == 'execute':
                arg = node.args[0] if node.args else None
                if not isinstance(arg, ast.Constant):
                    bad.append(f'auth.py:{node.lineno}: a built string is '
                               f'being executed as SQL')


def wiring_checks(bad, fx):
    """What dashboard.py has to do once these modules are wired in.

    ASLEEP UNTIL THEN, on purpose: this file is written before the routes
    are, and a check that fails on the day it is committed is a check
    somebody switches off. The moment dashboard.py imports auth, all of it
    arms itself.
    """
    try:
        src = open(DASHBOARD_PY, encoding='utf-8').read()
    except OSError as e:
        bad.append(f'could not read dashboard.py: {e}')
        return False
    # True the whole time, wired or not: a logged request line carries the
    # invite token out of the URL and into serve.log.
    if not re.search(r'def log_message\(self, \*a\):\s*\n\s*pass', src):
        bad.append('BoardHandler.log_message is no longer a no-op, so every '
                   'request line -- including /signup?t=<token> -- is being '
                   'written down')
    if not re.search(r'^\s*import auth\b', src, re.M):
        return False
    armed = True
    watched = re.search(r'for _m in \(([^)]*)\)', src)
    names = watched.group(1) if watched else ''
    for mod in ('auth.py', 'accounts.py'):
        if mod not in names:
            bad.append(f'{mod} is not in serve()\'s watch list, so an edit to '
                       f'it sits invisible behind a healthy-looking server '
                       f'until somebody restarts it by hand')
    if 'serve_request' not in src:
        bad.append('dashboard.py imports auth but never calls '
                   'auth.serve_request(), so nothing is actually gated')
    return armed


def import_checks(bad, fx):
    """The surface the router is about to be written against."""
    for name in ('COOKIE', 'LOGIN_PATH', 'LOGOUT_PATH', 'SIGNUP_PATH',
                 'ADMIN_PATH', 'PUBLIC_PATHS', 'PRIVATE_FILES', 'KEY_PATH'):
        if not hasattr(U, name):
            bad.append(f'auth.{name} is gone; the router reads it')
    for name in ('bootstrap', 'serve_request', 'guard', 'resolve', 'mint',
                 'read_session', 'set_cookie', 'clear_cookie', 'csrf_for',
                 'csrf_ok', 'safe_next', 'owns', 'is_public', 'usable',
                 'attempt_login', 'attempt_signup', 'peek_invite',
                 'admin_action', 'login_page', 'signup_page', 'admin_page'):
        if not callable(getattr(U, name, None)):
            bad.append(f'auth.{name}() is gone; the router calls it')
    if not hasattr(U.Request, 'from_handler') or not hasattr(U.Reply, 'send'):
        bad.append('Request.from_handler() / Reply.send() are how the router '
                   'talks to a socket; one of them is gone')
    # from_handler must read a body ONLY for the routes this module owns, or
    # it eats the JSON that /api/audit/verdict is about to read for itself.
    if not U.owns(U.LOGIN_PATH) or not U.owns(U.ADMIN_PATH + '/user'):
        bad.append('owns() does not claim this module\'s own routes')
    for p in ('/api/audit/verdict', '/api/review/seen', '/api/detect/flag',
              '/', '/audit/review'):
        if U.owns(p):
            bad.append(f'owns({p!r}) is True, so Request.from_handler() will '
                       f'read that route\'s body and the handler behind it '
                       f'will see an empty one')


class _Handler:
    """The three attributes Request.from_handler() reads, and nothing else."""

    def __init__(self, command, path, headers, body=b''):
        self.command = command
        self.path = path
        self.headers = headers
        self.rfile = io.BytesIO(body)
        self.client_address = ('10.0.0.9', 41234)


def handler_checks(bad, fx):
    """The bridge from a real request, including the body it must not eat."""
    body = b'username=zeta&password=hunter2hunter2'
    req = U.Request.from_handler(_Handler(
        'POST', '/login?next=/x',
        {'Content-Length': str(len(body)),
         'Content-Type': 'application/x-www-form-urlencoded',
         'Cookie': '%s=abc; other=1' % (U.COOKIE,), 'Host': 'dash:8080'},
        body))
    if req.one('username') != 'zeta' or req.one('password') != 'hunter2hunter2':
        bad.append('a login form did not survive the trip out of the socket')
    if req.arg('next') != '/x' or req.cookies.get(U.COOKIE) != 'abc':
        bad.append('the query string or the cookie was lost')
    if req.remote != '10.0.0.9' or req.host != 'dash:8080':
        bad.append('the client address or the Host header was lost')

    payload = b'{"key": "sequence#42", "verdict": "dog"}'
    h = _Handler('POST', '/api/audit/verdict',
                 {'Content-Length': str(len(payload)),
                  'Content-Type': 'application/json'}, payload)
    U.Request.from_handler(h)
    if h.rfile.read() != payload:
        bad.append('Request.from_handler() consumed the body of a route it '
                   'does not own; every verdict, flag and box correction the '
                   'dashboard posts would arrive empty')

    big = _Handler('POST', '/login',
                   {'Content-Length': str(U.MAX_BODY + 1),
                    'Content-Type': 'application/x-www-form-urlencoded'},
                   b'x' * 32)
    req = U.Request.from_handler(big)
    if not req.oversize or req.form:
        bad.append('a Content-Length larger than the cap was read anyway: an '
                   'unauthenticated POST can make this box read whatever it '
                   'says it is sending')
    if U.do_login(req).status != 413:
        bad.append('an oversized login was not refused')

    # A body that is not a form is not a form. A JSON login would bypass every
    # form-shaped check this file makes.
    h = _Handler('POST', '/login',
                 {'Content-Length': '30', 'Content-Type': 'application/json'},
                 b'{"username":"zeta","p":"x"}')
    if U.Request.from_handler(h).form:
        bad.append('a JSON body was parsed as a login form')


def reply_checks(bad, fx):
    """Reply.send() writes what it says it writes, and never logs."""
    class Sink:
        def __init__(self, command='GET'):
            self.command = command
            self.lines = []
            self.wfile = io.BytesIO()
            self.logged = 0

        def send_response(self, code):
            self.lines.append(('status', code))

        def send_header(self, k, v):
            self.lines.append((k, v))

        def end_headers(self):
            self.lines.append(('end', ''))

        def log_message(self, *a):
            self.logged += 1

    s = Sink()
    U.page_reply('<!doctype html>\n<html></html>').send(s)
    kinds = dict((k, v) for k, v in s.lines if k != 'end')
    if kinds.get('status') != 200:
        bad.append('page_reply did not send a 200')
    if kinds.get('Content-Length') != str(len(
            b'<!doctype html>\n<html></html>')):
        bad.append(f'the Content-Length is wrong: {kinds.get("Content-Length")}'
                   f' -- a wrong one truncates the page or hangs the browser')
    if s.wfile.getvalue() != b'<!doctype html>\n<html></html>':
        bad.append('the body that went down the socket is not the body')
    if s.logged:
        bad.append('sending a reply logged a request line')
    head = Sink('HEAD')
    U.page_reply('<!doctype html>\n<html></html>').send(head)
    if head.wfile.getvalue():
        bad.append('a HEAD request was answered with a body')
    # Two cookies, two headers -- a dict would silently keep one.
    r = U.Reply(200, b'', [U.set_cookie('a', 1), U.clear_cookie()])
    if len([1 for k, _ in r.headers if k == 'Set-Cookie']) != 2:
        bad.append('a reply cannot carry two Set-Cookie headers')


def identity_checks(bad, fx):
    """ONE identity strip, written once, on every page that has a header.

    It was written twice: dashboard.py kept a copy and admin_page() wrote a
    second one inline. They drifted exactly the way a duplicate does -- five
    pages got a bordered button, the sixth got an underlined link in a
    different sentence at a different height -- and the copy that mattered was
    the one carrying the CSRF token and the logout route, both of which live
    in this module. A second spelling is a form posting nowhere on the day
    either is renamed, and that fails as "the button did nothing".

    So: the markup comes from here, the pages splice it, and no page writes
    its own way out.
    """
    session = {'username': 'admin', 'csrf': 'TOKEN-ABC', 'role': 'admin'}
    strip = U.identity_html(session)
    if not strip:
        bad.append('identity_html renders nothing for a signed-in session')
        return
    for need, why in (
            ('method="post"', 'sign-out is a GET again, so any page the '
             'reader opens can end their session for them'),
            (U.CSRF_FIELD, 'the sign-out form carries no CSRF field'),
            ('TOKEN-ABC', "the form does not carry the session's own token"),
            (U.LOGOUT_PATH, 'the form posts somewhere other than the logout '
             'route this module owns'),
            ('class="hsep"', 'the divider does not ship with the strip, so a '
             'page draws a rule that hangs there when nobody is signed in'),
            ('class="whoi"', 'the monogram is gone — the one place a person '
             'appears on a page made of counts')):
        if need not in strip:
            bad.append('the identity strip: ' + why)
    if strip.count('sign out') != 1:
        bad.append('the identity strip offers the way out %d times'
                   % (strip.count('sign out'),))
    if U.identity_html(None) or U.identity_html({}):
        bad.append('a signed-out reader is shown an identity strip')
    # a name reaching the header is escaped, whatever USERNAME_RE allows today
    nasty = U.identity_html({'username': '<img src=x>', 'csrf': 'c'})
    if '<img' in nasty:
        bad.append('a username reaches the header unescaped')
    # AND THE MONOGRAM IS A WHOLE CHARACTER. Slicing after escaping cuts an
    # entity in half -- '<sam>' becomes '&lt;sam&gt;' and the disc shows a
    # bare '&'. USERNAME_RE forbids that character today, which is exactly
    # why nobody would see it until the day that rule loosens.
    mark = 'class="whoi" aria-hidden="true">'
    for raw, want in (('admin', 'a'), ('<img src=x>', '&lt;'),
                      ('&sam', '&amp;')):
        h = U.identity_html({'username': raw, 'csrf': 'c'})
        i = h.find(mark)
        got = h[i + len(mark):h.find('<', i + len(mark))] if i >= 0 else ''
        if got != want:
            bad.append('the monogram for %r is %r, not %r — an HTML entity '
                       'was cut in half' % (raw, got, want))

    # THE SECOND SPELLING. Every module that renders a header must get the
    # strip from here; none may write its own name-and-sign-out. The tokens
    # are MARKUP, not prose -- these files explain themselves at length, and a
    # check that fires on the word in a comment is a check nobody keeps.
    marks = ('>sign out<', 'class="whox"', 'action="%s"' % (U.LOGOUT_PATH,))
    for mod in ('dashboard.py', 'datasets.py', 'audit.py'):
        try:
            src = open(os.path.join(DASH, mod), encoding='utf-8').read()
        except OSError as e:
            bad.append(f'could not read {mod}: {e}')
            continue
        for tok in marks:
            if tok in src:
                bad.append(f'{mod} writes its own identity strip ({tok!r}) '
                           'instead of asking auth.py for the one')
    # and auth.py's own page must use the shared one, not a third copy
    src = open(AUTH_PY, encoding='utf-8').read()
    if src.count('>sign out<') != 1:
        bad.append('auth.py renders a way out %d times — the Accounts page is '
                   'writing its own again' % (src.count('>sign out<'),))
    if 'identity_html(session)' not in src:
        bad.append('the Accounts page does not render the shared strip')

    # THE CLUSTER ORDER, one contract across the four headers: where to GO,
    # then the hairline the strip brings, then whose session this is. The
    # strip standing before the controls is what made a header of three kinds
    # of thing read as a flat row of equals.
    for mod, mark in (('datasets.py', '__ACCOUNT__'),
                      ('audit.py', '__ACCOUNT__'),
                      ('dashboard.py', '<!--ACCT-->')):
        src = open(os.path.join(DASH, mod), encoding='utf-8').read()
        # the front page names the sentinel twice: once as the bytes the
        # handler splices at, once in the template. The template's copy is
        # the one with a position, so anchor on the action row.
        base = (src.index('<div class="hact">')
                if mod == 'dashboard.py' else 0)
        i = src.find(mark, base)
        if i < 0:
            bad.append(f'{mod} no longer splices the identity strip at all')
            continue
        if mod == 'dashboard.py':
            if 'class="upd"' not in src[base:i]:
                bad.append('the front page puts identity back in the middle '
                           'of the action row, ahead of what the page knows')
            continue
        line = src[src.rfind('\n', 0, i) + 1:src.find('\n', i)]
        if 'class="back"' not in line or line.index('back') > line.index(mark):
            bad.append(f'{mod} puts the identity strip ahead of the way out, '
                       'so the header reads as a flat row again')


def main():
    bad = []
    armed = False
    try:
        with Fixture() as fx:
            for fn in (key_checks, signature_checks, expiry_checks,
                       epoch_checks, cookie_checks, csrf_checks, gate_checks,
                       locked_checks, login_checks, logout_checks,
                       burst_checks, redirect_checks,
                       signup_checks, admin_checks, page_checks,
                       identity_checks,
                       silence_checks, source_checks, import_checks,
                       handler_checks, reply_checks, wiring_checks):
                try:
                    got = fn(bad, fx)
                    armed = armed or (fn is wiring_checks and bool(got))
                except Exception as e:      # noqa: BLE001 - report, not die
                    bad.append(f'{fn.__name__} threw {type(e).__name__}: {e}')
    except Exception as e:
        bad.append(f'the fixture would not build: {type(e).__name__}: {e}')
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('a cookie is a signature over every field and dies when the account '
          'changes, nothing at all is served without one, a member gets the '
          'same empty 404 at /admin as a stranger, and an invite is one '
          'account that never names who issued it'
          + ('' if armed else
             ' (the dashboard.py wiring checks are asleep until it imports '
             'auth)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
