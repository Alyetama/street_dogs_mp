#!/usr/bin/env python3
"""
The gate. Who is holding this browser, and may they see the harvest.

    import auth
    auth.bootstrap()                      # once, in serve()

    req = auth.Request.from_handler(self) # at the top of do_GET / do_POST
    reply = auth.serve_request(req)
    if reply is not None:
        if reply.status == 404 and not reply.body:
            self.send_error(404)          # byte-identical to any dead address
        else:
            reply.send(self)
        return
    ...                                   # req.session is the signed-in user

and add 'auth.py' and 'accounts.py' to serve()'s _watched set, or an edit to
either one sits invisible behind a healthy-looking server.

THIS MODULE IS THE GATE, NOT THE STORE. Every password, hash, invite and
lockout counter lives in accounts.py; nothing here writes SQL and nothing here
decides what a valid password is. What is here is the part that only makes
sense in front of a socket: a cookie that survives a restart, a form that
takes a password out of a POST body, three pages that render before anybody
is trusted, and one function the router asks "may this request continue".

NO SESSION TABLE. The dashboard re-execs itself whenever one of its source
files changes (serve()'s _reexec_if_stale), which happens while somebody is
editing -- so a table of live sessions in memory would sign everyone out
several times an afternoon, and a table on disk would be one more thing to
prune, back up and get wrong. A session is therefore a signed cookie: the
server keeps no record of it and verifies it from scratch on every request.

WHICH MEANS REVOCATION NEEDS AN ANSWER, and it has one. The payload carries
the account's session_epoch and every request compares it against the users
row; accounts.py bumps that number on set_password, set_active(False),
bump_session_epoch and an .env password change, and do_logout below bumps it
too. One UPDATE ends every live session for that account, everywhere, with
nothing to walk. Signing out is in that list on purpose: it was the one
revoking control that revoked nothing, which mattered because it is the one
a person reaches for when they think somebody has read their cookie.

THE COOKIE IS Secure WHEN THE SITE IS. Set unconditionally it would break a
plain-HTTP deployment outright -- the browser withholds a Secure cookie from a
non-HTTPS origin, so the login would appear to succeed and bounce straight
back to the form -- and left off it hands the session to anybody who can read
the wire. So it follows the deployment: DASHBOARD_HTTPS=1 in
the environment and the attribute goes on. Unset, it stays off, which is the
old behaviour and the right one on a tailnet.

The environment, not X-Forwarded-Proto: this module reads no forwarded header
anywhere, because nothing here can tell a proxy's word from a stranger's --
the same reasoning that keeps the login throttle on client_address. HttpOnly
and SameSite=Lax are always set, because they cost nothing and close the two
holes that do not need the wire.

FAIL CLOSED ON DATA, FAIL OPEN ON UPTIME. With no usable admin -- a fresh
clone, or a DASHBOARD_PASSWORD nobody has set -- this gate serves the login
page carrying accounts.ensure_admin()'s explanation and NOTHING else: no
page, no image, no /api answer. It does not refuse to start. The source
watcher re-execs this process unattended, so a build that exited on a missing
variable would take the dashboard off the air with nobody watching, and the
dashboard being down is not more secure than the dashboard asking for a
password.
"""

import base64
import binascii
import hashlib
import hmac
import html
import json
import ipaddress
import os
import re
import secrets
import sys
import time
from urllib.parse import parse_qs, quote

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    # dashboard.py is run as a script so its own directory is already on the
    # path, but the guard imports this module by file location and a plain
    # `import accounts` would fail there. Inserting the sibling directory is
    # what audit.py does for fn_audit, for the same reason.
    sys.path.insert(0, _HERE)
import accounts                                               # noqa: E402
import work_strip                                             # noqa: E402

REPO = os.path.dirname(os.path.dirname(_HERE))
OUT = os.path.join(REPO, 'data', 'dashboard')

# The HMAC key, when DASHBOARD_SECRET is not set. It sits beside the accounts
# database in the static handler's document root, which serves an ALLOW-list
# (dashboard.py's _static_allowed) matched against the name that will actually
# be opened rather than the one the client typed -- a prefix test on the typed
# path let a member fetch /recent_crops/../session.key and mint an admin
# cookie with it. This name is exported so the guard can fail if a future
# entry ever widens that list, the same protection accounts.PRIVATE_FILES
# gives the database.
KEY_NAME = 'session.key'
KEY_PATH = os.path.join(OUT, KEY_NAME)
PRIVATE_FILES = frozenset({KEY_NAME})

COOKIE = 'dash_session'
# Version prefix, inside the signature. It is what lets the payload's shape
# change without a token minted under the old shape being read as the new
# one -- an old cookie fails verification and its holder signs in again.
SESSION_VERSION = 's1'
# Domain separation. The session signature and the CSRF token are both an
# HMAC under the same secret, and without distinct labels a value produced
# for one is a valid value for the other.
SIGN_LABEL = b'dashboard-session-v1'
CSRF_LABEL = b'dashboard-csrf-v1'

ENV_SECRET = 'DASHBOARD_SECRET'
ENV_SESSION_HOURS = 'DASHBOARD_SESSION_HOURS'
# 16 characters of anything is 128 bits at worst and far more in practice.
# Below that the key is guessable and a guessable key is a forged cookie for
# any account, so a short one is refused rather than silently accepted.
SECRET_MIN = 16
SESSION_TTL_DEFAULT = 7 * 24 * 3600
SESSION_TTL_MIN, SESSION_TTL_MAX = 300, 90 * 24 * 3600

# A cookie a browser sends is at most 4KB; anything longer is not a session
# this server minted and is not worth an HMAC. Same for a form body: the
# three forms here carry a username, a password and a note, and an
# unauthenticated POST that makes the server read a gigabyte is a denial of
# service with no password required.
MAX_COOKIE = 4096
MAX_BODY = 64 * 1024

_DUE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

LOGIN_PATH = '/login'
LOGOUT_PATH = '/logout'
SIGNUP_PATH = '/signup'
ADMIN_PATH = '/admin'
ACCOUNT_PATH = '/account'
CSRF_FIELD = 'csrf'
TOKEN_FIELD = 't'
NEXT_FIELD = 'next'

# What an unauthenticated request may reach. An ALLOW-list, so a route added
# to dashboard.py tomorrow is behind the gate by default: the failure mode of
# a deny-list is a new page that nobody remembered to protect, and this server
# grows a new page most weeks.
PUBLIC_PATHS = frozenset({LOGIN_PATH, SIGNUP_PATH, LOGOUT_PATH})

# One message per outcome, redirected through the URL so a POST that changed
# something can answer with a redirect instead of a re-postable page. Codes,
# not sentences: a sentence in a query string is a sentence an attacker can
# put on your admin page.
NOTICES = {
    'invite_revoked': 'That invite was withdrawn.',
    'invite_expiry': 'That invite link now runs out at the time shown in the '
                     'table. The link itself is unchanged \u2014 whoever '
                     'holds it can still use it.',
    'password_changed': 'Your password is changed. Every other device signed '
                        'in as you has been signed out.',
    'assigned': 'That work is delegated. They will see it the next time they '
                'open a judging page.',
    'assign_cancelled': 'That target is called off. What was judged towards '
                        'it stays judged.',
    'assign_deleted': 'That target is deleted. Every annotation made towards '
                      'it is untouched \u2014 verdicts live in the ledgers, '
                      'not in this record.',
    'invite_deleted': 'That invite is off the list. Anyone who already used '
                      'it still has their account.',
    'user_disabled': 'That account is disabled and its sessions are over.',
    'user_deleted': 'That account is gone and cannot sign in again. Every '
                    'crop they judged stays judged, under their name \u2014 '
                    'verdicts live in the ledgers, not in this record.',
    'user_enabled': 'That account can sign in again.',
    'role_admin': 'That account is an admin now.',
    'role_member': 'That account is a member now.',
    'signed_out': 'You are signed out, on this device and on every other.',
    # Told apart from the one above on purpose: the cookie went either way,
    # but only one of them can promise the other devices went with it.
    'signed_out_here': 'You are signed out of this browser. The accounts '
                       'database could not be reached, so any other device '
                       'signed in as you may still be.',
    'session_over': 'That session has ended. Sign in again.',
}


# ── the signing key ─────────────────────────────────────────────────────────

# Read once and kept: the key file is opened on the first request and never
# again, because a session check runs on every image on the page and a file
# read per image is a syscall storm for a value that cannot change without a
# restart. Keyed by path so a test can hold its own without disturbing the
# server's.
_KEYS = {}


def secret(key_path=None):
    """The HMAC key. DASHBOARD_SECRET if usable, else a file, made once.

    A GENERATED KEY HAS TO PERSIST. The process re-execs itself when a source
    file changes, and a key held only in memory would sign every open session
    out on every edit -- which is exactly the annoyance the cookie design
    exists to avoid. So it is written once, 0600, beside the accounts
    database, and every later start reads it back.

    A key set in the environment WINS, because that is how you rotate: change
    the variable, restart, and every cookie in the world stops verifying. One
    that is too short is refused with a line naming the variable and never the
    value -- .env holds a hundred API keys and a "using DASHBOARD_SECRET=..."
    is how one of them ends up in a journal.
    """
    p = key_path or KEY_PATH
    got = _KEYS.get(p)
    if got is not None:
        return got
    raw = (os.environ.get(ENV_SECRET) or '').strip()
    if raw and len(raw) < SECRET_MIN:
        print(f'{ENV_SECRET} is shorter than {SECRET_MIN} characters and will '
              f'not be used; falling back to {KEY_NAME}', file=sys.stderr)
        raw = ''
    if raw:
        key = raw.encode('utf-8')
        _KEYS[p] = key
        return key
    key = _read_key(p) or _make_key(p)
    if key is None:
        # A filesystem that will not hold the key still gets a working login;
        # what it loses is sessions surviving a restart, which is a nuisance
        # and not an outage. Refusing to serve here would be the crash loop
        # this whole module is written to avoid.
        key = secrets.token_bytes(32)
        print(f'{KEY_NAME} could not be written; sessions will not survive a '
              f'restart', file=sys.stderr)
    _KEYS[p] = key
    return key


def _read_key(p):
    """The stored key, or None if there is not a usable one."""
    try:
        with open(p, 'rb') as fh:
            raw = fh.read(MAX_COOKIE).strip()
    except OSError:
        return None
    try:
        key = binascii.unhexlify(raw)
    except (binascii.Error, ValueError):
        return None
    return key if len(key) >= 32 else None


def _make_key(p):
    """Write a new key, 0600, without a window where it is not.

    O_EXCL is not only about the mode: two threads can reach this at once on
    the first two requests after a fresh install, and the loser must read the
    winner's key rather than overwrite it -- a second key would invalidate the
    session the first one had just minted.
    """
    d = os.path.dirname(p)
    if d:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            return None
    key = secrets.token_bytes(32)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return _read_key(p)
    except OSError:
        return None
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(binascii.hexlify(key) + b'\n')
    except OSError:
        return None
    try:
        os.chmod(p, 0o600)      # umask is 002 here; O_EXCL already set it,
    except OSError:             # this is the belt for a pre-existing file
        pass
    return key


def _mac(label, msg, key_path=None):
    """HMAC-SHA256 under a purpose-separated subkey."""
    sub = hmac.new(secret(key_path), label, hashlib.sha256).digest()
    return hmac.new(sub, msg, hashlib.sha256).digest()


# ── sessions ────────────────────────────────────────────────────────────────

def _b64(raw):
    return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')


def _unb64(text):
    return base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))


def session_ttl(env=None):
    """How long a session lasts, in seconds. DASHBOARD_SESSION_HOURS, or a week.

    A week rather than a day because the phone in the field is the client that
    matters, and a login screen in the middle of a review queue is how a queue
    stops being reviewed. The cost of a long session is bounded by the fact
    that any of them can be ended immediately -- accounts.bump_session_epoch()
    and the per-request epoch check are what make a long expiry affordable.
    """
    src = os.environ if env is None else env
    try:
        ttl = int(float(str(src.get(ENV_SESSION_HOURS, '')).strip()) * 3600)
    except (TypeError, ValueError):
        return SESSION_TTL_DEFAULT
    if ttl < SESSION_TTL_MIN or ttl > SESSION_TTL_MAX:
        return SESSION_TTL_DEFAULT
    return ttl


def mint(user, now=None, ttl=None, key_path=None):
    """(cookie value, max age) for one signed-in account.

    THE SIGNATURE COVERS THE ENCODED PAYLOAD, NOT A JOINED STRING OF FIELDS.
    The obvious version -- sign 'id|name|role|exp' -- is forgeable by anyone
    whose username may contain the delimiter: sign in as `bob|admin` and the
    verifier splits your name into a role. Here the fields are JSON, the JSON
    is base64url, and base64url's alphabet contains neither the '.' that
    separates the three parts nor anything else the parser looks at. No field
    value can end its own field.

    The payload is signed, not encrypted. Whoever holds the cookie can read
    their own id, name and role out of it, which they already know; there is
    nothing in it they do not.
    """
    ts = int(time.time() if now is None else now)
    ttl = int(session_ttl() if ttl is None else ttl)
    payload = {
        'uid': int(user['id']),
        'name': str(user['username']),
        'role': str(user['role']),
        'epoch': int(user['session_epoch']),
        'iat': ts,
        'exp': ts + ttl,
        # A nonce makes two sessions for one account in one second different
        # tokens, and gives the CSRF token below something session-specific
        # to hang off. Without it, two browsers signed into one account would
        # share a CSRF token and signing out of one would not invalidate the
        # other's forms.
        'nonce': _b64(secrets.token_bytes(12)),
    }
    body = SESSION_VERSION + '.' + _b64(
        json.dumps(payload, sort_keys=True, separators=(',', ':'))
        .encode('utf-8'))
    return body + '.' + _b64(_mac(SIGN_LABEL, body.encode('ascii'),
                                  key_path)), ttl


def read_session(value, now=None, key_path=None):
    """The payload of a cookie this server signed and that has not expired.

    None for anything else, with no distinction between a forged signature, a
    truncated cookie and a stale one: they are all "sign in again", and the
    only party who benefits from knowing which is the party holding a cookie
    they made up.

    EXPIRY IS READ OUT OF THE SIGNED PAYLOAD. The cookie also carries a
    Max-Age, and a browser is expected to drop it then -- but Max-Age is a
    request the client is free to ignore, and a stored cookie replayed a year
    later still arrives looking exactly like a fresh one. The signed 'exp' is
    the one the server cannot be lied to about.
    """
    ts = int(time.time() if now is None else now)
    if not value or len(value) > MAX_COOKIE:
        return None
    parts = value.split('.')
    if len(parts) != 3 or parts[0] != SESSION_VERSION:
        return None
    body = parts[0] + '.' + parts[1]
    try:
        given = _unb64(parts[2])
    except (binascii.Error, ValueError):
        return None
    want = _mac(SIGN_LABEL, body.encode('ascii'), key_path)
    # compare_digest, not ==: a byte-at-a-time comparison leaks how much of a
    # forged signature was right, and a signature is exactly the value an
    # attacker gets to retry a million times.
    if not hmac.compare_digest(given, want):
        return None
    try:
        payload = json.loads(_unb64(parts[1]).decode('utf-8'))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        if int(payload['exp']) <= ts or int(payload['iat']) > ts + 300:
            # A token issued in the future is a clock that moved, and honouring
            # it would let one stand for its whole lifetime plus the drift.
            # Five minutes of slack, because the drift is usually ntp settling.
            return None
        int(payload['uid']), int(payload['epoch'])
        str(payload['nonce'])
    except (KeyError, TypeError, ValueError):
        return None
    return payload


def resolve(value, now=None, path=None, key_path=None):
    """The account behind a cookie, checked against the database. None if not.

    THE ROW IS THE AUTHORITY, NOT THE PAYLOAD. The signed role is what the
    account was when it signed in; the role that decides whether the admin
    page opens is the one in the users table right now. accounts.set_role()
    does not bump session_epoch -- deliberately, since a demotion is not a
    reason to sign somebody out of the review queue -- so reading the row is
    the only thing that makes a demotion take effect before the cookie
    expires.

    One connect() per request. Measured at 0.145ms against the ~50ms a
    password costs, on a page that loads two dozen images.

    AND IT IS NOT CACHED, DELIBERATELY. The obvious saving here is to hold
    the row for a second or two in front of accounts.get_user, and it is the
    wrong trade: this lookup IS the revocation mechanism. There is no session
    table, so a disable, a password change, a demotion and a sign-out all
    take effect by this function reading the row again -- cache it for a
    second and every one of them has a second in which the account that was
    just shut off is still being served. The numbers, so nobody has to
    re-measure: read_session alone 0.005ms, resolve 0.157ms, connect+close
    0.113ms of it, and 200 threads resolving at once finish in 0.035s. A
    whole GET / is 1.6ms and a browser opens six connections to one origin.
    """
    payload = read_session(value, now=now, key_path=key_path)
    if payload is None:
        return None
    try:
        row = accounts.get_user(int(payload['uid']),
                                path=path or _state()['db'])
    except Exception:                    # noqa: BLE001 - see below
        # A store that has stopped answering -- an unmounted drive, a
        # permission that changed under the running process -- makes everybody
        # a stranger and sends them to the login page. It does not make every
        # request a traceback and a 500, on a process that re-execs itself
        # unattended.
        return None
    if row is None or not row['active']:
        return None
    if int(row['session_epoch']) != int(payload['epoch']):
        # The whole revocation mechanism, in one comparison: a password
        # change, a disable or an explicit sign-out moved the number and
        # every cookie minted before it stops verifying here.
        return None
    # BEING HERE IS WHAT "LAST SEEN" MEANS. A session lasts a week, and an
    # account that joined by redeeming an invite never calls verify_password
    # at all -- so a column stamped only at password time read NEVER for an
    # annotator who had judged crops every day since they signed up. The
    # write throttles itself inside the store (SEEN_THROTTLE) and never
    # raises, so the cost on this path -- which runs on every request -- is
    # one indexed read almost always, one small write occasionally.
    accounts.touch_seen(row['id'], now=now, path=path or _state()['db'])
    ses = dict(row)
    ses['nonce'] = str(payload['nonce'])
    ses['iat'] = int(payload['iat'])
    ses['exp'] = int(payload['exp'])
    ses['csrf'] = csrf_for(ses, key_path=key_path)
    return ses


def csrf_for(session, key_path=None):
    """The CSRF token for one session. Derived, never stored.

    SameSite=Lax already stops the cross-site POST this defends against, in
    every browser that implements it -- but Lax is a promise made by the
    client, and the server has no way to tell whether the client kept it. The
    token is the server's own check: it is an HMAC over the session's nonce,
    so it cannot be guessed, it is different for every session, and it can
    only be read by someone who can read a page rendered for that session --
    which an attacker on another origin cannot do, because the cookie is
    HttpOnly and nothing here sends CORS headers.

    What it is NOT is a defence against a stolen cookie. Whoever holds the
    cookie can fetch the admin page and read the token out of it. Cookie theft
    is what HttpOnly and the epoch check are for.
    """
    msg = ('%s|%s' % (session.get('nonce', ''), session.get('id', ''))).encode()
    return _b64(_mac(CSRF_LABEL, msg, key_path))


def csrf_ok(session, given, key_path=None):
    """Constant-time check of a form's CSRF field against its session."""
    if not session or not isinstance(given, str) or not given:
        return False
    return hmac.compare_digest(given, csrf_for(session, key_path=key_path))


# ── cookies ─────────────────────────────────────────────────────────────────

def parse_cookie(header):
    """{name: value} from a Cookie header, without http.cookies.

    SimpleCookie parses this and also unquotes, decodes and silently drops a
    whole header when one pair in it is malformed -- which for a shared host
    means an unrelated cookie set by something else on the same address can
    take the session cookie down with it. This reads the pairs and nothing
    else.
    """
    out = {}
    for piece in (header or '')[:MAX_COOKIE * 2].split(';'):
        k, sep, v = piece.partition('=')
        if not sep:
            continue
        k = k.strip()
        if k and k not in out:
            out[k] = v.strip().strip('"')
    return out


def over_https(req=None, env=None):
    """Is THIS REQUEST served over TLS?

    THE OPERATOR SAYS SO, NOT THE REQUEST. A reverse proxy terminates the TLS
    and what reaches this process is plain HTTP either way, so the obvious
    signal is X-Forwarded-Proto -- and this module reads no forwarded header
    at all, deliberately: nothing here can tell a proxy's word from a
    stranger's, and the same reasoning that keeps the login throttle on
    client_address keeps this on the environment. One variable, set once,
    where the deployment is described: DASHBOARD_HTTPS=1.

    Absent, no -- which is exactly how a tailnet deployment behaved before.

    WHAT THE VARIABLE MEANS IS "THE PROXY'S LEG IS TLS", and this process has
    two legs. The tunnel dials the origin over plain HTTP, and the same origin
    port is what a tailnet browser types in directly -- so a Secure flag on
    every reply locks that second browser out: it stores no Secure cookie from
    an http:// origin, sends nothing back, and the login form returns forever
    with no error to read. Which leg a request came in on is not a guess: a
    request through the proxy arrives from a peer the operator named AND
    carrying the client address that proxy appended. Sending that header
    yourself only makes your OWN cookie Secure, so there is nothing to gain by
    it; a plain tailnet request keeps exactly the cookie it had before there
    was a tunnel.
    """
    got = (env if env is not None else os.environ).get('DASHBOARD_HTTPS')
    if str(got or '').strip().lower() not in ('1', 'true', 'yes', 'on'):
        return False
    # AND WHEN NOBODY NAMED A PROXY, THE OPERATOR'S WORD IS ALL THERE IS.
    # DASHBOARD_TRUSTED_PROXY is what tells the two legs apart; a deployment
    # that terminates TLS somewhere this process cannot see, and never named
    # it, has no second leg to protect and every reply must carry the flag --
    # which is what it did before there were legs. Splitting them on a guess
    # would take Secure off a public HTTPS deployment whose only mistake was
    # leaving one variable unset.
    if req is None or not trusted_proxies(env):
        return True
    return bool(getattr(req, 'proxied', True))


def _flags(req=None, env=None):
    """The attributes every cookie this module sets carries."""
    return 'Path=/; HttpOnly; SameSite=Lax' + (
        '; Secure' if over_https(req, env) else '')


def set_cookie(value, max_age, req=None, env=None):
    """The Set-Cookie header for a fresh session.

    HttpOnly: the cookie is not readable from JavaScript, so an injected
    script on any page of this dashboard cannot walk off with a session.
    SameSite=Lax: a cross-site POST does not carry it, which is what stops a
    page on another origin from submitting this one's forms. Path=/: the gate
    covers every route, so the cookie has to reach every route. Secure when
    the deployment is -- see the module docstring.
    """
    return ('Set-Cookie',
            '%s=%s; Max-Age=%d; %s'
            % (COOKIE, value, int(max_age), _flags(req, env)))


def clear_cookie(req=None, env=None):
    """Expire the cookie. Max-Age=0 and an empty value, so a client that
    ignores one honours the other.

    Carries the SAME attributes as the cookie it is expiring: a browser
    matches them, and a Secure cookie is not cleared by a non-Secure header.
    """
    return ('Set-Cookie',
            '%s=; Max-Age=0; %s' % (COOKIE, _flags(req, env)))


# ── requests and replies ────────────────────────────────────────────────────

class Request:
    """What the gate needs to know about one HTTP request.

    A plain object rather than the handler itself, so every flow below can be
    exercised without a socket -- the guard drives all of them this way.
    """

    __slots__ = ('method', 'path', 'query', 'form', 'cookies', 'remote',
                 'host', 'oversize', 'session', 'proxied')

    def __init__(self, method='GET', path='/', query=None, form=None,
                 cookies=None, remote='', host='', oversize=False,
                 proxied=True):
        self.method = method
        self.path = path
        self.query = query or {}
        self.form = form or {}
        self.cookies = cookies or {}
        self.remote = remote
        self.host = host
        self.oversize = oversize
        # True unless somebody positively knows otherwise -- from_handler
        # decides it from the socket, and a Request built by hand is not a
        # plaintext leg, it is a test.
        self.proxied = proxied
        self.session = None

    @classmethod
    def from_handler(cls, handler):
        """Build one out of a BaseHTTPRequestHandler.

        THE BODY IS READ ONLY FOR THE ROUTES THIS MODULE OWNS. Reading it for
        every POST would consume the JSON that /api/audit/verdict and every
        other endpoint is about to read for itself, and the review queue would
        start losing verdicts the day the gate went in -- silently, since the
        handler would see an empty body rather than an error.
        """
        raw = handler.path or '/'
        path, _, qs = raw.partition('?')
        form, oversize = {}, False
        if handler.command == 'POST' and owns(path):
            try:
                n = int(handler.headers.get('Content-Length') or 0)
            except (TypeError, ValueError):
                n = 0
            if n > MAX_BODY:
                oversize = True
            elif n > 0:
                ctype = (handler.headers.get('Content-Type') or '').lower()
                # A client that announced a body and then went away raises
                # here, on the read -- and this runs before any route has been
                # chosen, so the traceback would come out of the gate itself.
                # An empty form is the honest reading: the request said
                # nothing, and the login flow refuses it the way it refuses
                # any other empty one.
                try:
                    data = handler.rfile.read(n)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # AND THE CONNECTION IS DONE WITH. socket.timeout IS an
                    # OSError, so a slow sender lands here on a socket that is
                    # still open with its body unread. Nothing reuses one
                    # today -- the handler answers HTTP/1.0 and closes after
                    # every reply -- so this is the belt: the day somebody
                    # sets protocol_version to HTTP/1.1, those unread bytes
                    # would be read as the next request line on it.
                    handler.close_connection = True
                    data = b''
                if ctype.startswith('application/x-www-form-urlencoded'):
                    form = parse_qs(data.decode('utf-8', 'replace'),
                                    keep_blank_values=True)
        return cls(
            method=handler.command,
            path=path,
            query=parse_qs(qs, keep_blank_values=True),
            form=form,
            cookies=parse_cookie(handler.headers.get('Cookie')),
            # WHO IS ASKING, and it is not always the socket. This used to
            # be client_address unconditionally, on the premise that nothing
            # proxies this server -- true on a tailnet, false the day it was
            # published: a tunnel now terminates the TLS and dials the origin
            # from this same box, so every visitor on earth arrives as one
            # address and the login lockout, which is keyed on it, stops
            # measuring the attacker and starts measuring the tunnel. Nine
            # volunteers with one typo each lock the tenth out.
            #
            # The header is read ONLY when the socket peer is a proxy the
            # operator named. From anybody else it is still what it always
            # was: a string the caller chose, worth nothing.
            remote=client_address(handler),
            host=handler.headers.get('Host') or '',
            proxied=arrived_by_proxy(handler),
            oversize=oversize)

    def one(self, name, default=''):
        """The first value of a form field, '' when it is not there."""
        v = self.form.get(name)
        return v[0] if isinstance(v, list) and v else default

    def arg(self, name, default=''):
        """The first value of a query parameter."""
        v = self.query.get(name)
        return v[0] if isinstance(v, list) and v else default


class Reply:
    """A response the router sends verbatim. Status, headers, bytes."""

    __slots__ = ('status', 'body', 'headers')

    def __init__(self, status=200, body=b'', headers=()):
        self.status = status
        self.body = body or b''
        self.headers = list(headers)

    def header(self, name):
        """The first value of one header, or '' -- for the guard's benefit."""
        for k, v in self.headers:
            if k.lower() == name.lower():
                return v
        return ''

    def send(self, handler):
        """Write it. Nothing is logged: BoardHandler.log_message is a no-op
        and must stay one, because a request line on the signup route carries
        the invite token in its query string."""
        # THE GATE ANSWERS EVERY UNAUTHENTICATED REQUEST, which makes it
        # the writer most likely to be talking to somebody who has already
        # gone -- a scanner that opens and drops, a browser that navigates
        # away from the login form. Without this, each one is a traceback in
        # the service log, and a public address supplies them all day.
        try:
            handler.send_response(self.status)
            for k, v in self.headers:
                handler.send_header(k, v)
            handler.send_header('Content-Length', str(len(self.body)))
            handler.end_headers()
            if handler.command != 'HEAD' and self.body:
                handler.wfile.write(self.body)
        except (BrokenPipeError, ConnectionResetError):
            handler.close_connection = True


# Every page this module serves carries these. no-store because a login form,
# an admin page and above all a freshly minted invite link have no business in
# a disk cache; no-referrer because the signup URL contains the token and a
# Referer header is how a URL walks off to somewhere else's log; DENY and
# frame-ancestors because a dashboard whose admin page can be framed is a
# dashboard whose disable buttons can be clicked by a page you were reading.
# The CSP is deliberately total: these pages fetch nothing at all, so
# 'none' plus the two inline sources is the whole policy they need.
SECURITY_HEADERS = (
    ('Cache-Control', 'no-store, max-age=0'),
    ('Referrer-Policy', 'no-referrer'),
    ('X-Content-Type-Options', 'nosniff'),
    ('X-Frame-Options', 'DENY'),
    # connect-src 'self': the admin page DOES fetch. It draws every
    # delegated target with an empty progress cell and fills them from
    # /api/assignments, because this module has no business reading the
    # annotation ledgers. With default-src 'none' and no connect-src, the
    # browser refused that request before it was made -- the endpoint was
    # right, answered correctly when asked directly, and every progress bar
    # on the page sat at "could not count" regardless. Nothing else here
    # fetches, and 'self' is the only origin that would ever be right.
    ('Content-Security-Policy',
     "default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
     "script-src 'unsafe-inline'; connect-src 'self'; form-action 'self'; "
     "base-uri 'none'; frame-ancestors 'none'"),
)


def page_reply(markup, status=200, extra=()):
    """An HTML page with the security headers on it."""
    body = markup.encode('utf-8')
    return Reply(status, body,
                 [('Content-Type', 'text/html; charset=utf-8')]
                 + list(SECURITY_HEADERS) + list(extra))


def json_reply(obj, status=200, extra=()):
    """A JSON answer, for the /api surface -- see guard() on why it is not a
    redirect."""
    body = json.dumps(obj).encode('utf-8')
    return Reply(status, body,
                 [('Content-Type', 'application/json')]
                 + list(SECURITY_HEADERS) + list(extra))


def redirect(location, status=303, extra=()):
    """303 after a POST so a refresh does not repeat it; 302 for a gate bounce."""
    return Reply(status, b'',
                 [('Location', location)] + list(SECURITY_HEADERS)
                 + list(extra))


def not_found():
    """An empty 404.

    The router should hand this to send_error(404) so it is byte-identical to
    every other unknown path: this is what a member gets for /admin, and a
    page that says "admins only" is a page that tells a member the address is
    worth attacking.
    """
    return Reply(404, b'', [])


def safe_next(value):
    """A path this server may redirect to after a login, or '/'.

    An open redirect on a login page is a phishing primitive: /login?next=
    somewhere-else sends somebody who has just typed their password to a page
    of the attacker's choosing, from a link that starts with the address they
    trust. Only a same-site absolute path is accepted, and '//host' is
    rejected explicitly -- a protocol-relative URL starts with a slash and
    goes to another host, and browsers fold a backslash into a slash before
    they parse it, so '/\\evil' is '//evil'.
    """
    v = (value or '').strip()
    if not v.startswith('/'):
        return '/'
    v = v.replace('\\', '/')
    if v.startswith('//') or '://' in v or '\n' in v or '\r' in v:
        return '/'
    if v.startswith(LOGIN_PATH) or v.startswith(LOGOUT_PATH):
        return '/'          # bouncing back to the form is not a destination
    return v


def owns(path):
    """Is this one of the gate's own routes?

    The prefix form matters for /admin, which carries its POST endpoints
    under it, and for nothing else -- /login/x is not a page and falls through
    to the dashboard's own 404.
    """
    return (path in (LOGIN_PATH, LOGOUT_PATH, SIGNUP_PATH, ACCOUNT_PATH)
            or path == ADMIN_PATH or path.startswith(ADMIN_PATH + '/'))


def is_public(path):
    """May an unauthenticated request reach this path at all?"""
    return path in PUBLIC_PATHS


# ── the state of the deployment ─────────────────────────────────────────────

_STATE = {'db': None, 'key': None, 'boot': None}


def _state():
    return _STATE


def bootstrap(db_path=None, key_path=None, env=None, now=None):
    """Read .env, make the .env admin real, and remember how it went.

    Called once from serve(), before the first request. Two things have to
    happen here and nowhere else: accounts.load_env(), because the systemd
    unit passes no Environment= beyond PYTHONUNBUFFERED and nothing else puts
    .env in front of this process; and accounts.ensure_admin(), which is what
    turns DASHBOARD_PASSWORD into a row that the login path can check like
    any other.

    NEVER RAISES. A database that will not open, a value in .env that will
    not parse -- every one of them comes back as a dict whose 'ok' is False,
    and the gate turns that into a page that says what to set. The source
    watcher re-execs this process unattended; an exception here is a crash
    loop nobody is watching.
    """
    _STATE['db'] = db_path or accounts.DB_PATH
    _STATE['key'] = key_path or KEY_PATH
    if env is None:
        accounts.load_env()
    try:
        got = accounts.ensure_admin(path=_STATE['db'], now=now, env=env)
    except Exception as e:               # noqa: BLE001 - uptime beats purity
        got = {'action': 'refused', 'ok': False,
               'username': accounts.admin_username(env), 'user_id': None,
               'admins': 0, 'others': [], 'demoted': [],
               'detail': 'The accounts database could not be opened (%s: %s). '
                         'Check that data/dashboard is writable, then restart.'
                         % (type(e).__name__, e)}
    _STATE['boot'] = got
    return got


def gate_state():
    """The bootstrap result, bootstrapping first if the router forgot to.

    A missing bootstrap() call must not become an unauthenticated dashboard,
    so the lazy path exists and does the same work. It is not the intended
    one: doing it here means the first request pays for a scrypt hash.
    """
    if _STATE['boot'] is None:
        bootstrap(_STATE['db'], _STATE['key'])
    return _STATE['boot']


def usable():
    """Is there an account anybody could sign in with?

    False is the locked state: the login page and its explanation, and no
    data at all. Note that this asks the STORE, not the environment -- an
    admin created by an earlier run whose DASHBOARD_PASSWORD has since been
    tidied out of .env can still sign in, and locking the dashboard over a
    variable that is no longer needed would be an outage of our own making.
    accounts.ensure_admin() reports exactly that case as ok=True with
    action='unset'.
    """
    got = gate_state() or {}
    return bool(got.get('ok'))


# ── the login flow ──────────────────────────────────────────────────────────

def trusted_proxies(env=None):
    """The socket peers whose forwarded client address may be believed.

    Named by the operator, because only they know what is in front of this
    process: DASHBOARD_TRUSTED_PROXY=127.0.0.1,<the address the proxy dials
    from>. Empty by default, which is a deployment with nothing in front and
    the old behaviour exactly.
    """
    raw = (env if env is not None else os.environ).get(
        'DASHBOARD_TRUSTED_PROXY') or ''
    return {p.strip() for p in raw.replace(';', ',').split(',') if p.strip()}


def forwarded_client(head):
    """The client address a trusted proxy appended, '' when it appended none.

    X-Forwarded-For FIRST, and the LAST hop of the LAST header of that name.
    Both parts matter and both were wrong. A client can send its own
    X-Forwarded-For, and a proxy that adds a second header rather than
    extending the first leaves two -- and Message.get() hands back the first
    one, which is the client's. Within a header, every hop but the last came
    from further out; only the last was written by the proxy in front of us.

    CF-Connecting-IP is the fallback rather than the preference. Cloudflare
    sets it at the edge and overwrites what a client sent, so it is good here
    -- but it is only good BECAUSE the proxy is Cloudflare, and nginx or
    caddy in that position passes a client's copy straight through. Reaching
    for it only when there is no X-Forwarded-For at all keeps the trusted
    deployment working and takes the forgeable path off the common one.
    """
    chain = []
    for one in (head.get_all('X-Forwarded-For') or []
                if hasattr(head, 'get_all')
                else [head.get('X-Forwarded-For') or '']):
        chain += [h.strip() for h in (one or '').split(',') if h.strip()]
    if chain:
        return chain[-1]
    return (head.get('CF-Connecting-IP') or '').strip()


def arrived_by_proxy(handler, env=None):
    """Did this request come in through a proxy the operator named?

    Both halves: the socket peer is on the list AND it appended a client
    address. A request straight off the origin port fails the second half
    even when the browser happens to sit on the same box as the tunnel.
    """
    peer = ''
    if getattr(handler, 'client_address', None):
        peer = handler.client_address[0] or ''
    if peer not in trusted_proxies(env):
        return False
    head = getattr(handler, 'headers', None)
    return bool(head is not None and forwarded_client(head))


def client_address(handler, env=None):
    """The address to hold responsible for this request.

    The socket peer, unless the peer is a proxy the operator named -- then the
    address that proxy says it is carrying, which is the client. Cloudflare
    sends CF-Connecting-IP; the general form is the LAST hop of
    X-Forwarded-For, the one the trusted proxy appended itself (the earlier
    ones came from the client and are worth nothing).

    An address that does not parse is refused rather than passed through: a
    throttle key is a string, and a caller who can choose it can choose
    somebody else's.
    """
    peer = ''
    if getattr(handler, 'client_address', None):
        peer = handler.client_address[0] or ''
    if peer not in trusted_proxies(env):
        return peer
    head = getattr(handler, 'headers', None)
    if head is None:
        return peer
    got = forwarded_client(head)
    try:
        return str(ipaddress.ip_address(got)) if got else peer
    except ValueError:
        return peer


def throttle_source(req):
    """The key a failed login is counted under: the client address.

    NOT the username. Counting failures per account hands anybody who knows
    the admin's name a way to lock the admin out of their own dashboard by
    typing a wrong password six times -- a denial of service delivered by the
    security feature. The address is the thing an attacker has to spend to
    keep guessing.

    Which address that is, on a proxied deployment, is client_address()'s
    problem -- and getting it wrong there is what turns this from a lockout on
    one guesser into a lockout on everybody at once.
    """
    return 'ip:' + (req.remote or '?')


def attempt_login(username, password, source, now=None, path=None):
    """Try one username and password. A dict, never an exception.

        {'ok', 'user', 'message', 'retry_after'}

    THE TWO WRONG ANSWERS ARE ONE ANSWER. An unknown username and a bad
    password produce the same sentence, the same status and -- because
    accounts.verify_password() derives a hash on both paths -- the same
    ~50ms. A login form that answers a miss faster than a hit is a user
    directory with a delay.

    A CORRECT PASSWORD DURING A LOCKOUT STILL FAILS, and is not even checked:
    verifying it would burn the 50ms the lockout exists to protect, and would
    make a locked-out attacker's timing tell them when they got it right.

    THE ATTEMPT IS COUNTED BEFORE IT IS CHECKED. Reading the counter, hashing
    for 40ms and writing the failure afterwards is check-then-act, and this
    runs on a thread per request: every attempt that arrived inside that
    window read the same pre-failure counter and was let through, which turned
    a 6-guess budget into 30-37 per burst. accounts.reserve_attempt() does
    the counting and the reading under one write lock instead.
    """
    try:
        return _attempt_login(username, password, source, now, path)
    except Exception as e:               # noqa: BLE001
        # A store that broke while the process was up -- the drive holding
        # data/dashboard went away, most likely. The person at the form gets a
        # sentence naming the fault instead of a 500, and the dashboard stays
        # up for whoever is already signed in.
        return {'ok': False, 'user': None, 'retry_after': 0,
                'message': 'The accounts database could not be read (%s).'
                           % (type(e).__name__,)}


def _attempt_login(username, password, source, now, path):
    """The attempt itself. See attempt_login() for every decision in here."""
    p = path or _state()['db']
    ts = int(time.time() if now is None else now)
    # One write, before anything expensive happens. It both counts this
    # attempt and reports whether the source was already locked when it
    # arrived; a locked source is refused on that reading, never on the one
    # its own increment just produced, or the account could never be checked
    # again once it crossed the line.
    st = accounts.reserve_attempt(source, now=ts, path=p)
    if st['was_locked']:
        # The wait is stated plainly. It is not a secret -- the person who has
        # to wait it out is usually the person who owns the account -- and a
        # lockout that looks like a wrong password sends them round the loop
        # again, extending it.
        return {'ok': False, 'user': None, 'retry_after': st['retry_after'],
                'message': 'Too many attempts from this device. Try again in '
                           '%s.' % (_span(st['retry_after']),)}
    pw = password if isinstance(password, str) else ''
    if len(pw) > accounts.PASSWORD_MAX:
        # Refused before hashing: scrypt hashes whatever it is handed, so an
        # unauthenticated POST with a megabyte in it is a way to make this
        # box do a megabyte of work per request. The length is the one thing
        # about the attempt the sender already knows, so this early exit
        # tells them nothing.
        return {'ok': False, 'user': None, 'retry_after': st['retry_after'],
                'message': _WRONG}
    user = accounts.verify_password(username, pw, now=ts, path=p)
    if user is None:
        return {'ok': False, 'user': None,
                'retry_after': st['retry_after'], 'message': _WRONG}
    accounts.clear_failures(source, path=p)
    return {'ok': True, 'user': user, 'retry_after': 0, 'message': ''}


_WRONG = 'That username and password do not match an account.'


def do_login(req, now=None):
    """GET renders the form; POST checks the body and mints the cookie.

    THE CREDENTIALS COME OUT OF THE BODY. Never out of the query string: a
    URL is written to browser history, to a bookmark, to the Referer of the
    next page and -- on a server that logged its request lines, which this one
    deliberately does not -- to disk. req.one() reads the form and nothing
    else, so ?username=&password= is simply an empty attempt.
    """
    nxt = safe_next(req.arg(NEXT_FIELD) or req.one(NEXT_FIELD))
    if not usable():
        # No account exists to sign in to. The page says which variable to
        # set, in the store's own words.
        return page_reply(login_page(nxt, locked=gate_state()))
    # Codes, looked up here -- never a sentence carried in the query string,
    # which is a sentence anybody can put on somebody else's login page.
    notice = NOTICES.get(req.arg('m'), '')
    if req.method != 'POST':
        return page_reply(login_page(nxt, notice=notice))
    if req.oversize:
        return page_reply(login_page(nxt, error='That form was too large.'),
                          status=413)
    got = attempt_login(req.one('username'), req.one('password'),
                        throttle_source(req), now=now)
    if not got['ok']:
        return page_reply(login_page(nxt, error=got['message'],
                                     username=req.one('username')),
                          status=401)
    value, ttl = mint(got['user'], now=now, key_path=_state()['key'])
    return redirect(nxt, extra=[set_cookie(value, ttl, req)])


def change_password(session, current, new, confirm, now=None, path=None):
    """One password change. A dict, never an exception.

        {'ok', 'user', 'message', 'retry_after'}

    THE CURRENT PASSWORD IS REQUIRED, and that is the whole point of the
    route. This cookie travels in the clear over the tailnet; somebody
    holding a copy can already read everything the owner can, and without
    this they could also take the account away from them -- change the
    password, bump the epoch, and be the only one left signed in. Knowing the
    old password is the one thing a stolen cookie does not carry.

    THROTTLED ON ITS OWN KEY. Guessing the current password from inside a
    session is worth doing, so it is counted -- but counted under the account
    rather than under the login source, or a stolen cookie could be used to
    lock the real owner out of the login form by failing here on purpose.
    """
    try:
        return _change_password(session, current, new, confirm, now, path)
    except accounts.AccountError as e:
        return {'ok': False, 'user': None, 'retry_after': 0,
                'message': e.message}
    except Exception as e:               # noqa: BLE001
        # Same shape as attempt_login's: a store that went away is a sentence
        # naming the fault, not a 500 in the middle of somebody's password.
        return {'ok': False, 'user': None, 'retry_after': 0,
                'message': 'The accounts database could not be read (%s).'
                           % (type(e).__name__,)}


def _change_password(session, current, new, confirm, now, path):
    """The change itself. See change_password() for every decision in here."""
    p = path or _state()['db']
    ts = int(time.time() if now is None else now)
    if not session or not session.get('username'):
        return {'ok': False, 'user': None, 'retry_after': 0,
                'message': 'Sign in first.'}
    who = session['username']
    # Checked before the store is touched, so a typo in the confirmation
    # costs nothing and does not count as a failed guess.
    if new != confirm:
        return {'ok': False, 'user': None, 'retry_after': 0,
                'message': 'Those two passwords are not the same.'}
    if new == current:
        return {'ok': False, 'user': None, 'retry_after': 0,
                'message': 'That is the password you already have.'}
    # the account's own name goes in: a password with your username in it is
    # one a guesser already half knows
    accounts.check_password(new, username=who)   # raises with the reason
    source = 'pw:' + accounts.normalise_username(who)
    st = accounts.reserve_attempt(source, now=ts, path=p)
    if st['was_locked']:
        return {'ok': False, 'user': None, 'retry_after': st['retry_after'],
                'message': 'Too many wrong passwords. Try again in %s.'
                           % (_span(st['retry_after']),)}
    # touch=False: this is not a login, and stamping last_login_at here would
    # make "last seen" mean "last changed their password" for anybody who did.
    user = accounts.verify_password(who, current, touch=False, path=p)
    if user is None:
        return {'ok': False, 'user': None, 'retry_after': 0,
                'message': 'That is not your current password.'}
    accounts.clear_failures(source, path=p)
    return {'ok': True, 'retry_after': 0, 'message': '',
            'user': accounts.set_password(who, new, now=ts, path=p)}


def do_account(req, now=None):
    """Your own account: the one thing you can change about it.

    A member had no way to change their password at all. It could only be
    done from a terminal, by an admin, with dashboard.py passwd -- so the
    answer to "somebody watched me type it" was to go and find the person who
    runs the machine.
    """
    session = req.session
    if session is None:
        return redirect(LOGIN_PATH, status=302)
    if req.method != 'POST':
        return page_reply(account_page(
            session, notice=NOTICES.get(req.arg('m'), '')))
    if req.oversize:
        return page_reply(account_page(session,
                                       error='That form was too large.'),
                          status=413)
    got = change_password(session, req.one('current'), req.one('password'),
                          req.one('confirm'), now=now)
    if not got['ok']:
        return page_reply(account_page(session, error=got['message']),
                          status=401 if got['retry_after'] else 400)
    # THE EPOCH JUST MOVED, which is the point -- every other device is
    # signed out. This browser has to be handed a cookie minted under the new
    # epoch or the person who just changed their password is bounced to the
    # login form by their own success, which reads exactly like a failure.
    value, ttl = mint(got['user'], now=now, key_path=_state()['key'])
    return redirect(ACCOUNT_PATH + '?m=password_changed',
                    extra=[set_cookie(value, ttl, req)])


def do_logout(req):
    """End the session on the server, not just in this browser.

    SIGNING OUT USED TO CHANGE NOTHING HERE. It sent
    Set-Cookie: dash_session=; Max-Age=0 and stopped, so the browser forgot
    its copy while the cookie itself stayed valid for the rest of its signed
    life -- seven days by default. This module says plainly at the top that
    the cookie travels in the clear and offers the epoch as the answer to
    that; the epoch is moved by set_password, set_active(False),
    bump_session_epoch and an .env password change, by every revoking action
    EXCEPT the one a worried person actually reaches for. Somebody who
    thought their cookie had been read off the wire clicked "sign out",
    watched it apparently work, and left the captured cookie live. Their
    only real remedy was to change their password.

    So it bumps the epoch, which ends every session for that account on every
    device -- and the control says so rather than pretending otherwise.

    WHICH IS WHY IT IS A POST WITH A TOKEN. As a GET, that same bump was one
    <img src="/logout"> on any page an annotator happened to visit, signing
    them out of their phone and their laptop from across the internet, over
    and over. The bookmark the old GET served is still served: a GET renders
    the button instead of acting on it, which is the shape a state change is
    supposed to have anyway.
    """
    if req.session is None:
        # Nothing to revoke -- no cookie, or one that no longer resolves.
        # There is no session to draw a token from and nothing a confirmation
        # would confirm, so the cookie goes and the form says so.
        return redirect(LOGIN_PATH + '?m=signed_out', status=302,
                        extra=[clear_cookie(req)])
    if req.method != 'POST':
        return page_reply(logout_page(req.session))
    if not csrf_ok(req.session, req.one(CSRF_FIELD), key_path=_state()['key']):
        # A form drawn for a session that has since been replaced. Ask again
        # rather than act on a token this session did not issue.
        return page_reply(logout_page(req.session,
                                      error='That form had expired — '
                                            'here it is again.'), status=400)
    try:
        accounts.bump_session_epoch(req.session['id'], path=_state()['db'])
    except Exception:                    # noqa: BLE001
        # The store stopped answering mid-click. This browser is still out,
        # because the cookie goes either way -- what cannot be promised is
        # the other devices, and saying "signed out everywhere" here would be
        # the same lie in a new place.
        return redirect(LOGIN_PATH + '?m=signed_out_here', status=303,
                        extra=[clear_cookie(req)])
    return redirect(LOGIN_PATH + '?m=signed_out', status=303,
                    extra=[clear_cookie(req)])


# ── the signup flow ─────────────────────────────────────────────────────────

def peek_invite(token, now=None, path=None):
    """What state an invite link is in, without spending it.

        {'state': 'open'|'used'|'revoked'|'expired'|'unknown', 'role', ...}

    WHAT IS NOT IN HERE IS THE POINT. No created_by, no note, no id. The
    person on this page is holding a link and nothing else -- they may be its
    intended holder, or they may be whoever the intended holder forwarded it
    to -- and "issued by alice, note: replacement for bob" is an org chart
    handed to an unauthenticated stranger.

    It reaches accounts._token_hash rather than hashing the token here on
    purpose: a second spelling of how a token is stored is a second thing to
    change, and the day accounts.py changes it, a private copy would go on
    cheerfully validating links that no longer redeem. The guard checks the
    two agree.
    """
    ts = int(time.time() if now is None else now)
    out = {'state': 'unknown', 'role': 'member', 'expires_at': 0}
    if not token or not isinstance(token, str) or len(token) > 256:
        return out
    try:
        con = accounts.connect(path or _state()['db'])
    except Exception:                    # noqa: BLE001 - see attempt_login
        return out                       # unknown, which draws no form
    try:
        row = con.execute('SELECT * FROM invites WHERE token_hash = ?',
                          (accounts._token_hash(token),)).fetchone()
        if row is None:
            return out
        out['state'] = accounts.invite_state(row, ts)
        out['role'] = row['role']
        out['expires_at'] = int(row['expires_at'])
        return out
    finally:
        con.close()


INVITE_WORDS = {
    'unknown': 'That invite link is not valid. Ask for a new one.',
    'used': 'That invite link has already been used. If the account is '
            'yours, sign in instead.',
    'revoked': 'That invite link was withdrawn. Ask for a new one.',
    'expired': 'That invite link has expired. Ask for a new one.',
}


def attempt_signup(token, username, password, confirm, source, now=None,
                   path=None):
    """Redeem an invite into an account. A dict, never an exception.

        {'ok', 'user', 'message'}

    The invite is claimed and the account created in ONE transaction inside
    accounts.redeem_invite(), so two people opening the same link in the same
    second produce one account, and a typo in the username does not spend the
    link.

    A GUESSED TOKEN COUNTS AGAINST THE SOURCE. Not a short password, not a
    taken username -- those are a real holder getting it wrong. A token that
    matches nothing is somebody trying links, and it is the only unauthenticated
    path here that touches the database.
    """
    try:
        return _attempt_signup(token, username, password, confirm, source,
                               now, path)
    except Exception as e:               # noqa: BLE001 - see attempt_login
        return {'ok': False, 'user': None,
                'message': 'The accounts database could not be read (%s).'
                           % (type(e).__name__,)}


def _attempt_signup(token, username, password, confirm, source, now, path):
    """The redemption itself. See attempt_signup() for the reasoning."""
    p = path or _state()['db']
    ts = int(time.time() if now is None else now)
    st = accounts.throttle_state(source, now=ts, path=p)
    if st['locked']:
        return {'ok': False, 'user': None,
                'message': 'Too many attempts from this device. Try again in '
                           '%s.' % (_span(st['retry_after']),)}
    if (password or '') != (confirm or ''):
        return {'ok': False, 'user': None,
                'message': 'The two passwords did not match.'}
    try:
        user = accounts.redeem_invite(token, username, password, now=ts,
                                      path=p)
    except accounts.AccountError as e:
        if e.code == 'invite_unknown':
            accounts.record_failure(source, now=ts, path=p)
        msg = (INVITE_WORDS.get(e.code[7:], e.message)
               if e.code.startswith('invite_') else e.message)
        return {'ok': False, 'user': None, 'message': msg}
    accounts.clear_failures(source, path=p)
    return {'ok': True, 'user': user, 'message': ''}


def do_signup(req, now=None):
    """GET validates the link and renders the form; POST creates the account.

    THE TOKEN IS IN THE URL, AND THAT IS A REAL COST. A link is the only way
    to hand somebody an invite without giving them a password over the same
    channel, and a link carries its token through the address bar: into
    browser history, into whatever a phone syncs, and into the Referer header
    of the next page the browser loads. What is done about it here:
    Referrer-Policy: no-referrer on this page, so the token does not leave in
    a header; no-store, so it does not sit in a disk cache; a default life of
    48 hours; single use, enforced by a compare-and-set in the database; and
    request lines are not logged, so it never reaches serve.log. What remains
    is the browser history of the person who opened it and any proxy on the
    path -- which on a tailnet is nobody, and off a tailnet this link should
    not be sent at all.
    """
    token = req.one(TOKEN_FIELD) or req.arg(TOKEN_FIELD)
    if not usable():
        return page_reply(login_page('/', locked=gate_state()))
    if req.method != 'POST':
        st = peek_invite(token, now=now)
        return page_reply(signup_page(token, st))
    if req.oversize:
        st = peek_invite(token, now=now)
        return page_reply(signup_page(token, st,
                                      error='That form was too large.'),
                          status=413)
    got = attempt_signup(token, req.one('username'), req.one('password'),
                         req.one('confirm'), throttle_source(req), now=now)
    if not got['ok']:
        st = peek_invite(token, now=now)
        return page_reply(signup_page(token, st, error=got['message'],
                                      username=req.one('username')),
                          status=400)
    # Straight in, no second login. The invite was the proof of who they are
    # and they have just chosen the password; making them type it again is a
    # form for the sake of a form.
    value, ttl = mint(got['user'], now=now, key_path=_state()['key'])
    return redirect('/', extra=[set_cookie(value, ttl, req)])


# ── the admin flow ──────────────────────────────────────────────────────────

def admin_action(action, req, session, now=None, path=None):
    """One state-changing admin request. A dict, never an exception.

        {'ok', 'notice', 'message', 'invite'}

    'invite' is set only by the mint action and carries the plaintext token
    exactly once -- see do_admin_post() on why that answer is a page and not
    a redirect.

    Every branch re-checks the actor's role from the session the gate
    resolved, which came from the users row. A member who has learned these
    endpoint names cannot reach this function at all, and if a future route
    ever calls it directly, it still refuses.
    """
    p = path or _state()['db']
    ts = int(time.time() if now is None else now)
    if not session or not accounts.is_admin(session.get('role')) \
            or not session.get(
            'active'):
        return {'ok': False, 'notice': '', 'invite': None,
                'message': 'Only an admin can do that.'}
    try:
        if action == 'invite':
            hours = req.one('hours')
            try:
                ttl = int(float(hours) * 3600) if hours.strip() else None
            except ValueError:
                return {'ok': False, 'notice': '', 'invite': None,
                        'message': 'That is not a number of hours.'}
            role = req.one('role') or 'member'
            inv = accounts.create_invite(session['id'], ttl=ttl,
                                         note=req.one('note'), role=role,
                                         now=ts, path=p)
            return {'ok': True, 'notice': '', 'invite': inv, 'message': ''}
        if action == 'revoke':
            accounts.revoke_invite(_int(req.one('id')), now=ts, path=p)
            return {'ok': True, 'notice': 'invite_revoked', 'invite': None,
                    'message': ''}
        if action == 'invite-expiry':
            # HOURS FROM NOW, the same unit the link was minted in, and read
            # the same way -- a link that ran out last week has no window left
            # to add to, so the clock starts again rather than continuing.
            hours = req.one('hours').strip()
            try:
                ttl = float(hours) * 3600
            except ValueError:
                return {'ok': False, 'notice': '', 'invite': None,
                        'message': 'That is not a number of hours.'}
            accounts.set_invite_expiry(_int(req.one('id')), ttl=ttl, now=ts,
                                       path=p)
            return {'ok': True, 'notice': 'invite_expiry', 'invite': None,
                    'message': ''}
        if action == 'forget-invite':
            # revoke withdraws a link that still means something; this drops
            # the line about one that does not
            gone = accounts.delete_invite(_int(req.one('id')), path=p)
            return {'ok': True, 'invite': None, 'message': '',
                    'notice': 'invite_deleted' if gone else ''}
        if action == 'assign':
            what = req.one('do') or 'new'
            if what == 'cancel':
                accounts.cancel_assignment(_int(req.one('id')), now=ts,
                                           path=p)
                return {'ok': True, 'notice': 'assign_cancelled',
                        'invite': None, 'message': ''}
            if what == 'delete':
                gone = accounts.delete_assignment(_int(req.one('id')),
                                                  path=p)
                # Already gone is not a failure. Two admins on the same row,
                # or a second click on a page that has not reloaded, is a
                # race rather than a mistake worth a red message.
                return {'ok': True, 'invite': None, 'message': '',
                        'notice': 'assign_deleted' if gone else ''}
            due = _day_end(req.one('due'))
            if req.one('due').strip() and due is None:
                return {'ok': False, 'notice': '', 'invite': None,
                        'message': 'That is not a date. Leave it empty for '
                                   'no deadline.'}
            got = accounts.create_assignment(
                req.one('who'), req.one('target'),
                surface=req.one('surface') or 'any',
                created_by=session['id'], due_at=due, note=req.one('note'),
                now=ts, path=p)
            if not got['ok']:
                return {'ok': False, 'notice': '', 'invite': None,
                        'message': got['message']}
            return {'ok': True, 'notice': 'assigned', 'invite': None,
                    'message': ''}
        if action == 'user':
            uid = _int(req.one('id'))
            what = req.one('do')
            if uid == session['id'] and what in ('disable', 'member',
                                                 'delete'):
                # Self-service lockout, refused. accounts.py stops the LAST
                # admin from stranding the dashboard, but with two admins
                # nothing stops one of them disabling themselves mid-click
                # and landing on the login page wondering what happened.
                return {'ok': False, 'notice': '', 'invite': None,
                        'message': 'That is the account you are signed in '
                                   'with. Ask the other admin.'}
            if what == 'disable':
                accounts.set_active(uid, False, path=p)
                return {'ok': True, 'notice': 'user_disabled', 'invite': None,
                        'message': ''}
            if what == 'enable':
                accounts.set_active(uid, True, path=p)
                return {'ok': True, 'notice': 'user_enabled', 'invite': None,
                        'message': ''}
            if what == 'delete':
                # The work stays. What goes is the ability to sign in.
                accounts.delete_user(uid, inherit_to=session['id'],
                                     path=p)
                return {'ok': True, 'notice': 'user_deleted', 'invite': None,
                        'message': ''}
            if what in accounts.ROLES:
                accounts.set_role(uid, what, path=p)
                return {'ok': True, 'notice': 'role_' + what, 'invite': None,
                        'message': ''}
        return {'ok': False, 'notice': '', 'invite': None,
                'message': 'That is not something this page can do.'}
    except accounts.AccountError as e:
        # The store's refusals are already sentences a person can act on
        # ("This is the last active admin. Promote somebody else first."), so
        # they are shown as they are rather than translated into a code.
        return {'ok': False, 'notice': '', 'invite': None,
                'message': e.message}
    except Exception as e:               # noqa: BLE001 - see attempt_login
        return {'ok': False, 'notice': '', 'invite': None,
                'message': 'That could not be written to the accounts '
                           'database (%s).' % (type(e).__name__,)}


def do_admin_get(req, session, now=None):
    """The admin page: invites, the users table, and a form to invite."""
    return page_reply(admin_page(session, req, now=now))


def do_admin_post(req, session, now=None):
    """A state-changing admin action, CSRF-checked, then the page again.

    A FRESH INVITE IS ANSWERED WITH A PAGE, NOT A REDIRECT. Every other
    action here follows POST/redirect/GET so a refresh cannot repeat it, but
    the redirect would have to carry the token in its Location header --
    which is a URL, which is history and a Referer, which is the one thing
    this whole flow is trying to keep the token out of. So the mint answers
    with the page directly, and a refresh mints a second invite that the
    admin can revoke. That is the cheaper mistake.
    """
    if req.oversize:
        return page_reply(admin_page(session, req, now=now,
                                     error='That form was too large.'),
                          status=413)
    if not csrf_ok(session, req.one(CSRF_FIELD), key_path=_state()['key']):
        # A missing or stale token is usually a page left open across a
        # sign-out, not an attack, so it says what to do rather than 403ing
        # into a dead end. Nothing was changed either way.
        return page_reply(
            admin_page(session, req, now=now,
                       error='That form had expired. It has been reloaded -- '
                             'try again.'),
            status=403)
    action = req.path[len(ADMIN_PATH) + 1:] or req.one('action')
    got = admin_action(action, req, session, now=now)
    if got['invite']:
        return page_reply(admin_page(session, req, now=now,
                                     minted=got['invite']))
    if not got['ok']:
        return page_reply(admin_page(session, req, now=now,
                                     error=got['message']), status=400)
    return redirect(ADMIN_PATH + '?m=' + quote(got['notice']))


# ── the gate itself ─────────────────────────────────────────────────────────

def guard(req, now=None):
    """None if this request may continue, or the Reply to send instead.

    Sets req.session on the way through, so the router does not look the user
    up a second time. The one exception is the locked deployment below, which
    never reads a cookie because there is no account for one to belong to.

    An /api path gets a 401 with a JSON body rather than a redirect. A page's
    fetch() follows a 302 by itself and hands the login HTML to JSON.parse,
    which surfaces as an unreadable console error on a dashboard that looks
    logged in; a 401 is something the client can act on.
    """
    # usable() first, because it is what lazily bootstraps the module: asking
    # it before the cookie is read means the paths below are set even if the
    # router forgot to call bootstrap(), and it means a deployment with no
    # store at all never reaches the database.
    if not usable():
        # Locked. The ONLY thing served is the login page and the sentence
        # explaining what to set -- no page, no image, no /api answer, and no
        # exception on the way past.
        if req.path == LOGIN_PATH:
            return None
        if req.path.startswith('/api/'):
            return json_reply({'error': 'the dashboard has no account '
                                        'configured yet'}, status=503)
        return redirect(LOGIN_PATH, status=302)
    req.session = resolve(req.cookies.get(COOKIE), now=now,
                          path=_state()['db'], key_path=_state()['key'])
    if req.session is not None:
        return None
    if is_public(req.path):
        return None
    if req.path.startswith('/api/'):
        return json_reply({'error': 'sign in'}, status=401)
    where = req.path
    if req.query:
        # The query goes along, so a bookmark of /audit/review?country=JPN
        # survives the detour through the login form.
        where += '?' + '&'.join(
            '%s=%s' % (quote(k), quote(v[0] if isinstance(v, list) and v
                                       else ''))
            for k, v in sorted(req.query.items()))
    tail = '?' + NEXT_FIELD + '=' + quote(where)
    if req.cookies.get(COOKIE):
        # They arrived WITH a cookie and it did not resolve: expired, or ended
        # by a password change, a disable or a sign-out-everywhere. Saying so
        # is the difference between "sign in again" and "did this thing ever
        # know who I was".
        tail += '&m=session_over'
    return redirect(LOGIN_PATH + tail, status=302)


def serve_request(req, now=None):
    """The whole gate in one call: None means "authenticated, carry on".

    The router calls this before it looks at self.path at all. Doing it the
    other way round -- routing first, gating the routes it recognises -- is
    how a page added next month ships unprotected, which is the failure this
    ordering exists to make impossible.
    """
    reply = guard(req, now=now)
    if reply is not None:
        return reply
    if req.path == LOGIN_PATH:
        if req.session is not None and req.method != 'POST':
            # Already signed in. Sending them back to a login form is how you
            # get a second account created by somebody who assumed the first
            # one had not worked.
            return redirect(safe_next(req.arg(NEXT_FIELD)), status=302)
        return do_login(req, now=now)
    if req.path == LOGOUT_PATH:
        return do_logout(req)
    if req.path == SIGNUP_PATH:
        if req.session is not None:
            # Somebody already signed in does not spend an invite. A second
            # account for one person is how a link gets burned by accident,
            # and signing out first is one click away.
            return redirect('/', status=302)
        return do_signup(req, now=now)
    if req.path == ACCOUNT_PATH:
        return do_account(req, now=now)
    if req.path == ADMIN_PATH or req.path.startswith(ADMIN_PATH + '/'):
        if not req.session or not accounts.is_admin(req.session.get('role')):
            # Gated on ROLE, not on being signed in, and answered with the
            # same empty 404 an unknown path gets. A member who guesses the
            # address learns nothing they did not already know.
            return not_found()
        if req.method == 'POST':
            return do_admin_post(req, req.session, now=now)
        if req.path != ADMIN_PATH:
            return not_found()
        return do_admin_get(req, req.session, now=now)
    return None


# ── the pages ───────────────────────────────────────────────────────────────
# Self-contained, every one of them: inline CSS, an inline data: favicon, no
# stylesheet, no script file, no font. The login and signup pages are the only
# surface an unauthenticated caller can reach, and every external asset on
# them would be one more thing served before anybody has proved who they are.
#
# The palette is audit.py's, restated rather than imported. These pages have
# to render when nothing else in the process is trusted -- including, on a bad
# day, before audit.py has loaded at all -- and a login screen that depends on
# the audit module is a login screen that breaks when the audit module does.

def esc(v):
    return html.escape('' if v is None else str(v), quote=True)


def _int(v, default=0):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _span(seconds):
    """'4 minutes', '2 hours' -- a wait a person can act on."""
    s = max(0, int(seconds))
    if s < 60:
        return '%d second%s' % (s, '' if s == 1 else 's')
    if s < 3600:
        m = (s + 29) // 60
        return '%d minute%s' % (m, '' if m == 1 else 's')
    h = (s + 1799) // 3600
    return '%d hour%s' % (h, '' if h == 1 else 's')


# The spans an invite is ever set to, in the unit the route reads (hours).
# A list rather than a free number because the question is "how long do they
# get", and the answers to that are a handful of round ones -- and because
# five spinners in five rows read as a column of data.
_EXPIRY_SPANS = ((12, '12 hours'), (24, '1 day'), (48, '2 days'),
                 (168, '1 week'), (720, '30 days'))


def _when(ts):
    """A timestamp as the operator's own clock reads it, or an em dash."""
    if not ts:
        return '\u2014'
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(int(ts)))


CSS = """
:root{--bg:#13151a;--panel:#1b2027;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;
--red:#ef5350;
--num:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);-webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  line-height:1.5}
a{color:inherit}
h1{font-size:20px;font-weight:660;letter-spacing:-.3px}
.sub{color:var(--dim);font-size:12.5px;margin-top:3px;max-width:56ch}
/* 76ch on the wide admin tables, 44ch inside the narrow auth card: one
   width was serving both, and under a full-width members table the note
   wrapped at forty-four characters into a strip a third of the panel. */
.note{font-size:12px;color:var(--dim);margin-top:14px;max-width:76ch}
.card .note{max-width:44ch}
/* Fields and buttons speak the vocabulary the rest of the dashboard does --
   the same radius, the same border, the same amber for the one control that
   commits. A login page in a different dialect reads as a different site,
   which is precisely the thing a person should be suspicious of. */
label{display:block;font-size:11px;text-transform:uppercase;
  letter-spacing:.07em;color:var(--dim);margin:0 0 5px}
input,select,textarea{width:100%;background:var(--panel2);
  border:1px solid var(--bd);color:var(--tx);border-radius:9px;
  padding:9px 11px;font-size:13.5px;font-family:inherit}
input:focus,select:focus{outline:0;border-color:rgba(232,166,69,.45)}
.field{margin-bottom:13px}
.btn{background:var(--panel2);border:1px solid var(--bd);color:var(--mut);
  border-radius:9px;padding:7px 13px;font-size:12.5px;cursor:pointer;
  font-family:inherit}
.btn:hover:not(:disabled){color:var(--tx);border-color:rgba(130,140,150,.32)}
.btn.go{color:var(--acc);border-color:rgba(232,166,69,.4);width:100%;
  padding:10px 13px;font-size:13.5px;font-weight:640}
.btn.small{padding:4px 9px;font-size:11.5px;border-radius:7px}
.btn.warn:hover{color:var(--red);border-color:rgba(239,83,80,.4)}
/* An error is not decoration: it is the whole answer the form came back
   with, so it sits above the fields it is about and keeps its colour. */
.msg{border-radius:10px;padding:10px 13px;font-size:12.5px;margin-bottom:14px}
.msg.bad{background:rgba(239,83,80,.1);border:1px solid rgba(239,83,80,.32);
  color:#f4a9a7}
.msg.ok{background:rgba(67,181,129,.1);border:1px solid rgba(67,181,129,.3);
  color:#8fd8b6}
.msg.warn{background:rgba(232,166,69,.09);
  border:1px solid rgba(232,166,69,.3);color:var(--acc)}
"""

CENTRED_CSS = """
body{display:flex;align-items:center;justify-content:center;min-height:100vh;
  padding:28px 20px}
.card{background:var(--panel);border:1px solid var(--bd);border-radius:14px;
  padding:26px 24px;width:100%;max-width:380px}
.card h1{margin-bottom:2px}
.card form{margin-top:20px}
"""

# ── the identity strip ──────────────────────────────────────────────────────
# ONE spelling of "who is signed in, and the way out", for every page with a
# header: the front page, /datasets, the three judging surfaces, and the
# Accounts page below. It used to be written twice -- dashboard.py's copy and
# a second one inline in admin_page() -- and the two drifted the way a
# duplicate always does. Five pages got a bordered button; this one got an
# underlined link, in a different sentence, floating at a different height
# because it centred itself against a header aligned to its top.
#
# It lives HERE because the strip is entirely about the session, and the
# session is this module's subject: LOGOUT_PATH and CSRF_FIELD are two feet
# away instead of read back through an import. dashboard.py asks for it.
#
# ONE PILL, NOT TWO WORDS. A monogram, the name, a hairline, the way out --
# bordered together so the eye takes the whole thing as a single object and
# skips it, the way it skips a page number. Loose in the row, "admin" and
# "sign out" read as two more controls standing beside the status line and
# the refresh button, which is what made a header holding three different
# KINDS of thing -- what to do, what the page knows, whose session this is --
# look like a flat row of six equal ones.
#
# The divider ships with it. A rule drawn by the page would be a rule left
# hanging on the day nobody is signed in.
IDENTITY_CSS = """/* A touch stronger than --bd, which is a 13% hairline meant for a
   BORDER -- a shape the eye completes from four sides. One free-standing
   pixel has no shape to complete, and at 13% it was invisible: the rule was
   drawn and the row still read as one flat run. */
.hsep{flex:none;width:1px;height:20px;background:rgba(130,140,150,.26)}
.who{display:inline-flex;align-items:center;flex:none;
border:1px solid var(--bd);border-radius:999px;
background:rgba(130,140,150,.06);overflow:hidden;
font-size:12px;line-height:1;color:var(--mut)}
/* The one round thing in a header of squared-off controls, and the only
   place a PERSON appears on a page otherwise made of counts -- which is the
   whole job: it answers "whose session is this" without being read. Muted,
   not accented: the accent on these pages means "the thing to do", and an
   account is context. */
.whoi{display:flex;align-items:center;justify-content:center;flex:none;
width:20px;height:20px;margin:3px 0 3px 3px;border-radius:50%;
background:rgba(130,140,150,.16);color:var(--mut);
font-size:10.5px;font-weight:700;text-transform:uppercase}
.whon{padding:0 10px 0 7px;color:var(--tx);font-weight:620;letter-spacing:.01em;
max-width:16ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
text-decoration:none;display:inline-block;line-height:26px}
.whon:hover{color:var(--acc)}
.whon:focus-visible{outline:2px solid var(--acc);outline-offset:-2px}
/* A form that takes up no room. Sign-out is a POST because it ends the
   session on every device by bumping session_epoch, and a state change a GET
   can make is one that any page the reader happens to open can make for
   them. Left as a block it drops the control onto its own line. */
.whof{display:flex;border-left:1px solid var(--bd)}
.whox{background:0;border:0;font:inherit;font-size:11.5px;color:var(--mut);
cursor:pointer;padding:7px 11px;transition:background .12s,color .12s}
.whox:hover{background:rgba(130,140,150,.12);color:var(--tx)}
.whox:focus-visible{outline:2px solid var(--acc);outline-offset:-2px}
@media(prefers-reduced-motion:reduce){.whox{transition:none}}"""


def identity_html(session):
    """Who is reading, and the way out -- or nothing at all.

    Empty for a signed-out reader, who cannot reach any page that renders
    this. The empty string leaves the page's sentinels in place for the next
    request, and takes the divider with it.
    """
    if not session:
        return ''
    # The username is [A-Za-z0-9._-] by accounts.USERNAME_RE, so this escape
    # has nothing to do today. It is here because the day that rule loosens
    # is not the day anybody will remember that a name reaches the header.
    raw = str(session.get('username') or '')
    name = esc(raw)
    if not name:
        return ''
    # THE FIRST CHARACTER OF THE NAME, escaped after it is taken. Escaping
    # first and slicing after cuts an entity in half: '<sam>' escapes to
    # '&lt;sam&gt;' and the monogram becomes a bare '&'. USERNAME_RE forbids
    # that character today, which is exactly why this would go unnoticed
    # until the day that rule loosens -- the day the comment above is about.
    return ('<span class="hsep" aria-hidden="true"></span>'
            '<div class="who"><span class="whoi" aria-hidden="true">'
            + esc(raw[:1]) + '</span>'
            # THE NAME IS THE WAY IN. A third control in the pill would
            # crowd it, and the one thing behind this link is the account the
            # name belongs to -- which is what a name in a corner means
            # everywhere else.
            '<a class="whon" href="' + esc(ACCOUNT_PATH)
            + '" title="Your account \u2014 change your password">'
            + name + '</a>'
            '<form class="whof" method="post" action="' + esc(LOGOUT_PATH)
            + '"><input type="hidden" name="' + esc(CSRF_FIELD)
            + '" value="' + esc(session.get('csrf') or '')
            + '"><button class="whox" type="submit" '
            'title="Ends this session on every device">sign out</button>'
            '</form></div>')


WIDE_CSS = IDENTITY_CSS + """
body{padding:0 22px 90px}
.wrap{max-width:1180px;margin:0 auto}
header{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;
  padding:22px 0 16px;border-bottom:1px solid var(--bd);margin-bottom:18px}
/* The same right-hand cluster the other pages carry, in the same order:
   where to GO, then a hairline, then whose session this is. It is one flex
   item so it lands at the end of the header instead of wherever the tagline
   beside it happens to stop wrapping. */
.hdrend{display:flex;align-items:center;gap:12px;margin-left:auto}
.back{font-size:12px;color:var(--mut);text-decoration:none;
  border:1px solid var(--bd);border-radius:8px;padding:6px 11px}
.back:hover{color:var(--tx);border-color:rgba(130,140,150,.3)}
section{margin-bottom:26px}
h2{font-size:13px;font-weight:640;color:var(--mut);margin-bottom:10px;
  text-transform:uppercase;letter-spacing:.07em}
.panel{background:var(--panel);border:1px solid var(--bd);border-radius:12px;
  padding:16px 18px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end}
.row .field{margin:0;flex:1 1 150px;min-width:120px}
.row .field.wide{flex:2 1 260px}
.row .go{width:auto}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;font-size:10.5px;text-transform:uppercase;
  letter-spacing:.07em;color:var(--dim);font-weight:600;padding:0 10px 7px 0}
td{padding:8px 10px 8px 0;border-top:1px solid var(--bd);vertical-align:middle;
  color:var(--mut)}
td.name{color:var(--tx);font-weight:600}
td.when,td.num{font-family:var(--num);font-variant-numeric:tabular-nums;
  font-size:11.5px;white-space:nowrap}
td.acts{text-align:right;white-space:nowrap}
td.acts form{display:inline}
/* The expiry box sits in a row of buttons, so it is sized like one rather
   than like a form field: wide enough for three digits and no wider. */
/* The expiry editor lives under the date it changes. Held at low contrast
   until the row is hovered or the control is focused: every open invite
   carries one, and five lit-up controls in five rows read as five different
   states rather than as one repeated affordance. */
form.exp{display:flex;gap:5px;margin-top:5px;align-items:center;
  opacity:.55;transition:opacity .12s ease}
tr:hover form.exp,form.exp:focus-within{opacity:1}
form.exp select{width:auto;flex:none;padding:2px 6px;font-size:11px;
  border-radius:7px;font-family:inherit}
form.exp .btn.small{padding:2px 8px;font-size:11px}
.tag{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
  border:1px solid var(--bd);border-radius:6px;padding:1px 6px;color:var(--dim)}
.tag.open{color:var(--green);border-color:rgba(67,181,129,.32)}
.tag.used{color:var(--mut)}
.tag.expired,.tag.revoked{color:var(--dim)}
.tag.admin{color:var(--acc);border-color:rgba(232,166,69,.35)}
.tag.off{color:var(--red);border-color:rgba(239,83,80,.3)}
.empty{color:var(--dim);font-size:12.5px;padding:6px 0}
/* The one thing on this page that is shown once. It gets the amber, the
   monospace and the whole width -- if it is missed it cannot be recovered,
   only reissued, and the line under it says so. */
.minted{background:rgba(232,166,69,.07);border:1px solid rgba(232,166,69,.34);
  border-radius:12px;padding:16px 18px;margin-bottom:20px}
.minted .lk{display:flex;gap:8px;margin:10px 0 8px}
.minted input{font-family:var(--num);font-size:12.5px}
.minted .once{font-size:12px;color:var(--acc)}
/* ── delegated work ── */
.tag.done{color:var(--green);border-color:rgba(67,181,129,.32)}
.tag.overdue{color:var(--red);border-color:rgba(239,83,80,.35)}
.tag.cancelled{color:var(--dim)}
/* The same calendar the judging pages use, so a date is picked the same way
   wherever this project asks for one -- but sized by the `input` rule above
   rather than by its own padding. A field 4px shorter than the ones beside it
   drags its label out of line, because the row aligns on the BOTTOM edge. */
.pdate{cursor:pointer;color-scheme:dark;font-variant-numeric:tabular-nums}
/* air between the form and the table under it: they are two different things
   and they were reading as one block of columns */
.panel .row+table{margin-top:18px}
/* A BAR AND THE TWO NUMBERS. The bar is for the glance down a column -- who
   is nearly there and who has not started -- and the numbers are for the
   answer, because a bar alone cannot tell 40 of 50 from 400 of 500. Width
   held so a short row and a long one line up down the page. */
td.prog{min-width:148px}
.pbar{height:4px;border-radius:3px;background:rgba(130,140,150,.16);
  overflow:hidden;margin-bottom:5px}
.pbar i{display:block;height:100%;background:var(--acc);border-radius:3px}
.pbar.full i{background:var(--green)}
.pnum{font-family:var(--num);font-variant-numeric:tabular-nums;
  font-size:11.5px;color:var(--mut);white-space:nowrap}
.pnum em{font-style:normal;color:var(--dim)}
"""

# A tab icon that is part of the document. dashboard.py serves the same dog at
# /favicon.ico, but that is a route behind this gate -- a login page that
# fetched it would either 302 into the login page again or need the icon made
# public, and a data: URI needs neither.
FAVICON = ("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'"
           "%20viewBox='0%200%20100%20100'%3E%3Ctext%20y='.9em'"
           "%20font-size='90'%3E%F0%9F%90%95%3C/text%3E%3C/svg%3E")


def _doc(title, css, body):
    """The shell every page here shares."""
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,'
            'initial-scale=1">\n'
            '<meta name="referrer" content="no-referrer">\n'
            '<title>%s</title>\n<link rel="icon" href="%s">\n'
            '<style>%s</style></head>\n<body>\n%s\n</body></html>\n'
            % (esc(title), FAVICON, CSS + css, body))


def login_page(nxt='/', error='', username='', locked=None, notice=''):
    """The form, or the explanation of why there is nothing to sign in to."""
    if locked is not None:
        # Every word of this comes from accounts.ensure_admin(), which names
        # variables and never values. It is the only page this server will
        # serve in that state, so it has to be enough to act on.
        body = (
            '<div class="card">\n'
            '  <h1>Nobody can sign in yet</h1>\n'
            '  <div class="sub">The dashboard is up and it is serving '
            'nothing until there is an account.</div>\n'
            '  <div class="msg warn" style="margin-top:18px">%s</div>\n'
            '  <div class="note">Set it in the repository\'s .env file, then '
            'restart the dashboard service. The password is written to the '
            'accounts database on the next start and read from there '
            'afterwards.</div>\n'
            '</div>' % (esc(locked.get('detail', '')),))
        return _doc('Sign in', CENTRED_CSS, body)
    bits = ['<div class="card">\n  <h1>Street dogs</h1>\n'
            '  <div class="sub">Sign in to see the dashboard.</div>\n']
    if error:
        bits.append('  <div class="msg bad">%s</div>\n' % (esc(error),))
    elif notice:
        bits.append('  <div class="msg ok">%s</div>\n' % (esc(notice),))
    bits.append(
        '  <form method="post" action="%s">\n'
        '    <input type="hidden" name="%s" value="%s">\n'
        '    <div class="field"><label for="u">username</label>\n'
        '      <input id="u" name="username" value="%s" autocomplete="username"'
        ' autocapitalize="none" autocorrect="off" spellcheck="false" '
        'autofocus required></div>\n'
        '    <div class="field"><label for="p">password</label>\n'
        '      <input id="p" name="password" type="password" '
        'autocomplete="current-password" required></div>\n'
        '    <button class="btn go" type="submit">Sign in</button>\n'
        '  </form>\n'
        '  <div class="note">Accounts are made from an invite link. If you '
        'need one, ask whoever runs this.</div>\n'
        '</div>'
        % (esc(LOGIN_PATH), esc(NEXT_FIELD), esc(nxt), esc(username)))
    return _doc('Sign in', CENTRED_CSS, ''.join(bits))


def logout_page(session, error=''):
    """What a GET /logout renders: the button, not the act.

    A bookmark of /logout still works -- it lands here -- and a page that
    merely renders cannot be fired by somebody else's <img> tag. The sentence
    is the honest one: this ends the session everywhere, because that is what
    the POST does, and a control that quietly did more than its label said
    would be the same defect wearing a different hat.
    """
    bits = ['<div class="card">\n  <h1>Sign out</h1>\n'
            '  <div class="sub">Signed in as <b>%s</b>.</div>\n'
            % (esc(str(session.get('username') or '')),)]
    if error:
        bits.append('  <div class="msg bad">%s</div>\n' % (esc(error),))
    bits.append(
        '  <form method="post" action="%s">\n'
        '    <input type="hidden" name="%s" value="%s">\n'
        '    <button class="btn go" type="submit">Sign out everywhere</button>'
        '\n  </form>\n'
        '  <div class="note">This ends the session on every device signed in '
        'as this account, not only in this browser. Nothing else about the '
        'account changes.</div>\n'
        '  <div class="note"><a href="/">&larr; back to the dashboard</a>'
        '</div>\n</div>'
        % (esc(LOGOUT_PATH), esc(CSRF_FIELD),
           esc(str(session.get('csrf') or ''))))
    return _doc('Sign out', CENTRED_CSS, ''.join(bits))


def signup_page(token, state, error='', username=''):
    """The invite form, or a plain sentence about why there is no form.

    A dead link gets the reason and a way out, and nothing else: no form to
    fill in, no note, and no word about who issued it.
    """
    st = (state or {}).get('state', 'unknown')
    if st != 'open':
        body = ('<div class="card">\n  <h1>This link is closed</h1>\n'
                '  <div class="msg warn" style="margin-top:16px">%s</div>\n'
                '  <div class="note"><a href="%s">Sign in</a> if you already '
                'have an account.</div>\n</div>'
                % (esc(INVITE_WORDS.get(st, INVITE_WORDS['unknown'])),
                   esc(LOGIN_PATH)))
        return _doc('Invite', CENTRED_CSS, body)
    role = (state or {}).get('role', 'member')
    bits = ['<div class="card">\n  <h1>Choose a username</h1>\n'
            '  <div class="sub">This invite makes %s account. It works '
            'once, and it expires %s.</div>\n'
            % ('an admin' if role == 'admin' else 'a member',
               esc(_when((state or {}).get('expires_at'))))]
    if error:
        bits.append('  <div class="msg bad">%s</div>\n' % (esc(error),))
    bits.append(
        '  <form method="post" action="%s">\n'
        '    <input type="hidden" name="%s" value="%s">\n'
        '    <div class="field"><label for="u">username</label>\n'
        '      <input id="u" name="username" value="%s" minlength="%d" '
        'maxlength="%d" autocomplete="username" autocapitalize="none" '
        'autocorrect="off" spellcheck="false" autofocus required></div>\n'
        '    <div class="field"><label for="p">password</label>\n'
        '      <input id="p" name="password" type="password" minlength="%d" '
        'autocomplete="new-password" required></div>\n'
        '    <div class="field"><label for="c">password again</label>\n'
        '      <input id="c" name="confirm" type="password" minlength="%d" '
        'autocomplete="new-password" required></div>\n'
        '    <button class="btn go" type="submit">Create the account</button>\n'
        '  </form>\n'
        '  <div class="note">Letters, digits, dot, dash and underscore in a '
        'username; %d characters or more in a password. Nobody can recover '
        'it for you \u2014 an admin can only set a new one.</div>\n</div>'
        % (esc(SIGNUP_PATH), esc(TOKEN_FIELD), esc(token), esc(username),
           accounts.USERNAME_MIN, accounts.USERNAME_MAX,
           accounts.PASSWORD_MIN, accounts.PASSWORD_MIN,
           accounts.PASSWORD_MIN))
    return _doc('Invite', CENTRED_CSS, ''.join(bits))


def invite_link(req, token):
    """The URL to hand somebody, built from the Host this request arrived on.

    Built from the header rather than from config because this server answers
    on two addresses at once -- the tailnet one for a phone and 127.0.0.1 for
    the machine itself -- and the right link is the one that matches however
    the admin got here. The value is echoed only into the admin's own page and
    is escaped on the way; a poisoned Host header poisons nothing but the copy
    button of the person who sent it.
    """
    host = (req.host or '').strip()
    tail = '%s?%s=%s' % (SIGNUP_PATH, TOKEN_FIELD, quote(token))
    if not host or '/' in host or ' ' in host:
        return tail
    return 'http://%s%s' % (host, tail)


def _day_end(s):
    """The instant a YYYY-MM-DD deadline runs out: the midnight AFTER it.

    "Due the 20th" means the reader has the 20th. A deadline landing at
    00:00 on the 20th would turn that whole day red, which is the same
    off-by-a-day the annotated-date window is built to avoid.
    """
    s = str(s or '').strip()
    if not _DUE_RE.match(s):
        return None
    y, m, d = int(s[:4]), int(s[5:7]), int(s[8:10])
    try:
        ts = time.mktime((y, m, d, 0, 0, 0, 0, 0, -1))
    except (OverflowError, ValueError):
        return None
    lt = time.localtime(ts)
    if (lt.tm_year, lt.tm_mon, lt.tm_mday) != (y, m, d):
        return None                       # mktime turns Feb 31 into March
    return int(ts) + 86400


def account_page(session, error='', notice=''):
    """Your own account. One form, because there is one thing to change.

    Deliberately not a settings page. A username is on every annotation this
    person has ever made and renaming them would either rewrite that or make
    it lie, and a role is somebody else's decision -- so the page offers the
    one thing that is genuinely theirs and says what it costs.
    """
    who = esc(str(session.get('username') or ''))
    role = session.get('role') if session.get('role') in accounts.ROLES \
        else 'member'
    bits = ['<div class="card">\n  <h1>Your account</h1>\n'
            '  <div class="sub">Signed in as <b>%s</b> \u2014 %s.</div>\n'
            % (who, role)]
    if error:
        bits.append('  <div class="msg bad">%s</div>\n' % (esc(error),))
    elif notice:
        bits.append('  <div class="msg ok">%s</div>\n' % (esc(notice),))
    bits.append(
        '  <form method="post" action="%s">\n'
        # The username, hidden and disabled: a password manager needs to know
        # WHICH login it is being asked to update, and without it some of
        # them save the new password as a second, nameless entry.
        '    <input type="text" name="username" value="%s" '
        'autocomplete="username" hidden disabled>\n'
        '    <div class="field"><label for="c0">current password</label>\n'
        '      <input id="c0" name="current" type="password" '
        'autocomplete="current-password" autofocus required></div>\n'
        '    <div class="field"><label for="p1">new password</label>\n'
        '      <input id="p1" name="password" type="password" minlength="%d" '
        'maxlength="%d" autocomplete="new-password" required></div>\n'
        '    <div class="field"><label for="p2">new password again</label>\n'
        '      <input id="p2" name="confirm" type="password" minlength="%d" '
        'maxlength="%d" autocomplete="new-password" required></div>\n'
        '    <button class="btn go" type="submit">Change password</button>\n'
        '  </form>\n'
        # Said before it happens, not after. Somebody changing a password on
        # a phone should know the laptop is about to ask them to sign in
        # again, or they will read it as the change having broken something.
        '  <div class="note">Changing it signs out every other device signed '
        'in as you \u2014 this one stays. There is no way to recover a '
        'forgotten password: an admin has to set a new one.</div>\n'
        '  <div class="note"><a href="/">&larr; dashboard</a></div>\n'
        '</div>'
        % (esc(ACCOUNT_PATH), who,
           accounts.PASSWORD_MIN, accounts.PASSWORD_MAX,
           accounts.PASSWORD_MIN, accounts.PASSWORD_MAX))
    return _doc('Your account', CENTRED_CSS, ''.join(bits))


def admin_page(session, req, now=None, error='', minted=None):
    """Invites and accounts, on one page, for an admin only."""
    ts = int(time.time() if now is None else now)
    p = _state()['db']
    csrf = esc(session.get('csrf', ''))
    notice = NOTICES.get(req.arg('m'), '')
    bits = ['<div class="wrap">\n<header>\n  <div><h1>Accounts</h1>\n'
            '    <div class="sub">Who may open this dashboard, the invite '
            'links that let them, and the work they have been asked '
            'for.</div></div>\n'
            '  <div class="hdrend"><a class="back" href="/">&larr; dashboard'
            '</a>%s</div>\n</header>\n' % (identity_html(session),)]
    if error:
        bits.append('<div class="msg bad">%s</div>\n' % (esc(error),))
    elif notice:
        bits.append('<div class="msg ok">%s</div>\n' % (esc(notice),))
    if minted:
        link = invite_link(req, minted['token'])
        bits.append(
            '<div class="minted">\n'
            '  <b>A new invite link, for %s.</b>\n'
            '  <div class="lk"><input id="lk" value="%s" readonly '
            'onfocus="this.select()">\n'
            '    <button class="btn" id="cp" type="button">copy</button></div>\n'
            '  <div class="once">This is the only time it is shown. It is not '
            'stored anywhere \u2014 if it is lost, revoke it below and issue '
            'another. It expires %s and works once.</div>\n</div>\n'
            % ('an admin account' if minted['role'] == 'admin'
               else 'a member account', esc(link),
               esc(_when(minted['expires_at']))))
    hours_default = _int(os.environ.get(accounts.ENV_INVITE_TTL_HOURS),
                         accounts.INVITE_TTL_DEFAULT // 3600)
    bits.append(
        '<section>\n<h2>Invite somebody</h2>\n<div class="panel">\n'
        '<form method="post" action="%s/invite" class="row">\n'
        '  <input type="hidden" name="%s" value="%s">\n'
        '  <div class="field wide"><label for="n">what it is for</label>\n'
        '    <input id="n" name="note" maxlength="200" placeholder="field '
        'team, phone" autocomplete="off"></div>\n'
        '  <div class="field"><label for="h">hours</label>\n'
        '    <input id="h" name="hours" type="number" min="1" max="720" '
        'step="1" value="%d"></div>\n'
        '  <div class="field"><label for="r">role</label>\n'
        '    <select id="r" name="role"><option value="member">member'
        '</option><option value="admin">admin</option></select></div>\n'
        '  <button class="btn go" type="submit">Create link</button>\n'
        '</form>\n</div>\n</section>\n'
        % (esc(ADMIN_PATH), esc(CSRF_FIELD), csrf, hours_default))

    try:
        invites = accounts.list_invites(path=p, now=ts)
    except Exception as e:                # noqa: BLE001 - a table, not the gate
        # A table that will not read is a broken table, not a broken gate: the
        # rest of the page -- and the invite form above it -- still works.
        invites = []
        bits.append('<div class="msg bad">%s</div>\n' % (esc(str(e)),))
    bits.append('<section>\n<h2>Invites</h2>\n<div class="panel">\n')
    if not invites:
        bits.append('<div class="empty">None yet.</div>\n')
    else:
        bits.append('<table><tr><th>state</th><th>role</th><th>for</th>'
                    '<th>issued by</th><th>issued</th><th>expires</th>'
                    '<th>taken by</th><th></th></tr>\n')
        for iv in invites:
            act = when = ''
            if iv['state'] in ('open', 'expired'):
                # WHEN IT RUNS OUT, CHANGED WITHOUT REISSUING IT. The link
                # went out in an email hours ago; the person it was for has
                # not clicked it yet, or clicked it a day late. Minting a
                # second link means chasing them with it, and leaves the first
                # one live. Hours from now, so a link that is already out of
                # time is the ordinary case rather than the impossible one.
                #
                # A LIST OF SPANS, NOT A NUMBER BOX. The number box was a
                # spinner with its arrows in every row and the same "48" in
                # each of them, which reads as a column of data rather than
                # as five copies of one control -- and nobody sets an invite
                # to 37 hours. It sits under the date it changes, not in the
                # actions cell, because that cell already held three buttons
                # and this is not a fourth: it edits the value beside it.
                # The deployment default may not be one of the spans --
                # DASHBOARD_INVITE_TTL_HOURS=72 is a choice an operator
                # actually makes -- and a <select> with nothing selected
                # falls back to its FIRST option, which is the SHORTEST.
                # An admin who clicked "set" without touching the control
                # would then hand out less time than their own default, the
                # opposite of what the button is for. The smallest span that
                # covers the default is selected instead; more time than
                # asked for beats a link that dies early.
                mark = next((h for h, _ in _EXPIRY_SPANS
                             if h >= hours_default), _EXPIRY_SPANS[-1][0])
                span = ''.join(
                    '<option value="%d"%s>%s</option>'
                    % (h, ' selected' if h == mark else '', w)
                    for h, w in _EXPIRY_SPANS)
                when = ('<form method="post" action="%s/invite-expiry"'
                        ' class="exp">'
                        '<input type="hidden" name="%s" value="%s">'
                        '<input type="hidden" name="id" value="%d">'
                        '<select name="hours" aria-label="new expiry, from '
                        'now">%s</select>'
                        '<button class="btn small" type="submit" '
                        'title="Move when this link stops working. The link '
                        'itself does not change.">set</button></form>'
                        % (esc(ADMIN_PATH), esc(CSRF_FIELD), csrf, iv['id'],
                           span))
                act = ('<form method="post" action="%s/revoke">'
                       '<input type="hidden" name="%s" value="%s">'
                       '<input type="hidden" name="id" value="%d">'
                       '<button class="btn small warn" type="submit">revoke'
                       '</button></form>'
                       % (esc(ADMIN_PATH), esc(CSRF_FIELD), csrf, iv['id']))
            # REVOKE WITHDRAWS A LINK THAT STILL WORKS. This drops the line
            # about one that does not: taken, expired, or already withdrawn.
            act += ('<form method="post" action="%s/forget-invite"'
                    ' onsubmit="return confirm(&quot;Remove this invite from '
                    'the list? Anyone who already used it keeps their '
                    'account.&quot;)">'
                    '<input type="hidden" name="%s" value="%s">'
                    '<input type="hidden" name="id" value="%d">'
                    '<button class="btn small" type="submit">remove'
                    '</button></form>'
                    % (esc(ADMIN_PATH), esc(CSRF_FIELD), csrf, iv['id']))
            bits.append(
                '<tr><td><span class="tag %s">%s</span></td>'
                '<td><span class="tag%s">%s</span></td>'
                '<td>%s</td><td class="name">%s</td>'
                '<td class="when">%s</td><td class="when">%s%s</td>'
                '<td>%s</td><td class="acts">%s</td></tr>\n'
                % (esc(iv['state']), esc(iv['state']),
                   ' admin' if iv['role'] == 'admin' else '', esc(iv['role']),
                   esc(iv['note'] or '\u2014'),
                   esc(iv['created_by_name'] or '\u2014'),
                   esc(_when(iv['created_at'])), esc(_when(iv['expires_at'])),
                   when, esc(iv['used_by_name'] or '\u2014'), act))
        bits.append('</table>\n')
    bits.append('</div>\n</section>\n')

    try:
        users = accounts.list_users(path=p)
    except Exception as e:                # noqa: BLE001
        users = []
        bits.append('<div class="msg bad">%s</div>' % (esc(str(e)),))
    bits.append('<section>\n<h2>Accounts</h2>\n<div class="panel">\n'
                '<table><tr><th>username</th><th>role</th><th>state</th>'
                '<th>joined</th><th>last seen</th><th></th></tr>\n')
    for u in users:
        me = u['id'] == session.get('id')
        acts = []
        # THE OWNER'S ROW CARRIES NO BUTTONS. Its tier follows DASHBOARD_USER
        # rather than anything on this page, and the store refuses every one
        # of these edits -- offering them would be offering an error. It also
        # used to read as a promotion: an owner is not an admin, so the button
        # said "make admin", which is a demotion wearing the wrong word.
        if not me and u['role'] != 'owner':
            acts.append(_useract(csrf, u['id'],
                                 'disable' if u['active'] else 'enable',
                                 'disable' if u['active'] else 'enable',
                                 warn=u['active']))
            acts.append(_useract(csrf, u['id'],
                                 'member' if u['role'] == 'admin' else 'admin',
                                 'make member' if u['role'] == 'admin'
                                 else 'make admin'))
            # DISABLE IS THE REVERSIBLE ONE; this is not. The work they did
            # stays where it is -- a verdict lives in a ledger, under the
            # name that made it.
            acts.append(_useract(
                csrf, u['id'], 'delete', 'remove', warn=True,
                confirm='Remove %s? They cannot sign in again. Every crop '
                        'they judged stays judged, under their name, and so '
                        'does every invite somebody joined through.'
                        % (u['username'],)))
        bits.append(
            '<tr><td class="name">%s%s</td>'
            '<td><span class="tag%s">%s</span></td>'
            '<td><span class="tag%s">%s</span></td>'
            '<td class="when">%s</td><td class="when">%s</td>'
            '<td class="acts">%s</td></tr>\n'
            % (esc(u['username']), ' <span class="tag">you</span>' if me else '',
               ' admin' if accounts.is_admin(u['role']) else '',
               esc(u['role']),
               '' if u['active'] else ' off',
               'active' if u['active'] else 'disabled',
               esc(_when(u['created_at'])),
               # last_seen_at, and last_login_at only as history: an invited
               # annotator may never have typed a password, and "last seen"
               # naming their signup week ago while they judged crops today
               # was this column lying under its own header.
               esc(_when(u.get('last_seen_at') or u['last_login_at'])),
               ' '.join(acts)))
    bits.append('</table>\n<div class="note">Disabling an account ends its '
                'open sessions immediately and can be undone. Removing one '
                'cannot: the sign-in goes, and what they judged stays judged '
                'under their name. Invite links they issued and nobody took '
                'go with them; the ones somebody joined through stay, so the '
                'people they brought in still have a record of how.</div>\n'
                '</div>\n</section>\n')
    bits.append(_delegation_section(session, csrf, users, p, ts))
    bits.append('</div>\n')
    if minted:
        # execCommand, not navigator.clipboard: the clipboard API is only
        # available in a secure context, and this page is plain HTTP on a
        # tailnet, so on the phone the modern call is simply undefined. The
        # field is selected either way, so the fallback is a manual copy.
        bits.append(
            '<script>\n'
            "(function(){var b=document.getElementById('cp'),"
            "f=document.getElementById('lk');if(!b||!f)return;\n"
            "b.onclick=function(){f.focus();f.select();"
            "f.setSelectionRange(0,f.value.length);\n"
            "var ok=false;try{ok=document.execCommand('copy')}catch(e){}\n"
            "b.textContent=ok?'copied':'press \\u2318C';};})();\n"
            '</script>\n')
    # THE PROGRESS CELLS, filled from the module that reads the ledgers. It
    # is a fetch rather than a render because this file must not learn where
    # the annotations live: the gate imports accounts.py and nothing else, and
    # the day it imports the ledgers too is the day a broken parquet stops
    # anybody signing in.
    bits.append(
        '<script>\n'
        # IRREVERSIBLE CONTROLS ASK FIRST. One delegated listener for every
        # button carrying data-confirm, rather than a line of JavaScript in
        # an attribute per row -- and it reads the question off the button,
        # so the sentence names the person and the row it is about.
        "document.addEventListener('click',function(e){\n"
        "var b=e.target&&e.target.closest&&e.target.closest('[data-confirm]');\n"
        "if(!b)return;\n"
        "if(!window.confirm(b.getAttribute('data-confirm')))"
        "{e.preventDefault();e.stopPropagation();}\n"
        "},true);\n"
        "(function(){var cs=document.querySelectorAll('.prog[data-a]');\n"
        'if(!cs.length)return;\n'
        "function n(v){return String(v).replace(/\\B(?=(\\d{3})+(?!\\d))/g,',')}\n"
        "function say(t){for(var i=0;i<cs.length;i++){"
        "var e=cs[i].querySelector('.pnum');if(e)e.textContent=t;}}\n"
        "fetch('/api/assignments',{credentials:'same-origin'})\n"
        '.then(function(r){if(!r.ok)throw 0;return r.json()})\n'
        '.then(function(j){var by={};\n'
        '(j.assignments||[]).forEach(function(a){by[a.id]=a});\n'
        'for(var i=0;i<cs.length;i++){var c=cs[i],\n'
        "a=by[c.getAttribute('data-a')];\n"
        "if(!a){var e=c.querySelector('.pnum');\n"
        "if(e)e.textContent='not counted';continue}\n"
        "c.innerHTML='<div class=\"pbar'+(a.pct>=100?' full':'')+'\">'+\n"
        "'<i style=\"width:'+(+a.pct||0)+'%\"></i></div>'+\n"
        "'<span class=\"pnum\">'+n(a.done)+' / '+n(a.target)+\n"
        "' <em>'+(+a.pct||0)+'%</em></span>';}})\n"
        "// A count that did not arrive must say so. Cells left reading a dash\n"
        "// would read as nobody having done anything, which is a different\n"
        "// and much more alarming thing than the server not answering.\n"
        ".catch(function(){say('could not count')});})();\n"
        '</script>\n')
    return _doc('Accounts', WIDE_CSS, ''.join(bits))


# The words the surfaces go by, out of the file the annotator's own strip
# reads them from. A second copy here said "any surface" where their bar said
# "every surface": one target, two names for it, and the page that set it
# disagreeing with the page it is measured on.
SURFACE_WORDS = work_strip.SURFACE_WORDS


def _due_day(due_at):
    """The last day somebody has, not the midnight that ends it.

    due_at is the instant the deadline runs out -- the midnight AFTER the due
    day -- so printing it raw names the morning after, and a column of those
    is every deadline on the page off by one.
    """
    if not due_at:
        return '\u2014'
    return time.strftime('%d %b %Y', time.localtime(int(due_at) - 1))


def _delegation_section(session, csrf, users, path, ts):
    """Work handed out, and how far along it is.

    THE PROGRESS NUMBERS ARE NOT RENDERED HERE. This module knows a target
    was set; it has no idea what anybody has judged, and giving it one would
    put the annotation ledgers behind the login gate's own import. The cells
    are stamped with their row id and filled from /api/assignments, which is
    served by the module that already reads those ledgers -- and answers a
    member with the same empty 404 an address that does not exist gets.
    """
    live = [u for u in users if u['active']]
    try:
        rows = accounts.list_assignments(path=path, now=ts)
    except Exception as e:                # noqa: BLE001 - a table, not the gate
        return ('<section>\n<h2>Delegated work</h2>\n<div class="panel">\n'
                '<div class="msg bad">%s</div>\n</div>\n</section>\n'
                % (esc(str(e)),))
    out = ['<section>\n<h2>Delegated work</h2>\n<div class="panel">\n']
    if not live:
        out.append('<div class="empty">Nobody to delegate to yet \u2014 '
                   'invite somebody first.</div>\n')
    else:
        out.append(
            '<form method="post" action="%s/assign" class="row">\n'
            '  <input type="hidden" name="%s" value="%s">\n'
            '  <input type="hidden" name="do" value="new">\n'
            '  <div class="field"><label for="aw">who</label>\n'
            '    <select id="aw" name="who">%s</select></div>\n'
            '  <div class="field"><label for="at">how many</label>\n'
            '    <input id="at" name="target" type="number" min="1" '
            'max="%d" step="1" value="500"></div>\n'
            '  <div class="field"><label for="as">on</label>\n'
            '    <select id="as" name="surface">%s</select></div>\n'
            '  <div class="field"><label for="ad">due (optional)</label>\n'
            '    <input id="ad" name="due" type="date" class="pdate"></div>\n'
            '  <div class="field wide"><label for="an">what it is for</label>\n'
            '    <input id="an" name="note" maxlength="200" '
            'placeholder="leash pass before the retrain" autocomplete="off">'
            '</div>\n'
            '  <button class="btn go" type="submit">Delegate</button>\n'
            '</form>\n'
            % (esc(ADMIN_PATH), esc(CSRF_FIELD), csrf,
               ''.join('<option value="%s">%s</option>'
                       % (esc(u['username']), esc(u['username']))
                       for u in live),
               accounts.MAX_TARGET,
               ''.join('<option value="%s">%s</option>'
                       % (esc(s), esc(SURFACE_WORDS[s]))
                       for s in accounts.SURFACES)))
    if not rows:
        out.append('<div class="empty">Nothing delegated yet.</div>\n')
    else:
        out.append('<table><tr><th>who</th><th>on</th><th>progress</th>'
                   '<th>for</th><th>set by</th><th>due</th><th>state</th>'
                   '<th></th></tr>\n')
        for a in rows:
            # TWO DIFFERENT ANSWERS, so two controls rather than one that
            # means both. "Call off" stops the work and keeps the record --
            # what was asked for, and where it got to. "Delete" is for a row
            # that should never have existed: the wrong person, or 5000 typed
            # for 500. Only the second is offered on a row that is already
            # finished or called off, because there is nothing left to stop.
            act = _assignact(csrf, a['id'], 'cancel', 'call off', warn=True) \
                if a['state'] in ('open', 'overdue') else ''
            act += _assignact(
                csrf, a['id'], 'delete', 'delete', warn=True,
                confirm='Delete this target for %s? The record goes; every '
                        'annotation made towards it stays exactly where it '
                        'is.' % (a['username'] or 'them',))
            out.append(
                '<tr><td class="name">%s</td><td>%s</td>'
                '<td class="prog" data-a="%d" data-target="%d">'
                '<span class="pnum">\u2014 / %s</span></td>'
                '<td>%s</td><td class="name">%s</td>'
                '<td class="when">%s</td>'
                '<td><span class="tag %s">%s</span></td>'
                '<td class="acts">%s</td></tr>\n'
                % (esc(a['username'] or '\u2014'),
                   esc(SURFACE_WORDS.get(a['surface'], a['surface'])),
                   a['id'], a['target'], esc('{:,}'.format(a['target'])),
                   esc(a['note'] or '\u2014'),
                   esc(a['created_by_name'] or '\u2014'),
                   esc(_due_day(a['due_at'])),
                   esc(a['state']), esc(a['state']), act))
        out.append('</table>\n')
    out.append(
        '<div class="note">A target counts only what is judged AFTER it is '
        'set \u2014 "five hundred" means five hundred more. One open target '
        'per person per surface. <b>Call off</b> stops the work and keeps the '
        'record of what was asked for; <b>delete</b> removes the record. '
        'Neither touches an annotation \u2014 verdicts live in the ledgers, '
        'and this table only ever asked for them.</div>\n'
        '</div>\n</section>\n')
    return ''.join(out)


def _assignact(csrf, aid, do, label, warn=False, confirm=''):
    """One button in the delegated-work table: a real form, so it is a POST.

    data-confirm rather than an inline onclick, so the page has one handler
    for every irreversible control instead of a string of JavaScript written
    into an attribute per row.
    """
    return ('<form method="post" action="%s/assign">'
            '<input type="hidden" name="%s" value="%s">'
            '<input type="hidden" name="do" value="%s">'
            '<input type="hidden" name="id" value="%d">'
            '<button class="btn small%s" type="submit"%s>%s</button>'
            '</form>'
            % (esc(ADMIN_PATH), esc(CSRF_FIELD), csrf, esc(do), aid,
               ' warn' if warn else '',
               (' data-confirm="%s"' % (esc(confirm),)) if confirm else '',
               esc(label)))


def _useract(csrf, uid, what, label, warn=False, confirm=''):
    """One button in the accounts table: a real form, so it is a POST.

    `confirm` puts a browser confirm in front of it, for the ones with no
    undo. It is a convenience, not a control: the refusals that matter --
    the last admin, your own account -- are enforced on the server.
    """
    return ('<form method="post" action="%s/user"%s>'
            '<input type="hidden" name="%s" value="%s">'
            '<input type="hidden" name="id" value="%d">'
            '<input type="hidden" name="do" value="%s">'
            '<button class="btn small%s" type="submit">%s</button></form>'
            % (esc(ADMIN_PATH),
               (' onsubmit="return confirm(&quot;%s&quot;)"' % (esc(confirm),))
               if confirm else '',
               esc(CSRF_FIELD), csrf, uid, esc(what),
               ' warn' if warn else '', esc(label)))
