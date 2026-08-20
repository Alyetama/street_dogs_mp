#!/usr/bin/env python3
"""The accounts store is the only thing between the tailnet and the harvest.

    python tools/detect/tests/adv_accounts.py

Everything here defends one of the claims tools/dashboard/accounts.py makes:

  * the database and BOTH of its sqlite sidecars are 0600 on disk, whatever
    the umask was, and neither the file nor the sidecars are anything the
    dashboard's static handler will hand out;
  * WAL, synchronous=FULL and foreign_keys are really on, the schema really
    is versioned, and migrate() run twice is a no-op that carries an existing
    database forward instead of rebuilding it;
  * a password is a salted, self-describing hash that verifies, that never
    repeats itself for two people with the same password, and that upgrades
    itself the next time its owner logs in;
  * a failed login costs the same whether the account is unknown, disabled or
    simply mistyped -- a login form that answers faster for a name nobody has
    is a user directory;
  * an invite is one-time under REAL concurrency: sixteen threads on one
    token produce one account, not sixteen, and expiry and revocation are
    honoured;
  * the token itself is nowhere in the file, so a copy of the database is not
    a set of working links;
  * the .env admin is created once, updated when .env changes, never
    duplicated by a change of case, and never left demoted or disabled;
  * the throttle table cannot grow without limit, since its key is a string
    the attacker chooses;
  * every query is parameterised.

The fixtures are temp directories. Nothing here reads, writes or measures
data/dashboard, and nothing anywhere near it is opened -- the one thing this
suite reads out of the repo is dashboard.py's static allow-list, as text.

Every check is written to fail if the defect it names comes back; a check that
cannot be made to fail is a certificate of nothing.
"""

import ast
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DASH = os.path.join(REPO, 'tools', 'dashboard')
if DASH not in sys.path:
    sys.path.insert(0, DASH)

import accounts as A                                          # noqa: E402

DASHBOARD_PY = os.path.join(DASH, 'dashboard.py')
ACCOUNTS_PY = os.path.join(DASH, 'accounts.py')

# Long enough to pass check_password everywhere, and distinct enough that a
# check comparing two hashes is comparing two different passwords.
PW = 'correct-horse-battery'
PW2 = 'a-completely-different-one'
ADMIN_ENV = {'DASHBOARD_USER': 'admin', 'DASHBOARD_PASSWORD': PW}


class Fixture:
    """A temp directory and a database in it. Never data/dashboard."""

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix='adv_accounts_')
        self.db = os.path.join(self.dir, 'accounts.db')
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.dir, ignore_errors=True)
        return False

    def fresh(self, name='fresh.db'):
        """A second, empty database -- for the checks that need a clean slate."""
        p = os.path.join(self.dir, name)
        for s in ('', '-wal', '-shm', '-journal'):
            try:
                os.remove(p + s)
            except OSError:
                pass
        return p

    def admin(self, path=None):
        """The .env admin, bootstrapped. Returns its row."""
        A.ensure_admin(path=path or self.db, env=dict(ADMIN_ENV))
        return A.get_user('admin', path=path or self.db)


def _mode(p):
    return oct(os.stat(p).st_mode & 0o777) if os.path.exists(p) else 'absent'


# ── the file on disk ────────────────────────────────────────────────────────

def file_checks(bad, fx):
    """0600 on the database AND on the sidecars, whatever the umask was.

    The sidecars matter as much as the database: -wal holds rows that have
    been written and not yet checkpointed, in the clear, and it is created
    fresh on every write after the last connection closed. A one-time chmod at
    install time would be right exactly once.
    """
    p = fx.fresh('modes.db')
    was = os.umask(0o000)          # the hostile case: nothing masked off
    try:
        con = A.connect(p)
        con.execute('SELECT 1').fetchone()
    finally:
        os.umask(was)
    try:
        for suffix in ('', '-wal', '-shm'):
            q = p + suffix
            if not os.path.exists(q):
                bad.append(f'{os.path.basename(q)} was never created, so its '
                           f'mode was never checked -- the check is asleep')
                continue
            if _mode(q) != '0o600':
                bad.append(f'{os.path.basename(q)} is {_mode(q)}, not 0o600: '
                           f'password hashes readable by anyone on the box')
    finally:
        con.close()

    # And again on a database that already exists with a bad mode, because
    # that is the real repair case -- an older build made one 0644.
    os.chmod(p, 0o644)
    con = A.connect(p)
    try:
        if _mode(p) != '0o600':
            bad.append(f'connect() left an existing database at {_mode(p)}; '
                       f'a mode that is only set at creation is not set')
    finally:
        con.close()


def pragma_checks(bad, fx):
    """WAL, synchronous=FULL, foreign keys and a busy timeout, per connection."""
    con = A.connect(fx.db)
    try:
        got = {
            'journal_mode': str(con.execute(
                'PRAGMA journal_mode').fetchone()[0]).lower(),
            'synchronous': int(con.execute(
                'PRAGMA synchronous').fetchone()[0]),
            'foreign_keys': int(con.execute(
                'PRAGMA foreign_keys').fetchone()[0]),
            'busy_timeout': int(con.execute(
                'PRAGMA busy_timeout').fetchone()[0]),
        }
        if got['journal_mode'] != 'wal':
            bad.append(f"journal_mode is {got['journal_mode']}, not wal: a "
                       f'threaded server\'s reader will block its writer')
        if got['synchronous'] != 2:
            bad.append(f"synchronous is {got['synchronous']}, not FULL(2): an "
                       f'account created before a power cut can vanish')
        if not got['foreign_keys']:
            bad.append('foreign_keys is off, so an invite may point at a user '
                       'id that never existed')
        if got['busy_timeout'] <= 0:
            bad.append('busy_timeout is 0, so the loser of a write race gets '
                       '"database is locked" instead of waiting')
    finally:
        con.close()

    # Enforcement, not just the pragma: prove both directions actually bite.
    fx.admin()
    con = A.connect(fx.db)
    try:
        try:
            con.execute('BEGIN IMMEDIATE')
            con.execute('INSERT INTO invites (token_hash, created_by, '
                        'created_at, expires_at) VALUES (?,?,?,?)',
                        ('nope', 10 ** 9, 1, 2))
            con.execute('COMMIT')
            bad.append('an invite was accepted with a created_by that is not '
                       'a user')
        except sqlite3.IntegrityError:
            con.execute('ROLLBACK')
        A.create_invite('admin', ttl=600, path=fx.db)
        try:
            con.execute('BEGIN IMMEDIATE')
            con.execute('DELETE FROM users WHERE username_norm = ?',
                        ('admin',))
            con.execute('COMMIT')
            bad.append('the admin who issued an invite could be deleted, '
                       'leaving the invite pointing at nobody')
        except sqlite3.IntegrityError:
            con.execute('ROLLBACK')
    finally:
        con.close()


def index_checks(bad, fx):
    """The three lookups on the request path use an index, not a scan.

    Not a micro-optimisation: throttle is keyed on a string the caller
    chooses, so a table scan per failed login is a denial of service you
    supply yourself.
    """
    fx.admin()
    con = A.connect(fx.db)
    try:
        for query, args, table in (
                ('SELECT * FROM users WHERE username_norm = ?', ('x',),
                 'users'),
                ('SELECT * FROM invites WHERE token_hash = ?', ('x',),
                 'invites'),
                ('SELECT * FROM throttle WHERE source = ?', ('x',),
                 'throttle')):
            plan = ' '.join(str(r[3]) for r in
                            con.execute('EXPLAIN QUERY PLAN ' + query, args))
            if 'USING INDEX' not in plan and 'USING COVERING INDEX' not in plan:
                bad.append(f'the {table} lookup is a scan, not an index '
                           f'search: {plan!r}')
    finally:
        con.close()


def private_file_checks(bad, fx):
    """The database sits in the static handler's document root. Prove it is
    not served, and that PRIVATE_FILES names every file sqlite makes.

    data/dashboard is where index.html, the map layers and the crop thumbnails
    come from, and SimpleHTTPRequestHandler hands out anything in its
    directory by name. dashboard.py answers from an allow-list, which is the
    only reason accounts.db is not a download; this check is what keeps a
    future entry from quietly widening it.
    """
    want = {os.path.basename(A.DB_PATH) + s
            for s in ('', '-wal', '-shm', '-journal')}
    if set(A.PRIVATE_FILES) != want:
        bad.append(f'PRIVATE_FILES is {sorted(A.PRIVATE_FILES)}, but sqlite '
                   f'writes {sorted(want)} -- a file nobody named is a file '
                   f'nobody guards')
    if os.path.dirname(A.DB_PATH) != os.path.join(REPO, 'data', 'dashboard'):
        bad.append('the database moved out of data/dashboard; the static '
                   'allow-list check below no longer describes the risk')

    try:
        src = open(DASHBOARD_PY, encoding='utf-8').read()
    except OSError as e:
        bad.append(f'could not read dashboard.py to check the static '
                   f'allow-list: {e}')
        return
    files, dirs = _literal(src, 'STATIC_FILES'), _literal(src, 'STATIC_DIRS')
    if files is None or dirs is None:
        bad.append('dashboard.py no longer defines STATIC_FILES/STATIC_DIRS '
                   'as literals; this check cannot see what is served and is '
                   'therefore not checking it')
        return
    for name in sorted(A.PRIVATE_FILES):
        url = '/' + name
        if url in files or any(url.startswith(d) for d in dirs):
            bad.append(f'{url} is in dashboard.py\'s static allow-list: the '
                       f'accounts database is downloadable')


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
                    # frozenset({...}) is a Call, not a literal
                    try:
                        if (isinstance(node.value, ast.Call)
                                and getattr(node.value.func, 'id', '')
                                in ('frozenset', 'set', 'tuple', 'list')):
                            return ast.literal_eval(node.value.args[0])
                    except (ValueError, IndexError):
                        return None
                    return None
    return None


# ── the schema ──────────────────────────────────────────────────────────────

def migrate_checks(bad, fx):
    """migrate() twice is a no-op, and it carries an existing database
    forward rather than starting again.

    The failure this names is the one that eats the users table: a migration
    that runs unconditionally, drops and recreates, and takes every account
    with it the first time the service restarts.
    """
    p = fx.fresh('migrate.db')
    con = A.connect(p)
    try:
        if A.schema_version(con) != A.SCHEMA_VERSION:
            bad.append(f'a new database stamps version '
                       f'{A.schema_version(con)}, not {A.SCHEMA_VERSION}')
        rows = list(con.execute('SELECT version FROM schema_version'))
        if [r['version'] for r in rows] != [A.SCHEMA_VERSION]:
            bad.append(f'schema_version holds {[r["version"] for r in rows]}, '
                       f'expected exactly [{A.SCHEMA_VERSION}]')
    finally:
        con.close()

    A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    before = A.list_users(p)
    con = A.connect(p)
    try:
        A.migrate(con)
        A.migrate(con)
        rows = [r['version'] for r in
                con.execute('SELECT version FROM schema_version')]
        if rows != [A.SCHEMA_VERSION]:
            bad.append(f'running migrate() again added rows: {rows}')
    finally:
        con.close()
    if A.list_users(p) != before:
        bad.append('migrate() run again changed the users table; it is '
                   'supposed to be a no-op')

    # The carry-forward case: a database whose stamp was lost (an older build,
    # or a restore that dropped the header). Nothing may be destroyed.
    con = sqlite3.connect(p)
    con.execute('PRAGMA user_version = 0')
    con.commit()
    con.close()
    con = A.connect(p)
    try:
        if A.schema_version(con) != A.SCHEMA_VERSION:
            bad.append('a database with an unstamped header did not come '
                       'forward to the current version')
    finally:
        con.close()
    if A.list_users(p) != before:
        bad.append('carrying an unstamped database forward changed the users '
                   'table -- the accounts were rebuilt, not migrated')


# ── passwords ───────────────────────────────────────────────────────────────

def hash_checks(bad, fx):
    """Salted, self-describing, verifying, and upgradeable."""
    h1, h2 = A.hash_password(PW), A.hash_password(PW)
    if h1 == h2:
        bad.append('two hashes of the same password are identical, so the '
                   'salt is shared or missing and one crack breaks everyone')
    if not A.verify_hash(h1, PW):
        bad.append('a password does not verify against its own hash')
    if A.verify_hash(h1, PW2):
        bad.append('the wrong password verifies')
    if PW in h1:
        bad.append('the password appears verbatim inside its own hash')

    try:
        algo, params, salt, dk = A._parse_hash(h1)
    except Exception as e:
        bad.append(f'a fresh hash does not parse: {type(e).__name__}: {e}')
        return
    if len(salt) < A.SALT_BYTES or len(dk) < A.DK_BYTES:
        bad.append(f'salt/digest are {len(salt)}/{len(dk)} bytes, wanted '
                   f'{A.SALT_BYTES}/{A.DK_BYTES}')
    if not params:
        bad.append('the encoding carries no parameters, so the cost can never '
                   'be raised without invalidating every existing login')
    if A.needs_rehash(h1):
        bad.append('a hash made right now already needs rehashing')

    # A hash at yesterday's settings must still verify AND be flagged.
    weak = A._encode('pbkdf2_sha256', {'i': 1000}, b'0' * A.SALT_BYTES,
                     A._derive('pbkdf2_sha256', {'i': 1000}, PW,
                               b'0' * A.SALT_BYTES))
    if not A.verify_hash(weak, PW):
        bad.append('a hash at older parameters no longer verifies -- raising '
                   'the cost would lock everybody out')
    if not A.needs_rehash(weak):
        bad.append('a hash at weaker parameters is not flagged for upgrade')
    if not A.needs_rehash('not a hash at all'):
        bad.append('an unreadable hash is not flagged for upgrade')
    if A.verify_hash('not a hash at all', PW):
        bad.append('an unreadable hash verifies')

    # One flipped byte in the digest must fail. This is the check that dies if
    # the comparison ever becomes a substring or a prefix test.
    algo, params, salt, dk = A._parse_hash(h1)
    flipped = bytes([dk[0] ^ 0xFF]) + dk[1:]
    if A.verify_hash(A._encode(algo, params, salt, flipped), PW):
        bad.append('a digest with a corrupted first byte still verifies')


def rehash_on_login_checks(bad, fx):
    """A weak hash is replaced the next time its owner signs in -- and the
    upgrade does not sign them out of everything."""
    p = fx.fresh('rehash.db')
    u = A.create_user('mallory', PW, path=p)
    weak = A._encode('pbkdf2_sha256', {'i': 1000}, b'1' * A.SALT_BYTES,
                     A._derive('pbkdf2_sha256', {'i': 1000}, PW,
                               b'1' * A.SALT_BYTES))
    con = A.connect(p)
    with A._tx(con):
        con.execute('UPDATE users SET pw_hash = ? WHERE id = ?', (weak, u['id']))
    con.close()

    if A.verify_password('mallory', PW, path=p) is None:
        bad.append('an account whose hash is at older parameters cannot log '
                   'in at all')
        return
    con = A.connect(p)
    now = con.execute('SELECT pw_hash, session_epoch FROM users WHERE id = ?',
                      (u['id'],)).fetchone()
    con.close()
    if now['pw_hash'] == weak:
        bad.append('a weak hash survived a successful login: needs_rehash() '
                   'is never acted on, so raising the cost never reaches '
                   'anybody who already has an account')
    if A.needs_rehash(now['pw_hash']):
        bad.append('the hash written by the upgrade still needs rehashing')
    if now['session_epoch'] != u['session_epoch']:
        bad.append('an automatic rehash signed the user out of every device; '
                   'it is the same password')
    if A.verify_password('mallory', PW, path=p) is None:
        bad.append('the account cannot log in again after its hash was '
                   'upgraded')


def timing_checks(bad, fx):
    """A miss costs what a wrong password costs.

    An early return for "no such user" answers in microseconds while a real
    account costs a full derivation, and the difference is readable over a
    LAN -- which turns the login form into a directory of who has an account
    here. Same for a disabled account, which must not be distinguishable from
    a live one with the wrong password.
    """
    p = fx.fresh('timing.db')
    A.create_user('present', PW, path=p)
    A.create_user('retired', PW, path=p, active=False)

    def cost(user, pw, n=5):
        got = []
        for _ in range(n):
            t = time.perf_counter()
            A.verify_password(user, pw, path=p, touch=False)
            got.append(time.perf_counter() - t)
        return sorted(got)[n // 2]

    wrong = cost('present', PW2)
    if wrong < 0.002:
        bad.append(f'a wrong password answers in {wrong * 1e3:.2f}ms, which is '
                   f'too fast for any password hash worth having')
        return
    for label, user, pw in (('an unknown username', 'ghost', PW2),
                            ('a disabled account', 'retired', PW),
                            ('an unusable username', '!!!!', PW2)):
        got = cost(user, pw)
        ratio = got / wrong
        if not 0.5 <= ratio <= 2.0:
            bad.append(
                f'{label} answers in {got * 1e3:.1f}ms against '
                f'{wrong * 1e3:.1f}ms for a wrong password ({ratio:.2f}x): '
                f'the login form tells you who exists')


# ── validation ──────────────────────────────────────────────────────────────

def validation_checks(bad, fx):
    """Each refusal is typed, and says what is wrong with the INPUT."""
    cases = [
        ('username_length', 'a', PW),
        ('username_length', 'z' * (A.USERNAME_MAX + 1), PW),
        ('username_charset', 'ali ce', PW),
        ('username_charset', 'alice<script>', PW),
        ('username_charset', '-alice', PW),
        ('username_charset', 'ali/ce', PW),
        ('username_missing', '   ', PW),
        ('password_short', 'alice', 'x' * (A.PASSWORD_MIN - 1)),
        ('password_long', 'alice', 'x' * (A.PASSWORD_MAX + 1)),
    ]
    p = fx.fresh('valid.db')
    for want, user, pw in cases:
        try:
            A.create_user(user, pw, path=p)
            bad.append(f'create_user({user!r}) was accepted; expected '
                       f'{want}')
        except A.AccountError as e:
            if e.code != want:
                bad.append(f'create_user({user!r}) refused with {e.code!r}, '
                           f'expected {want!r}')
            if not e.message or not str(e).strip():
                bad.append(f'{e.code} carries no message a person can read')
        except Exception as e:
            bad.append(f'create_user({user!r}) raised {type(e).__name__}, '
                       f'not an AccountError -- the server cannot tell a bad '
                       f'form from a broken store: {e}')
    try:
        A.set_role('alice', 'root', path=p)
        bad.append('set_role accepted a role that is not in ROLES')
    except A.AccountError as e:
        if e.code != 'role_unknown':
            bad.append(f'set_role refused with {e.code!r}, expected '
                       f'role_unknown')

    # Case and whitespace are one account, not several.
    A.create_user('  Alice  ', PW, path=p)
    try:
        A.create_user('ALICE', PW, path=p)
        bad.append('ALICE and Alice became two accounts: the store compares '
                   'usernames case-sensitively')
    except A.AccountError as e:
        if e.code != 'username_taken':
            bad.append(f'a duplicate username refused with {e.code!r}')
    if A.get_user('alice', path=p) is None:
        bad.append('an account created as "  Alice  " cannot be found as '
                   '"alice"')
    if A.verify_password(' ALICE ', PW, path=p) is None:
        bad.append('an account cannot log in under a different case of its '
                   'own name')


def leak_checks(bad, fx):
    """No password hash ever leaves this module in a returned row."""
    p = fx.fresh('leak.db')
    A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    u = A.create_user('bob', PW, path=p)
    inv = A.create_invite('admin', ttl=600, path=p)
    seen = [u, A.get_user('bob', path=p), A.verify_password('bob', PW, path=p),
            A.set_password('bob', PW2, path=p), A.set_active('bob', True,
                                                             path=p),
            A.set_role('bob', 'member', path=p),
            A.bump_session_epoch('bob', path=p)]
    seen += A.list_users(p)
    for d in seen:
        if d and 'pw_hash' in d:
            bad.append('a user row came back with pw_hash in it; one page '
                       'that prints a user prints a hash')
            break
    for d in [inv] + A.list_invites(p):
        if 'token_hash' in d:
            bad.append('an invite row came back with token_hash in it, which '
                       'is the lookup key for a live link')
            break


# ── users ───────────────────────────────────────────────────────────────────

def user_checks(bad, fx):
    """Roles, activation, and the session epoch that revokes a cookie."""
    p = fx.fresh('users.db')
    A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    bob = A.create_user('bob', PW, path=p)
    if bob['role'] != 'member' or not bob['active']:
        bad.append(f'a new account defaults to {bob["role"]}/'
                   f'active={bob["active"]}, not member/active')

    e0 = bob['session_epoch']
    if A.set_password('bob', PW2, path=p)['session_epoch'] <= e0:
        bad.append('changing a password left every existing session valid, '
                   'which is the one thing changing it is for')
    if A.verify_password('bob', PW, path=p) is not None:
        bad.append('the OLD password still works after set_password')
    if A.verify_password('bob', PW2, path=p) is None:
        bad.append('the new password does not work after set_password')

    e1 = A.get_user('bob', path=p)['session_epoch']
    if A.bump_session_epoch('bob', path=p)['session_epoch'] <= e1:
        bad.append('bump_session_epoch did not move the epoch, so "sign out '
                   'everywhere" signs nobody out')

    e2 = A.get_user('bob', path=p)['session_epoch']
    off = A.set_active('bob', False, path=p)
    if off['active']:
        bad.append('set_active(False) left the account active')
    if off['session_epoch'] <= e2:
        bad.append('disabling an account left its live sessions working, so '
                   'the button does not do what it says')
    if A.verify_password('bob', PW2, path=p) is not None:
        bad.append('a disabled account can still log in')
    if A.verify_password('bob', PW2, path=p, touch=False) is not None:
        bad.append('a disabled account passes verification without touch')
    A.set_active('bob', True, path=p)

    # The lockout guard, in both directions.
    for call, what in ((lambda: A.set_role('admin', 'member', path=p),
                        'demote the last admin'),
                       (lambda: A.set_active('admin', False, path=p),
                        'disable the last admin')):
        try:
            call()
            bad.append(f'it was possible to {what}, which leaves the '
                       f'dashboard with no way back in')
        except A.AccountError as e:
            if e.code != 'last_admin':
                bad.append(f'refusing to {what} used code {e.code!r}')
    # With a second admin it must be allowed again, or the guard is a wall.
    A.set_role('bob', 'admin', path=p)
    try:
        A.set_role('admin', 'member', path=p)
    except A.AccountError as e:
        bad.append(f'demoting an admin was refused ({e.code}) even though a '
                   f'second active admin exists')
    A.set_role('admin', 'admin', path=p)

    if A.get_user(bob['id'], path=p) is None:
        bad.append('a user cannot be looked up by id, only by name')
    if A.get_user('nobody-here', path=p) is not None:
        bad.append('get_user invented an account that does not exist')
    for fn in (A.set_password, A.set_role):
        try:
            fn('nobody-here', 'member' if fn is A.set_role else PW, path=p)
            bad.append(f'{fn.__name__} silently accepted a username that does '
                       f'not exist')
        except A.AccountError as e:
            if e.code != 'no_such_user':
                bad.append(f'{fn.__name__} on a missing user used code '
                           f'{e.code!r}')


# ── invites ─────────────────────────────────────────────────────────────────

def invite_checks(bad, fx):
    """One-time, expiring, revocable, and never stored in the clear."""
    p = fx.fresh('invites.db')
    A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    now = int(time.time())

    inv = A.create_invite('admin', ttl=600, note='field team', path=p)
    token = inv['token']
    if len(token) < 32:
        bad.append(f'an invite token is {len(token)} characters; that is '
                   f'guessable')
    if len(set(A.create_invite('admin', ttl=600, path=p)['token']
               for _ in range(5))) != 5:
        bad.append('two invites got the same token')

    # The token is nowhere on disk. Read the raw file, sidecars and all.
    blob = b''
    for suffix in ('', '-wal', '-shm'):
        try:
            with open(p + suffix, 'rb') as fh:
                blob += fh.read()
        except OSError:
            pass
    if token.encode() in blob:
        bad.append('the invite token is stored verbatim: a copy of the '
                   'database is a set of working signup links')

    if A.redeem_invite(token, 'carol', PW, path=p) is None:
        bad.append('a good invite could not be redeemed')
    for code, args in (
            ('invite_used', (token, 'dave', PW)),
            ('invite_unknown', ('a-token-nobody-issued', 'dave', PW))):
        try:
            A.redeem_invite(*args, path=p)
            bad.append(f'redeeming again was accepted; expected {code}')
        except A.AccountError as e:
            if e.code != code:
                bad.append(f'redeem refused with {e.code!r}, expected {code!r}')

    # Expiry, tested by minting one in the past rather than by sleeping.
    old = A.create_invite('admin', ttl=A.INVITE_TTL_MIN, now=now - 10 ** 5,
                          path=p)
    try:
        A.redeem_invite(old['token'], 'eve', PW, path=p)
        bad.append('an expired invite was redeemed')
    except A.AccountError as e:
        if e.code != 'invite_expired':
            bad.append(f'an expired invite refused with {e.code!r}')

    gone = A.create_invite('admin', ttl=600, path=p)
    A.revoke_invite(gone['id'], path=p)
    try:
        A.redeem_invite(gone['token'], 'frank', PW, path=p)
        bad.append('a revoked invite was redeemed')
    except A.AccountError as e:
        if e.code != 'invite_revoked':
            bad.append(f'a revoked invite refused with {e.code!r}')
    if A.revoke_invite(gone['id'], path=p)['state'] != 'revoked':
        bad.append('revoking twice is not idempotent')

    used = [r for r in A.list_invites(p) if r['state'] == 'used']
    if not used or used[0]['used_by_name'] != 'carol':
        bad.append('a redeemed invite does not record who redeemed it')
    try:
        A.revoke_invite(used[0]['id'], path=p)
        bad.append('a spent invite could be "revoked", which reads on the '
                   'page as though the account it made were gone too')
    except A.AccountError as e:
        if e.code != 'invite_used':
            bad.append(f'revoking a spent invite refused with {e.code!r}')

    # Only an active admin may mint one -- checked here, not only in a route.
    for who, why in (('carol', 'a member'), ('nobody-here', 'a missing user')):
        try:
            A.create_invite(who, ttl=600, path=p)
            bad.append(f'{why} could issue an invite')
        except A.AccountError as e:
            if e.code not in ('not_admin', 'no_such_user'):
                bad.append(f'{why} issuing an invite refused with {e.code!r}')
    A.set_active('carol', False, path=p)
    A.set_role('carol', 'admin', path=p)
    A.set_active('carol', False, path=p)
    try:
        A.create_invite('carol', ttl=600, path=p)
        bad.append('a DISABLED admin could still issue invites')
    except A.AccountError as e:
        if e.code != 'not_admin':
            bad.append(f'a disabled admin refused with {e.code!r}')

    for ttl in (0, -1, A.INVITE_TTL_MAX + 1):
        try:
            A.create_invite('admin', ttl=ttl, path=p)
            bad.append(f'an invite with ttl={ttl} was accepted')
        except A.AccountError as e:
            if e.code != 'ttl_range':
                bad.append(f'ttl={ttl} refused with {e.code!r}')


def race_checks(bad, fx):
    """Two people open the same invite link at the same moment.

    This is the check the whole compare-and-set exists for, and it has to be
    run under real threads against a real database -- a mocked lock proves
    nothing about what sqlite does when two writers meet.
    """
    p = fx.fresh('race.db')
    A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    inv = A.create_invite('admin', ttl=600, path=p)
    n = 16
    barrier = threading.Barrier(n)
    lock = threading.Lock()
    wins, losses, blew_up = [], [], []

    def go(i):
        barrier.wait()
        try:
            u = A.redeem_invite(inv['token'], 'racer%02d' % i, PW, path=p)
            with lock:
                wins.append(u['username'])
        except A.AccountError as e:
            with lock:
                losses.append(e.code)
        except Exception as e:                      # noqa: BLE001 - report it
            with lock:
                blew_up.append(f'{type(e).__name__}: {e}')

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)

    if blew_up:
        bad.append(f'{len(blew_up)} of {n} racing redemptions crashed instead '
                   f'of losing cleanly: {blew_up[0]}')
    if len(wins) != 1:
        bad.append(f'{len(wins)} of {n} threads redeemed the SAME invite: an '
                   f'invite link pasted into a group chat makes one account '
                   f'per reader')
    if set(losses) - {'invite_used'}:
        bad.append(f'the losers of the race saw {sorted(set(losses))}, not '
                   f'invite_used')
    made = [u['username'] for u in A.list_users(p) if u['username'] != 'admin']
    if len(made) != 1:
        bad.append(f'the race left {len(made)} accounts behind: {made}')
    con = A.connect(p)
    try:
        row = con.execute('SELECT used_at, used_by FROM invites').fetchone()
        if not row['used_at'] or not row['used_by']:
            bad.append('the winning redemption did not mark the invite used '
                       'and stamp who took it')
    finally:
        con.close()

    # The same race one level down: two signups on one username, no invite.
    p2 = fx.fresh('race2.db')
    A.connect(p2).close()
    ok, taken = [], []
    barrier2 = threading.Barrier(n)

    def go2(_):
        barrier2.wait()
        try:
            A.create_user('samename', PW, path=p2)
            with lock:
                ok.append(1)
        except A.AccountError as e:
            with lock:
                taken.append(e.code)
        except Exception as e:                      # noqa: BLE001 - report it
            with lock:
                blew_up.append(f'{type(e).__name__}: {e}')

    threads = [threading.Thread(target=go2, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    if len(ok) != 1 or len(A.list_users(p2)) != 1:
        bad.append(f'{len(ok)} of {n} concurrent signups on one username '
                   f'succeeded; the uniqueness check is a SELECT, not the '
                   f'index')


def rollback_checks(bad, fx):
    """A redemption that fails after claiming the invite leaves it usable.

    Somebody types a username that is already taken. If the claim is not
    rolled back with the rest, they have burnt their only invite on a typo.
    """
    p = fx.fresh('rollback.db')
    A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    A.create_user('taken', PW, path=p)
    inv = A.create_invite('admin', ttl=600, path=p)
    try:
        A.redeem_invite(inv['token'], 'taken', PW, path=p)
        bad.append('an invite was redeemed onto a username that already '
                   'exists')
    except A.AccountError as e:
        if e.code != 'username_taken':
            bad.append(f'redeeming onto a taken username gave {e.code!r}')
    state = [r for r in A.list_invites(p) if r['id'] == inv['id']][0]['state']
    if state != 'open':
        bad.append(f'after a failed redemption the invite is {state!r}, not '
                   f'open: a typo spends the link')
    if A.redeem_invite(inv['token'], 'nottaken', PW, path=p) is None:
        bad.append('the invite could not be used after the failed attempt')


# ── throttling ──────────────────────────────────────────────────────────────

def throttle_checks(bad, fx):
    """Backoff, forgiveness, and a hard ceiling on a table keyed by input."""
    p = fx.fresh('throttle.db')
    now = int(time.time())
    src = 'tester'

    st = A.throttle_state(src, now=now, path=p)
    if st['fails'] or st['locked']:
        bad.append('a source nobody has seen starts out throttled')
    for i in range(A.THROTTLE_FREE):
        st = A.record_failure(src, now=now, path=p)
    if st['locked']:
        bad.append(f'locked out after {A.THROTTLE_FREE} attempts, which is '
                   f'the number that is supposed to be free -- a typo costs a '
                   f'wait')
    st = A.record_failure(src, now=now, path=p)
    if not st['locked'] or st['retry_after'] <= 0:
        bad.append('the throttle never locks: an unlimited-rate login form is '
                   'an offline attack you can run online')
    first = st['retry_after']
    st = A.record_failure(src, now=now, path=p)
    if st['retry_after'] <= first:
        bad.append('the lockout does not grow with the number of failures')
    for _ in range(30):
        st = A.record_failure(src, now=now, path=p)
    if st['retry_after'] > A.THROTTLE_MAX:
        bad.append(f'the lockout ran past its cap to {st["retry_after"]}s; a '
                   f'lockout that grows without limit is a way to lock the '
                   f'real user out for good')

    if not A.clear_failures(src, path=p):
        bad.append('clear_failures reported nothing to clear after 30 '
                   'failures')
    if A.throttle_state(src, now=now, path=p)['locked']:
        bad.append('a successful login did not clear the failure count')

    # Forgiveness: a source last seen longer ago than the window starts over.
    A.record_failure(src, now=now - 10 * A.THROTTLE_WINDOW, path=p)
    A.record_failure(src, now=now - 10 * A.THROTTLE_WINDOW, path=p)
    st = A.record_failure(src, now=now, path=p)
    if st['fails'] != 1:
        bad.append(f'a source quiet for {10 * A.THROTTLE_WINDOW}s resumed at '
                   f'{st["fails"]} failures instead of starting over')

    # The ceiling. The key is a string the caller chooses, so without a bound
    # this table is a way to fill the disk from the login form.
    con = A.connect(p)
    with A._tx(con):
        con.executemany(
            'INSERT OR REPLACE INTO throttle (source, fails, first_at, '
            'last_at, locked_until) VALUES (?,1,?,?,?)',
            [(f'flood-{i}', now, now, now + A.THROTTLE_MAX)
             for i in range(A.THROTTLE_MAX_ROWS + 500)])
    con.close()
    A.record_failure('one-more', now=now, path=p)
    con = A.connect(p)
    n = con.execute('SELECT COUNT(*) c FROM throttle').fetchone()['c']
    con.close()
    if n > A.THROTTLE_MAX_ROWS + 1:
        bad.append(f'the throttle table holds {n} rows against a cap of '
                   f'{A.THROTTLE_MAX_ROWS}: it grows without limit on input '
                   f'an attacker chooses')

    left = A.prune_throttle(now=now + 10 * A.THROTTLE_WINDOW, path=p)
    if not left:
        bad.append('prune_throttle removed nothing from a table full of '
                   'stale rows')


# ── the .env admin ──────────────────────────────────────────────────────────

def bootstrap_checks(bad, fx):
    """ensure_admin creates once, follows .env, and never strands anybody."""
    p = fx.fresh('boot.db')

    got = A.ensure_admin(path=p, env={})
    if got['action'] != 'unset' or got['ok']:
        bad.append(f'with no credential configured ensure_admin returned '
                   f'{got["action"]}/ok={got["ok"]}; the server needs "unset" '
                   f'to know to serve the explain-yourself page')
    if not got['detail'] or A.ENV_PASSWORD not in got['detail']:
        bad.append('the "no credential" message does not name the variable to '
                   'set, which is the only thing it is for')
    if A.list_users(p):
        bad.append('ensure_admin created an account with no credential '
                   'configured')

    got = A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    if got['action'] != 'created' or not got['ok']:
        bad.append(f'the first ensure_admin returned {got["action"]}')
    if len(A.list_users(p)) != 1:
        bad.append('the first ensure_admin made more than one account')
    got = A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    if got['action'] != 'unchanged':
        bad.append(f'a second ensure_admin with the same .env returned '
                   f'{got["action"]}, not unchanged')
    # Case is not a new account. This is the duplicate-admin bug.
    A.ensure_admin(path=p, env={'DASHBOARD_USER': 'ADMIN',
                                'DASHBOARD_PASSWORD': PW})
    if len(A.list_users(p)) != 1:
        bad.append('a change of case in DASHBOARD_USER created a SECOND '
                   'admin account')

    u = A.get_user('admin', path=p)
    if u['role'] != 'admin' or not u['active']:
        bad.append('the bootstrapped account is not an active admin')
    if A.verify_password('admin', PW, path=p) is None:
        bad.append('the .env admin cannot log in with the .env password')

    # A changed password follows, and takes the old sessions with it.
    e0 = A.get_user('admin', path=p)['session_epoch']
    got = A.ensure_admin(path=p, env={'DASHBOARD_USER': 'admin',
                                      'DASHBOARD_PASSWORD': PW2})
    if got['action'] != 'updated':
        bad.append(f'changing DASHBOARD_PASSWORD returned {got["action"]}, so '
                   f'editing .env does not change the password')
    if A.verify_password('admin', PW, path=p) is not None:
        bad.append('the OLD .env password still works after .env changed')
    if A.verify_password('admin', PW2, path=p) is None:
        bad.append('the new .env password does not work')
    if A.get_user('admin', path=p)['session_epoch'] <= e0:
        bad.append('changing the .env password left every live session valid')

    # Never locked out: a demoted or disabled admin comes back.
    A.create_user('other', PW, role='admin', path=p)
    A.set_role('admin', 'member', path=p)
    A.set_active('admin', False, path=p)
    got = A.ensure_admin(path=p, env={'DASHBOARD_USER': 'admin',
                                      'DASHBOARD_PASSWORD': PW2})
    u = A.get_user('admin', path=p)
    if u['role'] != 'admin' or not u['active']:
        bad.append('a demoted or disabled .env admin was NOT restored, so the '
                   'documented way back in does not work')
    if 'other' not in got['others']:
        bad.append('ensure_admin did not report the other admins it left '
                   'alone')

    # A hash in .env instead of a password.
    p2 = fx.fresh('boot2.db')
    h = A.hash_password(PW)
    got = A.ensure_admin(path=p2, env={'DASHBOARD_USER': 'admin',
                                       'DASHBOARD_PASSWORD_HASH': h})
    if got['action'] != 'created' or A.verify_password('admin', PW,
                                                       path=p2) is None:
        bad.append('DASHBOARD_PASSWORD_HASH does not produce a working admin, '
                   'so there is no way to keep the plaintext out of .env')
    if A.ensure_admin(path=p2, env={'DASHBOARD_USER': 'admin',
                                    'DASHBOARD_PASSWORD_HASH': h}
                      )['action'] != 'unchanged':
        bad.append('a second run with the same DASHBOARD_PASSWORD_HASH did '
                   'not come back unchanged')

    # Configured but unusable: report, do not create, do not crash.
    for env, why in (
            ({'DASHBOARD_USER': 'a b', 'DASHBOARD_PASSWORD': PW},
             'a username with a space'),
            ({'DASHBOARD_USER': 'admin', 'DASHBOARD_PASSWORD': 'short'},
             'a password under the minimum'),
            ({'DASHBOARD_USER': 'admin',
              'DASHBOARD_PASSWORD_HASH': 'not-a-hash'},
             'a hash that will not parse')):
        p3 = fx.fresh('boot3.db')
        try:
            got = A.ensure_admin(path=p3, env=env)
        except Exception as e:                      # noqa: BLE001 - report it
            bad.append(f'{why} in .env made ensure_admin raise '
                       f'{type(e).__name__}, which takes the dashboard down '
                       f'unattended: {e}')
            continue
        if got['action'] != 'refused' or got['ok']:
            bad.append(f'{why} in .env returned {got["action"]}/'
                       f'ok={got["ok"]}, expected a refusal')
        if A.list_users(p3):
            bad.append(f'{why} in .env still created an account')
        if not got['detail']:
            bad.append(f'{why} in .env was refused with no explanation')

    # A credential later REMOVED from .env must not disown the existing admin.
    got = A.ensure_admin(path=p, env={})
    if got['action'] != 'unset' or not got['ok']:
        bad.append('after the credential was removed from .env the existing '
                   'admin stopped counting, so the login page would claim '
                   'nobody can sign in when somebody can')


def secret_checks(bad, fx):
    """No password, hash or token ever reaches stdout or stderr by accident.

    .env on this machine holds a hundred API keys. load_env() reading it and
    saying so, even once, is how one ends up in a journal that gets pasted
    into an issue.
    """
    import io
    import contextlib
    p = fx.fresh('quiet.db')
    env = {}
    out, err = io.StringIO(), io.StringIO()
    src = os.path.join(fx.dir, 'dotenv-sample')
    with open(src, 'w') as fh:
        fh.write('# a comment\nDASHBOARD_USER=admin\n'
                 'DASHBOARD_PASSWORD="%s"\nOTHER_KEY=sk-notreal\n' % (PW,))
    # BOTH parsers, because only one of them runs on any given machine and
    # the other is the one that ships broken. sys.modules['dotenv'] = None is
    # how an import is made to fail without uninstalling anything.
    stashed = sys.modules.get('dotenv', False)
    for label in ('python-dotenv', 'the built-in parser'):
        env = {}
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            found = A.load_env(src, env=env)
        noise = out.getvalue() + err.getvalue()
        if not found or env.get('DASHBOARD_PASSWORD') != PW:
            bad.append(f'{label} did not read a .env-shaped file, so the '
                       f'credential never reaches the process at all')
        if env.get('DASHBOARD_USER') != 'admin':
            bad.append(f'{label} did not read an unquoted value')
        for needle, what in ((PW, 'the password'), ('sk-notreal', 'an API key')):
            if needle in noise:
                bad.append(f'{what} was printed while loading .env with '
                           f'{label}')
        sys.modules['dotenv'] = None            # force the fallback next time
    if stashed is False:
        sys.modules.pop('dotenv', None)
    else:
        sys.modules['dotenv'] = stashed

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        A.ensure_admin(path=p, env={'DASHBOARD_USER': 'admin',
                                    'DASHBOARD_PASSWORD': PW})
        A.create_invite('admin', ttl=600, path=p)
    noise = out.getvalue() + err.getvalue()
    if PW in noise:
        bad.append('the password was printed while bootstrapping the admin')

    # An already-set variable wins: systemd's Environment= is deliberate.
    env2 = {'DASHBOARD_PASSWORD': 'set-by-systemd-not-env'}
    A.load_env(src, env=env2)
    if env2['DASHBOARD_PASSWORD'] != 'set-by-systemd-not-env':
        bad.append('load_env overwrote a variable that was already set, so a '
                   'value passed by systemd is silently replaced by the file')


# ── the source itself ───────────────────────────────────────────────────────

def sql_checks(bad, fx):
    """Every query is parameterised.

    An f-string in a query is how a username becomes SQL, and this store takes
    usernames from a signup form. The only interpolation allowed is a PRAGMA
    built from a module constant, which cannot carry user input.
    """
    try:
        tree = ast.parse(open(ACCOUNTS_PY, encoding='utf-8').read())
    except (OSError, SyntaxError) as e:
        bad.append(f'could not parse accounts.py: {e}')
        return
    # Names that are module-level string constants -- SCHEMA is passed to
    # executescript() by name, and that is the only indirection allowed. A
    # name assigned anything else could have been built out of input.
    constants = {t.id for node in tree.body if isinstance(node, ast.Assign)
                 for t in node.targets
                 if isinstance(t, ast.Name)
                 and isinstance(node.value, ast.Constant)
                 and isinstance(node.value.value, str)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, 'attr', '') not in ('execute', 'executescript',
                                                  'executemany'):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant):
            continue
        if isinstance(arg, ast.Name) and arg.id in constants:
            continue
        # 'PRAGMA busy_timeout=%d' % (CONST,) is the one interpolation sqlite
        # forces on us: a pragma will not take a bound parameter. It is safe
        # because the head is a literal PRAGMA, which no user input reaches.
        if (isinstance(arg, ast.BinOp) and isinstance(arg.left, ast.Constant)
                and str(arg.left.value).upper().startswith('PRAGMA')):
            continue
        how = ('an f-string' if isinstance(arg, ast.JoinedStr)
               else 'a built string')
        bad.append(f'accounts.py:{node.lineno}: {how} is being executed as '
                   f'SQL; a username off a signup form becomes SQL that way')


def import_checks(bad, fx):
    """The surface the server imports is all still there."""
    for name in ('DB_PATH', 'PRIVATE_FILES', 'SCHEMA_VERSION', 'ROLES'):
        if not hasattr(A, name):
            bad.append(f'accounts.{name} is gone; the server reads it')
    wanted = ('connect', 'migrate', 'create_user', 'verify_password',
              'set_password', 'list_users', 'set_active', 'set_role',
              'bump_session_epoch', 'create_invite', 'redeem_invite',
              'list_invites', 'revoke_invite', 'throttle_state',
              'record_failure', 'reserve_attempt', 'clear_failures',
              'ensure_admin',
              'admin_configured', 'load_env', 'needs_rehash', 'hash_password',
              'verify_hash')
    for name in wanted:
        if not callable(getattr(A, name, None)):
            bad.append(f'accounts.{name}() is gone; the gate calls it')
    if A.admin_configured({}) or not A.admin_configured(
            {'DASHBOARD_PASSWORD': PW}):
        bad.append('admin_configured() does not answer "is there a credential '
                   'in the environment", which is what decides whether the '
                   'server serves data at all')


def main():
    bad = []
    try:
        with Fixture() as fx:
            for fn in (file_checks, pragma_checks, index_checks,
                       private_file_checks, migrate_checks, hash_checks,
                       rehash_on_login_checks, timing_checks,
                       validation_checks, leak_checks, user_checks,
                       invite_checks, race_checks, rollback_checks,
                       throttle_checks, bootstrap_checks, secret_checks,
                       sql_checks, import_checks):
                try:
                    fn(bad, fx)
                except Exception as e:      # noqa: BLE001 - report, not die
                    bad.append(f'{fn.__name__} threw {type(e).__name__}: {e}')
    except Exception as e:
        bad.append(f'the fixture would not build: {type(e).__name__}: {e}')
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('the accounts database is 0600 and unserved, an invite is redeemable '
          'exactly once under sixteen threads, a miss costs what a wrong '
          'password costs, and the .env admin is one row however often it is '
          'bootstrapped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
