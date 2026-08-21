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

def _make_it_v2(path):
    """Turn a current store back into a v2-shaped one: the role CHECK
    narrowed to what v2 allowed, and the header lost the way a .dump and
    reload loses it. sqlite_sequence is put back to where it stood, because
    the copy this does resets it exactly as the migration would."""
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    cols = [r['name'] for r in con.execute('PRAGMA table_info(users)')]
    high = con.execute("SELECT seq FROM sqlite_sequence WHERE name = 'users'"
                       ).fetchone()['seq']
    sql = con.execute("SELECT sql FROM sqlite_master WHERE name = 'users'"
                      ).fetchone()['sql']
    old = sql.replace("'owner', 'admin'", "'admin'").replace(
        'CREATE TABLE users', 'CREATE TABLE users_v2', 1)
    assert "'owner'" not in old, 'the fixture did not narrow the CHECK'
    pick = ','.join("CASE role WHEN 'owner' THEN 'admin' ELSE role END"
                    if c == 'role' else c for c in cols)
    con.executescript(
        'PRAGMA foreign_keys = OFF; ' + old +
        '; INSERT INTO users_v2 (%s) SELECT %s FROM users;'
        % (','.join(cols), pick) +
        ' DROP TABLE users; ALTER TABLE users_v2 RENAME TO users;'
        " UPDATE sqlite_sequence SET seq = %d WHERE name = 'users';" % (high,) +
        ' PRAGMA user_version = 0;')
    con.commit()
    con.close()


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

    # THE SHAPE DECIDES, NOT THE STAMP. The case above kept a current table
    # and only lost the header. The one that bites is a v2 TABLE with a lost
    # header: `if have and have < 3` reads that 0 as "nothing to do", stamps
    # it v3, and leaves a CHECK that still refuses the word 'owner' -- after
    # which ensure_admin cannot write the tier and the dashboard never starts.
    q = fx.fresh('lostheader.db')
    A.ensure_admin(path=q, env=dict(ADMIN_ENV))
    A.create_user('volunteer', PW, path=q)
    A.create_user('secondvol', PW, path=q)
    A.delete_user('secondvol', path=q)      # id retired; high-water must stay
    kept = len(A.list_users(q))
    _make_it_v2(q)
    try:
        con = A.connect(q)
    except A.AccountError as e:
        bad.append('a v2 database with a lost header would not open: %s'
                   % (e.code,))
        return
    try:
        sql = con.execute("SELECT sql FROM sqlite_master WHERE name = 'users'"
                          ).fetchone()['sql']
        if "'owner'" not in sql:
            bad.append('a v2 users table with a lost header was stamped v3 '
                       'without widening the role CHECK, so the owner tier '
                       'can never be written and the service cannot start')
        seq = con.execute("SELECT seq FROM sqlite_sequence "
                          "WHERE name = 'users'").fetchone()
        if seq and seq['seq'] < 3:
            bad.append('the rebuild reset the AUTOINCREMENT high-water to '
                       '%d, so a removed account\'s id goes to the next '
                       'person invited -- and an id is what a session names'
                       % (seq['seq'],))
    finally:
        con.close()
    if len(A.list_users(q)) != kept:
        bad.append('the v2 carry-forward lost accounts')
    try:
        A.ensure_admin(path=q, env=dict(ADMIN_ENV))
    except Exception as e:                       # noqa: BLE001
        bad.append('the .env owner could not be restored after the v2 '
                   'carry-forward: %s: %s' % (type(e).__name__, e))
        return
    if A.get_user(ADMIN_ENV['DASHBOARD_USER'], path=q)['role'] != 'owner':
        bad.append('the .env account did not come back as the owner after a '
                   'v2 database was carried forward')


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

def owner_tier_checks(bad, fx):
    """The top tier is held by the .env, not by anybody who reaches the page.

    A role was inserted ABOVE admin, and the whole point of it is that exactly
    one account has it. Every route that writes a role had to learn that, or
    an admin could simply take it: promote themselves, then demote or delete
    the person whose machine this is. All four of those worked when the tier
    was first added.
    """
    p = fx.fresh('ownertier.db')
    A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    who = ADMIN_ENV['DASHBOARD_USER']
    A.create_user('adm', 'correct-horse-battery-staple', role='admin', path=p)

    owner = A.get_user(who, path=p)
    if owner['role'] != 'owner':
        bad.append('the .env account is %r, so this check grades nothing'
                   % (owner['role'],))
        return
    # NOBODY IS PROMOTED INTO IT
    for target in ('adm', who):
        try:
            A.set_role(target, 'owner', path=p)
            bad.append('%r was promoted to owner from the account store; an '
                       'admin can take the top tier' % (target,))
        except A.AccountError as e:
            if e.code != 'owner_not_grantable':
                bad.append('promoting to owner was refused as %r' % (e.code,))
    # NOR OUT OF IT
    for role in ('admin', 'member'):
        try:
            A.set_role(who, role, path=p)
            bad.append('the owner was demoted to %r, which is how an admin '
                       'strips the person whose machine this is' % (role,))
        except A.AccountError as e:
            if e.code != 'owner_not_demotable':
                bad.append('demoting the owner was refused as %r' % (e.code,))
    # NOR REMOVED
    try:
        A.delete_user(who, path=p)
        bad.append('the owner account was deleted outright')
    except A.AccountError as e:
        if e.code != 'owner_not_removable':
            bad.append('deleting the owner was refused as %r' % (e.code,))
    # NOR MINTED BY AN INVITE
    try:
        A.create_invite(A.get_user('adm', path=p)['id'], role='owner', path=p)
        bad.append('an invite was issued granting the owner tier')
    except A.AccountError as e:
        if e.code != 'owner_not_invitable':
            bad.append('an owner invite was refused as %r' % (e.code,))
    # NOR WRITTEN STRAIGHT IN. set_role, delete_user and create_invite were
    # each taught the tier; create_user and redeem_invite wrote whatever role
    # they were handed, which is three guarded doors and two open ones.
    try:
        A.create_user('sneak', 'correct-horse-battery-staple', role='owner',
                      path=p)
        bad.append('create_user wrote the owner tier straight into the table')
    except A.AccountError as e:
        if e.code != 'owner_not_grantable':
            bad.append('create_user(role=owner) was refused as %r' % (e.code,))
    got = A.create_invite(A.get_user('adm', path=p)['id'], role='admin',
                          path=p)
    con = sqlite3.connect(p)
    con.execute("UPDATE invites SET role = 'owner' WHERE id = ?",
                (got['id'],))
    con.commit()
    con.close()
    try:
        A.redeem_invite(got['token'], 'sneaky', 'correct-horse-battery-staple',
                        path=p)
        bad.append('an invite row saying owner was redeemed into an owner '
                   'account')
    except A.AccountError as e:
        if e.code != 'invite_owner':
            bad.append('redeeming an owner invite was refused as %r'
                       % (e.code,))
    # ...and the ordinary promotions still work
    try:
        A.set_role('adm', 'member', path=p)
        A.set_role('adm', 'admin', path=p)
    except A.AccountError as e:
        bad.append('an ordinary promotion broke: %s' % (e.code,))
    if A.get_user(who, path=p)['role'] != 'owner':
        bad.append('the owner did not survive this check')
    # NOR DISABLED -- the fourth verb, and the one that was open. Demoting
    # and removing the owner were refused; disabling reached the same end,
    # because _would_strand only fires when NO active admin would be left and
    # the admin doing the clicking is one.
    solo = fx.fresh('solo.db')
    A.ensure_admin(path=solo, env=dict(ADMIN_ENV))
    try:
        A.set_active(who, False, path=solo)
        bad.append('the only admin -- the owner -- was disabled, leaving no '
                   'way back into the dashboard')
    except A.AccountError as e:
        if e.code != 'owner_not_disablable':
            bad.append('disabling the only owner was refused as %r'
                       % (e.code,))
    # WITH A SECOND ADMIN PRESENT, which is the case the last-admin rule does
    # not cover and the one an annotation project is actually in: an admin
    # the owner invited disables them, and the owner's next request lands on
    # the login page until somebody restarts the service.
    A.create_user('deputy', 'correct-horse-battery-staple', role='admin',
                  path=solo)
    try:
        A.set_active(who, False, path=solo)
        bad.append('an admin disabled the owner, locking the person whose '
                   'machine this is out of their own dashboard')
    except A.AccountError as e:
        if e.code != 'owner_not_disablable':
            bad.append('disabling the owner beside a second admin was '
                       'refused as %r' % (e.code,))
    if not A.get_user(who, path=solo)['active']:
        bad.append('the owner ended this check disabled')
    # ...and an ordinary admin is still disablable, or the rule is a wall
    try:
        A.set_active('deputy', False, path=solo)
    except A.AccountError as e:
        bad.append('disabling an ordinary admin was refused (%s)' % (e.code,))


def removal_checks(bad, fx):
    """Removing an account, by the route accounts are actually made.

    A volunteer joins by REDEEMING an invite, which leaves a row pointing at
    them -- so the ordinary case is the one that broke: remove the person you
    invited and the foreign key refuses, with the store reporting an
    IntegrityError the page cannot explain.

    What survives is what still means something: the line about a redeemed
    invite keeps its timing and its role and loses only a pointer to an
    account that is gone; an invite this person ISSUED is a link nobody can
    use now.
    """
    p = fx.fresh('removal.db')
    A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    owner = A.get_user(ADMIN_ENV['DASHBOARD_USER'], path=p)
    inv = A.create_invite(owner['id'], path=p)
    A.redeem_invite(inv['token'], 'vol', 'correct-horse-battery-staple',
                    path=p)
    A.set_role('vol', 'admin', path=p)
    inv2 = A.create_invite(A.get_user('vol', path=p)['id'], path=p)
    A.redeem_invite(inv2['token'], 'vol2', 'tumbling-dice-in-june', path=p)
    # a THIRD one, issued by the same admin and never taken: the two have to
    # be told apart, and only one of them is a link that leads anywhere.
    A.create_invite(A.get_user('vol', path=p)['id'], note='still open',
                    path=p)
    before = len(A.list_invites(path=p))
    vol_id = A.get_user('vol', path=p)['id']
    try:
        A.delete_user('vol', path=p)
    except Exception as e:                # noqa: BLE001
        bad.append('removing an account that joined by invite failed with '
                   '%s -- which is every account: they are all made that way'
                   % (type(e).__name__,))
        return
    if A.get_user('vol', path=p) is not None:
        bad.append('the account is still there after being removed')
    if A.get_user('vol2', path=p) is None:
        bad.append('removing an admin took the person they invited with them')
    # THE LIVE LINK GOES, THE RECORD STAYS. An invite nobody took leads
    # nowhere once its author is gone; an invite somebody DID take is the
    # record of how a member who still has an account got in, and deleting
    # every row this admin ever issued emptied the invites page on a
    # deployment where one admin invited everybody.
    left = A.list_invites(path=p)
    if len(left) != before - 1:
        bad.append('removing the admin left %d invite(s) of %d, expected the '
                   'open one to go and the used ones to stay'
                   % (len(left), before))
    if any((i.get('note') or '') == 'still open' for i in left):
        bad.append('the open invite the removed admin issued is still on the '
                   'page, pointing at nobody')
    con = A.connect(p)
    try:
        if con.execute('PRAGMA foreign_key_check').fetchall():
            bad.append('removing an account left a dangling reference')
        rows = con.execute('SELECT used_at, used_by, created_by FROM '
                           'invites').fetchall()
        if len(rows) != 2:
            bad.append('the two redeemed invites did not both survive: %d'
                       % (len(rows),))
        for r in rows:
            if not r['used_at']:
                bad.append('an invite that was redeemed lost the fact it was '
                           'used')
            if r['created_by'] == vol_id:
                bad.append('an invite still names the account that is gone as '
                           'its author')
        gone = [r for r in rows if r['used_by'] is None]
        if not gone:
            bad.append('the invite the removed account redeemed still points '
                       'at it')
    finally:
        con.close()

    # AND THE WORK THEY HANDED OUT. assignments.created_by is NOT NULL and
    # points at users, so an admin who had ever given somebody a target could
    # not be removed at all: the foreign key refused and the page said
    # "IntegrityError" with nothing else in it. What must NOT happen instead
    # is the volunteer's job disappearing with the admin who set it.
    boss = A.create_user('boss', 'correct-horse-battery-staple', role='admin',
                         path=p)
    hand = A.create_user('hand', 'tumbling-dice-in-june', path=p)
    made = A.create_assignment('hand', '250', created_by=boss['id'], path=p)
    if not made['ok']:
        bad.append('the fixture could not hand out a target (%s), so this '
                   'check proves nothing' % (made['message'],))
        return
    try:
        A.delete_user('boss', inherit_to=owner['id'], path=p)
    except Exception as e:                # noqa: BLE001
        bad.append('an admin who had handed somebody a target could not be '
                   'removed: %s: %s' % (type(e).__name__, str(e)[:80]))
        return
    # ...and an heir named in some other alphabet gets the default one rather
    # than a ValueError out of a delete that half happened.
    A.create_user('third', 'rolling-stones-gather', role='admin', path=p)
    A.create_assignment('hand', '120', surface='gate',
                        created_by=A.get_user('third', path=p)['id'], path=p)
    try:
        A.delete_user('third', inherit_to='not-a-number', path=p)
    except A.AccountError as e:
        bad.append('removing an account with an unusable heir was refused '
                   '(%s) rather than falling back to the owner' % (e.code,))
    except Exception as e:                # noqa: BLE001
        bad.append('an heir that is not a number raised %s out of a delete '
                   'that had already started' % (type(e).__name__,))
    con = A.connect(p)
    try:
        rows = con.execute('SELECT user_id, created_by, target FROM '
                           'assignments WHERE target = 250').fetchall()
        if len(rows) != 1:
            bad.append('removing the admin who set a target took the '
                       "volunteer's job with it")
        elif rows[0]['user_id'] != hand['id'] or rows[0]['target'] != 250:
            bad.append('the job that survived is not the one that was set')
        elif rows[0]['created_by'] == boss['id']:
            bad.append('the job still names an account that no longer exists')
        if con.execute('PRAGMA foreign_key_check').fetchall():
            bad.append('removing a delegating admin left a dangling '
                       'reference')
    finally:
        con.close()


def strength_checks(bad, fx):
    """What a password has to be, now that the accounts belong to volunteers.

    Length first, and then the small set of shapes that clear a length rule
    while clearing no guessing at all. NOT composition rules: demanding a
    capital and a symbol produces `Password1!`, which is one guess wearing a
    costume, and it is why the passphrases below have to keep working.
    """
    # THE DECORATION IS THE VARIABLE, not the guess. A policy that demands a
    # capital, a digit and a symbol produces `P@ssw0rd1234` -- which is the
    # single most-guessed string there is, wearing exactly what the policy
    # asked for. Each of these is one guess in a costume.
    for pw in ('P@ssw0rd1234', 'p4ssw0rd1234', 'L3tm31n12345', 'M0nk3y123456',
               '$unshine12345', 'tru5tn01-2026', 'ch4ngem3-2026',
               'Dr@g0n!!12345', 'welcome-2026!', 'monkey123456'):
        try:
            A.check_password(pw, username='alice')
            bad.append('%r got through: it is a word off the guessing list '
                       'with the decoration a policy demanded' % (pw,))
        except A.AccountError as e:
            if e.code != 'password_common':
                bad.append('%r was refused as %r, not as a common password'
                           % (pw, e.code))
    # ...and a real password that merely CONTAINS such a word is not one
    for pw in ('masterpiece of cake', 'testing the waters daily',
               'rootless canal work', 'winter-harbour-glass-9'):
        try:
            A.check_password(pw, username='alice')
        except A.AccountError as e:
            bad.append('%r was refused (%s) -- a common word inside a real '
                       'passphrase is not the passphrase' % (pw, e.code))
    weak = [
        ('password1234', 'password_common'),
        ('letmein12345', 'password_common'),
        ('iloveyou1234', 'password_common'),
        ('aaaaaaaaaaaa', 'password_repetitive'),
        ('abababababab', 'password_repetitive'),
        ('123456789012', 'password_sequence'),
        ('987654321098', 'password_sequence'),
        ('abcdefghijkl', 'password_sequence'),
        ('qwertyuiop12', None),        # a keyboard row, however it is spelled
        ('asdfghjkl123', None),
        ('short', 'password_short'),
    ]
    for pw, code in weak:
        try:
            A.check_password(pw, username='alice')
            bad.append('a guesser gets in with %r on the first try' % (pw,))
        except A.AccountError as e:
            if code and e.code != code:
                bad.append('%r was refused as %r, want %r'
                           % (pw, e.code, code))
    # the account's own name, and the project's, are things a guesser knows
    for pw in ('alice-alice-alice', 'dashboard-is-fun-2', 'biodiv-forever-1'):
        try:
            A.check_password(pw, username='alice')
            bad.append('%r contains something the guesser already has'
                       % (pw,))
        except A.AccountError as e:
            if e.code not in ('password_guessable', 'password_repetitive'):
                bad.append('%r refused as %r' % (pw, e.code))
    # ...and a real one still goes through, passphrases included
    for pw in ('correct-horse-battery', 'tumbling dice in june',
               'xR4$mv9Qwe1z', 'my cat ate the router', 'ZebraCanoe19Fig'):
        try:
            A.check_password(pw, username='alice')
        except A.AccountError as e:
            bad.append('a strong password was refused: %r (%s)'
                       % (pw, e.code))
    # the rule reaches the places accounts are actually made
    p = fx.fresh('strength.db')
    A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    try:
        A.create_user('bob', 'password1234', path=p)
        bad.append('create_user took a password off the guessing list')
    except A.AccountError:
        pass
    try:
        A.create_user('carol', 'carol-carol-carol', path=p)
        bad.append('create_user took a password that is the username')
    except A.AccountError:
        pass


def unlock_checks(bad, fx):
    """There is a way out of a lockout that is not a successful login.

    Every other exit from the throttle is a login that works, and a lockout
    is exactly the refusal of one -- so an operator locked out of their own
    dashboard, or a volunteer locked out by somebody else's guessing, had
    sqlite on a live database as the only remedy.
    """
    import subprocess as _sp
    p = fx.fresh('unlock.db')
    A.ensure_admin(path=p, env=dict(ADMIN_ENV))
    for _ in range(9):
        A.record_failure('ip:203.0.113.5', path=p)
    A.record_failure('ip:198.51.100.2', path=p)
    if not A.throttle_state('ip:203.0.113.5', path=p).get('retry_after'):
        bad.append('nine failures did not lock the source, so this check '
                   'proves nothing')
        return
    here = os.path.join(REPO, 'tools', 'dashboard', 'accounts.py')
    got = _sp.run([sys.executable, here, '--db', p, '--unlock',
                   '203.0.113.5'], capture_output=True, text=True)
    if got.returncode:
        bad.append('--unlock failed: %s' % ((got.stderr or '')[:160],))
    if A.throttle_state('ip:203.0.113.5', path=p).get('retry_after'):
        bad.append('--unlock left the source locked, so the only way back in '
                   'is still a login the lockout is refusing')
    # ...and it clears the named one only
    if not A.throttle_state('ip:198.51.100.2', path=p).get('fails'):
        bad.append('--unlock on one source cleared another one too')
    # AN IPv6 ADDRESS IS AN ADDRESS. The prefix was guessed from "is there a
    # colon in it", which every IPv6 address answers yes to -- so the key
    # went in unprefixed, matched nothing, and printed the same sentence a
    # successful clear prints. The operator reads "cleared" and is still out.
    for _ in range(9):
        A.record_failure('ip:2001:db8::1', path=p)
    if not A.throttle_state('ip:2001:db8::1', path=p).get('retry_after'):
        bad.append('the v6 source did not lock, so this proves nothing')
    got = _sp.run([sys.executable, here, '--db', p, '--unlock',
                   '2001:db8::1'], capture_output=True, text=True)
    if A.throttle_state('ip:2001:db8::1', path=p).get('retry_after'):
        bad.append('--unlock on an IPv6 address left it locked, and said %r'
                   % ((got.stdout or '').strip()[:80],))
    # ...and a miss says so rather than borrowing the success wording
    got = _sp.run([sys.executable, here, '--db', p, '--unlock',
                   '192.0.2.77'], capture_output=True, text=True)
    if 'cleared' in (got.stdout or '').lower():
        bad.append('--unlock on a source that is not locked reported a '
                   'clear: %r' % ((got.stdout or '').strip()[:80],))

    got = _sp.run([sys.executable, here, '--db', p, '--unlock'],
                  capture_output=True, text=True)
    if got.returncode or A.throttle_state('ip:198.51.100.2',
                                          path=p).get('fails'):
        bad.append('--unlock with no source did not clear them all: %s'
                   % ((got.stdout or got.stderr or '')[:120],))


def impersonation_checks(bad, fx):
    """Names a volunteer must not be able to take.

    This dashboard is public and its annotators mostly do not know each other.
    A sign-up called `admin` or `support` can tell the rest of them anything
    and be believed, and it needs no permission to do it -- the name is the
    whole attack. The account that runs the deployment is exempt: it may hold
    its own name, whatever DASHBOARD_USER says that is.
    """
    # a deployment whose admin is NOT called admin, which is the case the
    # reservation exists for: with DASHBOARD_USER=admin the name belongs to
    # the deployment and is rightly allowed
    env = {'DASHBOARD_USER': 'theowner', 'DASHBOARD_PASSWORD': PW}
    p = fx.fresh('impersonate.db')
    A.ensure_admin(path=p, env=dict(env))
    for name in ('admin', 'Admin', 'ADMIN', 'root', 'support', 'staff',
                 'moderator', 'system', 'official'):
        try:
            A.check_username(name, env=dict(env))
            bad.append('a volunteer may sign up as %r, which reads as the '
                       'project speaking' % (name,))
        except A.AccountError as e:
            if getattr(e, 'code', None) != 'username_reserved':
                bad.append('%r was refused for the wrong reason: %s'
                           % (name, getattr(e, 'code', e)))
    # ...and an ordinary name still works
    for name in ('alice', 'bob_2', 'a.b-c', 'Zoe99'):
        try:
            A.check_username(name, env=dict(env))
        except A.AccountError as e:
            bad.append('an ordinary username %r was refused: %s'
                       % (name, getattr(e, 'code', e)))
    # ...and the deployment's own admin may hold its own name
    who = env['DASHBOARD_USER']
    try:
        A.check_username(who, env=dict(env))
    except A.AccountError as e:
        bad.append('the deployment admin cannot hold its own name %r: %s'
                   % (who, getattr(e, 'code', e)))
    try:
        A.check_username('admin', env=dict(ADMIN_ENV))
    except A.AccountError as e:
        bad.append('a deployment whose DASHBOARD_USER really is admin cannot '
                   'hold that name: %s' % (getattr(e, 'code', e),))


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

    # The lockout guard, in both directions. On an ORDINARY admin: the .env
    # account is the owner now, and its role is immutable by design -- see
    # owner_tier_checks -- so demoting it tests the wrong refusal.
    # IN A STORE WITH NO OWNER IN IT. The .env account cannot step aside any
    # more -- it is refused a demotion AND a disable, which is the point of
    # the tier -- so a fixture holding one never reaches the refusal under
    # test, because two active admins never strand anything.
    q = fx.fresh('lastadmin.db')
    A.create_user('onlyadmin', PW, role='admin', path=q)
    for call, what in ((lambda: A.set_role('onlyadmin', 'member', path=q),
                        'demote the last admin'),
                       (lambda: A.set_active('onlyadmin', False, path=q),
                        'disable the last admin')):
        try:
            call()
            bad.append(f'it was possible to {what}, which leaves the '
                       f'dashboard with no way back in')
        except A.AccountError as e:
            if e.code != 'last_admin':
                bad.append(f'refusing to {what} used code {e.code!r}')
    # With a second admin it must be allowed again, or the guard is a wall.
    A.create_user('seconds', PW, role='admin', path=q)
    try:
        A.set_role('onlyadmin', 'member', path=q)
    except A.AccountError as e:
        bad.append(f'demoting an admin was refused ({e.code}) even though a '
                   f'second active admin exists')

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

    # ── WHEN IT RUNS OUT, MOVED WITHOUT REISSUING THE LINK ──────────────
    # The link went out in an email hours ago and the person it was for has
    # not clicked it, or clicked it a day late. Minting a second one means
    # chasing them with it and leaves the first live.
    live = A.create_invite('admin', ttl=3600, note='live', now=now, path=p)
    got = A.set_invite_expiry(live['id'], ttl=72 * 3600, now=now, path=p)
    if got['expires_at'] != now + 72 * 3600:
        bad.append('extending an open invite set %r, expected %r'
                   % (got['expires_at'], now + 72 * 3600))
    if got['state'] != 'open':
        bad.append('an extended invite is %r, not open' % (got['state'],))
    # HOURS FROM NOW, NOT FROM WHEN IT WAS ISSUED -- a link that ran out last
    # week has no live window left to add to, and measuring from created_at
    # would refuse the one case anybody wants this for.
    # OLDER THAN THE LONGEST WINDOW ALLOWED, on purpose: at 45 days a rule
    # that measured the window from the day the link was issued would refuse
    # this, and one that measures from now allows it. A fixture an hour old
    # cannot tell the two apart.
    stale = A.create_invite('admin', ttl=A.INVITE_TTL_MIN,
                            now=now - 45 * 86400, path=p)
    if A.invite_state(stale, now) != 'expired':
        bad.append('the fixture invite is not expired, so this proves nothing')
    back = A.set_invite_expiry(stale['id'], ttl=24 * 3600, now=now, path=p)
    if back['state'] != 'open':
        bad.append('an expired invite given more time is still %r'
                   % (back['state'],))
    if A.redeem_invite(stale['token'], 'grace', PW, path=p) is None:
        bad.append('an invite given more time still would not redeem, so the '
                   'button moves a number and nothing else')
    # ...and an absolute moment, for a caller that has one
    at = A.set_invite_expiry(live['id'], at=now + 2 * 86400, now=now, path=p)
    if at['expires_at'] != now + 2 * 86400:
        bad.append('setting an absolute expiry wrote %r' % (at['expires_at'],))
    for args, code, why in (
            ({'ttl': 3600, 'at': now + 3600}, 'expiry_unclear', 'both at once'),
            ({}, 'expiry_unclear', 'neither'),
            ({'ttl': 60}, 'expiry_range', 'under the floor'),
            ({'ttl': 365 * 86400}, 'expiry_range', 'a year out'),
            ({'at': now - 1}, 'expiry_range', 'into the past'),
            ({'ttl': 'soon'}, 'expiry_range', 'not a number')):
        try:
            A.set_invite_expiry(live['id'], now=now, path=p, **args)
            bad.append('an expiry %s was accepted' % (why,))
        except A.AccountError as e:
            if e.code != code:
                bad.append('an expiry %s refused with %r, expected %r'
                           % (why, e.code, code))
    # A SPENT LINK AND A WITHDRAWN ONE ARE NOT MOVED. One is redeemed and the
    # account exists; the other was withdrawn on purpose, and quietly undoing
    # that leaves the trail saying something that did not happen.
    for iid, code, why in ((used[0]['id'], 'invite_used', 'a spent invite'),
                           (gone['id'], 'invite_revoked', 'a withdrawn one'),
                           (10 ** 6, 'invite_unknown', 'an id nobody has')):
        try:
            A.set_invite_expiry(iid, ttl=3600, now=now, path=p)
            bad.append('the expiry of %s was moved' % (why,))
        except A.AccountError as e:
            if e.code != code:
                bad.append('moving the expiry of %s refused with %r, expected '
                           '%r' % (why, e.code, code))
    if A.invite_state([r for r in A.list_invites(p)
                       if r['id'] == gone['id']][0], now) != 'revoked':
        bad.append('a withdrawn invite did not stay withdrawn')

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

    # THE .env ACCOUNT IS THE OWNER, which outranks an admin rather than
    # being one: it is the person whose machine this is.
    u = A.get_user('admin', path=p)
    if u['role'] != 'owner' or not u['active']:
        bad.append('the bootstrapped account is %r/active=%s, not an active '
                   'owner' % (u['role'], u['active']))
    if not A.is_admin(u['role']):
        bad.append('the owner does not carry admin rights, so the person who '
                   'owns the machine has less than the people they invite')
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

    # Never locked out: a demoted or disabled .env account comes back.
    #
    # Written straight into the table, because set_role() now REFUSES to
    # change the owner's tier -- which is the point of that refusal. The
    # state this restores from is one the app will not produce: an older
    # database, a hand-edit, a restore from before the tier existed.
    A.create_user('other', PW, role='admin', path=p)
    con = A.connect(p)
    with A._tx(con):
        con.execute("UPDATE users SET role = 'member', active = 0 "
                    "WHERE username_norm = 'admin'")
    con.close()
    got = A.ensure_admin(path=p, env={'DASHBOARD_USER': 'admin',
                                      'DASHBOARD_PASSWORD': PW2})
    u = A.get_user('admin', path=p)
    if u['role'] != 'owner' or not u['active']:
        bad.append('a demoted or disabled .env account was NOT restored to '
                   'owner (%r/active=%s), so the documented way back in does '
                   'not work' % (u['role'], u['active']))
    if 'other' not in got['others']:
        bad.append('ensure_admin did not report the other admins it left '
                   'alone')

    # EXACTLY ONE OWNER, AND IT IS THE ONE .env NAMES. Pointing DASHBOARD_USER
    # at somebody else promotes them -- and left the old account holding a
    # tier that set_role refuses to demote and delete_user refuses to remove,
    # so the only way to clear it was a sqlite shell. That is the remedy the
    # refusals themselves tell an operator to use, in those words.
    got = A.ensure_admin(path=p, env={'DASHBOARD_USER': 'nextowner',
                                      'DASHBOARD_PASSWORD': PW2})
    owners = [u['username'] for u in A.list_users(p) if u['role'] == 'owner']
    if owners != ['nextowner']:
        bad.append('after DASHBOARD_USER was pointed at somebody else the '
                   'owners are %r -- an account nobody can demote or remove '
                   'still holds the tier' % (owners,))
    if 'admin' not in got.get('demoted', []):
        bad.append('ensure_admin did not report which account stopped being '
                   'the owner')
    if A.get_user('admin', path=p)['role'] != 'admin':
        bad.append('the previous owner was not left as an admin')

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
                       validation_checks, leak_checks,
                       impersonation_checks, owner_tier_checks,
                       removal_checks,
                       strength_checks,
                       unlock_checks,
                       user_checks,
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
