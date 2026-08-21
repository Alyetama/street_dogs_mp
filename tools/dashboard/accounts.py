#!/usr/bin/env python3
"""
Who is allowed to look at the dashboard, in a database and not in a config file.

    python tools/dashboard/accounts.py                   # who exists
    python tools/dashboard/accounts.py --ensure-admin    # bootstrap from .env
    python tools/dashboard/accounts.py --hash            # a hash for .env
    python tools/dashboard/accounts.py --invite --hours 48 --note "field team"
    python tools/dashboard/accounts.py --invites
    python tools/dashboard/accounts.py --revoke 3
    python tools/dashboard/accounts.py --disable alice
    python tools/dashboard/accounts.py --set-password alice

THIS MODULE IS THE STORE, NOT THE GATE. No HTTP, no HTML, no cookies: those
belong to the server, and keeping them out of here means there is exactly one
place a credential is checked and exactly one place a password is written.
Everything below is a pure function over data/dashboard/accounts.db.

WHY A DATABASE AND NOT A PAIR OF ENVIRONMENT VARIABLES. The dashboard started
with one admin read out of .env, which is fine right up to the first invited
annotator: a second account means a second secret in a file that already holds
a hundred Mapillary keys, no way to take one back, and no way to tell whether
the person who left still has the password. Accounts have to be revocable
individually and they have to be creatable without editing a file the service
reads at boot. sqlite3 is in the standard library, so this costs a clone
nothing, and the conventions are the ones tools/detect/leash_store.py already
uses -- WAL, synchronous=FULL, a connection per call.

THE ADMIN STILL COMES FROM .env, and still LIVES HERE. ensure_admin() copies
the .env credential into the users table on every start and updates it when it
changes, so the login path never has two things to check. Change
DASHBOARD_PASSWORD in .env, restart, and the old sessions die with it (the
hash change bumps session_epoch). Nothing else about the admin is special: it
is a row, it can be joined against, it can issue invites, and it appears in
the user list like anybody else.

PASSWORDS ARE NEVER STORED AND NEITHER ARE INVITE TOKENS. A password becomes
a scrypt digest with its own salt and its parameters written next to it, so
the cost can be raised later without invalidating anybody's login --
needs_rehash() upgrades a weak old hash the next time its owner signs in. An
invite becomes a SHA-256 of the token and the token itself is returned exactly
once, from create_invite(), and then it is gone: a copy of this database is
not a set of working invite links.

THE DATABASE FILE IS 0600, AND SO ARE ITS SIBLINGS. sqlite writes
accounts.db-wal and accounts.db-shm beside it, deletes them on the last close
and recreates them on the next write, so their mode is not something you set
once -- connect() re-asserts it every time. This box's umask is 002; a
database created the ordinary way would be group-readable, and a group-
readable file of password hashes is an offline cracking target sitting in a
directory that gets rsynced.

AND THE DIRECTORY IS SERVED OVER HTTP. data/dashboard is the static handler's
root -- index.html, the map layers and the crop thumbnails all come out of it
by name. The handler serves an ALLOW-list (dashboard.py's _static_allowed), so
accounts.db is not reachable today; PRIVATE_FILES below names the files that
must never be added to it, and the guard checks that they are not. The
allow-list is matched against the name the handler will actually OPEN, not
the one the client typed -- matching the typed one let a signed-in member ask
for /recent_crops/../accounts.db, which passes a prefix test and resolves to
this file.
"""

import argparse
import base64
import contextlib
import getpass
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import sys
import time

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The name every annotation written before there were accounts is read as.
# Imported, never re-spelled -- fn_audit owns the word, and a second copy of
# it here is the day the ledgers say one thing and the signup form guards
# another. In-tree and standard-library-only, like audit.py's import of it.
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
from fn_audit import LEGACY_AUTHOR                              # noqa: E402

OUT = os.path.join(REPO, 'data', 'dashboard')
DB_PATH = os.path.join(OUT, 'accounts.db')
ENV_PATH = os.path.join(REPO, '.env')

# Files this store owns inside the static handler's document root. They are
# not in dashboard.py's allow-list and must never be put there; the -wal in
# particular carries recently written rows in the clear.
PRIVATE_FILES = frozenset({
    'accounts.db', 'accounts.db-wal', 'accounts.db-shm', 'accounts.db-journal',
})

# 10s, matching leash_store.py. It is a busy_timeout, not a deadline: under
# WAL a reader never waits at all, and a writer only waits behind another
# writer, which here means two invites being minted in the same instant.
DB_TIMEOUT = 10
SCHEMA_VERSION = 3

# THREE TIERS, MOST TO LEAST. `owner` is the person whose machine this is:
# one account, named by DASHBOARD_USER, holding the things an admin should not
# casually hold -- who has been visiting, and the rest of the operator's own
# surface. `admin` runs the annotation project. `member` annotates.
#
# Ranked rather than compared, because the bug this shape prevents is the
# obvious one: nineteen places asked `role == 'admin'`, and a new tier ABOVE
# admin that any of them missed would have left the owner with less than an
# admin.
ROLES = ('owner', 'admin', 'member')
RANK = {'owner': 2, 'admin': 1, 'member': 0}


def at_least(role, want):
    """True when `role` carries everything `want` does."""
    return RANK.get(str(role or ''), -1) >= RANK.get(want, 99)


def is_admin(role):
    """An admin, or the owner who outranks one."""
    return at_least(role, 'admin')


def is_owner(role):
    return at_least(role, 'owner')
# ASCII only, and deliberately narrow. A username is compared case-folded, is
# printed on a page, and ends up in a URL when an admin edits somebody; every
# character class outside this set turns one of those three into a question
# ("is Ali and ALI the same person", "does this need escaping", "is that a
# space or a non-breaking space"). Nothing here needs the answers.
USERNAME_RE = re.compile(r'^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$')
USERNAME_MIN, USERNAME_MAX = 2, 32
# 12 rather than 8: this login is the only thing between the open internet of
# the tailnet and every image in the harvest, and the throttle below buys time
# against online guessing, not against a leaked database.
PASSWORD_MIN, PASSWORD_MAX = 12, 1024

# scrypt at 32 MiB and ~50ms on this machine. maxmem MUST be passed: OpenSSL
# defaults it to 32MB, which n=2**15,r=8 exceeds by a few hundred kilobytes,
# and the call raises "memory limit exceeded" rather than falling back.
SCRYPT_N, SCRYPT_R, SCRYPT_P = 1 << 15, 8, 1
SCRYPT_MAXMEM = 96 * 1024 * 1024
PBKDF2_ROUNDS = 600_000        # only used where the build has no scrypt
SALT_BYTES, DK_BYTES = 16, 32
TOKEN_BYTES = 32               # 256 bits of CSPRNG; see _token_hash()

INVITE_TTL_DEFAULT = 48 * 3600
INVITE_TTL_MIN, INVITE_TTL_MAX = 300, 30 * 24 * 3600

# Failed logins are counted per source (the server decides what a source is --
# an address, a username, or both). FREE attempts cost nothing, then the wait
# doubles. WINDOW is how long a quiet period has to be before the count is
# forgiven, and MAX_ROWS is the hard cap that keeps a table keyed by something
# an attacker chooses from growing without limit.
THROTTLE_FREE = 5
THROTTLE_BASE = 5
THROTTLE_MAX = 900
THROTTLE_WINDOW = 3600
THROTTLE_MAX_ROWS = 4096

ENV_USER = 'DASHBOARD_USER'
ENV_PASSWORD = 'DASHBOARD_PASSWORD'
ENV_PASSWORD_HASH = 'DASHBOARD_PASSWORD_HASH'
ENV_INVITE_TTL_HOURS = 'DASHBOARD_INVITE_TTL_HOURS'
DEFAULT_ADMIN = 'admin'

# Names a stranger would read as staff. Refused for everyone except the
# account named by DASHBOARD_USER, which may hold its own name.
IMPERSONABLE = frozenset({
    'admin', 'admins', 'administrator', 'root', 'staff', 'support',
    'moderator', 'mod', 'system', 'security', 'help', 'official',
    # the pages print the word now -- an "owner" in the members list who is
    # not the owner is the same trick as an "admin" who is not an admin
    'owner', 'owners',
    'biodiv', 'dashboard', 'streetdogs',
})


class AccountError(ValueError):
    """A refusal with a machine-readable reason.

    ``code`` is what the server switches on; ``str()`` is what a person reads.
    The codes are deliberately about the INPUT and never about the store: a
    login that fails returns None with no code at all, because "no such user"
    and "wrong password" have to be the same answer. The two codes that do
    admit an account exists -- username_taken, from create_user and
    redeem_invite -- are only reachable by someone already holding a valid
    one-time invite, and a signup form that cannot say "that name is taken" is
    a signup form nobody can use.
    """

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL,
    username_norm TEXT    NOT NULL UNIQUE,
    pw_hash       TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'member'
                          CHECK (role IN ('owner', 'admin', 'member')),
    active        INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at    INTEGER NOT NULL,
    last_login_at INTEGER,
    session_epoch INTEGER NOT NULL DEFAULT 1
);
-- username_norm's UNIQUE constraint is already an index, and it is the one
-- every login uses. These two are for the pages: the admin list asks for
-- admins, the user table sorts by when people joined.
CREATE INDEX IF NOT EXISTS users_role    ON users(role, active);
CREATE INDEX IF NOT EXISTS users_created ON users(created_at);

CREATE TABLE IF NOT EXISTS invites (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    token_hash TEXT    NOT NULL UNIQUE,
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    used_at    INTEGER,
    used_by    INTEGER REFERENCES users(id),
    revoked_at INTEGER,
    role       TEXT    NOT NULL DEFAULT 'member'
                       CHECK (role IN ('owner', 'admin', 'member')),
    note       TEXT
);
-- No ON DELETE clause on either key on purpose: with foreign_keys=ON the
-- default is RESTRICT, so a user who issued or redeemed an invite cannot be
-- deleted out from under the record of it. Accounts are retired with
-- set_active(), which keeps the trail.
CREATE INDEX IF NOT EXISTS invites_by      ON invites(created_by);
CREATE INDEX IF NOT EXISTS invites_used_by ON invites(used_by);
CREATE INDEX IF NOT EXISTS invites_expires ON invites(expires_at);
CREATE INDEX IF NOT EXISTS invites_open    ON invites(used_at, revoked_at);

CREATE TABLE IF NOT EXISTS throttle (
    source       TEXT    PRIMARY KEY,
    fails        INTEGER NOT NULL DEFAULT 0,
    first_at     INTEGER NOT NULL,
    last_at      INTEGER NOT NULL,
    locked_until INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS throttle_last   ON throttle(last_at);
CREATE INDEX IF NOT EXISTS throttle_locked ON throttle(locked_until);

-- DELEGATED WORK: "judge five hundred of these" as a record rather than a
-- conversation. The row is the ASSIGNMENT and nothing else -- who, how many,
-- on which surface, from when. It holds no progress count, deliberately: the
-- annotations are the truth about how much has been done, they live in the
-- ledgers, and a number cached here would be a second answer to a question
-- that already has one. Progress is counted from the ledgers on demand.
--
-- start_at is why the record exists at all. "Five hundred" means five hundred
-- MORE; counting an annotator's whole history would hand somebody with four
-- hundred already an instant eighty per cent, which is not what anyone means
-- by delegating work.
CREATE TABLE IF NOT EXISTS assignments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id),
    surface      TEXT    NOT NULL
                         CHECK (surface IN ('any','review','gate','leash')),
    target       INTEGER NOT NULL CHECK (target > 0),
    start_at     INTEGER NOT NULL,
    created_at   INTEGER NOT NULL,
    created_by   INTEGER NOT NULL REFERENCES users(id),
    due_at       INTEGER,
    note         TEXT    NOT NULL DEFAULT '',
    done_at      INTEGER,
    cancelled_at INTEGER
);
-- ONE OPEN JOB PER PERSON PER SURFACE, enforced by the database rather than
-- by whichever page happens to be writing. Two open targets on one surface is
-- two progress bars over one pile of work, and no answer to "am I done".
CREATE UNIQUE INDEX IF NOT EXISTS assignments_one_open
    ON assignments(user_id, surface)
 WHERE done_at IS NULL AND cancelled_at IS NULL;
CREATE INDEX IF NOT EXISTS assignments_user ON assignments(user_id);
CREATE INDEX IF NOT EXISTS assignments_open ON assignments(done_at,
                                                          cancelled_at);
"""


# ── the file ────────────────────────────────────────────────────────────────

def _secure(p):
    """Re-assert 0600 on the database and on the sidecars sqlite just made.

    The -wal and -shm inherit the main file's mode, which is why the O_EXCL
    create below matters -- but they are deleted on the last close and made
    again on the next write, so this cannot be a one-time fixup at install
    time. Failures are swallowed: a database on a filesystem that has no modes
    to set is still a working database, and refusing to serve over it would
    trade a hardening measure for an outage.
    """
    for suffix in ('', '-wal', '-shm', '-journal'):
        try:
            os.chmod(p + suffix, 0o600)
        except OSError:
            pass


def connect(path=None):
    """Open the store, creating it if this is the first login.

    A CONNECTION PER CALL, closed in the caller's finally, exactly as
    leash_store.py does it. The alternative -- one connection per thread in a
    threading.local -- looks cheaper and is not: sqlite3 connections carry a
    thread affinity check so they cannot be shared, and the server this feeds
    is a ThreadingHTTPServer speaking HTTP/1.0, which means a fresh thread per
    REQUEST. A per-thread cache would therefore open exactly as many
    connections as this does, and additionally leave each one to be closed
    whenever the garbage collector got round to the dead thread. What made
    per-call cheap enough for an auth check on every image request is that
    migrate() is a single PRAGMA read once the schema is current, not an
    executescript: open, look a user up and close measures 0.15ms here, against
    the 50ms one password hash costs.

    WAL so a reader (the gate checking a cookie) never blocks the writer (the
    same server recording a failed login) -- they are different threads of one
    process, and the default journal turns that into "database is locked" at
    exactly the moment a lot of images load at once. synchronous=FULL because
    an account created and then lost to a power cut is an invite spent for
    nothing. foreign_keys=ON because sqlite does not enforce them otherwise
    and an invite pointing at a user id that never existed is a lie the audit
    page would print.
    """
    p = path or DB_PATH
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(p):
        # Make the file ourselves, before sqlite can. sqlite creates a new
        # database 0644 & ~umask, and this machine's umask is 002 -- so the
        # ordinary path leaves every password hash group-readable for the
        # window between creation and any chmod. O_EXCL|0600 has no window.
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass                      # another thread beat us to it, fine
    con = sqlite3.connect(p, timeout=DB_TIMEOUT)
    con.row_factory = sqlite3.Row
    # Explicit transaction control. redeem_invite has to hold a write lock
    # from the moment it reads an invite to the moment it claims it, and the
    # implicit transaction sqlite3 opens for DML only cannot cover the read.
    con.isolation_level = None
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=FULL')
    con.execute('PRAGMA foreign_keys=ON')
    con.execute('PRAGMA busy_timeout=%d' % (DB_TIMEOUT * 1000,))
    migrate(con)
    _secure(p)
    return con


@contextlib.contextmanager
def _tx(con):
    """One write transaction, IMMEDIATE so the lock is taken at the start.

    A deferred transaction takes its write lock at the first write, which for
    read-then-write work means two racers can both pass the read and only
    discover each other at COMMIT -- as SQLITE_BUSY, after the decision has
    been made. IMMEDIATE makes the loser wait at BEGIN instead, and it is what
    makes redeem_invite's compare-and-set a real one.
    """
    con.execute('BEGIN IMMEDIATE')
    try:
        yield con
    except BaseException:
        con.execute('ROLLBACK')
        raise
    else:
        con.execute('COMMIT')


# ── v3: the owner tier ──────────────────────────────────────────────────────
# A migration is a snapshot, so its SQL is written out here rather than read
# back out of SCHEMA: SCHEMA moves on, and a migration that follows it stops
# describing the step it actually performs. Literal statements, every one of
# them, which is also what the store's no-built-SQL rule asks for.
_V3_USERS = """
CREATE TABLE users_v3 (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL,
    username_norm TEXT    NOT NULL UNIQUE,
    pw_hash       TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'member'
                          CHECK (role IN ('owner', 'admin', 'member')),
    active        INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at    INTEGER NOT NULL,
    last_login_at INTEGER,
    session_epoch INTEGER NOT NULL DEFAULT 1
)"""
_V3_USERS_COPY = """
INSERT INTO users_v3 (id, username, username_norm, pw_hash, role, active,
                      created_at, last_login_at, session_epoch)
SELECT id, username, username_norm, pw_hash, role, active,
       created_at, last_login_at, session_epoch FROM users"""
_V3_USERS_DROP = 'DROP TABLE users'
_V3_USERS_RENAME = 'ALTER TABLE users_v3 RENAME TO users'

_V3_INVITES = """
CREATE TABLE invites_v3 (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    token_hash TEXT    NOT NULL UNIQUE,
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    used_at    INTEGER,
    used_by    INTEGER REFERENCES users(id),
    revoked_at INTEGER,
    role       TEXT    NOT NULL DEFAULT 'member'
                       CHECK (role IN ('owner', 'admin', 'member')),
    note       TEXT
)"""
_V3_INVITES_COPY = """
INSERT INTO invites_v3 (id, token_hash, created_by, created_at, expires_at,
                        used_at, used_by, revoked_at, role, note)
SELECT id, token_hash, created_by, created_at, expires_at,
       used_at, used_by, revoked_at, role, note FROM invites"""
_V3_INVITES_DROP = 'DROP TABLE invites'
_V3_INVITES_RENAME = 'ALTER TABLE invites_v3 RENAME TO invites'


def _accepts_owner(con, table):
    """Has this table already been rebuilt around the new CHECK?"""
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,)).fetchone()
    return bool(row) and "'owner'" in (row[0] or '')


def _restore_high(con, name, high):
    """Put back the AUTOINCREMENT high-water mark a rebuild reset.

    A fresh table with rows copied into it carries a mark of the largest id
    copied, which is lower than the store's whenever the newest account has
    been removed -- and sqlite would then hand that id to the next person
    invited. An id is what a session cookie names.
    """
    want = high.get(name)
    if not want:
        return
    con.execute('UPDATE sqlite_sequence SET seq = ? '
                'WHERE name = ? AND seq < ?', (want, name, want))


def _add_owner_role(con):
    """v3: let `role` be 'owner' as well.

    A CHECK constraint cannot be altered in SQLite, so each table is rebuilt
    around it -- the documented procedure, in its order: foreign keys OFF (a
    no-op inside a transaction, so before it), rebuild, confirm the references
    still hold, and only then back on. assignments.user_id and
    invites.created_by both point at users, and dropping the table under them
    is exactly what the check is for.

    Idempotent: a table that already accepts the value is skipped, which makes
    it safe to re-run after an interrupted migration.
    """
    users, invites = (not _accepts_owner(con, 'users'),
                      not _accepts_owner(con, 'invites'))
    if not (users or invites):
        return
    # WHAT WAS ALREADY BROKEN BEFORE THIS RAN. The check below is here to
    # catch a rebuild that dropped a table out from under a reference; it
    # cannot tell that from damage the database walked in with, and blaming
    # the migration for the latter sends whoever reads the message looking in
    # the wrong file. Counted first, compared after.
    was = len(con.execute('PRAGMA foreign_key_check').fetchall())
    # AUTOINCREMENT's high-water mark lives in sqlite_sequence, and a rebuild
    # that copies rows into a fresh table resets it to the largest id it
    # copied -- so a store whose newest account was removed would hand that
    # id straight back to the next person invited. Ids are what a session
    # cookie names.
    # sqlite_sequence does not exist until an AUTOINCREMENT table has taken
    # its first row, which is why this asks rather than selects.
    high = {}
    if con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' "
                   "AND name = 'sqlite_sequence'").fetchone():
        high = {r['name']: r['seq'] for r in con.execute(
            'SELECT name, seq FROM sqlite_sequence').fetchall()}
    con.execute('PRAGMA foreign_keys = OFF')
    try:
        # Spelled out rather than looped: every statement handed to execute()
        # is a module constant by name, which is the store's rule about SQL --
        # a query assembled from a variable is how a username becomes SQL, and
        # a reader should not have to prove these particular variables held
        # literals.
        # The high-water mark is put back INSIDE the transaction that reset
        # it. Restoring it afterwards is a second write that a kill between
        # the two loses for good: the table is already rebuilt, so the next
        # start sees a v3 shape, does nothing, and hands a removed account's
        # id to the next person invited.
        if users:
            with _tx(con):
                con.execute(_V3_USERS)
                con.execute(_V3_USERS_COPY)
                con.execute(_V3_USERS_DROP)
                con.execute(_V3_USERS_RENAME)
                _restore_high(con, 'users', high)
        if invites:
            with _tx(con):
                con.execute(_V3_INVITES)
                con.execute(_V3_INVITES_COPY)
                con.execute(_V3_INVITES_DROP)
                con.execute(_V3_INVITES_RENAME)
                _restore_high(con, 'invites', high)
        broken = con.execute('PRAGMA foreign_key_check').fetchall()
        if len(broken) > was:
            raise AccountError(
                'migration_broke_references',
                'The v3 migration left %d dangling reference(s).'
                % (len(broken) - was,))
    finally:
        con.execute('PRAGMA foreign_keys = ON')
    # the indexes named in SCHEMA went with the dropped tables
    con.executescript(SCHEMA)


def schema_version(con):
    """The version stamped in the file, 0 for a database nothing has touched."""
    return int(con.execute('PRAGMA user_version').fetchone()[0])


def migrate(con, now=None):
    """Bring the schema up to SCHEMA_VERSION. Running it twice does nothing.

    The fast path is one PRAGMA read, because connect() calls this every time
    and connect() is on the path of every authenticated request.

    Two things record the version and they are not redundant. user_version
    lives in the file header and is what the gate above reads, so the check
    costs no query planning at all. The schema_version TABLE is the readable
    trail -- one row per migration with when it ran -- which is what you want
    in front of you when a database from a month ago will not open.
    """
    have = schema_version(con)
    if have == SCHEMA_VERSION:
        return have
    # Every statement is CREATE ... IF NOT EXISTS, so this is also the repair
    # path for a database that was interrupted halfway through its first
    # migration: it re-runs, adds what is missing, and stamps the version.
    # executescript() commits any open transaction before it starts, which is
    # why it is not wrapped in one.
    con.executescript(SCHEMA)
    # NOT a truthiness test on the header. The header is the fast gate, not
    # the evidence: a database restored through .dump and reload, or copied by
    # something that does not carry PRAGMA user_version, arrives with a v2
    # shape and a 0 in front of it -- and `have and have < 3` reads that 0 as
    # "nothing to do", stamps it v3, and leaves a CHECK that still refuses the
    # word 'owner'. _add_owner_role asks the tables themselves and returns at
    # once when they already accept it, so this costs a lookup on the path
    # that had nothing to do anyway.
    if have < SCHEMA_VERSION:
        _add_owner_role(con)
    with _tx(con):
        con.execute(
            'INSERT OR IGNORE INTO schema_version (version, applied_at) '
            'VALUES (?, ?)',
            (SCHEMA_VERSION, int(time.time() if now is None else now)))
        con.execute('PRAGMA user_version = %d' % (SCHEMA_VERSION,))
    return SCHEMA_VERSION


# ── passwords ───────────────────────────────────────────────────────────────

def _b64(raw):
    return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')


def _unb64(text):
    return base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))


def _scrypt_available():
    """True where OpenSSL was built with scrypt. Some minimal builds are not."""
    try:
        hashlib.scrypt(b'x', salt=b'x' * SALT_BYTES, n=2, r=1, p=1, dklen=16,
                       maxmem=SCRYPT_MAXMEM)
        return True
    except (ValueError, AttributeError):        # pragma: no cover - build dep
        return False


def _derive(algo, params, password, salt):
    """The digest for one algorithm and one set of parameters."""
    pw = password.encode('utf-8')
    if algo == 'scrypt':
        return hashlib.scrypt(pw, salt=salt, n=int(params['n']),
                              r=int(params['r']), p=int(params['p']),
                              dklen=DK_BYTES, maxmem=SCRYPT_MAXMEM)
    if algo == 'pbkdf2_sha256':
        return hashlib.pbkdf2_hmac('sha256', pw, salt, int(params['i']),
                                   dklen=DK_BYTES)
    raise ValueError('unknown algorithm %r' % (algo,))


def current_params():
    """(algo, params) this build would use for a password set right now."""
    if _scrypt_available():
        return 'scrypt', {'n': SCRYPT_N, 'r': SCRYPT_R, 'p': SCRYPT_P}
    return 'pbkdf2_sha256', {'i': PBKDF2_ROUNDS}


def _encode(algo, params, salt, digest):
    body = ','.join('%s=%d' % (k, params[k]) for k in sorted(params))
    return '$'.join((algo, body, _b64(salt), _b64(digest)))


def _parse_hash(encoded):
    """(algo, params, salt, digest) from a stored hash. Raises on nonsense.

    Self-describing on purpose. The cost of a password hash has to be able to
    go up -- that is the whole point of choosing one -- and a bare digest with
    the parameters compiled into the verifier means the day you raise them,
    every existing login stops working. With the parameters written next to
    the digest, an old hash keeps verifying and needs_rehash() quietly
    replaces it the next time its owner signs in.
    """
    parts = (encoded or '').split('$')
    if len(parts) != 4:
        raise ValueError('malformed password hash')
    algo, body, salt_b64, dk_b64 = parts
    if algo not in ('scrypt', 'pbkdf2_sha256'):
        raise ValueError('unknown algorithm %r' % (algo,))
    params = {}
    for piece in body.split(','):
        k, _, v = piece.partition('=')
        params[k] = int(v)          # ValueError on junk, which is the point
    return algo, params, _unb64(salt_b64), _unb64(dk_b64)


def hash_password(password):
    """Encode a password. A fresh random salt every time, no exceptions.

    A shared or derived salt means two people who chose the same password have
    the same digest, which turns "did anyone here use hunter2" into a lookup.
    """
    algo, params = current_params()
    salt = secrets.token_bytes(SALT_BYTES)
    return _encode(algo, params, salt, _derive(algo, params, password, salt))


def verify_hash(encoded, password):
    """Does this password produce this hash? Constant-time in the comparison.

    A hash that will not parse still costs a full derivation before returning
    False. That is not politeness to corrupt data -- it is so that a row whose
    hash got truncated cannot be told apart from a wrong password by how long
    the answer took.
    """
    try:
        algo, params, salt, want = _parse_hash(encoded)
    except (ValueError, TypeError, base64.binascii.Error):
        algo, params = current_params()
        salt, want = b'\x00' * SALT_BYTES, b'\x00' * DK_BYTES
        _derive(algo, params, password or '', salt)
        return False
    try:
        got = _derive(algo, params, password or '', salt)
    except (ValueError, OverflowError, MemoryError):
        return False           # parameters this build cannot honour
    return hmac.compare_digest(got, want)


def needs_rehash(encoded):
    """True if this hash was made with anything other than today's settings."""
    try:
        algo, params, salt, _ = _parse_hash(encoded)
    except (ValueError, TypeError, base64.binascii.Error):
        return True            # unreadable counts as stale, not as fine
    want_algo, want_params = current_params()
    return (algo != want_algo or params != want_params
            or len(salt) < SALT_BYTES)


_dummy = {'hash': None}


def _dummy_hash():
    """A hash of a password nobody knows, for logins with no user behind them.

    Built once and kept, because building it costs a real derivation and this
    is called on the path where somebody is guessing usernames.
    """
    if _dummy['hash'] is None:
        _dummy['hash'] = hash_password(secrets.token_urlsafe(32))
    return _dummy['hash']


# ── validation ──────────────────────────────────────────────────────────────

def normalise_username(name):
    """The form a username is compared and stored uniquely under.

    Case-folded, not lower-cased, and it does not matter here -- USERNAME_RE
    is ASCII, so the two agree -- but the folded form is the one that stays
    right if the charset is ever widened. Whitespace goes first: a name pasted
    out of a spreadsheet arrives with a trailing space, and "alice " and
    "alice" must not be two accounts.
    """
    return (name or '').strip().casefold()


def _reserved_for(env=None):
    """The one account allowed to hold LEGACY_AUTHOR: the .env admin.

    Reads the FILE when the environment does not already carry the variable.
    Normally it does -- the server calls load_env() in bootstrap() before the
    first request -- but a process that skipped that (a CLI subcommand, a test
    harness) would fall back to DEFAULT_ADMIN, which is the very name being
    reserved, and the reservation would quietly permit everyone on exactly the
    deployment it exists to protect. Into a scratch dict, never os.environ:
    resolving a username must not import the rest of .env into the process,
    and nothing here is printed.
    """
    got = _env_str(ENV_USER, env)
    if not got and env is None:
        scratch = {}
        try:
            load_env(env=scratch)
        except OSError:                  # unreadable .env is not a crash here
            scratch = {}
        got = _env_str(ENV_USER, scratch)
    return got or DEFAULT_ADMIN


def check_username(name, env=None):
    """Return the normalised username, or raise AccountError saying why not.

    ONE NAME IS RESERVED. Every annotation made before the dashboard had
    accounts carries no author and is read as LEGACY_AUTHOR -- 3,247 of them,
    and they are read that way at the review page's byline, in the audit
    statistics and in the dataset manifest's judged_by. That word is also an
    ordinary username, and nothing else here would stop somebody taking it:
    the charset allows it and username_taken only fires once an account
    already holds it. On a deployment whose admin is called something else --
    DASHBOARD_USER renamed, or data/ restored beside a fresh accounts.db --
    the name is free, and an invited member claiming it from the signup form
    would be credited with every judgement made before they existed while
    their own new rows became indistinguishable from the founder's. So it is
    refusable for everyone except the account .env actually names, which is
    the one person it has always meant.
    """
    raw = (name or '').strip()
    if not raw:
        raise AccountError('username_missing', 'Choose a username.')
    if len(raw) < USERNAME_MIN or len(raw) > USERNAME_MAX:
        raise AccountError(
            'username_length',
            'A username is %d to %d characters.' % (USERNAME_MIN,
                                                    USERNAME_MAX))
    if not USERNAME_RE.match(raw):
        raise AccountError(
            'username_charset',
            'A username may use letters, digits, dot, dash and underscore, '
            'and must start and end with a letter or digit.')
    norm = normalise_username(raw)
    # NAMES THAT SPEAK FOR THE PROJECT. This dashboard is public now and its
    # annotators are volunteers who mostly do not know each other; a sign-up
    # called `admin` or `support` can tell the others anything and be believed,
    # and no permission is needed to do it. The account that runs the
    # deployment is exempt -- it may hold its own name whatever that is.
    if norm in IMPERSONABLE and \
            norm != normalise_username(_reserved_for(env)):
        raise AccountError(
            'username_reserved',
            'That username speaks for the project. Pick one that is yours.')
    if norm == normalise_username(LEGACY_AUTHOR) and \
            norm != normalise_username(_reserved_for(env)):
        raise AccountError(
            'username_reserved',
            'That username is reserved. It is the name every annotation made '
            'before this dashboard had accounts is recorded under, and only '
            'the %s account may use it.' % (ENV_USER,))
    return norm


# WHAT A GUESSER TRIES FIRST. Not a composition rule -- those produce
# `Password1!` and a sticky note. Length is the thing that matters, and on top
# of it the small set of shapes that survive a length rule while surviving no
# guessing at all: the words everybody picks, one character held down, a walk
# along the keyboard, a run of digits, and the name of the site or the account
# the password is for.
COMMON_PASSWORDS = frozenset("""
password passwd pass letmein welcome iloveyou monkey dragon football
baseball superman batman trustno trustnoone sunshine princess michael
shadow master qwerty qwertyuiop asdfghjkl zxcvbnm changeme secret default
whatever nothing test testing admin administrator root toor login access
starwars pokemon liverpool arsenal chelsea barcelona ferrari corvette
freedom whatever hello google facebook computer internet samsung
qazwsx zaqwsx killer jordan hunter buster soccer harley
ranger daniel hannah thomas summer winter autumn spring january february
""".split())

# runs to walk in either direction when looking for a keyboard or counting
_RUNS = ('abcdefghijklmnopqrstuvwxyz', '01234567890',
         'qwertyuiop', 'asdfghjkl', 'zxcvbnm')


# The substitutions everybody makes when a form asks for a symbol and a
# digit. `P@ssw0rd1234` is `password` wearing exactly this, and it is one of
# the most-guessed strings there has ever been.
_LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's',
                       '7': 't', '8': 'b', '9': 'g', '@': 'a', '$': 's',
                       '!': 'i', '|': 'l', '+': 't'})


def _common_prefix(pw):
    """A common word with a year stuck on the end is the same one guess.

    `trustno1-2026` and `ch4ngem3!` are the shape a password policy produces:
    the word somebody thought of, plus whatever the form demanded. Matched as
    a prefix with a SHORT tail -- long enough to catch a date and a symbol,
    short enough that `masterpiece of cake` is still a password rather than
    `master` with decoration.
    """
    for reading in _readings(pw):
        for word in COMMON_PASSWORDS:
            if len(word) >= 4 and reading.startswith(word) \
                    and len(reading) - len(word) <= 6:
                return True
    return False


def _readings(pw):
    """Every way this password could be read as one of the common words.

    A blocklist of BASE words rather than decorated ones, because the
    decoration is the variable: `monkey`, `monkey123`, `M0nk3y!` and
    `m0nk3y_2026` are one guess wearing four coats. Both readings of `1` are
    tried -- it stands in for `l` and for `i`, and `L3tm31n` needs the second.
    """
    low = pw.lower()
    out = {low, re.sub(r'[^a-z0-9]', '', low), _stem(pw)}
    plain = low.rstrip('0123456789!@#$%^&*_.-')
    for one in ('l', 'i'):
        table = dict(_LEET)
        table[ord('1')] = one
        out.add(re.sub(r'[^a-z]', '', plain.translate(table)))
        # ...and the whole string folded, digits kept: a word with a year on
        # the end still has to be recognisable as that word
        out.add(re.sub(r'[^a-z0-9]', '', low.translate(table)))
    return {x for x in out if x}


def _stem(pw):
    """The word somebody actually chose, before they were made to decorate it.

    `letmein12345` is `letmein` with the tax paid; so is `Password!!` and so
    is `P@ssw0rd`. The blocklist has to see the word, not the decoration --
    otherwise it refuses the honest spelling and waves through the one an
    attacker actually tries.
    """
    plain = pw.lower().rstrip('0123456789!@#$%^&*_.-')
    return re.sub(r'[^a-z]', '', plain.translate(_LEET))


def _is_a_run(pw):
    """True when it is a walk along a keyboard or the digits, decoration aside.

    Not only end to end: `qwertyuiop12` is a whole keyboard row with two
    digits after it, which is one guess, not twelve characters. Anything whose
    non-decoration part is a run counts, and so does a run repeated.
    """
    low = pw.lower()
    heads = {low, re.sub(r'[^a-z0-9]', '', low),
             low.rstrip('0123456789!@#$%^&*_.-')}
    for row in _RUNS:
        back = row[::-1]
        for head in heads:
            if len(head) < 4:
                continue
            if head in row or head in back:
                return True
        # a repeated walk -- 'abcabcabcabc' is one idea, not twelve characters
        for n in range(3, len(low) // 2 + 1):
            head = low[:n]
            if (head in row or head in back) and low == head * (len(low) // n) \
                    + head[:len(low) % n]:
                return True
    # digits that simply count, in either direction and from anywhere
    digits = re.sub(r'\D', '', low)
    if len(digits) >= 6 and len(digits) >= len(re.sub(r'[^a-z0-9]', '', low)) - 2:
        step = {(int(b) - int(a)) % 10 for a, b in zip(digits, digits[1:])}
        if step in ({1}, {9}):
            return True
    return False


def check_password(password, username='', extra=()):
    """Return the password unchanged, or raise AccountError saying why not.

    The upper bound is not a style rule. scrypt hashes whatever it is handed,
    so an unbounded field is an unauthenticated request that makes the server
    do unbounded work -- one megabyte of 'a' per POST, as many POSTs as they
    like.

    The lower bound is not the whole story either. This dashboard is public
    now and its accounts belong to volunteers, so the rule has to refuse what
    a guesser actually tries: `password1234` and `aaaaaaaaaaaa` and
    `qwertyuiop12` all cleared a twelve-character minimum, and each is one
    guess. What it deliberately does NOT do is demand a symbol and a capital:
    that produces one predictable shape, not an unpredictable password.
    """
    pw = password if isinstance(password, str) else ''
    if len(pw) < PASSWORD_MIN:
        raise AccountError(
            'password_short',
            'A password needs at least %d characters.' % (PASSWORD_MIN,))
    if len(pw) > PASSWORD_MAX:
        raise AccountError(
            'password_long',
            'A password may be at most %d characters.' % (PASSWORD_MAX,))
    low = pw.lower()
    if _readings(pw) & COMMON_PASSWORDS or _common_prefix(pw):
        raise AccountError(
            'password_common',
            'That is one of the first passwords anybody guesses. Pick '
            'something nobody else would write down.')
    if len(set(low)) < 5:
        raise AccountError(
            'password_repetitive',
            'That is the same few characters over and over. A password needs '
            'at least five different ones.')
    if _is_a_run(pw):
        raise AccountError(
            'password_sequence',
            'That is a walk along the keyboard. Pick something that is not '
            'in order.')
    # the name of the site, or of the account it belongs to: the guesser
    # knows both
    for word in [username] + list(extra or ()) + ['dashboard', 'streetdogs',
                                                  'street dogs', 'biodiv',
                                                  'mapillary']:
        word = normalise_username(word or '')
        if len(word) >= 4 and word in low:
            raise AccountError(
                'password_guessable',
                'That contains %r, which anybody guessing already knows. '
                'Leave it out.' % (word,))
    return pw


def check_role(role):
    if role not in ROLES:
        raise AccountError('role_unknown',
                           'A role is one of: %s.' % (', '.join(ROLES),))
    return role


# ── users ───────────────────────────────────────────────────────────────────

def _public(row):
    """A user row as the rest of the program may see it: no hash, ever.

    The hash is dropped here and not at the call sites, so a new page that
    prints a user cannot leak one by forgetting to. Anything that genuinely
    needs the hash reads the column itself, inside this module.
    """
    if row is None:
        return None
    d = dict(row)
    d.pop('pw_hash', None)
    d['active'] = bool(d.get('active'))
    return d


def _row_by_norm(con, norm):
    return con.execute('SELECT * FROM users WHERE username_norm = ?',
                       (norm,)).fetchone()


def _resolve(con, who):
    """A user row from an id, a username, or a dict that has either.

    The server holds an id in a session and a username in a form, and making
    every function take both means neither caller has to look the other up
    first -- one round trip instead of two, on a path that runs per request.
    """
    if isinstance(who, dict):
        who = who.get('id', who.get('username'))
    if isinstance(who, bool):
        who = None
    if isinstance(who, int):
        return con.execute('SELECT * FROM users WHERE id = ?',
                           (who,)).fetchone()
    return _row_by_norm(con, normalise_username(who))


def _need(con, who):
    row = _resolve(con, who)
    if row is None:
        raise AccountError('no_such_user', 'No such account.')
    return row


def count_admins(con=None, path=None):
    """How many ACTIVE admins exist. The number the lockout guards read."""
    own = con is None
    if own:
        con = connect(path)
    try:
        return int(con.execute(
            "SELECT COUNT(*) c FROM users WHERE role IN ('owner','admin') "
            "AND active = 1"
        ).fetchone()['c'])
    finally:
        if own:
            con.close()


def _would_strand(con, row, role=None, active=None):
    """True if this edit would leave the dashboard with no way back in.

    Discovered the obvious way: an admin demoting themselves to member to see
    what the member view looked like. There was no other admin, the admin page
    stopped existing, and the only route back was the sqlite3 shell. Every
    edit that can remove the last active admin refuses instead.
    """
    was_admin = is_admin(row['role']) and row['active']
    if not was_admin:
        return False
    still = (is_admin(role if role is not None else row['role'])
             and (row['active'] if active is None else active))
    return not still and count_admins(con) <= 1


def create_user(username, password, role='member', active=True, now=None,
                path=None, con=None):
    """Make an account. Raises AccountError for anything the input got wrong."""
    norm = check_username(username)
    check_password(password, username=norm)
    check_role(role)
    # THE OWNER TIER COMES FROM .env AND NOWHERE ELSE -- the same rule
    # set_role and create_invite hold, held here too so that the tier has one
    # source rather than three guarded doors and an unguarded one.
    if role == 'owner':
        raise AccountError(
            'owner_not_grantable',
            'The owner is whoever DASHBOARD_USER names. Set it in .env and '
            'restart.')
    raw = (username or '').strip()
    own = con is None
    if own:
        con = connect(path)
    try:
        pw = hash_password(password)
        ts = int(time.time() if now is None else now)
        try:
            with _tx(con):
                cur = con.execute(
                    'INSERT INTO users (username, username_norm, pw_hash, '
                    '  role, active, created_at, session_epoch) '
                    'VALUES (?,?,?,?,?,?,1)',
                    (raw, norm, pw, role, 1 if active else 0, ts))
                uid = cur.lastrowid
        except sqlite3.IntegrityError:
            # The UNIQUE index is the arbiter, not a SELECT beforehand: two
            # signups racing on one name both pass a check-then-insert.
            raise AccountError('username_taken',
                               'That username is already taken.')
        return _public(con.execute('SELECT * FROM users WHERE id = ?',
                                   (uid,)).fetchone())
    finally:
        if own:
            con.close()


def get_user(who, path=None):
    """One account, or None. No hash in what comes back."""
    con = connect(path)
    try:
        return _public(_resolve(con, who))
    finally:
        con.close()


def list_users(path=None):
    """Every account, oldest first. What the admin page's table renders."""
    con = connect(path)
    try:
        return [_public(r) for r in con.execute(
            'SELECT * FROM users ORDER BY created_at, id')]
    finally:
        con.close()


def verify_password(username, password, now=None, touch=True, path=None):
    """The account for this pair, or None. Never says which half was wrong.

    THE SAME WORK HAPPENS EITHER WAY. A name nobody has is verified against
    _dummy_hash() and a disabled account is verified against its real one, so
    the derivation runs before every return and the answer takes the same
    ~50ms whether the account exists, is disabled, or simply has a different
    password. Without that, a login form is a user directory: the miss returns
    in microseconds and the hit does not, and you can read the difference over
    a LAN. What is NOT constant-time is the indexed lookup of the username
    itself, which is tens of microseconds against tens of milliseconds of
    scrypt -- three orders of magnitude under the noise floor of the network
    this is served over.

    A successful login stamps last_login_at, and upgrades the stored hash if
    it was made with weaker parameters than today's. That upgrade does NOT
    bump session_epoch: it is the same password, so the sessions it opened are
    still that person's.
    """
    pw = password if isinstance(password, str) else ''
    con = connect(path)
    try:
        row = None
        norm = normalise_username(username)
        if norm:
            row = _row_by_norm(con, norm)
        encoded = row['pw_hash'] if row is not None else _dummy_hash()
        ok = verify_hash(encoded, pw)
        if row is None or not ok or not row['active']:
            return None
        ts = int(time.time() if now is None else now)
        fresh = hash_password(pw) if needs_rehash(encoded) else None
        if touch:
            with _tx(con):
                if fresh:
                    con.execute('UPDATE users SET pw_hash = ? WHERE id = ?',
                                (fresh, row['id']))
                con.execute('UPDATE users SET last_login_at = ? WHERE id = ?',
                            (ts, row['id']))
            row = con.execute('SELECT * FROM users WHERE id = ?',
                              (row['id'],)).fetchone()
        return _public(row)
    finally:
        con.close()


def set_password(who, password, now=None, path=None, bump=True):
    """Replace an account's password and, by default, kill its live sessions.

    bump=True is the safe default because the reason a password gets changed
    is usually that somebody else might know the old one, and a session cookie
    minted under it outlives the change otherwise.
    """
    check_password(password, username=str(who or ''))
    con = connect(path)
    try:
        row = _need(con, who)
        pw = hash_password(password)
        with _tx(con):
            con.execute('UPDATE users SET pw_hash = ? WHERE id = ?',
                        (pw, row['id']))
            if bump:
                con.execute('UPDATE users SET session_epoch = session_epoch + 1'
                            ' WHERE id = ?', (row['id'],))
        return _public(con.execute('SELECT * FROM users WHERE id = ?',
                                   (row['id'],)).fetchone())
    finally:
        con.close()


def _heir_for(con, want, going):
    """Who inherits the rows a removed account is named on.

    The account doing the removing when the caller names one, the owner
    otherwise, and the oldest remaining admin when there is no owner. Never
    the account on its way out, which is the whole point.
    """
    try:
        want = int(want) if want else 0
    except (TypeError, ValueError):
        # A caller who names an heir in some other alphabet gets the default
        # one rather than a ValueError out of a delete that half happened.
        want = 0
    if want and want != int(going) and con.execute(
            'SELECT 1 FROM users WHERE id = ?', (want,)).fetchone():
        return want
    row = con.execute(
        "SELECT id FROM users WHERE role IN ('owner','admin') "
        "AND active = 1 AND id != ? "
        "ORDER BY role = 'owner' DESC, id ASC LIMIT 1", (going,)).fetchone()
    if row is None:
        raise AccountError(
            'no_heir',
            'There is no other administrator to inherit what this account '
            'handed out.')
    return row['id']


def delete_user(who, inherit_to=None, path=None):
    """Remove an account outright. Disabling is the reversible one.

    WHAT THIS DOES NOT TOUCH: the work. Verdicts carry the annotator's name in
    a ledger, and those rows stay exactly as they are -- the person judged
    those crops and still did. What goes is the ability to sign in, which is
    what "remove" means here. A name with no account behind it reads on the
    audit pages as it always did.

    Refused for the last active admin, the same as demoting one: an account
    store nobody can sign into as an admin is a dashboard nobody can
    administer.
    """
    con = connect(path)
    try:
        with _tx(con):
            row = _need(con, who)
            if row['role'] == 'owner':
                raise AccountError(
                    'owner_not_removable',
                    'That is the owner account. Point DASHBOARD_USER at '
                    'somebody else and restart if it should change hands.')
            if is_admin(row['role']) and row['active']:
                left = con.execute(
                    "SELECT COUNT(*) c FROM users WHERE role IN "
                    "('owner','admin') AND active = 1 AND id != ?",
                    (row['id'],)).fetchone()['c']
                if not left:
                    raise AccountError(
                        'last_admin',
                        'This is the last active admin. Promote somebody '
                        'else first.')
            # EVERYTHING THAT POINTS AT THEM HAS TO STOP FIRST. invites
            # references users twice -- who issued one, and who redeemed one
            # -- and an account is normally created BY redeeming an invite,
            # so the ordinary case is the one that fails: remove the volunteer
            # you invited and the row recording their invite still points at
            # them.
            #
            # What survives is what still means something. The line about a
            # redeemed invite keeps its timing and its role and loses only the
            # pointer to an account that no longer exists. An invite this
            # person ISSUED and nobody took is a link that leads nowhere now,
            # so it goes with them -- but one that was TAKEN is the record of
            # how somebody who still has an account got in, and that is not
            # this person's to take with them. Their own open jobs do go.
            con.execute('DELETE FROM assignments WHERE user_id = ?',
                        (row['id'],))
            # A DELEGATION OUTLIVES THE DELEGATOR. assignments.created_by is
            # NOT NULL and points here, so an admin who had ever handed
            # somebody a target could not be removed at all: the foreign key
            # refused and the page said "IntegrityError" and nothing else.
            # Their own jobs went with them on the line above; the jobs they
            # handed to OTHER people are other people's work and stay, with
            # whoever did the removing named as the one who handed them out.
            con.execute('UPDATE assignments SET created_by = ? '
                        'WHERE created_by = ?',
                        (_heir_for(con, inherit_to, row['id']), row['id']))
            con.execute('UPDATE invites SET used_by = NULL WHERE used_by = ?',
                        (row['id'],))
            # THE LIVE LINKS ONLY. Deleting every invite they ever issued
            # emptied the whole table on a deployment where one admin invited
            # everybody -- and left the members that admin brought in with no
            # record of how they got here, which is the one thing the invites
            # page is for. A used or revoked row keeps its timing and its
            # role and is re-pointed at the heir, the same as a delegation.
            con.execute('DELETE FROM invites WHERE created_by = ? '
                        'AND used_at IS NULL AND revoked_at IS NULL',
                        (row['id'],))
            con.execute('UPDATE invites SET created_by = ? WHERE created_by = ?',
                        (_heir_for(con, inherit_to, row['id']), row['id']))
            con.execute('DELETE FROM users WHERE id = ?', (row['id'],))
            return {'id': row['id'], 'username': row['username'],
                    'role': row['role']}
    finally:
        con.close()


def delete_invite(invite_id, path=None):
    """Drop an invite from the record entirely -- revoked, used or expired.

    revoke_invite() withdraws an OPEN one and leaves the row, which is the
    right thing while it still means something. This is for afterwards: a
    used invite is a line about a person who now has an account, and a
    revoked one is a line about a link that no longer exists, and neither
    needs to sit on the page forever.
    """
    con = connect(path)
    try:
        with _tx(con):
            n = con.execute('DELETE FROM invites WHERE id = ?',
                            (invite_id,)).rowcount
            return bool(n)
    finally:
        con.close()


def set_active(who, active, path=None):
    """Enable or disable an account. Disabling also revokes its sessions.

    A disabled account whose cookie still works is not disabled; the
    session_epoch bump is what makes the button mean what it says.
    """
    con = connect(path)
    try:
        row = _need(con, who)
        active = bool(active)
        # Counted INSIDE the write transaction. Outside it, two admins
        # demoting each other in the same second both count two and both
        # succeed, and the count they read was true when they read it.
        with _tx(con):
            row = _need(con, row['id'])
            # THE OWNER IS NOT LOCKED OUT BY AN ADMIN. Removing them and
            # demoting them are both refused, and leaving this one open let an
            # admin reach the same end by the third button: disable, and the
            # owner's next request lands on the login page. It comes back on a
            # restart -- ensure_admin restores the .env account -- which is a
            # remedy that needs a shell on the machine.
            if row['role'] == 'owner' and not active:
                raise AccountError(
                    'owner_not_disablable',
                    'That is the owner account. Point DASHBOARD_USER at '
                    'somebody else and restart if it should change hands.')
            if _would_strand(con, row, active=active):
                raise AccountError(
                    'last_admin',
                    'This is the last active admin. Promote somebody else '
                    'first.')
            con.execute('UPDATE users SET active = ? WHERE id = ?',
                        (1 if active else 0, row['id']))
            if not active:
                con.execute('UPDATE users SET session_epoch = session_epoch + 1'
                            ' WHERE id = ?', (row['id'],))
        return _public(con.execute('SELECT * FROM users WHERE id = ?',
                                   (row['id'],)).fetchone())
    finally:
        con.close()


def set_role(who, role, path=None):
    """Promote or demote. Refuses to demote the last active admin.

    THE OWNER TIER IS NOT GRANTED HERE, and not taken away here either. It
    belongs to whoever DASHBOARD_USER names, and ensure_admin() is what puts
    it there -- so it is held by somebody with the .env file, not by anybody
    who can reach the accounts page. Without this an admin could POST their
    own id with do=owner and take the top tier, then demote the person whose
    machine it is.
    """
    check_role(role)
    if role == 'owner':
        raise AccountError(
            'owner_not_grantable',
            'The owner is the account named by DASHBOARD_USER. Change that '
            'and restart; it cannot be handed out from this page.')
    con = connect(path)
    try:
        row = _need(con, who)
        if row['role'] == 'owner':
            raise AccountError(
                'owner_not_demotable',
                'That is the owner account. Its role follows DASHBOARD_USER '
                'and cannot be changed from this page.')
        with _tx(con):                # see set_active on why this is inside
            row = _need(con, row['id'])
            if _would_strand(con, row, role=role):
                raise AccountError(
                    'last_admin',
                    'This is the last active admin. Promote somebody else '
                    'first.')
            con.execute('UPDATE users SET role = ? WHERE id = ?',
                        (role, row['id']))
        return _public(con.execute('SELECT * FROM users WHERE id = ?',
                                   (row['id'],)).fetchone())
    finally:
        con.close()


def bump_session_epoch(who, path=None):
    """Sign one account out everywhere, without changing its password.

    The gate stores the epoch in the cookie and compares it on every request,
    so incrementing it here is what "sign out all devices" costs -- no table
    of live sessions to walk, and no way for a session to survive because its
    row was missed.
    """
    con = connect(path)
    try:
        row = _need(con, who)
        with _tx(con):
            con.execute('UPDATE users SET session_epoch = session_epoch + 1 '
                        'WHERE id = ?', (row['id'],))
        return _public(con.execute('SELECT * FROM users WHERE id = ?',
                                   (row['id'],)).fetchone())
    finally:
        con.close()


# ── delegated work ──────────────────────────────────────────────────────────
# Storage only. This module knows a target was set and never what has been
# done about it: the annotations live in the ledgers, and a count kept here
# would be a second answer to a question the ledgers already answer -- one
# that goes wrong quietly, the first time somebody undoes a verdict.

SURFACES = ('any', 'review', 'gate', 'leash')
# A cap, so a typo in the box cannot write a target nobody could ever meet
# and leave a person staring at 0.004%.
MAX_TARGET = 1_000_000


def check_surface(surface):
    """The surface a target is set on, or a message saying why it is not."""
    s = str(surface or '').strip().lower()
    if s not in SURFACES:
        return None, ('Pick one of: '
                      + ', '.join(SURFACES).replace('any', 'any surface'))
    return s, ''


def check_target(target):
    """A whole number of annotations, or a message saying why it is not."""
    try:
        n = int(str(target).strip())
    except (TypeError, ValueError):
        return None, 'How many annotations? That is not a number.'
    if n < 1:
        return None, 'A target of nothing is not a target.'
    if n > MAX_TARGET:
        return None, 'That is more annotations than the project has.'
    return n, ''


def assignment_state(row, now=None):
    """'cancelled' | 'done' | 'overdue' | 'open', in the order that decides.

    Overdue is a state of an OPEN target, never of a finished one: work that
    landed late is still work, and a row that flips to red the day after it
    was completed is a scoreboard nobody trusts.
    """
    ts = int(time.time() if now is None else now)
    if row['cancelled_at']:
        return 'cancelled'
    if row['done_at']:
        return 'done'
    if row['due_at'] and ts > row['due_at']:
        return 'overdue'
    return 'open'


def _assignment(con, row, now=None):
    """One row, with the names the pages need spliced in.

    The usernames come from a join rather than being stored on the row: a
    person who is renamed is the same person, and a copy taken at the moment
    the work was handed out would say otherwise for as long as the record
    lasts.
    """
    if row is None:
        return None
    d = dict(row)
    d['state'] = assignment_state(row, now=now)
    for key, col in (('username', 'user_id'), ('created_by_name',
                                               'created_by')):
        got = con.execute('SELECT username FROM users WHERE id = ?',
                          (d[col],)).fetchone()
        d[key] = got['username'] if got else None
    return d


def create_assignment(who, target, surface='any', created_by=None,
                      due_at=None, note='', now=None, path=None):
    """Hand somebody a number of annotations to reach.

    Refused for an account that is retired: a target nobody can sign in to
    work on is a row that will read as unmet for ever.

    The one-open-per-surface rule is the database's, not this function's --
    two racing admins both passing a read-then-write check is exactly the
    hole a partial unique index does not have.
    """
    ts = int(time.time() if now is None else now)
    surface, why = check_surface(surface)
    if why:
        return {'ok': False, 'message': why, 'assignment': None}
    target, why = check_target(target)
    if why:
        return {'ok': False, 'message': why, 'assignment': None}
    if due_at is not None:
        try:
            due_at = int(due_at)
        except (TypeError, ValueError):
            return {'ok': False, 'message': 'That is not a date.',
                    'assignment': None}
        if due_at <= ts:
            return {'ok': False, 'assignment': None,
                    'message': 'That date has already been and gone.'}
    con = connect(path)
    try:
        row = _need(con, who)
        if not row['active']:
            return {'ok': False, 'assignment': None,
                    'message': 'That account is retired — bring it back '
                               'before handing it work.'}
        author = _need(con, created_by)['id'] if created_by is not None \
            else row['id']
        try:
            with _tx(con):
                cur = con.execute(
                    'INSERT INTO assignments (user_id, surface, target, '
                    ' start_at, created_at, created_by, due_at, note) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (row['id'], surface, target, ts, ts, author, due_at,
                     str(note or '')[:400]))
                new_id = cur.lastrowid
        except sqlite3.IntegrityError:
            return {'ok': False, 'assignment': None,
                    'message': '%s already has an open target on %s. '
                               'Finish or cancel that one first.'
                               % (row['username'],
                                  'every surface' if surface == 'any'
                                  else 'the ' + surface + ' surface')}
        return {'ok': True, 'message': '',
                'assignment': _assignment(
                    con, con.execute('SELECT * FROM assignments WHERE id = ?',
                                     (new_id,)).fetchone(), now=ts)}
    finally:
        con.close()


def list_assignments(who=None, open_only=False, now=None, path=None):
    """Targets, newest first. One person's, or everybody's."""
    con = connect(path)
    try:
        # FOUR WHOLE STATEMENTS, not one built from pieces. The values were
        # already bound, but a query assembled by string concatenation is a
        # shape this module does not have anywhere -- and the check that keeps
        # it that way cannot tell a safe concatenation from the one somebody
        # adds next year with a username in it.
        if who is None:
            rows = con.execute(
                'SELECT * FROM assignments WHERE done_at IS NULL '
                '  AND cancelled_at IS NULL '
                'ORDER BY created_at DESC, id DESC').fetchall() if open_only \
                else con.execute(
                'SELECT * FROM assignments '
                'ORDER BY created_at DESC, id DESC').fetchall()
        else:
            row = _resolve(con, who)
            if row is None:
                return []
            rows = con.execute(
                'SELECT * FROM assignments WHERE user_id = ? '
                '  AND done_at IS NULL AND cancelled_at IS NULL '
                'ORDER BY created_at DESC, id DESC',
                (row['id'],)).fetchall() if open_only \
                else con.execute(
                'SELECT * FROM assignments WHERE user_id = ? '
                'ORDER BY created_at DESC, id DESC',
                (row['id'],)).fetchall()
        return [_assignment(con, r, now=now) for r in rows]
    finally:
        con.close()


def get_assignment(assignment_id, now=None, path=None):
    """One target by id, or None."""
    con = connect(path)
    try:
        return _assignment(
            con, con.execute('SELECT * FROM assignments WHERE id = ?',
                             (_int_id(assignment_id),)).fetchone(), now=now)
    finally:
        con.close()


def cancel_assignment(assignment_id, now=None, path=None):
    """Call the work off. A finished target cannot be un-finished.

    Cancelling does not delete: the row stays, so "we asked for five hundred
    and stopped it at ninety" is still answerable a month later. Cancelling
    something already cancelled is not an error -- two admins clicking at
    once is a race, not a mistake worth a message.
    """
    ts = int(time.time() if now is None else now)
    con = connect(path)
    try:
        with _tx(con):
            con.execute('UPDATE assignments SET cancelled_at = ? '
                        'WHERE id = ? AND cancelled_at IS NULL '
                        '  AND done_at IS NULL',
                        (ts, _int_id(assignment_id)))
        return _assignment(
            con, con.execute('SELECT * FROM assignments WHERE id = ?',
                             (_int_id(assignment_id),)).fetchone(), now=ts)
    finally:
        con.close()


def complete_assignment(assignment_id, now=None, path=None):
    """Stamp a target as reached. Idempotent, and never un-stamps.

    Called by whoever counted the ledgers and found the number met. The stamp
    is what makes "done on the 14th" survive an annotation being undone
    afterwards -- reached is a thing that happened, not a thing that is
    currently true.
    """
    ts = int(time.time() if now is None else now)
    con = connect(path)
    try:
        with _tx(con):
            con.execute('UPDATE assignments SET done_at = ? '
                        'WHERE id = ? AND done_at IS NULL '
                        '  AND cancelled_at IS NULL',
                        (ts, _int_id(assignment_id)))
        return _assignment(
            con, con.execute('SELECT * FROM assignments WHERE id = ?',
                             (_int_id(assignment_id),)).fetchone(), now=ts)
    finally:
        con.close()


def delete_assignment(assignment_id, path=None):
    """Remove the record of a target. Says whether there was one to remove.

    CALLING OFF AND DELETING ARE DIFFERENT ANSWERS to different questions.
    Cancelling says "we stopped wanting this", and the row stays so that
    "we asked for five hundred and stopped it at ninety" is still answerable
    next month. Deleting says "this should never have been asked for" -- a
    target set on the wrong person, or a typo of 5000 for 500 -- and leaves
    nothing behind, because a row nobody meant is worse than no row.

    IT TOUCHES NO ANNOTATION. Assignments live in this database and verdicts
    live in the ledgers, and nothing in either points at the other: progress
    is counted by asking the ledgers who wrote what, so deleting the target
    removes the asking and not one answer. Somebody who judged four hundred
    crops towards a target deleted by mistake has still judged four hundred
    crops, and re-setting the target counts them from the new start_at.

    Nothing cascades onto this row either -- it is referenced by nothing --
    so the delete is the whole operation.
    """
    con = connect(path)
    try:
        with _tx(con):
            cur = con.execute('DELETE FROM assignments WHERE id = ?',
                              (_int_id(assignment_id),))
        return cur.rowcount > 0
    finally:
        con.close()


def _int_id(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


# ── invites ─────────────────────────────────────────────────────────────────

def _token_hash(token):
    """SHA-256 of an invite token, hex. Plain and fast, on purpose.

    A password gets scrypt because people choose passwords and people are
    guessable. An invite token is 256 bits out of secrets.token_bytes, so
    there is no dictionary to run and nothing a slow KDF would buy -- it would
    only make redemption expensive and hand an unauthenticated caller a way to
    burn 50ms of server time per made-up link.
    """
    return hashlib.sha256((token or '').encode('utf-8')).hexdigest()


def invite_state(row, now=None):
    """'used' | 'revoked' | 'expired' | 'open' -- in the order that decides."""
    ts = int(time.time() if now is None else now)
    if row['used_at']:
        return 'used'
    if row['revoked_at']:
        return 'revoked'
    if row['expires_at'] <= ts:
        return 'expired'
    return 'open'


def _invite_ttl_default(env=None):
    src = os.environ if env is None else env
    try:
        hours = float(str(src.get(ENV_INVITE_TTL_HOURS, '')).strip())
    except (TypeError, ValueError):
        return INVITE_TTL_DEFAULT
    return int(hours * 3600) if hours > 0 else INVITE_TTL_DEFAULT


def create_invite(created_by, ttl=None, note='', role='member', now=None,
                  path=None, env=None):
    """Mint a one-time signup token. The plaintext comes back exactly once.

    The return value carries 'token'; nothing else ever will, because only its
    SHA-256 is written. If the admin closes the dialog without copying it, the
    invite is dead and they issue another one -- which is the correct trade,
    and the reason it says so on the page.

    created_by must be an ACTIVE ADMIN. The check is here and not only in the
    route, because "only an admin can invite" is a property of the store; a
    new route that forgets the check should not be able to break it.
    """
    check_role(role)
    if role == 'owner':
        raise AccountError(
            'owner_not_invitable',
            'An invite cannot grant the owner tier. It belongs to the account '
            'DASHBOARD_USER names.')
    ts = int(time.time() if now is None else now)
    ttl = int(_invite_ttl_default(env) if ttl is None else ttl)
    if ttl < INVITE_TTL_MIN or ttl > INVITE_TTL_MAX:
        raise AccountError(
            'ttl_range',
            'An invite lasts between %d minutes and %d days.'
            % (INVITE_TTL_MIN // 60, INVITE_TTL_MAX // 86400))
    note = (note or '').strip()[:200]
    con = connect(path)
    try:
        row = _need(con, created_by)
        if not is_admin(row['role']) or not row['active']:
            raise AccountError('not_admin', 'Only an admin can invite.')
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with _tx(con):
            cur = con.execute(
                'INSERT INTO invites (token_hash, created_by, created_at, '
                '  expires_at, role, note) VALUES (?,?,?,?,?,?)',
                (_token_hash(token), row['id'], ts, ts + ttl, role, note))
            iid = cur.lastrowid
        got = dict(con.execute('SELECT * FROM invites WHERE id = ?',
                               (iid,)).fetchone())
        got.pop('token_hash', None)   # the caller has the token; see above
        got['state'] = 'open'
        got['token'] = token
        return got
    finally:
        con.close()


def redeem_invite(token, username, password, now=None, path=None):
    """Turn a token into an account. Exactly one caller can win a given token.

    ATOMIC, and it has to be: an invite link goes into a group chat and two
    people open it in the same second. The claim is a compare-and-set --
    UPDATE ... WHERE used_at IS NULL AND revoked_at IS NULL AND expires_at > ?
    -- inside one IMMEDIATE transaction, and the loser sees rowcount 0 rather
    than a stale row it read a moment earlier.

    THE INVITE IS CLAIMED BEFORE THE USER IS CREATED. The other order loses
    the race properly and still leaves an account behind. Because it is one
    transaction, a failure after the claim -- a taken username, most likely --
    rolls the claim back and the invite is usable again, which is what the
    person retyping their name expects.
    """
    norm = check_username(username)
    check_password(password, username=norm)
    raw = (username or '').strip()
    ts = int(time.time() if now is None else now)
    th = _token_hash(token)
    # Outside the transaction: this is the expensive part and holding a write
    # lock across 50ms of scrypt would serialise every concurrent signup.
    pw = hash_password(password)
    con = connect(path)
    try:
        with _tx(con):
            row = con.execute('SELECT * FROM invites WHERE token_hash = ?',
                              (th,)).fetchone()
            if row is None:
                raise AccountError('invite_unknown',
                                   'That invite link is not valid.')
            # create_invite refuses to mint one of these, so a row that says
            # 'owner' came from somewhere other than this program -- a hand
            # edit, or a database written by a build that had no owner tier.
            # Loud rather than clamped to 'admin': the person redeeming would
            # get an account, and nobody would be told the link claimed more
            # than it should.
            if row['role'] == 'owner':
                raise AccountError(
                    'invite_owner',
                    'That invite link is not valid.')
            cur = con.execute(
                'UPDATE invites SET used_at = ? WHERE id = ? '
                '  AND used_at IS NULL AND revoked_at IS NULL '
                '  AND expires_at > ?', (ts, row['id'], ts))
            if cur.rowcount != 1:
                # Say WHICH, because the person holding a dead link needs to
                # know whether to ask for a new one or to sign in instead.
                raise AccountError(
                    'invite_' + invite_state(row, ts),
                    {'used': 'That invite link has already been used.',
                     'revoked': 'That invite link was withdrawn.',
                     'expired': 'That invite link has expired.'}.get(
                         invite_state(row, ts),
                         'That invite link is not valid.'))
            try:
                ins = con.execute(
                    'INSERT INTO users (username, username_norm, pw_hash, '
                    '  role, active, created_at, session_epoch) '
                    'VALUES (?,?,?,?,1,?,1)',
                    (raw, norm, pw, row['role'], ts))
            except sqlite3.IntegrityError:
                raise AccountError('username_taken',
                                   'That username is already taken.')
            con.execute('UPDATE invites SET used_by = ? WHERE id = ?',
                        (ins.lastrowid, row['id']))
            uid = ins.lastrowid
        return _public(con.execute('SELECT * FROM users WHERE id = ?',
                                   (uid,)).fetchone())
    finally:
        con.close()


def list_invites(path=None, now=None, open_only=False):
    """Every invite, newest first, each with its state and who took it.

    No token_hash in what comes back. It is not the token, but it is the
    lookup key for one, and there is no page that needs it.
    """
    ts = int(time.time() if now is None else now)
    con = connect(path)
    try:
        rows = con.execute(
            'SELECT i.*, c.username AS created_by_name, '
            '       u.username AS used_by_name '
            'FROM invites i '
            'LEFT JOIN users c ON c.id = i.created_by '
            'LEFT JOIN users u ON u.id = i.used_by '
            'ORDER BY i.created_at DESC, i.id DESC')
        out = []
        for r in rows:
            d = dict(r)
            d['state'] = invite_state(r, ts)
            d.pop('token_hash', None)
            if open_only and d['state'] != 'open':
                continue
            out.append(d)
        return out
    finally:
        con.close()


def set_invite_expiry(invite_id, ttl=None, at=None, now=None, path=None):
    """Move when an invite link stops working. One of ttl or at, not both.

    ttl is seconds FROM NOW; at is the moment itself. The window is measured
    from now rather than from the day the link was issued, which is the whole
    point of the call: a link that expired last week has no live window left
    to extend, and measuring from created_at would refuse the one case that
    sends anybody looking for this button.

    OPEN AND EXPIRED ONLY. A used invite is spent -- moving its expiry would
    read on the page as though the account it made could be taken back, and it
    cannot. A revoked one was withdrawn on purpose; mint a new link rather
    than quietly undoing that, so the trail keeps saying what happened.

    Reviving an expired link is not a way in by itself. The plaintext token
    was shown once, at creation, and is not stored -- so this only matters to
    somebody who already holds the link, which is exactly the person the admin
    is trying to let in.
    """
    if (ttl is None) == (at is None):
        raise AccountError(
            'expiry_unclear',
            'Say either how long from now the link should last, or when it '
            'should stop working -- not both.')
    ts = int(time.time() if now is None else now)
    try:
        want = int(ts + float(ttl)) if at is None else int(at)
    except (TypeError, ValueError):
        raise AccountError('expiry_range', 'That is not a time.')
    if want - ts < INVITE_TTL_MIN or want - ts > INVITE_TTL_MAX:
        raise AccountError(
            'expiry_range',
            'An invite lasts between %d minutes and %d days from now. To '
            'stop one sooner than that, withdraw it.'
            % (INVITE_TTL_MIN // 60, INVITE_TTL_MAX // 86400))
    con = connect(path)
    try:
        with _tx(con):
            row = con.execute('SELECT * FROM invites WHERE id = ?',
                              (invite_id,)).fetchone()
            if row is None:
                raise AccountError('invite_unknown', 'No such invite.')
            state = invite_state(row, ts)
            if state == 'used':
                raise AccountError(
                    'invite_used',
                    'That invite was already redeemed, so when it expires no '
                    'longer decides anything.')
            if state == 'revoked':
                raise AccountError(
                    'invite_revoked',
                    'That invite was withdrawn. Issue a new link rather than '
                    'bringing this one back.')
            # used_at IS NULL in the WHERE as well as the check above: the
            # read and the write are one transaction, but the condition that
            # matters is worth stating where the write happens.
            con.execute('UPDATE invites SET expires_at = ? WHERE id = ? '
                        'AND used_at IS NULL AND revoked_at IS NULL',
                        (want, invite_id))
        d = dict(con.execute('SELECT * FROM invites WHERE id = ?',
                             (invite_id,)).fetchone())
        d['state'] = invite_state(d, ts)
        d.pop('token_hash', None)
        return d
    finally:
        con.close()


def revoke_invite(invite_id, now=None, path=None):
    """Withdraw an unused invite. Revoking a used one is refused, not silent.

    A revoke that "succeeded" on an invite somebody had already redeemed would
    read on the page as though the account it created were gone too. It is
    not, and set_active() is the thing that removes an account.
    """
    ts = int(time.time() if now is None else now)
    con = connect(path)
    try:
        row = con.execute('SELECT * FROM invites WHERE id = ?',
                          (invite_id,)).fetchone()
        if row is None:
            raise AccountError('invite_unknown', 'No such invite.')
        state = invite_state(row, ts)
        if state == 'used':
            raise AccountError(
                'invite_used',
                'That invite was already redeemed. Disable the account '
                'instead.')
        if state == 'revoked':
            d = dict(row)
            d['state'] = 'revoked'
            d.pop('token_hash', None)
            return d                     # idempotent: a second click is fine
        with _tx(con):
            con.execute('UPDATE invites SET revoked_at = ? WHERE id = ? '
                        'AND used_at IS NULL', (ts, invite_id))
        d = dict(con.execute('SELECT * FROM invites WHERE id = ?',
                             (invite_id,)).fetchone())
        d['state'] = invite_state(d, ts)
        d.pop('token_hash', None)
        return d
    finally:
        con.close()


# ── throttling ──────────────────────────────────────────────────────────────

def _lock_for(fails):
    """Seconds of lockout after ``fails`` failures. Doubling, capped."""
    over = fails - THROTTLE_FREE
    if over <= 0:
        return 0
    return min(THROTTLE_BASE * (2 ** (over - 1)), THROTTLE_MAX)


def _state(row, ts):
    if row is None:
        return {'fails': 0, 'locked': False, 'locked_until': 0,
                'retry_after': 0}
    locked_until = int(row['locked_until'] or 0)
    retry = max(0, locked_until - ts)
    return {'fails': int(row['fails']), 'locked': retry > 0,
            'locked_until': locked_until, 'retry_after': retry}


def throttle_state(source, now=None, path=None, con=None):
    """How much trouble this source is in. Read-only; safe on the hot path."""
    ts = int(time.time() if now is None else now)
    own = con is None
    if own:
        con = connect(path)
    try:
        return _state(con.execute('SELECT * FROM throttle WHERE source = ?',
                                  (str(source),)).fetchone(), ts)
    finally:
        if own:
            con.close()


def _count(con, src, ts):
    """One more attempt against ``src``, inside an already-open transaction.

    Returns the state AFTER the increment with 'was_locked' carrying the
    state BEFORE it -- the two readings a caller can need, taken while the
    write lock is held, so no third party can slip between them.
    """
    row = con.execute('SELECT * FROM throttle WHERE source = ?',
                      (src,)).fetchone()
    before = _state(row, ts)
    if row is None or ts - int(row['last_at']) > THROTTLE_WINDOW:
        fails, first = 1, ts
    else:
        fails, first = int(row['fails']) + 1, int(row['first_at'])
    until = ts + _lock_for(fails)
    con.execute(
        'INSERT INTO throttle (source, fails, first_at, last_at, '
        '  locked_until) VALUES (?,?,?,?,?) '
        'ON CONFLICT(source) DO UPDATE SET fails=excluded.fails, '
        '  first_at=excluded.first_at, last_at=excluded.last_at, '
        '  locked_until=excluded.locked_until',
        (src, fails, first, ts, until))
    _prune(con, ts)
    out = _state(con.execute('SELECT * FROM throttle WHERE source = ?',
                             (src,)).fetchone(), ts)
    out['was_locked'] = before['locked']
    return out


def record_failure(source, now=None, path=None):
    """Count one failed attempt and return the resulting lockout.

    The count restarts after THROTTLE_WINDOW of quiet, so somebody who
    mistyped their password twice last week does not start today one attempt
    from a fifteen-minute wait.

    For a caller that has ALREADY decided the attempt failed. A caller that
    is about to spend 40ms deciding wants reserve_attempt() below instead.
    """
    ts = int(time.time() if now is None else now)
    src = str(source)
    con = connect(path)
    try:
        with _tx(con):
            out = _count(con, src, ts)
        return out
    finally:
        con.close()


def reserve_attempt(source, now=None, path=None):
    """Count an attempt BEFORE it is checked, and say whether the source was
    already locked when it arrived.

        {'fails', 'locked', 'locked_until', 'retry_after', 'was_locked'}

    CHECK-THEN-ACT IS NOT A THROTTLE ON A THREADED SERVER. Reading
    throttle_state(), spending ~40ms deriving a scrypt hash, and only then
    calling record_failure() leaves a 40ms window in which every other thread
    reads the same pre-failure counter and passes the same check -- and this
    store feeds a ThreadingHTTPServer, one thread per request. Fired one at a
    time the free budget was the 6 password checks the doubling schedule
    intends; fired all at once it was 30 to 37. The lockout still capped the
    damage, but one burst per lockout window bought roughly 3,500 guesses a
    day from a single address instead of about 96.

    So the counting and the reading are ONE transaction: the Nth caller in a
    burst reads N, not 0, and the lock the (N-1)th wrote is already in the
    row by the time the Nth looks.

    WAS_LOCKED IS THE STATE BEFORE THIS ATTEMPT, and it is the one to branch
    on. Refusing anything whose lock is non-zero AFTER the increment would
    mean an account that has just crossed the threshold can never be checked
    again: every later attempt would extend its own lockout before the
    password was looked at, including the correct one, and the real user
    would be shut out permanently.
    """
    ts = int(time.time() if now is None else now)
    src = str(source)
    con = connect(path)
    try:
        with _tx(con):
            out = _count(con, src, ts)
        return out
    finally:
        con.close()


def clear_failures(source, path=None):
    """Forget a source, on a login that worked. Returns rows removed."""
    con = connect(path)
    try:
        with _tx(con):
            cur = con.execute('DELETE FROM throttle WHERE source = ?',
                              (str(source),))
        return cur.rowcount
    finally:
        con.close()


def _prune(con, ts):
    """Drop what is stale, then whatever is left over the cap. Returns rows.

    Called from record_failure, so the bound holds without anybody
    remembering to run a cleanup: the key is a string an attacker chooses,
    and a table keyed on attacker input with no ceiling is a disk-filling
    primitive dressed up as a security feature.
    """
    gone = con.execute(
        'DELETE FROM throttle WHERE last_at < ? AND locked_until < ?',
        (ts - THROTTLE_WINDOW, ts)).rowcount
    n = int(con.execute('SELECT COUNT(*) c FROM throttle').fetchone()['c'])
    if n > THROTTLE_MAX_ROWS:
        # Oldest first: the rows worth keeping under pressure are the ones
        # somebody is hammering right now.
        gone += con.execute(
            'DELETE FROM throttle WHERE source IN ('
            '  SELECT source FROM throttle ORDER BY last_at ASC LIMIT ?)',
            (n - THROTTLE_MAX_ROWS,)).rowcount
    return gone


def prune_throttle(now=None, path=None):
    """The same sweep, callable on its own (a cron, or a slow-day rebuild)."""
    ts = int(time.time() if now is None else now)
    con = connect(path)
    try:
        with _tx(con):
            return _prune(con, ts)
    finally:
        con.close()


# ── the .env admin ──────────────────────────────────────────────────────────

def load_env(path=None, env=None):
    """Read .env into the environment. Already-set variables WIN.

    The dashboard runs under systemd with no Environment= beyond
    PYTHONUNBUFFERED, so nothing puts .env in front of the process -- it has
    to read the file itself or DASHBOARD_PASSWORD is simply not there and the
    gate has no credential to check.

    python-dotenv where it is installed; a small parser where it is not, so a
    fresh clone with only the standard library still boots with a login. In
    both cases an existing variable is left alone: a value passed by systemd
    or by the shell is a deliberate override of the file.

    Returns True if the file was found. Nothing is printed either way -- this
    file holds a hundred API keys, and a helpful "loaded X=Y" is how they end
    up in a journal.
    """
    p = path or ENV_PATH
    if not os.path.exists(p):
        return False
    target = os.environ if env is None else env
    try:
        from dotenv import dotenv_values
    except ImportError:
        dotenv_values = None
    if dotenv_values is not None:
        try:
            for k, v in dotenv_values(p).items():
                if v is not None and k not in target:
                    target[k] = v
            return True
        except Exception:
            pass          # fall through to the parser below rather than die
    try:
        with open(p, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('export '):
                    line = line[7:].lstrip()
                k, sep, v = line.partition('=')
                k = k.strip()
                if not sep or not k:
                    continue
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
                    v = v[1:-1]
                if k not in target:
                    target[k] = v
    except OSError:
        return False
    return True


def _env_str(name, env=None, strip=True):
    src = os.environ if env is None else env
    v = src.get(name)
    if not isinstance(v, str):
        return ''
    # A password is stripped of line endings only. An editor adds those; a
    # person may well have chosen a password that ends in a space, and
    # silently trimming it means a password that works in their manager and
    # not here.
    return v.strip() if strip else v.strip('\r\n')


def admin_username(env=None):
    """The username the .env admin will have. DEFAULT_ADMIN if unset."""
    return _env_str(ENV_USER, env) or DEFAULT_ADMIN


def admin_configured(env=None):
    """Is there a credential in the environment at all?

    False is the state the server has to survive without going down: the
    source watcher re-execs this process unattended when a file changes, so a
    build that refused to start without a password would take the dashboard
    off the air with nobody watching. The gate serves the login page and a
    line naming the variable to set, and no data at all.
    """
    return bool(_env_str(ENV_PASSWORD_HASH, env)
                or _env_str(ENV_PASSWORD, env, strip=False))


def ensure_admin(path=None, now=None, env=None):
    """Make the .env admin real, and keep it in step with .env. Idempotent.

    Returns a dict the server can render straight onto the "nobody can log in
    yet" page:

        action  'unset'     no credential configured; nothing was written
                'refused'   configured but unusable; detail says what to fix
                'created'   the account did not exist and now does
                'updated'   the .env credential changed; hash replaced
                'unchanged' already in step
        ok      True when there IS a usable admin afterwards
        detail  one sentence, naming variables and never values

    NEVER LOCKS ANYONE OUT. If the row exists but was disabled or demoted, it
    is put back: .env is the master credential, and the recovery story for
    "I demoted myself" has to be something other than a sqlite shell.

    NEVER CREATES A SECOND ADMIN BY ACCIDENT. The lookup is on the normalised
    username, so 'Admin' on Tuesday and 'admin' on Wednesday are one account
    and the second start updates it instead of adding a rival. Renaming
    DASHBOARD_USER outright DOES create the new account -- that is what was
    asked for -- and the old one is DEMOTED to admin and named in 'demoted',
    rather than being deleted behind the operator's back or left holding a
    tier that neither the pages nor the store will take back off it.

    A CHANGED PASSWORD REVOKES LIVE SESSIONS. That is the whole reason to
    change it, and a cookie minted under the old one outlives the edit
    otherwise.
    """
    ts = int(time.time() if now is None else now)
    name = admin_username(env)
    given_hash = _env_str(ENV_PASSWORD_HASH, env)
    given_pw = _env_str(ENV_PASSWORD, env, strip=False)
    out = {'action': 'unset', 'ok': False, 'username': name, 'user_id': None,
           'admins': 0, 'others': [], 'demoted': [], 'detail': ''}

    con = connect(path)
    try:
        out['admins'] = count_admins(con)
        if not (given_hash or given_pw):
            out['detail'] = (
                'No dashboard credential is configured. Set %s (and '
                'optionally %s) in .env, then restart.'
                % (ENV_PASSWORD, ENV_USER))
            # An admin created by an earlier run with a credential that has
            # since been removed from .env still works, and saying otherwise
            # would send somebody looking for a fault that is not there.
            out['ok'] = out['admins'] > 0
            return out
        try:
            # with THIS env, not the process's: the reserved-name rule asks
            # who the admin is, and a test (or a re-exec) that passes an
            # environment in has to be judged against the one it passed
            norm = check_username(name, env=env)
        except AccountError as e:
            out.update(action='refused',
                       detail='%s is not a usable username: %s' % (ENV_USER,
                                                                   e.message))
            out['ok'] = out['admins'] > 0
            return out
        if given_hash:
            try:
                _parse_hash(given_hash)
            except (ValueError, TypeError, base64.binascii.Error):
                out.update(
                    action='refused',
                    detail='%s is not a hash this build can read. Make one '
                           'with "accounts.py --hash".' % (ENV_PASSWORD_HASH,))
                out['ok'] = out['admins'] > 0
                return out
            want = given_hash
        else:
            try:
                check_password(given_pw)
            except AccountError as e:
                out.update(action='refused',
                           detail='%s is not usable: %s' % (ENV_PASSWORD,
                                                            e.message))
                out['ok'] = out['admins'] > 0
                return out
            want = None                 # decided below, against the stored one

        # EXACTLY ONE OWNER, AND IT IS THIS ONE. Renaming DASHBOARD_USER
        # promotes the new account; without this the old one keeps the tier
        # and becomes unreachable -- set_role refuses to demote an owner and
        # delete_user refuses to remove one, so the only way back would be a
        # sqlite shell. The message on those two refusals says to point
        # DASHBOARD_USER somewhere else and restart, which is this.
        stale = [r['username'] for r in con.execute(
            "SELECT username FROM users WHERE role = 'owner' "
            'AND username_norm != ?', (norm,)).fetchall()]
        if stale:
            with _tx(con):
                con.execute("UPDATE users SET role = 'admin' "
                            "WHERE role = 'owner' AND username_norm != ?",
                            (norm,))
            out['demoted'] = stale

        row = _row_by_norm(con, norm)
        if row is None:
            pw = want or hash_password(given_pw)
            with _tx(con):
                cur = con.execute(
                    'INSERT INTO users (username, username_norm, pw_hash, '
                    "  role, active, created_at, session_epoch) "
                    "VALUES (?,?,?,'owner',1,?,1)",
                    ((name or '').strip(), norm, pw, ts))
                uid = cur.lastrowid
            out.update(action='created', ok=True, user_id=uid,
                       admins=count_admins(con),
                       detail='Created the %s admin from .env.' % (name,))
        else:
            # "Did the credential change?" is answered differently by the two
            # variables. A hash compares byte for byte; a password can only be
            # checked by verifying it against what is stored, and that same
            # verification tells us whether the stored hash is merely old.
            if want is not None:
                changed = row['pw_hash'] != want
                stale = False
            else:
                matched = verify_hash(row['pw_hash'], given_pw)
                changed = not matched
                stale = matched and needs_rehash(row['pw_hash'])
                if changed or stale:
                    want = hash_password(given_pw)
            # THE .env ACCOUNT IS THE OWNER. It is the person whose machine
            # this is, which is a different thing from an admin who runs the
            # annotation project -- and without this, every start would
            # quietly demote them back.
            restore = (row['role'] != 'owner') or (not row['active'])
            if changed or stale or restore:
                with _tx(con):
                    if changed or stale:
                        con.execute('UPDATE users SET pw_hash = ? WHERE id = ?',
                                    (want, row['id']))
                    if restore:
                        con.execute("UPDATE users SET role = 'owner', "
                                    'active = 1 WHERE id = ?', (row['id'],))
                    if changed or restore:
                        # A rehash at the same password keeps its sessions; a
                        # new password or a re-enabled account does not.
                        con.execute('UPDATE users SET session_epoch = '
                                    'session_epoch + 1 WHERE id = ?',
                                    (row['id'],))
                bits = []
                if changed:
                    bits.append('password changed in .env')
                elif stale:
                    bits.append('password hash upgraded')
                if restore:
                    bits.append('account re-enabled as admin')
                out.update(action='updated', ok=True, user_id=row['id'],
                           admins=count_admins(con),
                           detail='Updated the %s admin (%s).'
                                  % (name, '; '.join(bits)))
            else:
                out.update(action='unchanged', ok=True, user_id=row['id'],
                           detail='The %s admin is in step with .env.' % (name,))
        others = [r['username'] for r in con.execute(
            "SELECT username FROM users WHERE role IN ('owner','admin') "
            "AND active = 1 "
            'AND username_norm != ? ORDER BY username_norm', (norm,))]
        out['others'] = others
        out['admins'] = count_admins(con)
        return out
    finally:
        con.close()


# ── the command line ────────────────────────────────────────────────────────

def _fmt_when(ts):
    import datetime as dt
    return dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts \
        else '-'


def _ask_password(prompt='password: '):
    """Read a password from the terminal. NEVER from argv.

    Everything on a command line is in /proc and in whatever shell history the
    operator keeps, readable by every other account on the box. A password
    that has to be typed twice is a small tax next to that.
    """
    a = getpass.getpass(prompt)
    b = getpass.getpass('again: ')
    if a != b:
        raise AccountError('mismatch', 'The two entries did not match.')
    return check_password(a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--db', default=DB_PATH)
    ap.add_argument('--ensure-admin', action='store_true',
                    help='create or update the .env admin')
    ap.add_argument('--hash', action='store_true',
                    help='print a hash to paste into ' + ENV_PASSWORD_HASH)
    ap.add_argument('--invite', action='store_true', help='mint a token')
    ap.add_argument('--as-user', default=None,
                    help='which admin issues the invite (default: the .env one)')
    ap.add_argument('--hours', type=float, default=None)
    ap.add_argument('--note', default='')
    ap.add_argument('--role', choices=ROLES, default='member')
    ap.add_argument('--invites', action='store_true', help='list invites')
    ap.add_argument('--revoke', type=int, metavar='ID')
    ap.add_argument('--disable', metavar='USER')
    ap.add_argument('--enable', metavar='USER')
    ap.add_argument('--set-password', metavar='USER')
    ap.add_argument('--sign-out', metavar='USER',
                    help='revoke every live session for one account')
    ap.add_argument('--unlock', nargs='?', const='*', metavar='SOURCE',
                    help='clear the login lockout: one source, or all of '
                         'them. The lockout has no other way out -- it is '
                         'cleared by a SUCCESSFUL login, which is the thing '
                         'it is refusing.')
    a = ap.parse_args(argv)

    try:
        if a.hash:
            print(hash_password(_ask_password()))
            return 0
        if a.ensure_admin:
            load_env()
            got = ensure_admin(path=a.db)
            print('%s: %s' % (got['action'], got['detail']))
            if got.get('demoted'):
                print('no longer the owner: ' + ', '.join(got['demoted']))
            if got['others']:
                print('other active admins: ' + ', '.join(got['others']))
            return 0 if got['ok'] else 1
        if a.set_password:
            set_password(a.set_password, _ask_password(), path=a.db)
            print('password changed; live sessions signed out')
            return 0
        if a.sign_out:
            bump_session_epoch(a.sign_out, path=a.db)
            print('signed out everywhere')
            return 0
        if a.unlock:
            # THE WAY BACK IN. Every other exit from a lockout is a
            # successful login, and a lockout is precisely the refusal of
            # one -- so an operator locked out of their own dashboard had
            # sqlite as their only remedy, on a live database, with nothing
            # on the page telling them what had happened.
            con = connect(a.db)
            try:
                with _tx(con):
                    if a.unlock == '*':
                        n = con.execute('DELETE FROM throttle').rowcount
                        print('cleared %d lockout(s)' % (n,))
                    else:
                        # NOT "does it contain a colon". Every IPv6 address
                        # does, so `--unlock 2001:db8::1` was read as an
                        # already-prefixed key, matched nothing, and said so
                        # in the same words a successful clear uses.
                        src = a.unlock if a.unlock.startswith(('ip:', 'pw:')) \
                            else 'ip:' + a.unlock
                        n = con.execute(
                            'DELETE FROM throttle WHERE source = ?',
                            (src,)).rowcount
                        if n:
                            print('cleared %d lockout(s) for %s' % (n, src))
                        else:
                            # The operator is locked out and guessing at the
                            # key. Show them what is actually held.
                            print('no lockout for %s' % (src,))
                            for r in con.execute(
                                    'SELECT source FROM throttle '
                                    'ORDER BY source').fetchall():
                                print('  held: %s' % (r['source'],))
            finally:
                con.close()
            return 0
        if a.disable or a.enable:
            u = set_active(a.disable or a.enable, bool(a.enable), path=a.db)
            print('%s is now %s' % (u['username'],
                                    'enabled' if u['active'] else 'disabled'))
            return 0
        if a.invite:
            load_env()
            who = a.as_user or admin_username()
            ttl = int(a.hours * 3600) if a.hours else None
            inv = create_invite(who, ttl=ttl, note=a.note, role=a.role,
                                path=a.db)
            print('token (shown once): ' + inv['token'])
            print('expires %s, role %s' % (_fmt_when(inv['expires_at']),
                                           inv['role']))
            return 0
        if a.revoke:
            inv = revoke_invite(a.revoke, path=a.db)
            print('invite %d is %s' % (inv['id'], inv['state']))
            return 0
        if a.invites:
            rows = list_invites(path=a.db)
            print('%d invite(s)  [%s]' % (len(rows), a.db))
            for r in rows:
                print('  #%-4d %-8s issued %s by %-12s expires %s  %s'
                      % (r['id'], r['state'], _fmt_when(r['created_at']),
                         r['created_by_name'] or '?',
                         _fmt_when(r['expires_at']), r['note'] or ''))
            return 0

        users = list_users(path=a.db)
        print('%d account(s)  [%s]' % (len(users), a.db))
        for u in users:
            print('  %-20s %-7s %-9s joined %s  last seen %s'
                  % (u['username'], u['role'],
                     'active' if u['active'] else 'disabled',
                     _fmt_when(u['created_at']), _fmt_when(u['last_login_at'])))
        if not users:
            print('nobody yet -- run --ensure-admin after setting %s in .env'
                  % (ENV_PASSWORD,))
        return 0
    except AccountError as e:
        print('refused (%s): %s' % (e.code, e.message), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
