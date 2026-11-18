"""AUDIT_PY / DASH_PY / FA_PY: run a test against an edited COPY of
tools/dashboard/audit.py, tools/dashboard/dashboard.py or
tools/detect/fn_audit.py without installing it.

WHY NOT JUST INSTALL IT. The dashboard re-execs itself the moment one of its
watched files is saved, and lazily imports tools/detect modules while it
serves: an edit to test IS a restart of the live page, and a half-finished
one is an outage. So a change is written to a copy first and the suites are
pointed at it.

HOW. install(repo) makes the copy stand in for the live file inside this
process, for both ways a test reaches it:

  * an import (`import audit`, `import dashboard as dash`, a
    spec_from_file_location of the live path) compiles the COPY's source
    under the LIVE path -- so its __file__, and every REPO-relative path it
    derives at import, are exactly what they will be once it is installed;
  * an open() of the live path for READING (a source scan) reads the copy.
    A write to it is refused: nothing here may edit a watched file.

Nothing happens when none of the three variables is set, so a test calls
install() unconditionally only through its own `if` on them, and a plain
run is byte for byte what it was. Child processes are not covered: a test
that spawns a python to import one of these files tests the live one there.
"""

import builtins
import importlib.machinery
import io
import os
import sys

# env var -> the file it stands in for, relative to the repo root
ENV = {'AUDIT_PY': ('tools', 'dashboard', 'audit.py'),
       'DASH_PY': ('tools', 'dashboard', 'dashboard.py'),
       'FA_PY': ('tools', 'detect', 'fn_audit.py')}
_DONE = {}


def install(repo):
    """{live path: copy path} now standing in (empty when none is set)."""
    want = {}
    for env, rel in ENV.items():
        p = os.environ.get(env)
        live = os.path.realpath(os.path.join(repo, *rel))
        if p and os.path.realpath(p) != live:
            if not os.path.isfile(p):
                raise SystemExit(f'{env}={p} is not a file')
            want[live] = os.path.realpath(p)
    if not want or _DONE:
        return dict(_DONE)
    _DONE.update(want)
    real_open = builtins.open

    def opener(f, *a, **k):
        if isinstance(f, (str, bytes, os.PathLike)):
            rp = os.path.realpath(os.fsdecode(f))
            if rp in _DONE:
                mode = a[0] if a else k.get('mode', 'r')
                if any(c in mode for c in 'wax+'):
                    raise PermissionError(f'{rp} is stood in for by '
                                          f'{_DONE[rp]}; refusing to write it')
                f = _DONE[rp]
        return real_open(f, *a, **k)
    builtins.open = opener
    io.open = opener

    loader = importlib.machinery.SourceFileLoader
    real_code = loader.get_code

    def get_code(self, fullname):
        rp = os.path.realpath(self.path)
        if rp in _DONE:
            with real_open(_DONE[rp], 'rb') as fh:
                return compile(fh.read(), rp, 'exec', dont_inherit=True)
        return real_code(self, fullname)
    loader.get_code = get_code
    # a module imported before this ran is the live one: drop it, so the
    # next import is the copy (a test calls install() before importing them)
    for name in ('audit', 'dashboard', 'fn_audit'):
        m = sys.modules.get(name)
        if m is not None and os.path.realpath(
                getattr(m, '__file__', '') or '') in _DONE:
            del sys.modules[name]
    for live, copy in sorted(want.items()):
        print(f'stand-in: {os.path.relpath(live, repo)} <- {copy}')
    return dict(_DONE)
