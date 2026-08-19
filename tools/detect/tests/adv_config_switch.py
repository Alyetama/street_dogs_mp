#!/usr/bin/env python3
"""A config switch written any natural way flips, and never dies in silence.

    python tools/detect/tests/adv_config_switch.py

cfg_bool() exists because cfg() returns strings only, so `"llm_page": true`
-- the natural JSON spelling -- fell through to the default and the feature
silently stayed as it was (d0c6de82). The pre-ship review found the same
fall-through one spelling over: JSON 1 and 0 (numbers, not strings) hit the
default with no warning printed, which for a switch means "on" written the
natural way leaves the page off with nothing on stderr to say why. And an
unreadable value -- 'banana' in the environment variable, a list in the file
-- was swallowed whole.

The contract this file holds cfg_bool to:

  * every spelling a person would write flips the switch: JSON true/false,
    JSON 1/0, and the string forms '1'/'0'/'true'/'false'/'yes'/'no'/
    'on'/'off' in any case, in the file or the environment;
  * the environment beats the file, and an EMPTY environment variable
    falls through to the file rather than reading as false;
  * a value that is present but unreadable warns on stderr, naming the
    source, and then uses the default -- a written key never dies silently.

The config path is redirected into a temp directory; the machine's real
dashboard.config.json is never read and never touched.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(REPO, 'tools', 'dashboard'),
           os.path.join(REPO, 'tools', 'detect')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dashboard as db                                          # noqa: E402

ENV = 'DASHBOARD_ADV_SWITCH_TEST'
KEY = 'adv_switch_test'


def run_case(tmp, file_val, env_val, default):
    """cfg_bool under one (file value, env value) pair; returns (got, stderr).

    A sentinel object marks 'key absent from the file'.
    """
    cfg = os.path.join(tmp, 'cfg.json')
    if file_val is ABSENT:
        with open(cfg, 'w') as fh:
            json.dump({}, fh)
    else:
        with open(cfg, 'w') as fh:
            json.dump({KEY: file_val}, fh)
    # a fresh mtime per case, or the cache would answer for the previous one
    db._cfg_cache.update(mtime=None, data={})
    old_env = os.environ.pop(ENV, None)
    if env_val is not None:
        os.environ[ENV] = env_val
    old_path = db.CFG_PATH
    db.CFG_PATH = cfg
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            got = db.cfg_bool(KEY, default, env=ENV)
    finally:
        db.CFG_PATH = old_path
        db._cfg_cache.update(mtime=None, data={})
        os.environ.pop(ENV, None)
        if old_env is not None:
            os.environ[ENV] = old_env
    return got, err.getvalue()


ABSENT = object()


def main():
    bad = []
    tmp = tempfile.mkdtemp(prefix='adv_config_switch_')
    try:
        # every spelling a person would write, in the file
        for val, want in ((True, True), (False, False), (1, True), (0, False),
                          ('1', True), ('0', False), ('true', True),
                          ('false', False), ('yes', True), ('no', False),
                          ('on', True), ('off', False), ('TRUE', True),
                          (' On ', True)):
            for default in (True, False):
                got, err = run_case(tmp, val, None, default)
                if got is not want:
                    bad.append(f'file {val!r} with default {default} read as '
                               f'{got}, want {want} — a switch written the '
                               f'natural way did not flip')
                elif err:
                    bad.append(f'file {val!r} warned ({err.strip()!r}) about '
                               f'a value that reads fine')
        # ...and in the environment, which beats the file
        for val, want in (('1', True), ('0', False), ('true', True),
                          ('off', False)):
            got, _ = run_case(tmp, not want, val, not want)
            if got is not want:
                bad.append(f'env {val!r} over file {not want} read as {got}, '
                           f'want {want} — the environment lost to the file')
        # an EMPTY env var is "unset", not "false": the file decides
        got, _ = run_case(tmp, True, '', False)
        if got is not True:
            bad.append('an empty environment variable overrode the file '
                       'value instead of falling through to it')
        # a key absent everywhere is the default, silently
        for default in (True, False):
            got, err = run_case(tmp, ABSENT, None, default)
            if got is not default or err:
                bad.append(f'an absent key gave {got!r} with stderr '
                           f'{err.strip()!r}; want the default, silently')
        # present but unreadable: the default, WITH a warning naming the source
        for file_val, env_val, src_word in (
                ('banana', None, 'cfg.json'),
                ([True], None, 'cfg.json'),
                (2.5, None, 'cfg.json'),
                (True, 'banana', ENV)):
            got, err = run_case(tmp, file_val, env_val, False)
            if got is not False:
                bad.append(f'unreadable ({file_val!r}, env {env_val!r}) read '
                           f'as {got}, not the default')
            if not err:
                bad.append(f'unreadable ({file_val!r}, env {env_val!r}) '
                           f'produced no stderr warning — a written key died '
                           f'in silence, the exact failure cfg_bool exists '
                           f'to remove')
            elif src_word not in err:
                bad.append(f'the warning does not name where the bad value '
                           f'came from: {err.strip()!r}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('a switch flips under every natural spelling, the environment '
          'beats the file, and an unreadable value warns before defaulting')
    return 0


if __name__ == '__main__':
    sys.exit(main())
