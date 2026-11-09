#!/usr/bin/env python3
"""DETECT_ROOT env override: one process can write a separate store.

A sweep with a candidate detector runs with DETECT_ROOT pointing at its own
store, so the live store (named by data/detect_root.txt) and everything that
reads it stay untouched. Every reader of the pointer -- store, status,
worklist -- must agree, or a sweep would write its shards to one root and its
status or worklist to another.

    python tools/detect/tests/adv_detect_root_env.py
"""

import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import status  # noqa: E402
import store  # noqa: E402
import worklist  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def no_env():
    os.environ.pop('DETECT_ROOT', None)


def main():
    no_env()
    try:
        file_root = store.get_detect_root()
    except store.StoreError:
        file_root = None
    status_file_root = status.detect_root()
    wl_file_root = (worklist.resolve_detect_root(
        argparse.Namespace(detect_root=None)) if file_root else None)

    # e1: no env -> every reader answers what it answered before
    if file_root:
        check(
            'e1 store without env reads the pointer file', file_root == open(
                store._REPO_ROOT / 'data' / 'detect_root.txt').read().strip())
        check('e1 status agrees with store without env',
              status_file_root == file_root.rstrip('/'), status_file_root)
        check('e1 worklist agrees with store without env',
              wl_file_root.rstrip('/') == file_root.rstrip('/'), wl_file_root)
    else:
        print('skip e1: no data/detect_root.txt on this machine')

    with tempfile.TemporaryDirectory() as td:
        alt = os.path.join(td, 'alt_store')
        os.environ['DETECT_ROOT'] = alt + '/'
        try:
            # e2: env wins, the same answer from all three, trailing / dropped
            check('e2 store honours DETECT_ROOT',
                  store.get_detect_root() == alt, store.get_detect_root())
            check('e2 status honours DETECT_ROOT',
                  status.detect_root() == alt, status.detect_root())
            check(
                'e2 status.json lands under DETECT_ROOT',
                status.default_status_path() == os.path.join(
                    alt, 'status.json'), status.default_status_path())
            check(
                'e2 worklist honours DETECT_ROOT',
                worklist.resolve_detect_root(
                    argparse.Namespace(detect_root=None)) == alt)
            check(
                'e2 --detect-root still beats the env',
                worklist.resolve_detect_root(
                    argparse.Namespace(detect_root='/x/y')) == '/x/y')
            check(
                'e2 store paths follow the env',
                store.pair_dir(store.get_detect_root(), 2, 'r', 'c',
                               'd').startswith(alt + os.sep))

            # e3: a relative env is refused, never resolved against the cwd
            os.environ['DETECT_ROOT'] = 'data/detect_alt'
            try:
                store.get_detect_root()
                check('e3 store refuses a relative DETECT_ROOT', False)
            except store.StoreError:
                check('e3 store refuses a relative DETECT_ROOT', True)

            # e4: an empty env means unset
            os.environ['DETECT_ROOT'] = '   '
            if file_root:
                check('e4 blank DETECT_ROOT falls back to the file',
                      store.get_detect_root() == file_root)
        finally:
            no_env()

    if file_root:
        check('e5 unsetting restores the file root',
              store.get_detect_root() == file_root)
    print(f'{len(FAILS)} failed' if FAILS else 'all DETECT_ROOT checks passed')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
